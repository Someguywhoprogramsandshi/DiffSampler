#include <windows.h>
#include <shlobj.h>
#include <dxgi.h>
#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <world/harvest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cwctype>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>
#include <vector>


namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

namespace ds {
constexpr char VERSION[] = "1.0.6";
constexpr wchar_t MODEL_RELATIVE_PATH[] = L"model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx";
constexpr int SR = 44100;
constexpr int HOP = 512;
constexpr int N_FFT = 2048;
constexpr int N_MELS = 128;
constexpr double FMIN = 40.0;
constexpr double FMAX = 16000.0;
constexpr double HOP_MS = HOP * 1000.0 / SR;
constexpr float MEL_FLOOR = -11.512925464970229f; // log(1e-5)
constexpr int BUCKET = 64;
constexpr wchar_t PIPE_NAME[] = L"\\\\.\\pipe\\DiffSampler.Native.v1";
constexpr wchar_t MUTEX_NAME[] = L"Local\\DiffSampler.Native.Worker.v1";
constexpr unsigned int IDLE_DEFAULT = 900;
constexpr size_t IPC_BUFFER_SIZE = 64 * 1024;
constexpr uint32_t IPC_RENDER = 1;
constexpr uint32_t IPC_INFO = 2;
constexpr uint32_t IPC_STOP = 3;
constexpr uint32_t IPC_PING = 4;

static std::mutex g_logMutex;

std::wstring envW(const wchar_t* name, const std::wstring& fallback = L"") {
    DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
    if (!n) return fallback;
    std::wstring value(n, L'\0');
    DWORD got = GetEnvironmentVariableW(name, value.data(), n);
    if (!got) return fallback;
    value.resize(got);
    return value;
}

std::string wideToUtf8(const std::wstring& s) {
    if (s.empty()) return {};
    int n = WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, s.data(), (int)s.size(), nullptr, 0, nullptr, nullptr);
    if (n <= 0) throw std::runtime_error("Invalid Unicode argument.");
    std::string out((size_t)n, '\0');
    if (!WideCharToMultiByte(CP_UTF8, WC_ERR_INVALID_CHARS, s.data(), (int)s.size(), out.data(), n, nullptr, nullptr))
        throw std::runtime_error("Could not convert command-line argument to UTF-8.");
    return out;
}

std::wstring utf8ToWide(const std::string& s) {
    if (s.empty()) return {};
    int n = MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), nullptr, 0);
    if (n <= 0) throw std::runtime_error("Invalid UTF-8 path in worker request.");
    std::wstring out((size_t)n, L'\0');
    if (!MultiByteToWideChar(CP_UTF8, MB_ERR_INVALID_CHARS, s.data(), (int)s.size(), out.data(), n))
        throw std::runtime_error("Could not convert worker path.");
    return out;
}

fs::path pathFromUtf8(const std::string& s) { return fs::path(utf8ToWide(s)); }

fs::path exePath() {
    std::wstring buf(32768, L'\0');
    DWORD n = GetModuleFileNameW(nullptr, buf.data(), (DWORD)buf.size());
    if (!n || n >= buf.size()) throw std::runtime_error("Could not determine executable path.");
    buf.resize(n);
    return fs::path(buf);
}

fs::path appDir() {
    std::wstring base = envW(L"LOCALAPPDATA");
    if (base.empty()) base = envW(L"USERPROFILE", L".") + L"\\.cache";
    fs::path p = fs::path(base) / L"DiffSampler";
    std::error_code ec;
    fs::create_directories(p, ec);
    return p;
}

void log(const std::string& message) noexcept {
    try {
        std::lock_guard<std::mutex> lock(g_logMutex);
        fs::path p = appDir() / L"daemon.log";
        std::error_code ec;
        if (fs::exists(p, ec) && fs::file_size(p, ec) > 1024 * 1024) fs::remove(p, ec);
        SYSTEMTIME st{}; GetLocalTime(&st);
        std::ofstream f(p, std::ios::binary | std::ios::app);
        if (f) {
            f << std::setfill('0') << std::setw(2) << st.wHour << ':'
              << std::setw(2) << st.wMinute << ':' << std::setw(2) << st.wSecond
              << ' ' << message << "\r\n";
        }
    } catch (...) {}
}

std::string basenameUtf8(const std::string& p) {
    try { return wideToUtf8(pathFromUtf8(p).filename().wstring()); }
    catch (...) { return p; }
}

double parseDouble(const std::vector<std::string>& a, size_t i, double fallback) {
    if (i >= a.size() || a[i].empty()) return fallback;
    try {
        size_t used = 0;
        double v = std::stod(a[i], &used);
        return used == a[i].size() && std::isfinite(v) ? v : fallback;
    } catch (...) { return fallback; }
}

int parseInt(const std::wstring& s, int fallback) {
    if (s.empty()) return fallback;
    try { return std::stoi(s); } catch (...) { return fallback; }
}

const std::string B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
const std::map<char, int> NOTE = {{'C',0},{'D',2},{'E',4},{'F',5},{'G',7},{'A',9},{'B',11}};

double noteToHz(const std::string& raw) {
    std::string s = raw;
    auto first = s.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return 440.0;
    s = s.substr(first);
    try {
        size_t used = 0;
        double midi = std::stod(s, &used);
        if (used == s.size()) return 440.0 * std::pow(2.0, (midi - 69.0) / 12.0);
    } catch (...) {}
    if (s.empty()) return 440.0;
    char letter = (char)std::toupper((unsigned char)s[0]);
    auto it = NOTE.find(letter);
    if (it == NOTE.end()) return 440.0;
    int note = it->second;
    size_t i = 1;
    while (i < s.size() && (s[i] == '#' || s[i] == 'b')) {
        note += s[i] == '#' ? 1 : -1;
        ++i;
    }
    if (i >= s.size()) return 440.0;
    try {
        size_t used = 0;
        int octave = std::stoi(s.substr(i), &used);
        if (used != s.size() - i) return 440.0;
        int midi = 12 * (octave + 1) + note;
        return 440.0 * std::pow(2.0, (midi - 69.0) / 12.0);
    } catch (...) { return 440.0; }
}

std::vector<int> decodePitchbend(const std::string& s) {
    std::vector<int> out;
    size_t begin = 0;
    int partIndex = 0;
    while (begin <= s.size()) {
        size_t end = s.find('#', begin);
        if (end == std::string::npos) end = s.size();
        std::string part = s.substr(begin, end - begin);
        if ((partIndex & 1) == 0) {
            for (size_t i = 0; i + 1 < part.size(); i += 2) {
                size_t a = B64.find(part[i]), b = B64.find(part[i + 1]);
                if (a == std::string::npos || b == std::string::npos) continue;
                int v = (int)((a << 6) | b);
                out.push_back(v >= 2048 ? v - 4096 : v);
            }
        } else if (!part.empty() && !out.empty()) {
            int count = 0;
            try { count = std::clamp(std::stoi(part), 0, 1000000); } catch (...) { count = 0; }
            if (count) out.insert(out.end(), (size_t)count, out.back());
        }
        ++partIndex;
        if (end == s.size()) break;
        begin = end + 1;
    }
    return out;
}

struct Flags { double g = 0.0; double t = 0.0; };
Flags parseFlags(const std::string& s) {
    Flags out;
    for (size_t i = 0; i < s.size();) {
        char c = s[i];
        if (c == 'g' || c == 't') {
            size_t j = i + 1;
            while (j < s.size() && (s[j] == '-' || s[j] == '+' || s[j] == '.' || std::isdigit((unsigned char)s[j]))) ++j;
            try {
                double v = std::stod(s.substr(i + 1, j - i - 1));
                if (std::isfinite(v)) (c == 'g' ? out.g : out.t) = v;
            } catch (...) {}
            i = j;
        } else ++i;
    }
    return out;
}

// The upstream UTAU renderer uses a Slaney mel scale and Slaney area normalization.
double hzToMel(double f) {
    constexpr double fsp = 200.0 / 3.0;
    constexpr double minHz = 1000.0;
    const double minMel = minHz / fsp;
    const double logstep = std::log(6.4) / 27.0;
    return f >= minHz ? minMel + std::log(std::max(f, 1e-9) / minHz) / logstep : f / fsp;
}
double melToHz(double m) {
    constexpr double fsp = 200.0 / 3.0;
    constexpr double minHz = 1000.0;
    const double minMel = minHz / fsp;
    const double logstep = std::log(6.4) / 27.0;
    return m >= minMel ? minHz * std::exp(logstep * (m - minMel)) : fsp * m;
}

struct MelBank {
    std::array<double, N_MELS> centers{};
    std::array<float, N_MELS * (N_FFT / 2 + 1)> weights{};
    std::array<int, N_MELS> firstBin{}, lastBin{};
    MelBank() {
        constexpr int bins = N_FFT / 2 + 1;
        std::array<double, N_MELS + 2> mf{};
        double lo = hzToMel(FMIN), hi = hzToMel(FMAX);
        for (int i = 0; i < N_MELS + 2; ++i)
            mf[i] = melToHz(lo + (hi - lo) * i / (N_MELS + 1));
        for (int m = 0; m < N_MELS; ++m) {
            centers[m] = mf[m + 1];
            double left = mf[m + 1] - mf[m];
            double right = mf[m + 2] - mf[m + 1];
            double norm = 2.0 / (mf[m + 2] - mf[m]);
            for (int k = 0; k < bins; ++k) {
                double hz = (double)k * SR / N_FFT;
                double lower = -(mf[m + 1] - hz) / left;
                double upper = (mf[m + 2] - hz) / right;
                weights[(size_t)m * bins + k] = (float)(std::max(0.0, std::min(lower, upper)) * norm);
            }
            firstBin[m] = 0; lastBin[m] = -1;
            for (int k = 0; k < bins; ++k) {
                if (weights[(size_t)m * bins + k] != 0.0f) { if (lastBin[m] < 0) firstBin[m] = k; lastBin[m] = k; }
            }
        }
    }
};
const MelBank& melBank() { static const MelBank b; return b; }

void fft(std::array<std::complex<float>, N_FFT>& a) {
    // In-place radix-2 Cooley-Tukey FFT.
    for (unsigned i = 1, j = 0; i < N_FFT; ++i) {
        unsigned bit = N_FFT >> 1;
        for (; j & bit; bit >>= 1) j ^= bit;
        j ^= bit;
        if (i < j) std::swap(a[i], a[j]);
    }
    for (unsigned len = 2; len <= N_FFT; len <<= 1) {
        const float angle = (float)(-2.0 * 3.14159265358979323846 / len);
        std::complex<float> wlen(std::cos(angle), std::sin(angle));
        for (unsigned i = 0; i < N_FFT; i += len) {
            std::complex<float> w(1.0f, 0.0f);
            for (unsigned j = 0; j < len / 2; ++j) {
                auto u = a[i + j];
                auto v = a[i + j + len / 2] * w;
                a[i + j] = u + v;
                a[i + j + len / 2] = u - v;
                w *= wlen;
            }
        }
    }
}

int reflectIndex(int i, int n) {
    if (n <= 1) return 0;
    while (i < 0 || i >= n) {
        if (i < 0) i = -i;
        if (i >= n) i = 2 * n - 2 - i;
    }
    return i;
}

std::vector<float> wavToMel(const std::vector<float>& x) {
    const int pad = (N_FFT - HOP) / 2;
    const size_t frames = std::max<size_t>(1, x.size() / HOP);
    const size_t need = (frames - 1) * HOP + N_FFT;
    std::vector<float> xp(x.size() + 2 * pad, 0.0f);
    if (!x.empty()) {
        for (int i = 0; i < pad; ++i) {
            if ((int)x.size() > pad) xp[(size_t)i] = x[(size_t)reflectIndex(i - pad, (int)x.size())];
        }
        std::copy(x.begin(), x.end(), xp.begin() + pad);
        for (int i = 0; i < pad; ++i) {
            if ((int)x.size() > pad) xp[pad + x.size() + (size_t)i] = x[(size_t)reflectIndex((int)x.size() + i, (int)x.size())];
        }
    }
    if (xp.size() < need) xp.resize(need, 0.0f);

    std::array<float, N_FFT> window{};
    for (int i = 0; i < N_FFT; ++i)
        window[i] = (float)(0.5 - 0.5 * std::cos(2.0 * 3.14159265358979323846 * i / N_FFT));
    const auto& bank = melBank();
    constexpr int bins = N_FFT / 2 + 1;
    std::vector<float> out(frames * N_MELS, MEL_FLOOR);
    std::array<std::complex<float>, N_FFT> spec{};

    for (size_t frame = 0; frame < frames; ++frame) {
        const size_t start = frame * HOP;
        for (int k = 0; k < N_FFT; ++k) {
            float sample = start + (size_t)k < xp.size() ? xp[start + (size_t)k] : 0.0f;
            spec[k] = std::complex<float>(sample * window[k], 0.0f);
        }
        fft(spec);
        std::array<float, bins> mag{};
        for (int k = 0; k < bins; ++k)
            mag[k] = std::sqrt(spec[k].real() * spec[k].real() + spec[k].imag() * spec[k].imag() + 1e-9f);
        for (int m = 0; m < N_MELS; ++m) {
            const float* w = bank.weights.data() + (size_t)m * bins;
            double sum = 0.0;
            for (int k = bank.firstBin[m]; k <= bank.lastBin[m]; ++k) sum += (double)mag[k] * w[k];
            out[frame * N_MELS + m] = (float)std::log(std::max(sum, 1e-5));
        }
    }
    return out;
}

// Windowed-sinc resampler (wide Kaiser-like cosine window); deterministic and dependency-free.
std::vector<float> resampleHQ(const std::vector<float>& in, int inRate, int outRate) {
    if (in.empty() || inRate <= 0 || outRate <= 0) return {};
    if (inRate == outRate) return in;
    size_t outCount = (size_t)std::max<long long>(1, (long long)std::llround((double)in.size() * outRate / inRate));
    std::vector<float> out(outCount);
    constexpr int radius = 32;
    constexpr int taps = 2 * radius;
    const double ratio = (double)outRate / inRate;
    const double cutoff = std::min(1.0, ratio) * 0.95;
    const int g = std::gcd(inRate, outRate);
    const size_t P = (size_t)(outRate / g);
    const long long Q = inRate / g;
    std::vector<double> coef(P * taps), normFull(P, 0.0);
    std::vector<int> centerOf(P);
    for (size_t r = 0; r < P; ++r) {
        const double pos = (double)r / ratio;
        const int center = (int)std::floor(pos);
        centerOf[r] = center;
        for (int t = 0; t < taps; ++t) {
            const int k = center - radius + 1 + t;
            const double d = pos - k;
            const double z = d * cutoff;
            const double sinc = std::abs(z) < 1e-12 ? 1.0 : std::sin(3.14159265358979323846 * z) / (3.14159265358979323846 * z);
            const double winArg = d / radius;
            double c = 0.0;
            if (std::abs(winArg) < 1.0) c = cutoff * sinc * (0.5 + 0.5 * std::cos(3.14159265358979323846 * winArg));
            coef[r * taps + t] = c;
            normFull[r] += c;
        }
    }
    const long long n = (long long)in.size();
    for (size_t j = 0; j < outCount; ++j) {
        const size_t r = j % P;
        const long long m = (long long)(j / P);
        const long long first = m * Q + centerOf[r] - radius + 1;
        const double* c = &coef[r * taps];
        double sum = 0.0, norm = 0.0;
        if (first >= 0 && first + taps <= n) {
            const float* x = &in[(size_t)first];
            for (int t = 0; t < taps; ++t) sum += x[t] * c[t];
            norm = normFull[r];
        } else {
            for (int t = 0; t < taps; ++t) {
                const long long k = first + t;
                if (k < 0 || k >= n) continue;
                sum += in[(size_t)k] * c[t];
                norm += c[t];
            }
        }
        out[j] = (float)(std::abs(norm) > 1e-12 ? sum / norm : 0.0);
    }
    return out;
}

uint16_t floatToHalf(float value) {
    uint32_t bits; std::memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000;
    int exp = (int)((bits >> 23) & 0xff) - 127 + 15;
    uint32_t mant = bits & 0x7fffff;
    if (exp <= 0) {
        if (exp < -10) return (uint16_t)sign;
        mant = (mant | 0x800000) >> (1 - exp);
        return (uint16_t)(sign | ((mant + 0x1000) >> 13));
    }
    if (exp >= 31) return (uint16_t)(sign | 0x7c00);
    return (uint16_t)(sign | ((uint32_t)exp << 10) | ((mant + 0x1000) >> 13));
}
float halfToFloat(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16;
    int exp = (h >> 10) & 0x1f;
    uint32_t mant = h & 0x3ff;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) bits = sign;
        else {
            exp = 1;
            while ((mant & 0x400) == 0) { mant <<= 1; --exp; }
            mant &= 0x3ff;
            bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000 | (mant << 13);
    else bits = sign | ((uint32_t)(exp + 127 - 15) << 23) | (mant << 13);
    float v; std::memcpy(&v, &bits, sizeof(v)); return v;
}

struct Features {
    std::vector<float> mel; // row-major [frames, 128]
    std::vector<float> f0;
    size_t frames = 0;
    size_t samples = 0;
};

uint64_t fnv1a64(const std::string& s) {
    uint64_t h = 1469598103934665603ull;
    for (unsigned char c : s) { h ^= c; h *= 1099511628211ull; }
    return h;
}
std::string hex64(uint64_t v) { std::ostringstream o; o << std::hex << std::setw(16) << std::setfill('0') << v; return o.str(); }

// Minimal RIFF/WAVE reader supporting PCM 8/16/24/32 and IEEE float 32/64.
uint16_t readU16(const unsigned char* p) { return (uint16_t)(p[0] | (p[1] << 8)); }
uint32_t readU32(const unsigned char* p) { return (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24); }
int32_t signExtend24(uint32_t x) { return (x & 0x800000) ? (int32_t)(x | 0xff000000) : (int32_t)x; }

std::vector<float> loadAudio(const fs::path& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("Cannot open input WAV: " + wideToUtf8(path.wstring()));
    std::array<unsigned char, 12> riff{};
    f.read((char*)riff.data(), riff.size());
    if (f.gcount() != (std::streamsize)riff.size() || std::memcmp(riff.data(), "RIFF", 4) || std::memcmp(riff.data() + 8, "WAVE", 4))
        throw std::runtime_error("Input is not a RIFF/WAVE file.");

    uint16_t format = 0, channels = 0, bits = 0, blockAlign = 0;
    uint32_t sampleRate = 0;
    std::vector<unsigned char> data;
    while (f && !f.eof()) {
        unsigned char ch[8]{};
        f.read((char*)ch, 8);
        if (f.gcount() != 8) break;
        uint32_t size = readU32(ch + 4);
        if (!std::memcmp(ch, "fmt ", 4)) {
            if (size < 16 || size > 65536) throw std::runtime_error("Invalid WAV fmt chunk.");
            std::vector<unsigned char> fmt(size);
            f.read((char*)fmt.data(), size);
            if ((uint32_t)f.gcount() != size) throw std::runtime_error("Truncated WAV fmt chunk.");
            format = readU16(fmt.data()); channels = readU16(fmt.data() + 2);
            sampleRate = readU32(fmt.data() + 4); blockAlign = readU16(fmt.data() + 12); bits = readU16(fmt.data() + 14);
            if (format == 0xfffe && size >= 40) format = readU16(fmt.data() + 24); // WAVE_FORMAT_EXTENSIBLE sub-format tag
        } else if (!std::memcmp(ch, "data", 4)) {
            if (size > 1024u * 1024u * 1024u) throw std::runtime_error("WAV is too large.");
            data.resize(size);
            f.read((char*)data.data(), size);
            if ((uint32_t)f.gcount() != size) throw std::runtime_error("Truncated WAV data chunk.");
        } else {
            f.seekg(size, std::ios::cur);
        }
        if (size & 1) f.seekg(1, std::ios::cur);
    }
    if (!channels || channels > 32 || !sampleRate || !blockAlign || data.empty()) throw std::runtime_error("WAV is missing valid audio format/data.");
    if (format != 1 && format != 3) throw std::runtime_error("Unsupported WAV encoding (expected PCM or IEEE float).");
    if (bits != 8 && bits != 16 && bits != 24 && bits != 32 && !(format == 3 && bits == 64))
        throw std::runtime_error("Unsupported WAV bit depth.");
    const size_t bytesPerSample = bits / 8;
    if (blockAlign < channels * bytesPerSample) throw std::runtime_error("Invalid WAV block alignment.");
    size_t count = data.size() / blockAlign;
    std::vector<float> mono(count, 0.0f);
    for (size_t i = 0; i < count; ++i) {
        double acc = 0.0;
        for (uint16_t c = 0; c < channels; ++c) {
            const unsigned char* p = data.data() + i * blockAlign + c * bytesPerSample;
            double v = 0.0;
            if (format == 3 && bits == 32) { float t; std::memcpy(&t, p, 4); v = std::isfinite(t) ? t : 0.0; }
            else if (format == 3 && bits == 64) { double t; std::memcpy(&t, p, 8); v = std::isfinite(t) ? t : 0.0; }
            else if (bits == 8) v = ((int)p[0] - 128) / 128.0;
            else if (bits == 16) v = (int16_t)readU16(p) / 32768.0;
            else if (bits == 24) v = signExtend24((uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16)) / 8388608.0;
            else if (bits == 32) v = (int32_t)readU32(p) / 2147483648.0;
            acc += v;
        }
        mono[i] = (float)(acc / channels);
    }
    if (sampleRate != SR) mono = resampleHQ(mono, (int)sampleRate, SR);
    return mono;
}

void writeWav16(const fs::path& path, const std::vector<float>& audio) {
    std::error_code ec;
    if (path.has_parent_path()) fs::create_directories(path.parent_path(), ec);
    if (ec) throw std::runtime_error("Cannot create output directory: " + ec.message());
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("Cannot create output WAV: " + wideToUtf8(path.wstring()));
    const uint32_t dataSize = (uint32_t)(audio.size() * 2);
    auto w16 = [&f](uint16_t v) { char b[2] = {(char)(v & 255), (char)(v >> 8)}; f.write(b, 2); };
    auto w32 = [&f](uint32_t v) { char b[4] = {(char)v, (char)(v >> 8), (char)(v >> 16), (char)(v >> 24)}; f.write(b, 4); };
    f.write("RIFF", 4); w32(36 + dataSize); f.write("WAVE", 4);
    f.write("fmt ", 4); w32(16); w16(1); w16(1); w32(SR); w32(SR * 2); w16(2); w16(16);
    f.write("data", 4); w32(dataSize);
    for (float x : audio) {
        double v = std::clamp((double)x, -1.0, 1.0);
        int q = (int)std::lrint(v * (v < 0 ? 32768.0 : 32767.0));
        w16((uint16_t)(int16_t)q);
    }
    if (!f) throw std::runtime_error("Failed while writing output WAV.");
}

void extractF0(const std::vector<float>& x, size_t nFrames, std::vector<float>& out) {
    const size_t skip = std::min<size_t>(HOP / 2, x.size());
    std::vector<float> tail(x.begin() + skip, x.end());
    auto x22f = resampleHQ(tail, SR, SR / 2);
    std::vector<double> x22(x22f.begin(), x22f.end());
    if (x22.size() < 4) { out.assign(nFrames, 0.0f); return; }
    HarvestOption option;
    InitializeHarvestOption(&option);
    option.f0_floor = 50.0;
    option.f0_ceil = 1100.0;
    option.frame_period = HOP_MS;
    int n = GetSamplesForHarvest(SR / 2, (int)x22.size(), option.frame_period);
    if (n <= 0) { out.assign(nFrames, 0.0f); return; }
    std::vector<double> temporal((size_t)n), f0((size_t)n);
    Harvest(x22.data(), (int)x22.size(), SR / 2, &option, temporal.data(), f0.data());
    out.assign(nFrames, 0.0f);
    for (size_t i = 0; i < std::min(nFrames, f0.size()); ++i)
        out[i] = std::isfinite(f0[i]) && f0[i] > 0.0 ? (float)f0[i] : 0.0f;
}

bool loadFeatureCache(const fs::path& path, Features& ft) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char magic[8]{}; uint64_t frames = 0, samples = 0;
    f.read(magic, 8); f.read((char*)&frames, 8); f.read((char*)&samples, 8);
    if (!f || std::memcmp(magic, "DSFC0001", 8) || frames == 0 || frames > 10000000 || samples > 1000000000ull) return false;
    size_t melCount = (size_t)frames * N_MELS;
    std::vector<uint16_t> half(melCount);
    std::vector<float> f0((size_t)frames);
    f.read((char*)half.data(), (std::streamsize)(half.size() * sizeof(uint16_t)));
    f.read((char*)f0.data(), (std::streamsize)(f0.size() * sizeof(float)));
    if (!f) return false;
    ft.frames = (size_t)frames; ft.samples = (size_t)samples;
    ft.mel.resize(melCount);
    for (size_t i = 0; i < melCount; ++i) ft.mel[i] = halfToFloat(half[i]);
    ft.f0 = std::move(f0);
    return true;
}
void saveFeatureCache(const fs::path& path, const Features& ft) {
    try {
        fs::path temp = path; temp += L"." + std::to_wstring(GetCurrentProcessId()) + L".tmp";
        std::ofstream f(temp, std::ios::binary | std::ios::trunc);
        if (!f) return;
        f.write("DSFC0001", 8);
        uint64_t frames = ft.frames, samples = ft.samples;
        f.write((const char*)&frames, 8); f.write((const char*)&samples, 8);
        std::vector<uint16_t> half(ft.mel.size());
        for (size_t i = 0; i < ft.mel.size(); ++i) half[i] = floatToHalf(ft.mel[i]);
        f.write((const char*)half.data(), (std::streamsize)(half.size() * sizeof(uint16_t)));
        f.write((const char*)ft.f0.data(), (std::streamsize)(ft.f0.size() * sizeof(float)));
        f.close();
        if (!f) { std::error_code ec; fs::remove(temp, ec); return; }
        std::error_code ec; fs::rename(temp, path, ec);
        if (ec) { fs::remove(path, ec); ec.clear(); fs::rename(temp, path, ec); }
    } catch (...) {}
}

class FeatureCache {
public:
    std::shared_ptr<Features> get(const fs::path& wavPath) {
        std::error_code ec;
        fs::path abs = fs::absolute(wavPath, ec);
        if (ec) abs = wavPath;
        uint64_t size = fs::file_size(abs, ec);
        if (ec) throw std::runtime_error("Cannot stat source WAV: " + wideToUtf8(abs.wstring()));
        auto stamp = fs::last_write_time(abs, ec).time_since_epoch().count();
        if (ec) stamp = 0;
        std::string fingerprint = wideToUtf8(abs.lexically_normal().wstring()) + "|" + std::to_string(size) + "|" + std::to_string(stamp) + "|v1";
        std::string key = hex64(fnv1a64(fingerprint));
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = memory_.find(key);
            if (it != memory_.end()) return it->second;
        }
        std::shared_ptr<std::mutex> keyLock;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto& slot = inflight_[key];
            if (!slot) slot = std::make_shared<std::mutex>();
            keyLock = slot;
        }
        struct InflightCleanup {
            FeatureCache* self; const std::string& key; const std::shared_ptr<std::mutex>& mine;
            ~InflightCleanup() {
                std::lock_guard<std::mutex> lock(self->mutex_);
                auto it = self->inflight_.find(key);
                if (it != self->inflight_.end() && it->second == mine) self->inflight_.erase(it);
            }
        } cleanup{this, key, keyLock};
        std::lock_guard<std::mutex> computeLock(*keyLock);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto it = memory_.find(key);
            if (it != memory_.end()) return it->second;
        }
        fs::path cacheFile = appDir() / L"cache" / (utf8ToWide(key) + L".dscache");
        fs::create_directories(cacheFile.parent_path(), ec);
        auto ft = std::make_shared<Features>();
        if (!loadFeatureCache(cacheFile, *ft)) {
            auto x = loadAudio(abs);
            ft->samples = x.size();
            ft->mel = wavToMel(x);
            ft->frames = ft->mel.size() / N_MELS;
            extractF0(x, ft->frames, ft->f0);
            saveFeatureCache(cacheFile, *ft);
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (memory_.size() >= 512) memory_.erase(memory_.begin());
            memory_[key] = ft;
        }
        return ft;
    }
private:
    std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Features>> memory_;
    std::unordered_map<std::string, std::shared_ptr<std::mutex>> inflight_;
};

int pickDmlDevice(std::string& adapterName) {
    adapterName.clear();
    IDXGIFactory1* factory = nullptr;
    if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), reinterpret_cast<void**>(&factory))) || !factory) return 0;
    int best = 0;
    bool found = false;
    SIZE_T bestMemory = 0;
    for (UINT i = 0;; ++i) {
        IDXGIAdapter1* adapter = nullptr;
        if (factory->EnumAdapters1(i, &adapter) == DXGI_ERROR_NOT_FOUND) break;
        DXGI_ADAPTER_DESC1 desc{};
        const bool ok = SUCCEEDED(adapter->GetDesc1(&desc));
        adapter->Release();
        if (!ok || (desc.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)) continue;
        if (!found || desc.DedicatedVideoMemory > bestMemory) {
            found = true; best = (int)i; bestMemory = desc.DedicatedVideoMemory;
            try { adapterName = wideToUtf8(desc.Description); } catch (...) { adapterName.clear(); }
        }
    }
    factory->Release();
    return found ? best : 0;
}

class Engine {
public:
    Engine() : env_(ORT_LOGGING_LEVEL_WARNING, "DiffSampler") {
        loadModelAndSession();
        features_ = std::make_unique<FeatureCache>();
        try {
            std::vector<float> m((size_t)BUCKET * N_MELS, MEL_FLOOR), f(BUCKET, 220.0f);
            (void)vocode(m, f);
            log("warmup complete");
        } catch (const std::exception& e) { log(std::string("warmup failed: ") + e.what()); }
    }
    const std::string& provider() const { return provider_; }
    const std::string& deviceInfo() const { return deviceInfo_; }
    void render(const std::vector<std::string>& argv) {
        auto t0 = Clock::now();
        if (argv.size() < 2) throw std::runtime_error("Expected input and output WAV paths.");
        fs::path inp = pathFromUtf8(argv[0]);
        fs::path outp = pathFromUtf8(argv[1]);
        double noteHz = argv.size() > 2 ? noteToHz(argv[2]) : 440.0;
        double vel = parseDouble(argv, 3, 100.0);
        Flags flags = parseFlags(argv.size() > 4 ? argv[4] : "");
        double offset = parseDouble(argv, 5, 0.0);
        double length = std::max(10.0, parseDouble(argv, 6, 500.0));
        double cons = std::max(0.0, parseDouble(argv, 7, 0.0));
        double cutoff = parseDouble(argv, 8, 0.0);
        double volume = parseDouble(argv, 9, 100.0);
        double modulation = parseDouble(argv, 10, 0.0);
        double tempo = 120.0;
        if (argv.size() > 11) {
            std::string s = argv[11];
            if (!s.empty() && s[0] == '!') s.erase(s.begin());
            try { tempo = std::stod(s); } catch (...) { tempo = 120.0; }
        }
        if (!std::isfinite(tempo) || tempo <= 0.0) tempo = 120.0;
        std::vector<int> pb = decodePitchbend(argv.size() > 12 ? argv[12] : "");

        auto ft = features_->get(inp);
        const auto& mel = ft->mel;
        const auto& f0s = ft->f0;
        const size_t T = ft->frames;
        const double totalMs = ft->samples * 1000.0 / SR;
        double start = std::clamp(offset, 0.0, std::max(totalMs - 2.0 * HOP_MS, 0.0));
        double end = cutoff < 0.0 ? start - cutoff : totalMs - cutoff;
        end = std::min(std::max(end, start + 2.0 * HOP_MS), totalMs);
        cons = std::min(cons, end - start);
        double consOut = cons * std::pow(2.0, std::clamp(1.0 - vel / 100.0, -20.0, 20.0));
        consOut = std::min(consOut, length);

        size_t N = std::max<size_t>(1, (size_t)std::ceil(length / HOP_MS));
        std::vector<float> m(N * N_MELS, MEL_FLOOR), f0(N, 0.0f);
        std::vector<double> tOut(N), tSrc(N), wSrc(N);
        const double restOut = std::max(length - consOut, 1e-3);
        for (size_t i = 0; i < N; ++i) {
            tOut[i] = (i + 0.5) * HOP_MS;
            if (consOut > 0.0 && tOut[i] < consOut) tSrc[i] = start + tOut[i] / consOut * cons;
            else tSrc[i] = start + cons + (tOut[i] - consOut) / restOut * (end - start - cons);
        }
        for (size_t i = 0; i < N; ++i) {
            double fr = std::clamp(tSrc[i] / HOP_MS - 0.5, 0.0, (double)T - 1.0);
            size_t i0 = (size_t)std::floor(fr), i1 = std::min(i0 + 1, T - 1);
            float w = (float)(fr - i0);
            for (int k = 0; k < N_MELS; ++k) {
                float a = mel[i0 * N_MELS + k], b = mel[i1 * N_MELS + k];
                m[i * N_MELS + k] = a * (1.0f - w) + b * w;
            }
            float fa = f0s[i0], fb = f0s[i1];
            if (fa > 0 && fb > 0) f0[i] = (float)std::exp((1.0 - w) * std::log(fa) + w * std::log(fb));
            else if (fa > 0 && w < 0.5f) f0[i] = fa;
            else if (fb > 0 && w >= 0.5f) f0[i] = fb;
        }

        std::vector<double> selected;
        for (size_t i = 0; i < T; ++i) {
            double seg = (i + 0.5) * HOP_MS;
            if (seg >= start + cons && seg < end && f0s[i] > 0) selected.push_back(std::log(f0s[i]));
        }
        if (selected.empty()) {
            for (size_t i = 0; i < T; ++i) {
                double seg = (i + 0.5) * HOP_MS;
                if (seg >= start && seg < end && f0s[i] > 0) selected.push_back(std::log(f0s[i]));
            }
        }
        if (!selected.empty()) {
            auto mid = selected.begin() + selected.size() / 2;
            std::nth_element(selected.begin(), mid, selected.end());
            double medianLog = *mid;
            if ((selected.size() & 1) == 0) {
                auto lower = std::max_element(selected.begin(), mid);
                medianLog = (medianLog + *lower) * 0.5;
            }
            double f0Ref = std::exp(medianLog);
            double base = noteHz * std::pow(2.0, flags.t / 1200.0);
            double dt = 60.0 / (tempo * 96.0) * 1000.0;
            for (size_t i = 0; i < N; ++i) {
                if (f0[i] <= 0) continue;
                double bend = 0.0;
                if (!pb.empty()) {
                    double index = tOut[i] / dt;
                    if (index <= 0) bend = pb.front();
                    else if (index >= pb.size() - 1) bend = pb.back();
                    else {
                        size_t a = (size_t)std::floor(index), b = a + 1;
                        double w = index - a;
                        bend = pb[a] * (1.0 - w) + pb[b] * w;
                    }
                }
                f0[i] = (float)(base * std::pow((double)f0[i] / f0Ref, modulation / 100.0) * std::pow(2.0, bend / 1200.0));
            }
            double shiftSt = std::abs(12.0 * std::log2(base / f0Ref));
            if (std::isfinite(shiftSt) && shiftSt > 12.0)
                log("note shifted " + std::to_string(shiftSt) + " st; model is trained for +/-12");
        } else std::fill(f0.begin(), f0.end(), 0.0f);

        if (flags.g != 0.0) {
            const auto& centers = melBank().centers;
            double ratio = std::pow(2.0, std::clamp(flags.g, -2000.0, 2000.0) / 200.0);
            std::array<int, N_MELS> j0{}, j1{};
            std::array<float, N_MELS> wj{};
            for (int k = 0; k < N_MELS; ++k) {
                double src = std::clamp((double)k, 0.0, (double)N_MELS - 1.0);
                double target = centers[k] / ratio;
                if (target <= centers.front()) src = 0;
                else if (target >= centers.back()) src = N_MELS - 1;
                else {
                    auto it = std::upper_bound(centers.begin(), centers.end(), target);
                    int b = (int)(it - centers.begin()), a = b - 1;
                    src = a + (target - centers[a]) / (centers[b] - centers[a]);
                }
                j0[k] = (int)std::floor(src);
                j1[k] = std::min(j0[k] + 1, N_MELS - 1);
                wj[k] = (float)(src - j0[k]);
            }
            for (size_t i = 0; i < N; ++i) {
                float row[N_MELS];
                std::copy(m.begin() + i * N_MELS, m.begin() + (i + 1) * N_MELS, row);
                for (int k = 0; k < N_MELS; ++k)
                    m[i * N_MELS + k] = row[j0[k]] * (1.0f - wj[k]) + row[j1[k]] * wj[k];
            }
        }

        auto wav = vocode(m, f0);
        size_t nOut = (size_t)std::llround(length * SR / 1000.0);
        if (wav.size() < nOut) wav.resize(nOut, 0.0f); else wav.resize(nOut);
        // Match the usual UTAU resampler output level before applying VOL.
        // A high percentile is steadier than a single-sample peak, which may
        // be a consonant spike or neural-vocoder transient.
        std::wstring autoGainSetting = envW(L"DIFFSAMPLER_AUTOGAIN", L"1");
        std::transform(autoGainSetting.begin(), autoGainSetting.end(), autoGainSetting.begin(),
                       [](wchar_t c) { return (wchar_t)std::towlower(c); });
        const bool autoGain = autoGainSetting != L"0" && autoGainSetting != L"false" &&
                              autoGainSetting != L"off" && autoGainSetting != L"no";
        double gain = 1.0;
        if (autoGain && !wav.empty()) {
            std::vector<float> levels;
            levels.reserve(wav.size());
            for (float x : wav) if (std::isfinite(x)) levels.push_back(std::abs(x));
            if (!levels.empty()) {
                size_t q = std::min(levels.size() - 1, (size_t)(0.995 * (levels.size() - 1)));
                std::nth_element(levels.begin(), levels.begin() + q, levels.end());
                double referencePeak = levels[q];
                if (referencePeak > 1e-6) gain = std::min(16.0, 0.50 / referencePeak);
            }
        }
        gain *= std::max(0.0, volume) / 100.0;
        for (float& x : wav) {
            if (!std::isfinite(x)) x = 0.0f;
            x = (float)std::clamp((double)x * gain, -1.0, 1.0);
        }
        writeWav16(outp, wav);
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        log(basenameUtf8(argv[0]) + " -> " + basenameUtf8(argv[1]) + " " + std::to_string((int)std::round(length)) + "ms in " + std::to_string((int)std::round(ms)) + "ms [" + provider_ + "]");
    }

    std::shared_ptr<Features> getFeatures(const fs::path& p) { return features_->get(p); }

private:
    void loadModelAndSession() {
        // Keep model weights as a normal, inspectable sidecar file rather than
        // embedding a very large opaque data blob in the executable. This also
        // lets users verify the model independently of the program binary.
        const std::wstring modelOverride = envW(L"DIFFSAMPLER_MODEL");
        const fs::path modelPath = modelOverride.empty()
            ? (exePath().parent_path() / MODEL_RELATIVE_PATH)
            : fs::path(modelOverride);
        std::ifstream modelFile(modelPath, std::ios::binary);
        if (!modelFile) {
            throw std::runtime_error("Could not open ONNX model file: " + wideToUtf8(modelPath.wstring()));
        }
        modelFile.seekg(0, std::ios::end);
        const std::streamoff modelLength = modelFile.tellg();
        if (modelLength <= 0 || static_cast<unsigned long long>(modelLength) >
                static_cast<unsigned long long>(std::numeric_limits<size_t>::max()) ||
            static_cast<unsigned long long>(modelLength) >
                static_cast<unsigned long long>(std::numeric_limits<std::streamsize>::max())) {
            throw std::runtime_error("ONNX model file has an invalid size.");
        }
        modelFile.seekg(0, std::ios::beg);
        modelBytes_.resize(static_cast<size_t>(modelLength));
        modelFile.read(reinterpret_cast<char*>(modelBytes_.data()), static_cast<std::streamsize>(modelLength));
        if (!modelFile) throw std::runtime_error("Could not read the complete ONNX model file.");
        const unsigned char* modelData = modelBytes_.data();
        const size_t modelSize = modelBytes_.size();

        std::string want = "auto";
        std::wstring w = envW(L"DIFFSAMPLER_PROVIDER", L"auto");
        for (auto& c : w) c = (wchar_t)towlower(c);
        try { want = wideToUtf8(w); } catch (...) {}
        const std::wstring deviceEnv = envW(L"DIFFSAMPLER_DEVICE");
        const bool deviceAuto = deviceEnv.empty() || deviceEnv == L"auto";
        int device = deviceAuto ? 0 : parseInt(deviceEnv, 0);
        if (device < 0) device = 0;
        bool tryDml = (want == "auto" || want == "dml");

        if (tryDml) {
            try {
                std::string adapterName;
                if (deviceAuto) device = pickDmlDevice(adapterName);
                Ort::SessionOptions options;
                options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
                options.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);
                options.DisableMemPattern();
                Ort::ThrowOnError(OrtSessionOptionsAppendExecutionProvider_DML(options, device));
                session_ = std::make_unique<Ort::Session>(env_, modelData, modelSize, options);
                provider_ = "DmlExecutionProvider";
                deviceInfo_ = "device " + std::to_string(device) + (adapterName.empty() ? std::string() : ": " + adapterName);
            } catch (const std::exception& e) {
                log(std::string("DirectML session failed; using CPU: ") + e.what());
                session_.reset();
            }
        } else if (want != "cpu") {
            log("Unknown provider requested; using CPU");
        }
        if (!session_) {
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
            options.SetIntraOpNumThreads(0);
            session_ = std::make_unique<Ort::Session>(env_, modelData, modelSize, options);
            provider_ = "CPUExecutionProvider";
        }
        if (session_->GetInputCount() < 2 || session_->GetOutputCount() < 1)
            throw std::runtime_error("ONNX model must have mel/f0 inputs and at least one output.");
        Ort::AllocatorWithDefaultOptions allocator;
        auto outName = session_->GetOutputNameAllocated(0, allocator);
        outputName_ = outName.get();
        log("model ready on " + provider_ + (deviceInfo_.empty() ? std::string() : " (" + deviceInfo_ + ")"));
    }

    std::vector<float> vocode(const std::vector<float>& mel, const std::vector<float>& f0) {
        size_t T = f0.size();
        size_t Tp = std::max<size_t>(BUCKET, ((T + BUCKET - 1) / BUCKET) * BUCKET);
        std::vector<float> m(Tp * N_MELS, MEL_FLOOR), f(Tp, 0.0f);
        if (T == 0 || mel.size() != T * N_MELS) throw std::runtime_error("Internal mel tensor shape mismatch.");
        std::copy(mel.begin(), mel.end(), m.begin());
        std::copy(f0.begin(), f0.end(), f.begin());
        // Keep the vocoder's right-hand receptive field on the final phoneme
        // instead of conditioning it on artificial silence. This stabilizes
        // note tails for the wavtool's overlap/crossfade stage.
        for (size_t i = T; i < Tp; ++i) {
            std::copy(mel.end() - N_MELS, mel.end(), m.begin() + i * N_MELS);
            f[i] = f0.back();
        }
        std::array<int64_t, 3> mShape{1, (int64_t)Tp, N_MELS};
        std::array<int64_t, 2> fShape{1, (int64_t)Tp};
        auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        std::array<Ort::Value, 2> inputs = {
            Ort::Value::CreateTensor<float>(memory, m.data(), m.size(), mShape.data(), mShape.size()),
            Ort::Value::CreateTensor<float>(memory, f.data(), f.size(), fShape.data(), fShape.size())
        };
        const char* inputNames[] = {"mel", "f0"};
        const char* outputNames[] = {outputName_.c_str()};
        Ort::RunOptions runOptions{nullptr};
        std::vector<Ort::Value> outputs;
        {
            std::lock_guard<std::mutex> lock(inferenceMutex_);
            outputs = session_->Run(runOptions, inputNames, inputs.data(), inputs.size(), outputNames, 1);
        }
        if (outputs.empty() || !outputs[0].IsTensor()) throw std::runtime_error("Vocoder returned no tensor output.");
        auto info = outputs[0].GetTensorTypeAndShapeInfo();
        if (info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) throw std::runtime_error("Vocoder output is not float32.");
        size_t count = info.GetElementCount();
        const float* data = outputs[0].GetTensorData<float>();
        if (!data || count == 0) throw std::runtime_error("Vocoder returned an empty output tensor.");
        // Match the Python implementation: discard samples generated for the
        // 64-frame bucket padding. Never let padded frames leak into a note.
        const size_t wanted = T * (size_t)HOP;
        const size_t usable = std::min(count, wanted);
        return std::vector<float>(data, data + usable);
    }

    Ort::Env env_;
    std::vector<unsigned char> modelBytes_;
    std::unique_ptr<Ort::Session> session_;
    std::unique_ptr<FeatureCache> features_;
    std::mutex inferenceMutex_;
    std::string provider_ = "CPUExecutionProvider";
    std::string deviceInfo_;
    std::string outputName_;
};

void appendU32(std::vector<unsigned char>& b, uint32_t v) {
    for (int i = 0; i < 4; ++i) b.push_back((unsigned char)((v >> (i * 8)) & 255));
}
uint32_t takeU32(const unsigned char*& p, const unsigned char* end) {
    if (end - p < 4) throw std::runtime_error("Malformed worker request.");
    uint32_t v = (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24);
    p += 4; return v;
}
std::vector<unsigned char> makeRequest(uint32_t cmd, const std::vector<std::string>& args = {}) {
    std::vector<unsigned char> b{'D','S','R','1'};
    appendU32(b, cmd); appendU32(b, (uint32_t)args.size());
    for (const auto& s : args) {
        if (s.size() > 16 * 1024) throw std::runtime_error("Worker argument is too long.");
        appendU32(b, (uint32_t)s.size()); b.insert(b.end(), s.begin(), s.end());
    }
    return b;
}
struct Request { uint32_t cmd = 0; std::vector<std::string> args; };
Request parseRequest(const std::vector<unsigned char>& b) {
    if (b.size() < 12 || std::memcmp(b.data(), "DSR1", 4)) throw std::runtime_error("Invalid worker request header.");
    const unsigned char* p = b.data() + 4; const unsigned char* end = b.data() + b.size();
    Request r; r.cmd = takeU32(p, end); uint32_t n = takeU32(p, end);
    if (n > 64) throw std::runtime_error("Too many worker arguments.");
    for (uint32_t i = 0; i < n; ++i) {
        uint32_t len = takeU32(p, end);
        if (len > 16 * 1024 || (size_t)(end - p) < len) throw std::runtime_error("Invalid worker argument length.");
        r.args.emplace_back((const char*)p, (size_t)len); p += len;
    }
    if (p != end) throw std::runtime_error("Trailing data in worker request.");
    return r;
}
std::vector<unsigned char> makeResponse(bool ok, const std::string& message) {
    std::vector<unsigned char> b{'D','S','P','1'};
    appendU32(b, ok ? 1u : 0u); appendU32(b, (uint32_t)message.size());
    b.insert(b.end(), message.begin(), message.end()); return b;
}
struct Response { bool ok = false; std::string message; };
Response parseResponse(const std::vector<unsigned char>& b) {
    if (b.size() < 12 || std::memcmp(b.data(), "DSP1", 4)) throw std::runtime_error("Invalid worker response.");
    const unsigned char* p = b.data() + 4; const unsigned char* end = b.data() + b.size();
    Response r; r.ok = takeU32(p, end) != 0; uint32_t len = takeU32(p, end);
    if ((size_t)(end - p) != len) throw std::runtime_error("Invalid worker response length.");
    r.message.assign((const char*)p, len); return r;
}

bool transact(uint32_t cmd, const std::vector<std::string>& args, Response& response, DWORD timeoutMs = 600000) {
    auto request = makeRequest(cmd, args);
    DWORD start = GetTickCount();
    HANDLE h = INVALID_HANDLE_VALUE;
    for (;;) {
        h = CreateFileW(PIPE_NAME, GENERIC_READ | GENERIC_WRITE, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        if (h != INVALID_HANDLE_VALUE) break;
        DWORD e = GetLastError();
        if (e != ERROR_PIPE_BUSY && e != ERROR_FILE_NOT_FOUND) return false;
        if (GetTickCount() - start >= timeoutMs) return false;
        if (!WaitNamedPipeW(PIPE_NAME, 250)) Sleep(25);
    }
    DWORD mode = PIPE_READMODE_MESSAGE;
    SetNamedPipeHandleState(h, &mode, nullptr, nullptr);
    DWORD written = 0;
    BOOL ok = WriteFile(h, request.data(), (DWORD)request.size(), &written, nullptr);
    if (!ok || written != request.size()) { CloseHandle(h); return false; }
    std::vector<unsigned char> buffer(IPC_BUFFER_SIZE);
    DWORD read = 0;
    ok = ReadFile(h, buffer.data(), (DWORD)buffer.size(), &read, nullptr);
    DWORD err = ok ? ERROR_SUCCESS : GetLastError();
    CloseHandle(h);
    if (!ok || err == ERROR_MORE_DATA) return false;
    buffer.resize(read);
    try { response = parseResponse(buffer); return true; }
    catch (...) { return false; }
}

bool workerExists() {
    HANDLE h = OpenMutexW(SYNCHRONIZE, FALSE, MUTEX_NAME);
    if (!h) return false;
    CloseHandle(h); return true;
}

bool spawnWorker() {
    if (workerExists()) return true;
    fs::path exe = exePath();
    std::wstring cmd = L"\"" + exe.wstring() + L"\" --daemon";
    STARTUPINFOW si{}; si.cb = sizeof(si);
    PROCESS_INFORMATION pi{};
    std::vector<wchar_t> writable(cmd.begin(), cmd.end()); writable.push_back(L'\0');
    BOOL ok = CreateProcessW(exe.c_str(), writable.data(), nullptr, nullptr, FALSE,
        CREATE_NO_WINDOW, nullptr, exe.parent_path().c_str(), &si, &pi);
    if (!ok) {
        log("CreateProcess for worker failed: " + std::to_string(GetLastError()));
        return false;
    }
    CloseHandle(pi.hThread); CloseHandle(pi.hProcess);
    log("worker process started");
    return true;
}

bool pipeIo(HANDLE pipe, HANDLE event, bool write, void* buffer, DWORD size, DWORD& transferred, DWORD timeoutMs) {
    transferred = 0;
    ResetEvent(event);
    OVERLAPPED ov{}; ov.hEvent = event;
    BOOL ok = write ? WriteFile(pipe, buffer, size, nullptr, &ov) : ReadFile(pipe, buffer, size, nullptr, &ov);
    if (!ok) {
        if (GetLastError() != ERROR_IO_PENDING) return false;
        if (WaitForSingleObject(event, timeoutMs) != WAIT_OBJECT_0) {
            CancelIoEx(pipe, &ov);
            GetOverlappedResult(pipe, &ov, &transferred, TRUE);
            return false;
        }
    }
    return GetOverlappedResult(pipe, &ov, &transferred, FALSE) != FALSE;
}

unsigned int idleSeconds() {
    std::wstring w = envW(L"DIFFSAMPLER_IDLE", std::to_wstring(IDLE_DEFAULT));
    int v = std::clamp(parseInt(w, IDLE_DEFAULT), 60, 86400);
    return (unsigned int)v;
}

struct WorkerState {
    std::atomic<int> active{0};
    std::atomic<bool> stop{false};
    std::atomic<long long> lastActivityMs{0};
    HANDLE stopEvent = nullptr;
    Engine* engine = nullptr;
    std::string engineError;
};

long long nowMs() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now().time_since_epoch()).count();
}

void serveConnection(HANDLE pipe, WorkerState* st) {
    HANDLE event = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    bool requestStop = false;
    try {
        if (event) {
            std::vector<unsigned char> input(IPC_BUFFER_SIZE);
            DWORD got = 0;
            bool readOk = pipeIo(pipe, event, false, input.data(), (DWORD)input.size(), got, 10000);
            Response response;
            try {
                if (!readOk || got == 0) throw std::runtime_error("Could not read worker request.");
                input.resize(got);
                Request req = parseRequest(input);
                if (req.cmd == IPC_PING) { response = {true, "ready"}; }
                else if (req.cmd == IPC_INFO) {
                    if (!st->engine) response = {false, st->engineError};
                    else response = {true, "provider=" + st->engine->provider() +
                        (st->engine->deviceInfo().empty() ? std::string() : " (" + st->engine->deviceInfo() + ")")};
                } else if (req.cmd == IPC_STOP) {
                    response = {true, "stopping"}; requestStop = true;
                } else if (req.cmd == IPC_RENDER) {
                    if (!st->engine) response = {false, st->engineError};
                    else { st->engine->render(req.args); response = {true, "ok"}; }
                } else response = {false, "unknown worker command"};
            } catch (const std::exception& e) {
                response = {false, e.what()}; log(std::string("worker request failed: ") + e.what());
            } catch (...) { response = {false, "unknown worker error"}; }
            auto output = makeResponse(response.ok, response.message);
            DWORD written = 0;
            if (pipeIo(pipe, event, true, output.data(), (DWORD)output.size(), written, 10000))
                FlushFileBuffers(pipe);
        }
    } catch (...) { log("worker connection handler failed"); }
    if (event) CloseHandle(event);
    DisconnectNamedPipe(pipe);
    CloseHandle(pipe);
    st->lastActivityMs = nowMs();
    if (requestStop) { st->stop = true; SetEvent(st->stopEvent); }
    st->active.fetch_sub(1);
}

int daemonMain() {
    HANDLE mutex = CreateMutexW(nullptr, TRUE, MUTEX_NAME);
    if (!mutex) return 1;
    if (GetLastError() == ERROR_ALREADY_EXISTS) { CloseHandle(mutex); return 0; }
    log("worker starting; loading model");
    std::unique_ptr<Engine> engine;
    std::string engineError;
    try { engine = std::make_unique<Engine>(); }
    catch (const std::exception& e) { engineError = e.what(); log(std::string("engine initialization failed: ") + e.what()); }

    WorkerState st;
    st.engine = engine.get();
    st.engineError = engineError;
    st.stopEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    HANDLE ioEvent = CreateEventW(nullptr, TRUE, FALSE, nullptr);
    if (!st.stopEvent || !ioEvent) {
        if (st.stopEvent) CloseHandle(st.stopEvent);
        if (ioEvent) CloseHandle(ioEvent);
        CloseHandle(mutex);
        return 1;
    }
    st.lastActivityMs = nowMs();
    const long long idleMs = (long long)idleSeconds() * 1000;

    while (!st.stop) {
        HANDLE pipe = CreateNamedPipeW(PIPE_NAME, PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
            PIPE_TYPE_MESSAGE | PIPE_READMODE_MESSAGE | PIPE_WAIT | PIPE_REJECT_REMOTE_CLIENTS,
            PIPE_UNLIMITED_INSTANCES, (DWORD)IPC_BUFFER_SIZE, (DWORD)IPC_BUFFER_SIZE, 0, nullptr);
        if (pipe == INVALID_HANDLE_VALUE) {
            log("CreateNamedPipe failed: " + std::to_string(GetLastError()));
            Sleep(500); continue;
        }
        ResetEvent(ioEvent);
        OVERLAPPED connectOv{}; connectOv.hEvent = ioEvent;
        BOOL connected = ConnectNamedPipe(pipe, &connectOv);
        DWORD connectErr = connected ? ERROR_SUCCESS : GetLastError();
        if (!connected && connectErr == ERROR_PIPE_CONNECTED) connected = TRUE;
        bool idleExit = false;
        if (!connected && connectErr == ERROR_IO_PENDING) {
            const HANDLE waits[2] = {ioEvent, st.stopEvent};
            for (;;) {
                DWORD w = WaitForMultipleObjects(2, waits, FALSE, 1000);
                if (w == WAIT_OBJECT_0) {
                    DWORD ignored = 0;
                    connected = GetOverlappedResult(pipe, &connectOv, &ignored, FALSE);
                    break;
                }
                if (w == WAIT_TIMEOUT) {
                    if (st.active.load() == 0 && nowMs() - st.lastActivityMs.load() >= idleMs) { idleExit = true; break; }
                    continue;
                }
                break;
            }
            if (!connected) {
                CancelIoEx(pipe, &connectOv);
                DWORD ignored = 0;
                GetOverlappedResult(pipe, &connectOv, &ignored, TRUE);
            }
        }
        if (idleExit) {
            CloseHandle(pipe);
            log("worker idle timeout; exiting normally"); break;
        }
        if (!connected) {
            CloseHandle(pipe);
            if (!st.stop) Sleep(100);
            continue;
        }
        st.lastActivityMs = nowMs();
        st.active.fetch_add(1);
        try {
            std::thread(serveConnection, pipe, &st).detach();
        } catch (...) {
            log("could not start a worker connection thread");
            st.active.fetch_sub(1);
            DisconnectNamedPipe(pipe);
            CloseHandle(pipe);
            Sleep(50);
        }
    }
    for (int i = 0; i < 6000 && st.active.load() > 0; ++i) Sleep(10);
    if (st.active.load() > 0) log("worker exiting while requests were still running");
    CloseHandle(ioEvent);
    CloseHandle(st.stopEvent);
    CloseHandle(mutex);
    return 0;
}

bool requestWithStartup(uint32_t cmd, const std::vector<std::string>& args, Response& response, DWORD timeoutMs = 90000) {
    if (workerExists()) {
        if (transact(cmd, args, response, 1500)) return true;
    } else {
        spawnWorker();
    }
    DWORD start = GetTickCount();
    while (GetTickCount() - start < timeoutMs) {
        if (transact(cmd, args, response, 3000)) return true;
        if (!workerExists()) {
            spawnWorker();
        }
        Sleep(150);
    }
    return false;
}

size_t physicalCoreCount() {
    DWORD length = 0;
    GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &length);
    if (length == 0) return 0;
    std::vector<unsigned char> buffer(length);
    if (!GetLogicalProcessorInformationEx(RelationProcessorCore,
            reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data()), &length)) return 0;
    size_t cores = 0;
    for (size_t offset = 0; offset < length;) {
        auto* info = reinterpret_cast<PSYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX>(buffer.data() + offset);
        if (info->Size == 0) break;
        if (info->Relationship == RelationProcessorCore) ++cores;
        offset += info->Size;
    }
    return cores;
}

void precache(const fs::path& root, const std::wstring& prefix = L"") {
    if (!fs::exists(root) || !fs::is_directory(root)) throw std::runtime_error("Precache path is not a directory.");
    auto lower = [](std::wstring s) { for (auto& c : s) c = (wchar_t)towlower(c); return s; };
    const std::wstring wantPrefix = lower(prefix);
    std::vector<fs::path> wavs;
    for (auto it = fs::recursive_directory_iterator(root, fs::directory_options::skip_permission_denied); it != fs::recursive_directory_iterator(); ++it) {
        std::error_code ec;
        if (it->is_regular_file(ec) && !ec) {
            if (lower(it->path().extension().wstring()) != L".wav") continue;
            if (!wantPrefix.empty() && lower(it->path().filename().wstring()).rfind(wantPrefix, 0) != 0) continue;
            wavs.push_back(it->path());
        }
    }
    std::cout << wavs.size() << " wav files\n";
    FeatureCache cache;
    std::atomic<size_t> next{0}, done{0};
    std::mutex printMutex;
    const unsigned hw = std::thread::hardware_concurrency();
    const size_t physical = physicalCoreCount();
    const size_t cores = physical ? physical : (hw ? hw : 2u);
    const size_t threads = std::max<size_t>(1, std::min<size_t>(cores, wavs.size()));
    auto work = [&]() {
        for (;;) {
            const size_t i = next.fetch_add(1);
            if (i >= wavs.size()) return;
            try { cache.get(wavs[i]); }
            catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(printMutex);
                std::cerr << "skip " << wideToUtf8(wavs[i].wstring()) << ": " << e.what() << "\n";
            }
            const size_t d = done.fetch_add(1) + 1;
            if (d % 25 == 0 || d == wavs.size()) {
                std::lock_guard<std::mutex> lock(printMutex);
                std::cout << "  " << d << "/" << wavs.size() << "\n";
            }
        }
    };
    std::vector<std::thread> pool;
    for (size_t t = 0; t < threads; ++t) pool.emplace_back(work);
    for (auto& t : pool) t.join();
    std::cout << "done\n";
}

std::string noticeText() {
    try {
        std::ifstream f(exePath().parent_path() / L"model" / L"NOTICE.txt", std::ios::binary);
        if (!f) throw std::runtime_error("Model notice is missing.");
        std::ostringstream out;
        out << f.rdbuf();
        if (!f.good() && !f.eof()) throw std::runtime_error("Could not read model notice.");
        return out.str();
    } catch (...) {
        return "DiffSampler uses the OpenVPI PC-NSF-HiFiGAN vocoder model. See https://github.com/openvpi/vocoders and model/NOTICE.txt beside this executable.\r\n";
    }
}

int mainImpl(int argc, wchar_t** wargv) {
    std::vector<std::wstring> wa;
    for (int i = 1; i < argc; ++i) wa.emplace_back(wargv[i]);
    if (wa.empty() || wa[0] == L"--help" || wa[0] == L"-h") {
        std::cout << "DiffSampler " << VERSION << "\n"
            << "Usage: DiffSampler.exe <in.wav> <out.wav> <pitch> <vel> <flags> <offset> <length> <consonant> <cutoff> <volume> <modulation> <tempo> <pitchbend>\n"
            << "Options: --daemon --stop --info --precache <dir> [name-prefix] --notice --version --where\n"
            << "Environment: DIFFSAMPLER_PROVIDER=auto|dml|cpu, DIFFSAMPLER_DEVICE=auto|<index>, DIFFSAMPLER_IDLE=900, DIFFSAMPLER_NODAEMON=1, DIFFSAMPLER_MODEL=<onnx> (default: model\\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx)\n";
        return 0;
    }
    if (wa[0] == L"--version") { std::cout << VERSION << "\n"; return 0; }
    if (wa[0] == L"--notice") { std::cout << noticeText(); return 0; }
    if (wa[0] == L"--where") { std::cout << wideToUtf8(exePath().wstring()) << "\n"; return 0; }
    if (wa[0] == L"--daemon") return daemonMain();
    if (wa[0] == L"--stop") {
        Response r;
        if (transact(IPC_STOP, {}, r, 3000) && r.ok) std::cout << "stopped\n";
        else std::cout << "worker is not running\n";
        return 0;
    }
    if (wa[0] == L"--info") {
        Response r;
        if (requestWithStartup(IPC_INFO, {}, r, 90000) && r.ok) std::cout << r.message << "\n";
        else { std::cerr << "DiffSampler: " << (r.message.empty() ? "worker did not start" : r.message) << "\n"; return 1; }
        return 0;
    }
    if (wa[0] == L"--precache") {
        if (wa.size() < 2) throw std::runtime_error("Usage: DiffSampler.exe --precache <directory> [file-name-prefix]");
        precache(fs::path(wa[1]), wa.size() > 2 ? wa[2] : std::wstring()); return 0;
    }
    if (wa.size() < 2) throw std::runtime_error("Expected input and output WAV paths. Run --help for usage.");
    std::vector<std::string> args;
    args.reserve(wa.size());
    for (const auto& a : wa) args.push_back(wideToUtf8(a));
    try {
        if (envW(L"DIFFSAMPLER_NODAEMON") == L"1") {
            Engine e; e.render(args); return 0;
        }
        Response r;
        if (requestWithStartup(IPC_RENDER, args, r, 90000) && r.ok) return 0;
        std::string reason = r.message.empty() ? "worker did not start" : r.message;
        log("worker unavailable; falling back in-process: " + reason);
        Engine e; e.render(args); return 0;
    } catch (const std::exception& e) {
        std::cerr << "DiffSampler error: " << e.what() << "\n";
        return 1;
    }
}
} // namespace ds

int wmain(int argc, wchar_t** argv) {
    try { return ds::mainImpl(argc, argv); }
    catch (const std::exception& e) { std::cerr << "DiffSampler error: " << e.what() << "\n"; return 1; }
    catch (...) { std::cerr << "DiffSampler error: unknown failure\n"; return 1; }
}
