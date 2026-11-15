#!/usr/bin/env python3
"""
DiffSampler - fast neural UTAU resampler on top of the DiffSinger community
PC-NSF-HiFiGAN vocoder (44.1k / hop 512 / 128 mel).

One file, one executable:

  DiffSampler.exe <in.wav> <out.wav> <pitch> <vel> <flags> <offset> <length>
                  <consonant> <cutoff> <volume> <modulation> <tempo> <pitchbend>
      -> normal UTAU resampler call.  This is a *thin client*: it forwards the
         job to a background copy of itself that keeps the model loaded on the
         GPU, so every note costs milliseconds instead of "start Python, load
         the model, build the GPU session" seconds.

  DiffSampler.exe --daemon            run the background worker (auto-started)
  DiffSampler.exe --stop              stop the background worker
  DiffSampler.exe --info              show which device the model runs on
  DiffSampler.exe --precache <dir>    pre-compute mel/F0 for a whole voicebank
  DiffSampler.exe --notice            print the vocoder licence notice

Environment (all optional):
  DIFFSAMPLER_PROVIDER = auto | dml | cuda | cpu      (default auto)
  DIFFSAMPLER_DEVICE   = GPU index for dml/cuda       (default 0)
  DIFFSAMPLER_IDLE     = seconds before the worker quits when idle (default 900)
  DIFFSAMPLER_NODAEMON = 1  render inside this process (no worker)
  DIFFSAMPLER_MODEL    = path to another PC-NSF-HiFiGAN ONNX file
  DIFFSAMPLER_AUTOGAIN = 0 to disable per-note level matching (default 1)
"""
import os
import sys
import json
import time
import socket
import hashlib

VERSION = "1.0.6"
SR = 44100
HOP = 512
N_FFT = 2048
N_MELS = 128
FMIN = 40.0
FMAX = 16000.0
HOP_MS = HOP * 1000.0 / SR
MEL_FLOOR = float(__import__("math").log(1e-5))
BUCKET = 64  # frames; keeps the number of distinct GPU graph shapes small


# --------------------------------------------------------------------------
# paths / resources
# --------------------------------------------------------------------------
def app_dir():
    base = os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache")
    d = os.path.join(base, "DiffSampler")
    os.makedirs(d, exist_ok=True)
    return d


def resource(*parts):
    roots = []
    if getattr(sys, "_MEIPASS", None):
        roots.append(sys._MEIPASS)
    roots.append(os.path.dirname(os.path.abspath(__file__)))
    roots.append(os.path.dirname(os.path.abspath(sys.argv[0] or ".")))
    roots.append(os.path.dirname(os.path.abspath(sys.executable)))
    for r in roots:
        p = os.path.join(r, *parts)
        if os.path.exists(p):
            return p
    return os.path.join(roots[0], *parts)


MODEL_NAME = "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"


def model_path():
    env = os.environ.get("DIFFSAMPLER_MODEL")
    if env:
        return env
    return resource("model", MODEL_NAME)


def log(msg):
    try:
        p = os.path.join(app_dir(), "daemon.log")
        if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
            os.remove(p)
        with open(p, "a", encoding="utf-8") as f:
            f.write(time.strftime("%H:%M:%S ") + str(msg) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------
# UTAU argument helpers
# --------------------------------------------------------------------------
_B64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
_NOTE = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def note_to_hz(s):
    s = str(s).strip()
    try:  # OpenUTAU may hand over a plain MIDI number
        return 440.0 * 2 ** ((float(s) - 69) / 12)
    except ValueError:
        pass
    n = _NOTE[s[0].upper()]
    i = 1
    while i < len(s) and s[i] in "#b":
        n += 1 if s[i] == "#" else -1
        i += 1
    midi = 12 * (int(s[i:]) + 1) + n
    return 440.0 * 2 ** ((midi - 69) / 12)


def decode_pitchbend(s):
    out = []
    if not s:
        return out
    parts = s.strip().split("#")
    for i, part in enumerate(parts):
        if i % 2 == 0:
            for j in range(0, len(part) - 1, 2):
                v = (_B64.index(part[j]) << 6) | _B64.index(part[j + 1])
                out.append(v - 4096 if v >= 2048 else v)
        elif part and out:
            out.extend([out[-1]] * int(part))
    return out


def fnum(args, i, default):
    try:
        return float(args[i])
    except (IndexError, ValueError):
        return default


def parse_flags(flags):
    """g<n> formant shift, t<n> tune in cents.  Unknown flags are ignored."""
    res = {"g": 0.0, "t": 0.0}
    s = str(flags or "")
    i = 0
    while i < len(s):
        c = s[i]
        if c in "gt":
            j = i + 1
            while j < len(s) and (s[j] in "-+." or s[j].isdigit()):
                j += 1
            try:
                res[c] = float(s[i + 1:j])
            except ValueError:
                pass
            i = j
        else:
            i += 1
    return res


# --------------------------------------------------------------------------
# DSP (numpy only)
# --------------------------------------------------------------------------
def _slaney_hz_to_mel(f):
    import numpy as np
    f = np.asarray(f, dtype=np.float64)
    f_sp = 200.0 / 3
    mel = f / f_sp
    min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(f >= min_log_hz, min_log_mel + np.log(np.maximum(f, 1e-9) / min_log_hz) / logstep, mel)


def _slaney_mel_to_hz(m):
    import numpy as np
    m = np.asarray(m, dtype=np.float64)
    f_sp = 200.0 / 3
    f = f_sp * m
    min_log_hz, min_log_mel = 1000.0, 1000.0 / f_sp
    logstep = np.log(6.4) / 27.0
    return np.where(m >= min_log_mel, min_log_hz * np.exp(logstep * (m - min_log_mel)), f)


_MELFB = None


def mel_filterbank():
    """librosa.filters.mel(sr, n_fft, n_mels, fmin, fmax) (slaney scale + norm)."""
    global _MELFB
    if _MELFB is None:
        import numpy as np
        fft_f = np.linspace(0, SR / 2, N_FFT // 2 + 1)
        mel_f = _slaney_mel_to_hz(np.linspace(_slaney_hz_to_mel(FMIN), _slaney_hz_to_mel(FMAX), N_MELS + 2))
        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fft_f)
        lower = -ramps[:-2] / fdiff[:-1, None]
        upper = ramps[2:] / fdiff[1:, None]
        w = np.maximum(0, np.minimum(lower, upper))
        w *= (2.0 / (mel_f[2:N_MELS + 2] - mel_f[:N_MELS]))[:, None]
        _MELFB = (w.astype(np.float32), mel_f[1:-1].astype(np.float64))
    return _MELFB


def wav_to_mel(x):
    """x: float32 mono @44.1k -> (T,128) log-mel, same recipe as the vocoder training."""
    import numpy as np
    from numpy.lib.stride_tricks import sliding_window_view
    fb, _ = mel_filterbank()
    pad = (N_FFT - HOP) // 2
    n_frames = max(1, len(x) // HOP)
    need = (n_frames - 1) * HOP + N_FFT
    xp = np.pad(x, (pad, pad), mode="reflect") if len(x) > pad else np.pad(x, (pad, pad))
    if len(xp) < need:
        xp = np.pad(xp, (0, need - len(xp)))
    win = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)  # periodic hann
    out = np.empty((n_frames, N_MELS), dtype=np.float32)
    view = sliding_window_view(xp, N_FFT)[::HOP]
    for s in range(0, n_frames, 512):
        fr = view[s:s + 512][: n_frames - s] * win
        spec = np.fft.rfft(fr, axis=-1)
        mag = np.sqrt(spec.real ** 2 + spec.imag ** 2 + 1e-9)
        out[s:s + len(fr)] = np.log(np.maximum(mag @ fb.T, 1e-5))
    return out


def load_audio(path):
    import numpy as np
    import soundfile as sf
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1) if x.shape[1] > 1 else x[:, 0]
    if sr != SR:
        import soxr
        x = soxr.resample(x, sr, SR, quality="HQ")
    return np.ascontiguousarray(x, dtype=np.float32)


def apply_output_gain(wav, volume, auto_gain=True):
    """Match UTAU's usual note peak level, then apply the requested VOL."""
    import numpy as np

    y = np.asarray(wav, dtype=np.float32).copy()
    if y.size == 0:
        return y
    y[~np.isfinite(y)] = 0.0

    gain = 1.0
    if auto_gain:
        levels = np.abs(y)
        q_index = min(levels.size - 1, int(0.995 * (levels.size - 1)))
        reference_peak = float(np.partition(levels, q_index)[q_index])
        if reference_peak > 1e-6:
            # A robust near-peak avoids letting one transient/noise spike
            # determine the loudness of the entire phoneme.
            gain = min(16.0, 0.50 / reference_peak)

    gain *= max(0.0, float(volume)) / 100.0
    y *= gain
    return np.clip(y, -1.0, 1.0).astype(np.float32, copy=False)


def extract_f0(x, n_frames):
    """Harvest F0 aligned to mel frame centres ((i+0.5)*HOP). 0 = unvoiced."""
    import numpy as np
    import types
    # pyworld imports pkg_resources only to read its own version; a stub keeps
    # setuptools out of the bundle and saves its slow import.
    stub = types.ModuleType("pkg_resources")
    stub.get_distribution = lambda name: types.SimpleNamespace(version="0")
    sys.modules.setdefault("pkg_resources", stub)
    import pyworld as pw
    import soxr
    x22 = soxr.resample(x[HOP // 2:], SR, SR // 2, quality="HQ").astype(np.float64)
    f0, _ = pw.harvest(x22, SR // 2, f0_floor=50.0, f0_ceil=1100.0, frame_period=HOP_MS)
    f0 = f0.astype(np.float32)
    if len(f0) < n_frames:
        f0 = np.pad(f0, (0, n_frames - len(f0)))
    return f0[:n_frames]


class Features:
    def __init__(self):
        from collections import OrderedDict
        import threading
        self.mem = OrderedDict()
        self.locks = {}
        self.mutex = threading.Lock()
        self.cache_dir = os.path.join(app_dir(), "cache")
        os.makedirs(self.cache_dir, exist_ok=True)

    def get(self, path):
        import numpy as np
        st = os.stat(path)
        key = hashlib.sha1(f"{os.path.abspath(path)}|{st.st_size}|{st.st_mtime_ns}|v1".encode()).hexdigest()
        with self.mutex:
            if key in self.mem:
                self.mem.move_to_end(key)
                return self.mem[key]
            lock = self.locks.setdefault(key, __import__("threading").Lock())
        with lock:
            with self.mutex:
                if key in self.mem:
                    return self.mem[key]
            f = os.path.join(self.cache_dir, key + ".npz")
            feat = None
            if os.path.exists(f):
                try:
                    z = np.load(f)
                    feat = {"mel": z["mel"].astype(np.float32), "f0": z["f0"], "n": int(z["n"])}
                except Exception:
                    feat = None
            if feat is None:
                x = load_audio(path)
                mel = wav_to_mel(x)
                f0 = extract_f0(x, len(mel))
                feat = {"mel": mel, "f0": f0, "n": len(x)}
                try:
                    np.savez(f + ".tmp.npz", mel=mel.astype(np.float16), f0=f0, n=len(x))
                    os.replace(f + ".tmp.npz", f)
                except Exception:
                    pass
            with self.mutex:
                self.mem[key] = feat
                while len(self.mem) > 512:
                    self.mem.popitem(last=False)
                self.locks.pop(key, None)
            return feat


# --------------------------------------------------------------------------
# GPU session
# --------------------------------------------------------------------------
def make_session():
    import onnxruntime as ort
    want = os.environ.get("DIFFSAMPLER_PROVIDER", "auto").lower()
    dev = int(os.environ.get("DIFFSAMPLER_DEVICE", "0") or 0)
    avail = ort.get_available_providers()
    names = {"cuda": ["CUDAExecutionProvider"], "dml": ["DmlExecutionProvider"], "cpu": []}
    order = names.get(want, ["CUDAExecutionProvider", "DmlExecutionProvider"]) + ["CPUExecutionProvider"]
    provs = [p for p in order if p in avail]
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    if "DmlExecutionProvider" in provs:
        so.enable_mem_pattern = False
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    spec = [(p, {"device_id": dev}) if p in ("CUDAExecutionProvider", "DmlExecutionProvider") else p for p in provs]
    try:
        sess = ort.InferenceSession(model_path(), so, providers=spec)
    except Exception as e:
        log(f"GPU session failed ({e}); falling back to CPU")
        sess = ort.InferenceSession(model_path(), so, providers=["CPUExecutionProvider"])
    return sess


# --------------------------------------------------------------------------
# renderer
# --------------------------------------------------------------------------
class Engine:
    def __init__(self):
        import threading
        self.feats = Features()
        self.sess = make_session()
        self.gpu_lock = threading.Lock()
        self.provider = self.sess.get_providers()[0]
        # warm-up: builds the GPU graph so the first real note is fast
        try:
            self.vocode(None, None, warm=True)
        except Exception as e:
            log(f"warmup failed: {e}")

    def vocode(self, mel, f0, warm=False):
        import numpy as np
        if warm:
            T = BUCKET
            mel = np.full((T, N_MELS), MEL_FLOOR, np.float32)
            f0 = np.full(T, 220.0, np.float32)
        T = len(mel)
        Tp = max(BUCKET, -(-T // BUCKET) * BUCKET)
        m = np.empty((1, Tp, N_MELS), np.float32)
        f = np.empty((1, Tp), np.float32)
        m[0, :T] = mel
        f[0, :T] = f0
        # Keep the vocoder's right-hand receptive field on the final phoneme
        # instead of conditioning it on artificial silence. This makes the
        # rendered tail much more stable when the wavtool crossfades notes.
        if T < Tp:
            m[0, T:] = mel[-1]
            f[0, T:] = f0[-1]
        with self.gpu_lock:
            wav = self.sess.run(None, {"mel": m, "f0": f})[0][0]
        return wav[: T * HOP]

    def render(self, argv):
        import numpy as np
        import soundfile as sf
        t0 = time.time()
        inp, outp = argv[0], argv[1]
        note_hz = note_to_hz(argv[2]) if len(argv) > 2 else 440.0
        vel = fnum(argv, 3, 100.0)
        flags = parse_flags(argv[4] if len(argv) > 4 else "")
        offset = fnum(argv, 5, 0.0)
        length = max(10.0, fnum(argv, 6, 500.0))
        cons = max(0.0, fnum(argv, 7, 0.0))
        cutoff = fnum(argv, 8, 0.0)
        volume = fnum(argv, 9, 100.0)
        modul = fnum(argv, 10, 0.0)
        tempo_s = argv[11] if len(argv) > 11 else "!120"
        try:
            tempo = float(str(tempo_s).lstrip("!"))
        except ValueError:
            tempo = 120.0
        pb = decode_pitchbend(argv[12] if len(argv) > 12 else "")

        ft = self.feats.get(inp)
        mel, f0s, nsamp = ft["mel"], ft["f0"], ft["n"]
        T = len(mel)
        total_ms = nsamp * 1000.0 / SR
        start = min(max(offset, 0.0), max(total_ms - 2 * HOP_MS, 0.0))
        end = start - cutoff if cutoff < 0 else total_ms - cutoff
        end = min(max(end, start + 2 * HOP_MS), total_ms)
        cons = min(cons, end - start)
        cons_out = cons * 2.0 ** (1.0 - vel / 100.0)
        cons_out = min(cons_out, length)

        N = max(1, int(np.ceil(length / HOP_MS)))
        t_out = (np.arange(N) + 0.5) * HOP_MS
        t_src = np.empty(N)
        a = t_out < cons_out
        if cons_out > 0:
            t_src[a] = start + t_out[a] / cons_out * cons
        rest_out = max(length - cons_out, 1e-3)
        t_src[~a] = start + cons + (t_out[~a] - cons_out) / rest_out * (end - start - cons)
        fr = np.clip(t_src / HOP_MS - 0.5, 0, T - 1)

        i0 = np.floor(fr).astype(int)
        i1 = np.minimum(i0 + 1, T - 1)
        w = (fr - i0).astype(np.float32)
        m = mel[i0] * (1 - w)[:, None] + mel[i1] * w[:, None]

        # --- F0 (voiced-aware interpolation, log domain)
        fa, fb = f0s[i0], f0s[i1]
        va, vb = fa > 0, fb > 0
        both = va & vb
        f0 = np.zeros(N, np.float32)
        with np.errstate(divide="ignore", invalid="ignore"):
            f0[both] = np.exp((1 - w[both]) * np.log(fa[both]) + w[both] * np.log(fb[both]))
        only_a, only_b = va & ~vb, vb & ~va
        f0[only_a] = np.where(w[only_a] < 0.5, fa[only_a], 0)
        f0[only_b] = np.where(w[only_b] >= 0.5, fb[only_b], 0)

        seg = (np.arange(T) + 0.5) * HOP_MS
        sel = (seg >= start + cons) & (seg < end) & (f0s > 0)
        if not sel.any():
            sel = (seg >= start) & (seg < end) & (f0s > 0)
        if sel.any():
            f0_ref = float(np.exp(np.median(np.log(f0s[sel]))))
            base = note_hz * 2 ** (flags["t"] / 1200.0)
            pbc = np.zeros(N)
            if pb:
                dt = 60.0 / (tempo * 96.0) * 1000.0
                pbc = np.interp(t_out, np.arange(len(pb)) * dt, np.asarray(pb, float))
            scale = modul / 100.0
            v = f0 > 0
            f0[v] = base * (f0[v] / f0_ref) ** scale * 2 ** (pbc[v] / 1200.0)
            shift_st = abs(12 * np.log2(base / f0_ref))
            if shift_st > 12:
                log(f"note: {inp} shifted {shift_st:.1f} st (model is trained for +-12)")
        else:
            f0[:] = 0

        # --- formant (gender) flag: warp the mel axis
        g = flags["g"]
        if g:
            _, centers = mel_filterbank()
            r = 2.0 ** (g / 200.0)
            src = np.interp(centers / r, centers, np.arange(N_MELS))
            j0 = np.clip(np.floor(src).astype(int), 0, N_MELS - 1)
            j1 = np.minimum(j0 + 1, N_MELS - 1)
            wj = (src - j0).astype(np.float32)
            m = m[:, j0] * (1 - wj) + m[:, j1] * wj

        wav = self.vocode(m.astype(np.float32), f0)
        n_out = int(round(length * SR / 1000.0))
        wav = wav[:n_out]
        auto_gain = os.environ.get("DIFFSAMPLER_AUTOGAIN", "1").strip().lower() not in ("0", "false", "off", "no")
        wav = apply_output_gain(wav, volume, auto_gain=auto_gain)
        os.makedirs(os.path.dirname(os.path.abspath(outp)) or ".", exist_ok=True)
        sf.write(outp, wav, SR, subtype="PCM_16", format="WAV")
        log(f"{os.path.basename(inp)} -> {os.path.basename(outp)} {length:.0f}ms in {(time.time() - t0) * 1000:.0f}ms [{self.provider}]")


# --------------------------------------------------------------------------
# daemon + client
# --------------------------------------------------------------------------
def token_file():
    return os.path.join(app_dir(), "daemon.json")


def daemon_main():
    import threading
    import secrets
    if _rpc({"cmd": "ping"}, timeout=3) is not None:
        return 0  # a healthy worker is already running
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.bind(("127.0.0.1", 0))  # OS-assigned port: a fixed one can collide with outgoing connections
    srv.listen(64)
    token = secrets.token_hex(16)
    tmp = token_file() + f".{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump({"token": token, "pid": os.getpid(), "port": srv.getsockname()[1]}, f)
    os.replace(tmp, token_file())
    log(f"daemon up, pid {os.getpid()}, port {srv.getsockname()[1]}")
    state = {"last": time.time(), "busy": 0}
    idle = float(os.environ.get("DIFFSAMPLER_IDLE", "900"))
    engine_box = {}
    ready = threading.Event()

    def load():
        try:
            engine_box["e"] = Engine()
            log(f"model ready on {engine_box['e'].provider}")
        except Exception as e:
            engine_box["err"] = repr(e)
            log(f"engine failed: {e!r}")
        ready.set()

    threading.Thread(target=load, daemon=True).start()

    def quit_now():
        try:
            with open(token_file()) as f:
                mine = json.load(f).get("pid") == os.getpid()
            if mine:
                os.remove(token_file())
        except Exception:
            pass
        os._exit(0)

    def watchdog():
        while True:
            time.sleep(5)
            if state["busy"] == 0 and time.time() - state["last"] > idle:
                log("idle timeout")
                quit_now()

    threading.Thread(target=watchdog, daemon=True).start()

    def handle(conn):
        state["busy"] += 1
        try:
            conn.settimeout(600)
            data = b""
            while not data.endswith(b"\n"):
                chunk = conn.recv(65536)
                if not chunk:
                    break
                data += chunk
            req = json.loads(data.decode("utf-8"))
            if req.get("token") != token:
                resp = {"ok": False, "error": "bad token"}
            elif req.get("cmd") == "stop":
                conn.sendall(b'{"ok":true}\n')
                quit_now()
            elif req.get("cmd") == "ping":
                resp = {"ok": True}
            else:
                ready.wait()
                if "e" not in engine_box:
                    resp = {"ok": False, "error": engine_box.get("err", "engine failed")}
                elif req.get("cmd") == "info":
                    resp = {"ok": True, "provider": engine_box["e"].provider}
                else:
                    engine_box["e"].render(req["argv"])
                    resp = {"ok": True}
        except Exception as e:
            import traceback
            log(traceback.format_exc())
            resp = {"ok": False, "error": repr(e)}
        finally:
            state["busy"] -= 1
            state["last"] = time.time()
        try:
            conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
        except OSError:
            pass
        conn.close()

    while True:
        c, _ = srv.accept()
        threading.Thread(target=handle, args=(c,), daemon=True).start()


def _rpc(req, timeout=600):
    try:
        with open(token_file()) as f:
            info = json.load(f)
        req["token"] = info["token"]
    except Exception:
        return None
    try:
        s = socket.create_connection(("127.0.0.1", int(info["port"])), timeout=3)
    except (OSError, KeyError, ValueError):
        return None
    s.settimeout(timeout)
    s.sendall((json.dumps(req) + "\n").encode("utf-8"))
    data = b""
    while not data.endswith(b"\n"):
        c = s.recv(65536)
        if not c:
            break
        data += c
    s.close()
    return json.loads(data.decode("utf-8")) if data else None


def self_command():
    """Command line that starts another copy of *this* program (source, PyInstaller or Nuitka)."""
    import shutil
    comp = globals().get("__compiled__")
    if comp is not None:  # Nuitka (onefile: the launcher path is argv0 as originally typed)
        cands = [getattr(comp, "original_argv0", None), sys.argv[0], sys.executable]
    elif getattr(sys, "frozen", False):  # PyInstaller
        cands = [sys.executable]
    else:
        return [sys.executable, os.path.abspath(__file__)]
    for c in cands:
        if not c:
            continue
        p = c if os.path.isabs(c) else (shutil.which(c) or os.path.abspath(c))
        if os.path.isfile(p):
            return [p]
    return [sys.executable]


def spawn_daemon():
    import subprocess
    cmd = self_command() + ["--daemon"]
    kw = {}
    if os.name == "nt":
        kw["creationflags"] = 0x00000008 | 0x00000200 | 0x08000000  # DETACHED | NEW_GROUP | NO_WINDOW
    else:
        kw["start_new_session"] = True
    log(f"spawning worker: {cmd}")
    return subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, **kw)


def client_render(argv):
    req = {"cmd": "render", "argv": argv}
    resp = _rpc(dict(req))
    if resp is not None:
        return resp
    # Several notes may start at once: only the client holding the lock spawns the worker.
    lock = os.path.join(app_dir(), "spawn.lock")
    proc = None
    try:
        if os.path.exists(lock) and time.time() - os.path.getmtime(lock) > 60:
            os.remove(lock)  # stale
        os.close(os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
        proc = spawn_daemon()
    except OSError:
        pass  # someone else is already starting it
    try:
        deadline = time.time() + 90
        while resp is None and time.time() < deadline:
            time.sleep(0.15)
            resp = _rpc(dict(req))
            if resp is None and proc is not None and proc.poll() is not None:  # worker died
                time.sleep(0.3)
                resp = _rpc(dict(req))
                break
    finally:
        if proc is not None:
            try:
                os.remove(lock)
            except OSError:
                pass
    return resp


def render_inprocess(argv):
    Engine().render(argv)


def precache(root):
    from concurrent.futures import ThreadPoolExecutor
    feats = Features()
    files = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.lower().endswith(".wav")]
    print(f"{len(files)} wav files")
    done = [0]

    def one(p):
        try:
            feats.get(p)
        except Exception as e:
            print("skip", p, e)
        done[0] += 1
        if done[0] % 25 == 0 or done[0] == len(files):
            print(f"  {done[0]}/{len(files)}")

    with ThreadPoolExecutor(max_workers=max(1, (os.cpu_count() or 2))) as ex:
        list(ex.map(one, files))
    print("done")


def main():
    a = sys.argv[1:]
    if not a or a[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if a[0] == "--version":
        print(VERSION)
        return 0
    if a[0] == "--notice":
        p = resource("model", "NOTICE.txt")
        print(open(p, encoding="utf-8").read() if os.path.exists(p) else "see https://github.com/openvpi/vocoders")
        print("\nThe separately distributed vocoder model is licensed CC BY-NC-SA 4.0:\n"
              "https://creativecommons.org/licenses/by-nc-sa/4.0/\n"
              "Official release page: https://github.com/openvpi/vocoders/releases/tag/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02\n"
              "DiffSampler only converts the model for use (no weights were modified).")
        return 0
    if a[0] == "--where":
        comp = globals().get("__compiled__")
        print({"executable": sys.executable, "argv0": sys.argv[0], "frozen": getattr(sys, "frozen", False),
               "nuitka": comp is not None, "original_argv0": getattr(comp, "original_argv0", None),
               "spawn": self_command()})
        return 0
    if a[0] == "--daemon":
        return daemon_main()
    if a[0] == "--stop":
        _rpc({"cmd": "stop"}, timeout=5)
        print("stopped")
        return 0
    if a[0] == "--info":
        r = _rpc({"cmd": "info"})
        if r is None:
            spawn_daemon()
            for _ in range(600):
                time.sleep(0.15)
                r = _rpc({"cmd": "info"})
                if r:
                    break
        print(r)
        return 0
    if a[0] == "--precache":
        precache(a[1])
        return 0

    argv = [os.path.abspath(a[0]), os.path.abspath(a[1])] + a[2:]
    try:
        if os.environ.get("DIFFSAMPLER_NODAEMON") == "1":
            render_inprocess(argv)
            return 0
        resp = client_render(argv)
        if resp is None or not resp.get("ok"):
            err = (resp or {}).get("error", "worker did not start")
            log(f"client falling back to in-process: {err}")
            render_inprocess(argv)
        return 0
    except Exception as e:
        sys.stderr.write(f"DiffSampler error: {e!r}\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
