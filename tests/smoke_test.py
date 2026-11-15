"""Synthetic end-to-end check: makes a vowel-like sample, renders notes through
the real CLI path, and verifies pitch / length / level of the results."""
import os, sys, subprocess, time, tempfile
import numpy as np, soundfile as sf, pyworld as pw

HERE = os.path.dirname(os.path.abspath(__file__))
EXE = [os.environ["DS_EXE"]] if os.environ.get("DS_EXE") else [sys.executable, os.path.join(HERE, "..", "diffsampler.py")]
SR = 44100
tmp = tempfile.mkdtemp()

def make_sample(path):
    t = np.arange(int(SR * 2.0)) / SR
    f0 = 220 * (1 + 0.01 * np.sin(2 * np.pi * 5 * t))
    ph = 2 * np.pi * np.cumsum(f0) / SR
    x = np.zeros_like(t)
    for h in range(1, 60):
        fh = h * 220
        if fh > 9000: break
        amp = (np.exp(-((fh - 700) / 250) ** 2) + 0.6 * np.exp(-((fh - 1200) / 300) ** 2) + 0.05) / h ** 0.3
        x += amp * np.sin(h * ph)
    x /= np.abs(x).max() / 0.5
    n = int(0.15 * SR)                       # noisy "consonant"
    x[:n] = 0.15 * np.random.RandomState(0).randn(n) * np.hanning(n)
    sf.write(path, x.astype(np.float32), SR, subtype="PCM_16")

def f0_of(path):
    # harmonic-sum pitch estimate (Harvest is unreliable on this artificial vowel)
    y, sr = sf.read(path)
    seg = y[len(y) // 3: len(y) // 3 + 16384]
    sp = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))); fr = np.fft.rfftfreq(len(seg), 1 / sr)
    cand = np.arange(60, 900, 0.5)
    score = [sum(np.interp(h * f, fr, sp) for h in range(1, 12)) for f in cand]
    return float(cand[int(np.argmax(score))]), y

src = os.path.join(tmp, "a.wav"); make_sample(src)
def run(name, *args, env=None):
    out = os.path.join(tmp, name)
    t0 = time.time()
    r = subprocess.run(EXE + [src, out, *map(str, args)], capture_output=True, text=True, env={**os.environ, **(env or {})})
    assert r.returncode == 0, r.stderr
    return out, time.time() - t0

# args: pitch vel flags offset length cons cutoff vol mod tempo pb
o, dt1 = run("e4.wav", "E4", 100, "", 0, 800, 150, -1500, 100, 100, "!120", "")
f, y = f0_of(o); print(f"E4  target 329.6  got {f:.1f}  len {len(y)/SR*1000:.0f}ms  rms {np.sqrt((y**2).mean()):.3f}  first-call {dt1:.2f}s")
assert abs(12*np.log2(f/329.63)) < 0.5 and abs(len(y)/SR*1000-800) < 2
o, dt2 = run("c3.wav", "C3", 100, "", 0, 1200, 150, -1500, 100, 100, "!120", "")
f, y = f0_of(o); print(f"C3  target 130.8  got {f:.1f}  len {len(y)/SR*1000:.0f}ms  warm-call {dt2:.2f}s")
assert abs(12*np.log2(f/130.81)) < 0.5
o, _ = run("g.wav", "A3", 100, "g60", 0, 600, 150, -1500, 100, 100, "!120", "")
f, y = f0_of(o); print(f"A3+g60 target 220 got {f:.1f}")
# pitch bend: +1200 cents constant ("Bg" == +? use explicit): 'AA#20#' = 0 ; build +100 cents run
pbs = "BG#40#"   # 1*64+6=70 cents held
o, _ = run("pb.wav", "A3", 100, "", 0, 600, 150, -1500, 100, 100, "!120", pbs)
f, y = f0_of(o); print(f"A3 +70c  target {220*2**(70/1200):.1f} got {f:.1f}")
o1, _ = run("v100.wav", "A3", 100, "", 0, 600, 150, -1500, 100, 100, "!120", "")
o2, _ = run("v50.wav", "A3", 100, "", 0, 600, 150, -1500, 50, 100, "!120", "")
y100 = f0_of(o1)[1]; y50 = f0_of(o2)[1]
r = np.sqrt((y50**2).mean()) / max(np.sqrt((y100**2).mean()), 1e-9)
print(f"volume 50 -> rms ratio {r:.3f}"); assert abs(r - 0.5) < 0.02
p995 = float(np.quantile(np.abs(y100), 0.995))
print(f"auto-level p99.5 peak {p995:.3f}"); assert 0.35 <= p995 <= 0.65
subprocess.run(EXE + ["--stop"])
print("OK", tmp)

# --- cold-start burst: 6 notes launched at once must share ONE worker
import concurrent.futures as cf, json
LOG = os.path.join(os.environ.get("LOCALAPPDATA") or os.path.join(os.path.expanduser("~"), ".cache"), "DiffSampler", "daemon.log")
subprocess.run(EXE + ["--stop"]); time.sleep(0.5)
before = open(LOG).read().count("daemon up")
def one(i):
    return run(f"burst{i}.wav", "C4", 100, "", 0, 500, 150, -1500, 100, 100, "!120", "")[1]
with cf.ThreadPoolExecutor(6) as ex:
    times = list(ex.map(one, range(6)))
print("burst times", [round(t, 2) for t in times])
log = open(LOG).read()
ups = log.count("daemon up") - before; print("workers started during burst:", ups)
assert ups == 1 and "falling back" not in log.split("daemon up")[-1]
subprocess.run(EXE + ["--stop"]); print("BURST OK")
