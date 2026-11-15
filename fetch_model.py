"""Downloads the PC-NSF-HiFiGAN ONNX model + its licence notice into ./model (stdlib only)."""
import io, os, sys, urllib.request, zipfile
URL = ("https://github.com/openvpi/vocoders/releases/download/"
       "pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
os.makedirs(out, exist_ok=True)
if os.path.exists(os.path.join(out, "pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx")):
    print("model already present"); sys.exit(0)
print("downloading", URL)
data = urllib.request.urlopen(URL, timeout=300).read()
with zipfile.ZipFile(io.BytesIO(data)) as z:  # .oudep is a plain zip
    for n in z.namelist():
        if n.endswith(".onnx") or n == "NOTICE.txt":
            with open(os.path.join(out, n), "wb") as f:
                f.write(z.read(n))
            print("extracted", n)
