# DiffSampler — native C++ build

This is the native Windows C++ build of DiffSampler 1.0.6. It keeps the same model, 44.1 kHz / 512-hop / 128-bin feature format, Harvest F0 analysis, UTAU argument order, note timing, pitch/modulation/formant controls, and ONNX vocoder inference.

## What changed

- The renderer is a conventional x64 PE executable with standard Windows version information and an `asInvoker` application manifest. There is no Python runtime, PyInstaller, Nuitka, one-file unpacker, or temporary extraction directory.
- The ONNX weights are distributed as a normal sidecar file at `model\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx`. The executable reads the file directly; it does not unpack the weights into a temporary directory.
- DirectML and ONNX Runtime DLLs are placed beside `DiffSampler.exe`; users do not need to install Python, ONNX Runtime, or a separate GPU toolkit.
- The persistent worker uses a per-user Windows named pipe instead of an ephemeral localhost TCP port, JSON token file, and random authentication token.
- The worker exits normally after the configured idle period. `--stop` asks it to stop cleanly.
- The worker keeps a listening pipe instance open and serves every connection on its own thread, so as many notes are rendered at once as the host sends. Feature extraction, interpolation, gain and WAV writing overlap; only the vocoder call itself runs one note at a time.
- The DirectML adapter is selected automatically: the non-software adapter with the most dedicated video memory (see `DIFFSAMPLER_DEVICE`). Adapter 0 is often the integrated GPU, which is several times slower than a discrete one.
- Feature extraction for new source WAVs is faster: the windowed-sinc resampler precomputes its kernel per phase and the mel filterbank sums only the non-zero part of each filter. The mel output is identical to the previous implementation.
- Source-WAV features are cached under `%LOCALAPPDATA%\DiffSampler\cache` using a small versioned binary cache format.
- Each note is peak-matched to a consistent, conservative level before UTAU volume is applied. The normalizer uses a near-peak percentile to avoid one transient dominating the gain.
- Unused ONNX bucket frames repeat the final phoneme frame instead of feeding artificial silence into the vocoder right after the note. This gives the rendered tail a more stable context for the wavtool overlap stage.
- The executable does not contact the network. The build script downloads version-pinned dependencies and the licensed ONNX model only while building, then packages the model as a sidecar file.

The native renderer uses WORLD's Harvest implementation for F0 extraction. The native build uses the original neural PC-NSF-HiFiGAN ONNX vocoder for synthesis. WORLD Harvest is used only for F0 analysis, as in the Python implementation; it is not used to synthesize audio. The native input-rate converter is a windowed-sinc implementation rather than `soxr`, so resampled source WAVs may differ slightly from the Python build. Validate output on representative voicebanks before replacing your existing resampler.

## Build (Windows 10/11, x64)

Build-machine requirements:

1. Visual Studio 2026 with the **Desktop development with C++** workload and a Windows 10/11 SDK.
2. CMake 4.2 or newer.
3. `curl.exe` (included with current Windows 10/11 installations).
4. An internet connection during the build only.

Run `build_windows.bat` from this folder. It downloads ONNX Runtime DirectML 1.20.1 and the official OpenVPI PC-NSF-HiFiGAN model archive; CMake fetches the WORLD v1.0.1 source and builds `build\dist\DiffSampler.exe`.

The output folder contains:

- `DiffSampler.exe` — native renderer.
- `model\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx` — model weights loaded directly from disk.
- ONNX Runtime / DirectML redistributable DLLs — keep these next to the EXE.
- `README.txt` and `model\NOTICE.txt` — keep the attribution and model license with any redistributed copy.

No Python, C++ compiler, or model download is required on the user's machine; keep the included `model` folder with the executable. The build script also copies the Visual C++ runtime DLLs from the build machine when they are present. To verify a file later, run `certutil -hashfile DiffSampler.exe SHA256` from the distribution folder and compare it with the matching line in `SHA256SUMS.txt`; repeat for any other file you want to check. The user's Windows graphics driver and Windows DirectX/DirectML components still need to be functional for GPU inference; CPU fallback is available.

## UTAU use

Set `DiffSampler.exe` as the resampler in UTAU / OpenUtau, using the normal UTAU resampler arguments:

```text
DiffSampler.exe <in.wav> <out.wav> <pitch> <vel> <flags> <offset> <length> <consonant> <cutoff> <volume> <modulation> <tempo> <pitchbend>
```

Supported flags and parameters are inherited from the supplied Python implementation:

- Pitch: note name (for example `E4`) or MIDI number.
- `vel`: consonant-region timing adjustment.
- `g<n>`: formant/gender-axis shift.
- `t<n>`: tuning in cents.
- `offset`, `length`, `consonant`, `cutoff`, `volume`, `modulation`, tempo and UTAU base64/RLE pitchbend.

Other UTAU flags remain ignored, just as in the supplied source.

## Commands

```text
DiffSampler.exe --help
DiffSampler.exe --version
DiffSampler.exe --info
DiffSampler.exe --stop
DiffSampler.exe --precache "C:\path\to\voicebank"
DiffSampler.exe --precache "C:\path\to\OpenUtau\Cache" src-
DiffSampler.exe --notice
DiffSampler.exe --where
```

`--precache` pre-computes the feature cache for every `.wav` under the given folder using one thread per physical CPU core (never more threads than files). The optional second argument limits it to files whose name starts with that prefix. In the OpenUtau builds we tested, each note's source is a `src-*.wav` file in OpenUtau's `Cache` folder, so the second form pre-computes everything that has already been used.

Environment variables:

- `DIFFSAMPLER_PROVIDER=auto|dml|cpu` (default `auto`). `cuda` is not included in this DirectML build; use `dml` or `cpu`.
- `DIFFSAMPLER_DEVICE=auto|<index>` chooses the DirectML adapter (default `auto`). `auto` picks the non-software adapter with the most dedicated video memory, so a laptop's discrete GPU is used instead of the integrated one (adapter 0 is often the integrated GPU and was about 4-5x slower in testing). Set a number to force a specific adapter. `DiffSampler.exe --info` shows which adapter was selected.
- `DIFFSAMPLER_IDLE=900` sets worker idle shutdown in seconds (60–86400): the worker exits after this long without requests.
- `DIFFSAMPLER_NODAEMON=1` disables the worker and renders inside each UTAU process. This is useful for diagnosis, but normally slower because it reloads the ONNX model each time.
- `DIFFSAMPLER_MODEL=<path>` uses an alternate external ONNX model instead of the default `model\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx`.
- `DIFFSAMPLER_AUTOGAIN=0` disables the default per-note level matching; UTAU volume is still applied.

Log: `%LOCALAPPDATA%\DiffSampler\daemon.log`.

Performance notes:

- Run `DiffSampler.exe --info`. It shows the provider and the adapter in use. `CPUExecutionProvider` means DirectML could not be initialised and the vocoder runs on the CPU, which is much slower.
- The first note that uses a source WAV has to compute its features, mostly WORLD Harvest (about two seconds for a ten second file). Later notes use the cache, and `--precache` avoids these pauses while editing.
- Raise `DIFFSAMPLER_IDLE` if the worker should stay loaded through long pauses.

## Antivirus false-positive notes

The root-level `diffsampler.py` is retained as a legacy Python/reference renderer for source comparison and its tests; it is not compiled or bundled by `build_windows.bat`. It retains the Python edition's older loopback-TCP worker implementation. The 1.0.6 native Windows distribution is built only from `cpp/diffsampler.cpp` and uses the named-pipe worker described below. Do not use a PyInstaller/Nuitka build of the legacy script as a substitute for the native 1.0.6 release.

Some heuristic scanners scrutinize unsigned programs that start a background worker. DiffSampler keeps this worker because it avoids reloading the neural model for every UTAU note. The worker is started only by DiffSampler itself, communicates over a per-user Windows named pipe, writes a local diagnostic log, and exits after an idle period or `--stop`. It does not contact the network.

Version 1.0.6 also stops embedding the large ONNX weight file in the PE executable. The model is a separate, user-inspectable file in the `model` directory; no self-extraction or temporary unpacking is involved. The build script no longer invokes PowerShell with an execution-policy bypass. The EXE has standard version metadata and an `asInvoker` manifest; it does not request administrator privileges.
The MSVC project embeds that manifest once through `resources/app.rc.in`; `/MANIFEST:NO` disables the linkers automatic second manifest resource, preventing the duplicate `RT_MANIFEST` resource that caused `CVTRES Cvt1100`. `NOMINMAX` and `WIN32_LEAN_AND_MEAN` are defined only by CMake, avoiding duplicate macro-definition warnings.

These changes remove avoidable behaviors but cannot guarantee that every antivirus vendor will classify an unsigned build as clean. Do not disable antivirus, add broad exclusions, pack/obfuscate the executable, or modify the binary to fool scanners. For a public release, publish the source, build instructions, SHA-256 manifest, and exact detection details. Submit false positives directly to the affected vendors. Code signing can establish publisher identity and improve reputation, but it does not prove that a program is harmless or guarantee zero detections.

## Model license

The PC-NSF-HiFiGAN weights are provided by the OpenVPI Team under **CC BY-NC-SA 4.0**. Do not sell or charge for redistribution of the model. Include the complete `model/NOTICE.txt` and the license reference when redistributing a build. The C++ port does not modify the model weights. Keep the model file beside the executable at `model\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx`. Keep third-party notices for ONNX Runtime, DirectML, and WORLD with source or release packages. The build output includes the WORLD modified-BSD license and a `SHA256SUMS.txt` manifest. The manifest allows you to verify that the distributed files have not changed after the checksums were generated; it is not a vendor signature or a guarantee that the software is malware-free.


### Build dependency download note

The Windows build script downloads ONNX Runtime DirectML from the official NuGet package endpoint (`Microsoft.ML.OnnxRuntime.DirectML`). The old GitHub release-asset URL for a `onnxruntime-win-x64-directml-1.20.1.zip` file does not exist and returns HTTP 404. The script stages the NuGet package's headers, import library, and runtime DLLs into the directory layout expected by CMake.


### DirectML header layout correction

The DirectML NuGet package exposes `dml_provider_factory.h` separately from the standard ONNX Runtime headers. The build script now searches the extracted package for that header and stages it at `include/dml_provider_factory.h`, matching the source include (`#include <dml_provider_factory.h>`). This avoids assuming it lives under the internal `onnxruntime/core/providers/dml` source-tree path.


Build-system note: this package targets Visual Studio 2026 using CMake generator `Visual Studio 18 2026`; CMake 4.2 or newer is required for that generator. The Windows build script requests the x64 platform. It removes no antivirus settings and does not change Windows security policies.


## Note overlap behavior

DiffSampler is a resampler, not a wavtool. It creates one rendered phoneme at a time; the selected UTAU/OpenUtau wavtool combines those clips and performs the actual overlap/crossfade using the voicebank's OTO `preutter` and `overlap` values. Those overlap values are not part of the standard resampler argument list, so DiffSampler cannot independently change the crossfade length. This release improves the note-tail conditioning supplied to the vocoder so the wavtool receives a less abrupt end. If transitions are still mistimed, inspect the OTO overlap/preutter settings and the selected wavtool.
