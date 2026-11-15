@echo off
setlocal EnableExtensions
rem Native C++ build for DiffSampler. Run this file from Explorer or a terminal.
rem Requires Visual Studio 2026 C++ workload, CMake 4.2+, PowerShell 5.1+.

set "ROOT=%~dp0"
cd /d "%ROOT%" || goto :err
set "DEPS=%ROOT%_deps"
set "ORT_VERSION=1.20.1"
set "ORT_DIR=%DEPS%\onnxruntime-win-x64-directml-%ORT_VERSION%"
set "ORT_NUPKG=%DEPS%\Microsoft.ML.OnnxRuntime.DirectML.%ORT_VERSION%.nupkg"
set "ORT_ZIP=%DEPS%\Microsoft.ML.OnnxRuntime.DirectML.%ORT_VERSION%.zip"
set "MODEL=model\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"
set "MODEL_ARCHIVE=%DEPS%\pc-nsf-hifigan.oudep"
set "MODEL_ZIP=%DEPS%\pc-nsf-hifigan.zip"
set "MODEL_URL=https://github.com/openvpi/vocoders/releases/download/pc-nsf-hifigan-44.1k-hop512-128bin-2025.02/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.oudep"

 echo [1/6] Checking build tools...
where cmake.exe >nul 2>nul
if errorlevel 1 (echo ERROR: CMake 4.2 or newer was not found on PATH.& goto :err)
where curl.exe >nul 2>nul
if errorlevel 1 (echo ERROR: curl.exe was not found on PATH.& goto :err)
where powershell.exe >nul 2>nul
if errorlevel 1 (echo ERROR: Windows PowerShell was not found.& goto :err)

cmake --version
if errorlevel 1 goto :err
if not exist "%DEPS%" mkdir "%DEPS%"
if errorlevel 1 (echo ERROR: Could not create dependency directory: "%DEPS%".& goto :err)

if not exist "%ORT_DIR%\include\dml_provider_factory.h" (
  echo [2/6] Downloading ONNX Runtime DirectML %ORT_VERSION% NuGet package...
  rem ONNX Runtime DirectML is distributed as a NuGet package, not the ZIP
  rem asset URL used by older scripts. The old GitHub URL returns HTTP 404.
  if not exist "%ORT_NUPKG%" (
    echo Downloading ONNX Runtime DirectML NuGet package...
    curl.exe --fail --show-error --location --retry 3 --retry-all-errors --output "%ORT_NUPKG%" "https://www.nuget.org/api/v2/package/Microsoft.ML.OnnxRuntime.DirectML/%ORT_VERSION%"
    if errorlevel 1 (echo ERROR: ONNX Runtime DirectML NuGet download failed.& goto :err)
  ) else (
    echo Reusing the previously downloaded NuGet package...
  )
  copy /y "%ORT_NUPKG%" "%ORT_ZIP%" >nul
  if errorlevel 1 (echo ERROR: Could not stage ONNX Runtime NuGet package.& goto :err)
  if exist "%DEPS%\ort-package" rmdir /s /q "%DEPS%\ort-package"
  if exist "%ORT_DIR%" rmdir /s /q "%ORT_DIR%"
  powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath '%ORT_ZIP%' -DestinationPath '%DEPS%\ort-package' -Force; $pkg='%DEPS%\ort-package'; $inc=Join-Path $pkg 'build\native\include'; $native=Join-Path $pkg 'runtimes\win-x64\native'; if (!(Test-Path (Join-Path $inc 'onnxruntime_cxx_api.h'))) { throw 'NuGet package is missing build/native/include headers' }; if (!(Test-Path (Join-Path $native 'onnxruntime.dll'))) { throw 'NuGet package is missing runtimes/win-x64/native/onnxruntime.dll' }; $dml=Get-ChildItem -LiteralPath $pkg -Filter 'dml_provider_factory.h' -File -Recurse | Select-Object -First 1; if (!$dml) { throw 'NuGet package does not contain dml_provider_factory.h' }; New-Item -ItemType Directory -Force -Path '%ORT_DIR%\include','%ORT_DIR%\lib','%ORT_DIR%\bin' | Out-Null; Copy-Item -Path (Join-Path $inc '*') -Destination '%ORT_DIR%\include' -Recurse -Force; Copy-Item -LiteralPath $dml.FullName -Destination '%ORT_DIR%\include\dml_provider_factory.h' -Force; Copy-Item -Path (Join-Path $native '*') -Destination '%ORT_DIR%\bin' -Recurse -Force; Copy-Item -Path (Join-Path $native '*') -Destination '%ORT_DIR%\lib' -Recurse -Force"
  if errorlevel 1 (echo ERROR: Could not extract or stage ONNX Runtime DirectML SDK.& goto :err)
)
if not exist "%ORT_DIR%\include\onnxruntime_cxx_api.h" (echo ERROR: ONNX Runtime headers are missing after extraction.& goto :err)
if not exist "%ORT_DIR%\include\dml_provider_factory.h" (echo ERROR: DirectML provider header is missing after staging.& goto :err)
if not exist "%ORT_DIR%\lib\onnxruntime.lib" (echo ERROR: ONNX Runtime import library is missing from the NuGet package.& goto :err)
if not exist "%ORT_DIR%\bin\onnxruntime.dll" (echo ERROR: ONNX Runtime DLL is missing from the NuGet package.& goto :err)

if not exist "%MODEL%" (
  echo [3/6] Downloading licensed OpenVPI PC-NSF-HiFiGAN model...
  curl.exe --fail --show-error --location --retry 3 --output "%MODEL_ARCHIVE%" "%MODEL_URL%"
  if errorlevel 1 (echo ERROR: Vocoder model download failed.& goto :err)
  copy /y "%MODEL_ARCHIVE%" "%MODEL_ZIP%" >nul
  if errorlevel 1 (echo ERROR: Could not stage model archive.& goto :err)
  if exist "%DEPS%\model-extract" rmdir /s /q "%DEPS%\model-extract"
  powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; Expand-Archive -LiteralPath '%MODEL_ZIP%' -DestinationPath '%DEPS%\model-extract' -Force; $m=Get-ChildItem -LiteralPath '%DEPS%\model-extract' -Filter '*.onnx' -File -Recurse | Select-Object -First 1; if (-not $m) { throw 'ONNX model not found in model archive' }; Copy-Item -LiteralPath $m.FullName -Destination '%MODEL%' -Force; $n=Get-ChildItem -LiteralPath '%DEPS%\model-extract' -Filter 'NOTICE.txt' -File -Recurse | Select-Object -First 1; if ($n) { Copy-Item -LiteralPath $n.FullName -Destination 'model\NOTICE.txt' -Force }"
  if errorlevel 1 (echo ERROR: Model archive extraction failed. The .oudep download may be incomplete or invalid.& goto :err)
)
if not exist "%MODEL%" (echo ERROR: Model download did not produce %MODEL%.& goto :err)
if not exist "model\NOTICE.txt" (echo ERROR: Model NOTICE.txt is missing.& goto :err)

 echo [4/6] Configuring Visual Studio 2026 x64 build...
if exist build rmdir /s /q build
if errorlevel 1 (echo ERROR: Could not remove the previous build directory.& goto :err)
rem Visual Studio 2026 uses CMake generator "Visual Studio 18 2026".
rem CMake 4.2+ is required for this generator.
cmake -S . -B build -G "Visual Studio 18 2026" -A x64 -DONNXRUNTIME_ROOT="%ORT_DIR%"
if errorlevel 1 (echo ERROR: CMake configuration failed. See the diagnostic above.& goto :err)

 echo [5/6] Compiling native DiffSampler...
cmake --build build --config Release --parallel
if errorlevel 1 (echo ERROR: C++ compilation failed. See the compiler diagnostic above.& goto :err)

 echo [6/6] Checking package...
if not exist "build\dist\DiffSampler.exe" (echo ERROR: Build completed without build\dist\DiffSampler.exe.& goto :err)
if not exist "build\dist\model\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx" (echo ERROR: Packaged ONNX model is missing.& goto :err)
echo Writing SHA-256 checksums...
powershell.exe -NoLogo -NoProfile -NonInteractive -Command "$ErrorActionPreference='Stop'; $root=(Resolve-Path 'build\dist').Path; $out=Join-Path $root 'SHA256SUMS.txt'; $lines=Get-ChildItem -LiteralPath $root -File -Recurse | Where-Object { $_.FullName -ne $out } | Sort-Object FullName | ForEach-Object { $h=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash.ToLowerInvariant(); $rel=$_.FullName.Substring($root.Length).TrimStart([char[]]@('\','/')).Replace('\','/'); $h + '  ' + $rel }; [System.IO.File]::WriteAllLines($out,[string[]]$lines,[System.Text.UTF8Encoding]::new($false))"
if errorlevel 1 (echo ERROR: Could not create distribution checksums.& goto :err)
echo.
echo Native build complete: "%ROOT%build\dist\DiffSampler.exe"
echo Model sidecar: build\dist\model\pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx
echo Run: build\dist\DiffSampler.exe --version
echo Verify EXE hash: certutil -hashfile build\dist\DiffSampler.exe SHA256
echo.
exit /b 0

:err
echo.
echo BUILD FAILED. The last error above identifies the failed step.
echo Project folder: "%ROOT%"
echo.
echo If this window was opened by double-clicking, press any key to close it.
pause >nul
exit /b 1
