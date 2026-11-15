$ErrorActionPreference = 'Stop'
$exe = (Resolve-Path (Join-Path $PSScriptRoot '..\build\dist\DiffSampler.exe')).Path
$tmp = Join-Path $env:TEMP ('DiffSamplerNativeSmoke-' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $tmp | Out-Null
$in = Join-Path $tmp 'source.wav'
$out = Join-Path $tmp 'render.wav'

# Write a small standards-compliant mono PCM16 test WAV without Python.
$sampleRate = 44100
$seconds = 1.8
$n = [int]($sampleRate * $seconds)
$stream = [System.IO.File]::Create($in)
$writer = [System.IO.BinaryWriter]::new($stream)
$writer.Write([System.Text.Encoding]::ASCII.GetBytes('RIFF'))
$writer.Write([int](36 + 2 * $n))
$writer.Write([System.Text.Encoding]::ASCII.GetBytes('WAVE'))
$writer.Write([System.Text.Encoding]::ASCII.GetBytes('fmt '))
$writer.Write([int]16); $writer.Write([short]1); $writer.Write([short]1)
$writer.Write([int]$sampleRate); $writer.Write([int]($sampleRate * 2)); $writer.Write([short]2); $writer.Write([short]16)
$writer.Write([System.Text.Encoding]::ASCII.GetBytes('data')); $writer.Write([int](2 * $n))
for ($i = 0; $i -lt $n; $i++) {
    $t = $i / $sampleRate
    $v = [math]::Sin(2 * [math]::PI * 220.0 * $t) * 0.35
    $writer.Write([short][math]::Round($v * 32767))
}
$writer.Dispose(); $stream.Dispose()

$env:DIFFSAMPLER_PROVIDER = 'cpu'
& $exe $in $out 'E4' 100 '' 0 600 120 -1500 100 100 '!120' ''
if ($LASTEXITCODE -ne 0) { throw "Render failed with exit code $LASTEXITCODE" }
if (!(Test-Path $out)) { throw 'Output WAV was not created.' }
$bytes = [System.IO.File]::ReadAllBytes($out)
if ($bytes.Length -lt 1000 -or [System.Text.Encoding]::ASCII.GetString($bytes, 0, 4) -ne 'RIFF') {
    throw 'Output is not a valid non-empty RIFF/WAVE file.'
}
& $exe --stop | Out-Null
Write-Host "Native smoke test passed: $out ($($bytes.Length) bytes)"
Remove-Item -LiteralPath $tmp -Recurse -Force
