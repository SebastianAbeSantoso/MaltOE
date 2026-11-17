<#
.SYNOPSIS
  Task 1.8 (gate G4): does a VRAM ballast create real pressure on Windows, and what does a normal
  llama.cpp run do when the GPU is nearly full?

.DESCRIPTION
  For each ballast size, the script holds that much VRAM with ballast_probe.exe, then runs the same
  llama-bench command (experts of the first -NCpuMoe layers on the CPU, the rest on the GPU, sized so
  the run fits in 8 GB with no ballast). It records:

    * did the ballast really hold the memory (nvidia-smi used, ballast's own READY line)
    * did llama.cpp fail with an out-of-memory error, or run slower?
    * how much "Shared GPU memory" (system RAM used as GPU memory) appeared: this is the WDDM
      sysmem-fallback signature
    * decode and prompt speed

  Run it twice and keep both tables:
    1. NVIDIA Control Panel > Manage 3D settings > "CUDA - Sysmem Fallback Policy" = Driver Default
    2. the same setting = "Prefer No Sysmem Fallback"
  Pass -Label so the two result files are named differently.

  Build the ballast first:   nvcc -O2 -o ballast_probe.exe ballast_probe.cu

.EXAMPLE
  .\run_g4.ps1 -LlamaBin C:\src\llama.cpp\build\bin -Model D:\models\Qwen3-30B-A3B-Q4_K_M.gguf -Label default
  .\run_g4.ps1 -LlamaBin C:\src\llama.cpp\build\bin -Model D:\models\Qwen3-30B-A3B-Q4_K_M.gguf -Label no-fallback
#>
[CmdletBinding()]
param(
    [string]$LlamaBin   = "C:\src\llama.cpp\build\bin",
    [string]$Model      = "D:\models\Qwen3-30B-A3B-Q4_K_M.gguf",
    [string]$Ballast    = (Join-Path $PSScriptRoot "ballast_probe.exe"),
    [double[]]$BallastGiB = @(0, 1, 2, 3, 4),
    [int]$NCpuMoe       = 36,        # first 36 of 48 layers' experts on CPU; about 12 layers of experts on the GPU
    [string]$Label      = "run",
    [string]$OutDir     = ".\g4-results",
    [int]$TimeoutSec    = 900,
    [int]$TouchMs       = 0,         # 0: idle filler; >0: the filler rewrites its memory every TouchMs ms
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\common.ps1")

$bench = Join-Path $LlamaBin "llama-bench.exe"
if (-not $DryRun) {
    foreach ($f in $bench, $Model, $Ballast) { if (-not (Test-Path $f)) { throw "missing $f" } }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

function Start-Ballast([double]$gib, [string]$logBase) {
    if ($gib -le 0) { return $null }
    $outFile = "$logBase.ballast.txt"
    $p = Start-Process -FilePath $Ballast -ArgumentList @("$gib", "1800", "$TouchMs") -RedirectStandardOutput $outFile -PassThru -NoNewWindow
    $deadline = (Get-Date).AddSeconds(90)
    while ((Get-Date) -lt $deadline) {
        if ((Test-Path $outFile) -and ((Get-Content $outFile -Raw) -match "READY")) { break }
        if ($p.HasExited) { break }
        Start-Sleep -Milliseconds 500
    }
    return [pscustomobject]@{ Proc = $p; File = $outFile }
}

$rows = @()
foreach ($g in $BallastGiB) {
    $base = Join-Path $OutDir ("$Label-ballast$g")
    $b = $null
    $held = "n/a"
    $sharedBefore = $null
    if ($DryRun) {
        Say "(dry run) ballast $g GiB; $bench -m $Model -ngl 99 -ncmoe $NCpuMoe -p 128 -n 32 -r 2 -o json"
        continue
    }
    try {
        $b = Start-Ballast $g $base
        if ($b) {
            $txt = Get-Content $b.File -Raw
            if ($txt -match "READY allocated (\d+) MiB of requested (\d+) MiB") { $held = "$($Matches[1]) of $($Matches[2]) MiB" } else { $held = "no READY: " + (Tail $txt 3) }
        }
        $sharedBefore = Get-CounterSum '\GPU Adapter Memory(*)\Shared Usage'

        $res = Invoke-Measured -Exe $bench -Arguments @("-m", $Model, "-ngl", "99", "-ncmoe", "$NCpuMoe", "-p", "128", "-n", "32", "-r", "2", "-o", "json") `
                               -LogBase $base -TimeoutSec $TimeoutSec
        $pp = $null; $tg = $null; $status = "ok"
        if ($res.TimedOut) { $status = "TIMEOUT" }
        elseif ($res.ExitCode -ne 0) { $status = "FAIL(exit $($res.ExitCode)): " + (Tail $res.Text 3) }
        else {
            try {
                $json = $res.StdOut | ConvertFrom-Json
                foreach ($e in $json) {
                    if ($e.n_gen -eq 0 -and $e.n_prompt -gt 0) { $pp = [math]::Round($e.avg_ts, 1) }
                    if ($e.n_gen -gt 0) { $tg = [math]::Round($e.avg_ts, 1) }
                }
            } catch { $status = "ran, but output not parsed; see $base.out.txt" }
        }
        $sharedDeltaMiB = $null
        if ($null -ne $res.MaxSharedGpuMiB -and $null -ne $sharedBefore) { $sharedDeltaMiB = $res.MaxSharedGpuMiB - [math]::Round($sharedBefore / 1MB) }
        $rows += [pscustomobject][ordered]@{
            ballast_gib = $g; ballast_held = $held; status = $status; prompt_tps = $pp; decode_tps = $tg
            peak_gpu_used_mib = $res.PeakGpuMiB; shared_gpu_delta_mib = $sharedDeltaMiB; seconds = $res.Seconds
        }
    } finally {
        if ($b -and -not $b.Proc.HasExited) { try { $b.Proc.Kill() } catch {} }
    }
}
if ($DryRun) { return }

$rows | Export-Csv -NoTypeInformation -Path (Join-Path $OutDir "g4-$Label.csv")
$md = @("| ballast GiB | ballast held | status | prompt tok/s | decode tok/s | peak GPU used MiB | shared GPU memory delta MiB | s |", "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: |")
foreach ($r in $rows) { $md += "| $($r.ballast_gib) | $($r.ballast_held) | $($r.status) | $($r.prompt_tps) | $($r.decode_tps) | $($r.peak_gpu_used_mib) | $($r.shared_gpu_delta_mib) | $($r.seconds) |" }
$mdText = $md -join "`n"
Set-Content -Path (Join-Path $OutDir "g4-$Label.md") -Value $mdText -Encoding UTF8
Write-Host ""; Write-Host "==== paste this back (label: $Label) ====" -ForegroundColor Green; Write-Host $mdText
Write-Host "Also say which 'CUDA - Sysmem Fallback Policy' setting was active, and your NVIDIA driver version."
