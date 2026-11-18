<#
.SYNOPSIS
  Task 1.7 (gate G2b): how well does llama.cpp's own experimental MoE cache work on this machine?

.DESCRIPTION
  At the pinned commit, llama.cpp already has --moe-cache-mib: a VRAM cache for experts that live in
  host memory (-ncmoe). It uploads only the experts a token routes to, and keeps the frequently and
  recently used ones on the GPU. MaltOE would build on it, so before building anything we measure:

    * static placement, no cache:        -ncmoe N                       (the stock baseline, BL-2)
    * cache of increasing size:          -ncmoe 48 --moe-cache-mib M    (all experts on the host)

  Every run generates the same fixed prompt with greedy decoding, so routing is realistic and the
  runs are comparable. The script records decode and prompt speed, the cache hit rate the engine
  prints, peak GPU memory, and (approximately) bytes read from disk, which shows the SSD tier at work
  when the model is bigger than free RAM.

  The first run of the first configuration also warms the OS file cache. Each configuration runs
  -Repeat times; keep both numbers when you paste results.

.EXAMPLE
  .\run_moe_cache.ps1 -LlamaBin C:\src\llama.cpp\build\bin -Model D:\models\Qwen3-30B-A3B-Q4_K_M.gguf
#>
[CmdletBinding()]
param(
    [string]$LlamaBin = "C:\src\llama.cpp\build\bin",
    [string]$Model    = "D:\models\Qwen3-30B-A3B-Q4_K_M.gguf",
    [string]$Prompt   = (Join-Path $PSScriptRoot "probe_prompt.txt"),
    [string]$OutDir   = ".\moe-cache-results",
    [int]$NPredict    = 128,
    [int]$Ctx         = 2048,
    [int[]]$StaticNcmoe = @(40, 34, 28),        # experts of the first N layers on CPU; fewer = more on the GPU
    [int[]]$CacheMiB    = @(1024, 2048, 3072, 4096),
    [double[]]$CpuShares = @(0),                # --moe-cache-cpu-share: share of cache misses the CPU computes instead of uploading
    [int]$Repeat      = 2,
    [int]$TimeoutSec  = 1800,
    [switch]$DryRun
)
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "..\common.ps1")

$exe = Join-Path $LlamaBin "llama-completion.exe"
if (-not $DryRun) {
    if (-not (Test-Path $exe)) { throw "missing $exe" }
    if (-not (Test-Path $Model)) { throw "model not found: $Model" }
    if (-not (Test-Path $Prompt)) { throw "prompt file not found: $Prompt" }
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$configs = @()
foreach ($n in $StaticNcmoe) { $configs += [pscustomobject]@{ Name = "static ncmoe=$n"; Ncmoe = $n; Cache = 0 } }
foreach ($m in $CacheMiB) {
    foreach ($sh in $CpuShares) {
        $configs += [pscustomobject]@{ Name = "cache ${m} MiB share $sh (ncmoe=48)"; Ncmoe = 48; Cache = $m; Share = $sh }
    }
}

function Parse-Run([string]$text) {
    $r = [ordered]@{ decode_tps = $null; prompt_tps = $null; hit_rate = $null; uploaded_mib = $null; cpu_misses = $null; oom = $false }
    if ($text -match "(?<!prompt )eval time =\s+[0-9.]+ ms /\s+\d+ runs\s+\(\s*[0-9.]+ ms per token,\s+([0-9.]+) tokens per second\)") { $r.decode_tps = [double]$Matches[1] }
    if ($text -match "prompt eval time =\s+[0-9.]+ ms /\s+\d+ tokens\s+\(\s*[0-9.]+ ms per token,\s+([0-9.]+) tokens per second\)") { $r.prompt_tps = [double]$Matches[1] }
    if ($text -match "llama_moe_cache: ubatch <= 8: hits = \d+, misses = \d+ \((\d+) on the CPU\), hit rate = ([0-9.]+)%, uploaded = ([0-9.]+) MiB") {
        $r.cpu_misses = [int]$Matches[1]; $r.hit_rate = [double]$Matches[2]; $r.uploaded_mib = [double]$Matches[3]
    }
    if ($text -match "(?i)out of memory|failed to allocate|unable to allocate|cudaMalloc failed") { $r.oom = $true }
    return $r
}

$rows = @()
foreach ($c in $configs) {
    for ($i = 1; $i -le $Repeat; $i++) {
        $label = ($c.Name -replace "[^a-zA-Z0-9]+", "-").Trim("-") + "-run$i"
        $args2 = @("-m", $Model, "-ngl", "99", "-ncmoe", "$($c.Ncmoe)", "-f", $Prompt, "-n", "$NPredict",
                   "-c", "$Ctx", "--temp", "0", "--seed", "1", "-no-cnv", "--no-warmup",
                   "-lv", "4")   # verbose: the cache's hit-rate line is filtered out at the default log level
        if ($c.Cache -gt 0) {
            $args2 += @("--moe-cache-mib", "$($c.Cache)", "--moe-cache-cpu-share", ([string]::Format([cultureinfo]::InvariantCulture, "{0}", $c.Share)))
        }

        if ($DryRun) { Say ("$exe " + ($args2 -join " ")); continue }
        $res = Invoke-Measured -Exe $exe -Arguments $args2 -LogBase (Join-Path $OutDir $label) -TimeoutSec $TimeoutSec
        $p = Parse-Run $res.Text
        $status = if ($res.TimedOut) { "TIMEOUT" } elseif ($res.ExitCode -ne 0) { "FAIL(exit $($res.ExitCode))" } elseif ($p.oom) { "OOM" } else { "ok" }
        if ($status -ne "ok") { Say "  $status : $(Tail $res.Text 4)" }
        $rows += [pscustomobject][ordered]@{
            config = $c.Name; run = $i; status = $status
            decode_tps = $p.decode_tps; prompt_tps = $p.prompt_tps; cache_hit_pct = $p.hit_rate; uploaded_mib = $p.uploaded_mib; cpu_misses = $p.cpu_misses
            peak_gpu_mib = $res.PeakGpuMiB; disk_read_gib = $res.DiskReadGiB; min_avail_ram_mib = $res.MinAvailMiB; seconds = $res.Seconds
        }
    }
}

if ($DryRun) { return }
$rows | Export-Csv -NoTypeInformation -Path (Join-Path $OutDir "moe-cache-results.csv")
$md = @("| config | run | status | decode tok/s | prompt tok/s | cache hit % | misses on CPU | uploaded MiB | peak GPU MiB | disk read GiB | min free RAM MiB | s |", "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
foreach ($r in $rows) { $md += "| $($r.config) | $($r.run) | $($r.status) | $($r.decode_tps) | $($r.prompt_tps) | $($r.cache_hit_pct) | $($r.cpu_misses) | $($r.uploaded_mib) | $($r.peak_gpu_mib) | $($r.disk_read_gib) | $($r.min_avail_ram_mib) | $($r.seconds) |" }
$mdText = $md -join "`n"
Set-Content -Path (Join-Path $OutDir "moe-cache-results.md") -Value $mdText -Encoding UTF8
Write-Host ""; Write-Host "==== paste this back ====" -ForegroundColor Green; Write-Host $mdText
Write-Host "(also paste: nvidia-smi output, and the first 'llama_model_loader' lines of one *.err.txt log)"
