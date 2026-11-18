<#
.SYNOPSIS
  Task 1.5 (gate G3): measure what happens to quality when experts run at 33% / 67% width.

.DESCRIPTION
  1. Builds a calibration text and runs llama-imatrix once on the base model.
  2. For each variant (full width, k=2, k=1, k=1 with RANDOM neurons as a control):
       - creates the variant GGUF (copy of the base with the dropped neurons zeroed),
       - runs llama-perplexity on the wikitext-2 test split,
       - starts llama-server and runs the 5 coding + 5 reasoning smoke prompts,
       - deletes the variant file.
  3. Prints a Markdown table to paste back into the chat.

  Experts stay on the CPU (-ncmoe) so the model fits an 8 GB GPU. Nothing here changes the
  engine: width is emulated by zeroing neurons, which gives the same output as running a
  narrower expert (see make_variant.py).

  Disk: about 2 x the base file at once (base + one variant) plus the dataset cache.
  Time: roughly an hour to several hours depending on CPU speed, mostly the imatrix run and
  the CPU-expert decode in the smoke eval. Use -Variants to run fewer.

.EXAMPLE
  .\run_g3.ps1 -LlamaBin C:\src\llama.cpp\build\bin\Release -LlamaCppSrc C:\src\llama.cpp `
      -BaseGguf D:\models\Qwen3-30B-A3B-Q4_K_M.gguf -WorkDir D:\maltoe-work

.EXAMPLE
  .\run_g3.ps1 -DryRun   # print every command without running it
#>
[CmdletBinding()]
param(
    [string]$LlamaBin    = "C:\src\llama.cpp\build\bin\Release",
    [string]$LlamaCppSrc = "C:\src\llama.cpp",
    [string]$BaseGguf    = "D:\models\Qwen3-30B-A3B-Q4_K_M.gguf",
    [string]$WorkDir     = ".\work",
    [int]$NCpuMoe        = 48,      # layers whose experts stay on the CPU; 48 = all of them
    [int]$ImatrixChunks  = 250,     # 250 x 512 tokens
    [int]$PplChunks      = 64,      # 64 x 512 tokens of wikitext-2 test
    [int]$Port           = 8080,
    [string[]]$Variants  = @("k3", "k2", "k1", "k1-random"),
    [switch]$SkipImatrix,
    [switch]$DryRun
)

$ErrorActionPreference = "Stop"
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$WorkDir = [System.IO.Path]::GetFullPath($WorkDir)

function Say($msg) { Write-Host "[g3] $msg" -ForegroundColor Cyan }

function Run([string]$exe, [string[]]$arguments, [string]$logFile = $null) {
    $line = "$exe " + ($arguments -join " ")
    Say $line
    if ($DryRun) { return "" }
    # llama.cpp writes its progress to stderr. In Windows PowerShell 5.1, stderr lines of a native
    # program become error records, which would abort the script under "Stop". Relax it for the call.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        if ($logFile) {
            # tee so progress is visible and the log is kept
            & $exe @arguments 2>&1 | Tee-Object -FilePath $logFile | Out-Host
        } else {
            & $exe @arguments 2>&1 | Out-Host
        }
        $code = $LASTEXITCODE
    } finally {
        $ErrorActionPreference = $prev
    }
    if ($code -ne 0) { throw "command failed with exit code ${code}: $line" }
    if ($logFile) { return (Get-Content $logFile -Raw) }
    return ""
}

# Start-Process joins -ArgumentList without quoting in Windows PowerShell 5.1.
function Quote-Args([string[]]$a) { $a | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } } }

$IsWin = ($PSVersionTable.PSEdition -eq "Desktop") -or ($IsWindows -eq $true)

function PyRun([string]$script, [string[]]$scriptArgs, [string]$logFile = $null) {
    $ggufPy = Join-Path $LlamaCppSrc "gguf-py"
    $args2 = @("run", "--no-project", "--with", $ggufPy, "python", (Join-Path $here $script)) + $scriptArgs
    return Run "uv" $args2 $logFile
}

# ---- preflight ----
if (-not $DryRun) {
    foreach ($b in "llama-imatrix.exe", "llama-perplexity.exe", "llama-server.exe") {
        if (-not (Test-Path (Join-Path $LlamaBin $b))) { throw "missing $b in $LlamaBin (build llama.cpp first, see ..\README.md)" }
    }
    if (-not (Test-Path $BaseGguf)) { throw "base model not found: $BaseGguf" }
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { throw "uv not found; install from https://docs.astral.sh/uv/" }
    if ($IsWin) {
        $freeGB = (Get-PSDrive -Name ([System.IO.Path]::GetPathRoot($WorkDir).Substring(0,1))).Free / 1GB
        $needGB = [math]::Round((Get-Item $BaseGguf).Length / 1GB * 1.2 + 3, 1)
        if ($freeGB -lt $needGB) { throw "need about $needGB GB free on the work drive, have $([math]::Round($freeGB,1)) GB" }
    }
}
New-Item -ItemType Directory -Force -Path $WorkDir | Out-Null
$results = Join-Path $WorkDir "results"
New-Item -ItemType Directory -Force -Path $results | Out-Null

# ---- 1. calibration + imatrix ----
$calib = Join-Path $WorkDir "calibration.txt"
$wikiTest = Join-Path $WorkDir "wiki.test.raw"
$imatrix = Join-Path $WorkDir "imatrix.gguf"

if (-not (Test-Path $calib) -or -not (Test-Path $wikiTest)) {
    Say "building calibration text"
    PyRun "make_calibration.py" @("--llama-cpp", $LlamaCppSrc, "--out-dir", $WorkDir) | Out-Null
}
if ($SkipImatrix -and (Test-Path $imatrix)) {
    Say "reusing $imatrix"
} else {
    Say "running llama-imatrix (this is the longest single step)"
    Run (Join-Path $LlamaBin "llama-imatrix.exe") @(
        "-m", $BaseGguf, "-f", $calib, "-o", $imatrix,
        "-c", "512", "--chunks", "$ImatrixChunks", "-ngl", "99", "-ncmoe", "$NCpuMoe"
    ) (Join-Path $results "imatrix.log") | Out-Null
}

# ---- 2. variants ----
$summary = @()
foreach ($v in $Variants) {
    Say "=== variant $v ==="
    $variantFile = Join-Path $WorkDir "variant-$v.gguf"
    $model = $BaseGguf
    $meta = $null

    switch ($v) {
        "k3"        { }  # full width = the unmodified base model
        "k2"        { $mk = @("--k", "2") }
        "k1"        { $mk = @("--k", "1") }
        "k1-random" { $mk = @("--k", "1", "--select", "random", "--seed", "7") }
        default {
            # "c4k3" = keep 3 of 4 chunks (75%), "c4k2" = 2 of 4 (50%), and so on
            if ($v -match '^c(\d+)k(\d+)$' -and [int]$Matches[2] -ge 1 -and [int]$Matches[2] -le [int]$Matches[1]) {
                $mk = @("--chunks", $Matches[1], "--k", $Matches[2])
            } else { throw "unknown variant $v" }
        }
    }
    if ($v -ne "k3") {
        PyRun "make_variant.py" (@("--base", $BaseGguf, "--imatrix", $imatrix, "--out", $variantFile, "--force") + $mk) (Join-Path $results "make-$v.log") | Out-Null
        $model = $variantFile
        $metaFile = [System.IO.Path]::ChangeExtension($variantFile, ".variant.json")
        if ((-not $DryRun) -and (Test-Path $metaFile)) { $meta = Get-Content $metaFile -Raw | ConvertFrom-Json }
    }

    # perplexity
    $pplLog = Join-Path $results "ppl-$v.log"
    $pplText = Run (Join-Path $LlamaBin "llama-perplexity.exe") @(
        "-m", $model, "-f", $wikiTest, "-c", "512", "--chunks", "$PplChunks", "-ngl", "99", "-ncmoe", "$NCpuMoe"
    ) $pplLog
    $ppl = $null; $pplErr = $null
    if ($pplText -match "Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)") { $ppl = [double]$Matches[1]; $pplErr = [double]$Matches[2] }

    # smoke eval through llama-server
    $evalJson = Join-Path $results "smoke-$v.json"
    $srvLog = Join-Path $results "server-$v.log"
    $vram = $null
    if ($DryRun) {
        Say "(dry run) would start llama-server on port $Port and run smoke_eval.py"
    } else {
        $srvArgs = Quote-Args @("-m", $model, "-ngl", "99", "-ncmoe", "$NCpuMoe", "-c", "4096", "--port", "$Port", "--jinja")
        $sp = @{ FilePath = (Join-Path $LlamaBin "llama-server.exe"); ArgumentList = $srvArgs; PassThru = $true
                 RedirectStandardError = $srvLog; RedirectStandardOutput = ($srvLog + ".out") }
        if ($IsWin) { $sp.WindowStyle = "Hidden" }
        $srv = Start-Process @sp
        try {
            PyRun "smoke_eval.py" @("--server", "http://127.0.0.1:$Port", "--out", $evalJson, "--cache", (Join-Path $WorkDir "datasets"), "--label", $v) | Out-Null
            $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
            if ($smi) { $vram = (& nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | Select-Object -First 1).Trim() }
        } finally {
            if (-not $srv.HasExited) { Stop-Process -Id $srv.Id -Force }
        }
    }

    $row = [ordered]@{ variant = $v; ppl = $ppl; ppl_err = $pplErr; coding = $null; reasoning = $null; mass_kept_pct = $null; vram_used_mib = $vram }
    if ((-not $DryRun) -and (Test-Path $evalJson)) {
        $e = Get-Content $evalJson -Raw | ConvertFrom-Json
        $row.coding = "$($e.coding.solved)/$($e.coding.n)"; $row.reasoning = "$($e.reasoning.solved)/$($e.reasoning.n)"
    }
    if ($meta) { $row.mass_kept_pct = [math]::Round(100 * $meta.importance_mass_kept_mean, 1) }
    $summary += [pscustomobject]$row

    if ((Test-Path $variantFile) -and -not $DryRun) { Remove-Item $variantFile -Force }
}

# ---- 3. report ----
$md = @("| variant | wikitext-2 PPL (64x512) | coding | reasoning | importance mass kept | GPU MiB used |", "| --- | ---: | ---: | ---: | ---: | ---: |")
foreach ($r in $summary) {
    $p = if ($null -ne $r.ppl) { "{0:N3} +/- {1:N3}" -f $r.ppl, $r.ppl_err } else { "n/a" }
    $m = if ($null -ne $r.mass_kept_pct) { "$($r.mass_kept_pct)%" } else { "n/a" }
    $md += "| $($r.variant) | $p | $($r.coding) | $($r.reasoning) | $m | $($r.vram_used_mib) |"
}
$mdText = $md -join "`n"
Set-Content -Path (Join-Path $results "g3-summary.md") -Value $mdText -Encoding UTF8
$summary | ConvertTo-Json | Set-Content -Path (Join-Path $results "g3-summary.json") -Encoding UTF8
Write-Host ""
Write-Host "==== paste this back ====" -ForegroundColor Green
Write-Host $mdText
Write-Host "(files are in $results; also paste the first lines of imatrix.log and calibration.json)"
