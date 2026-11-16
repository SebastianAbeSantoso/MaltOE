<#
  Shared helpers for the Task 1 hardware spikes. Dot-source it:   . "$PSScriptRoot\..\common.ps1"
  Works in Windows PowerShell 5.1 and PowerShell 7.
#>

function Say([string]$msg) { Write-Host "[spike] $msg" -ForegroundColor Cyan }

$script:IsWin = ($PSVersionTable.PSEdition -eq "Desktop") -or ($IsWindows -eq $true)

# Windows performance counters. Counter names are localized on non-English Windows; every read is
# optional and returns $null when a counter is missing.
function Get-CounterSum([string]$path) {
    try {
        $s = (Get-Counter -Counter $path -ErrorAction Stop).CounterSamples
        return ($s | Measure-Object -Property CookedValue -Sum).Sum
    } catch { return $null }
}

function Quote-Args([string[]]$a) {
    $a | ForEach-Object { if ($_ -match "\s") { '"' + $_ + '"' } else { $_ } }
}

<#
  Run a program, capture stdout+stderr to files, and sample system counters once a second while it runs.
  Returns an object with: ExitCode, TimedOut, Text (stdout+stderr), StdOut, StdErr, Seconds,
  PeakGpuMiB, DiskReadGiB (sum of per-second read rates, approximate), MinAvailMiB, MaxSharedGpuMiB.
#>
function Invoke-Measured {
    param(
        [Parameter(Mandatory)][string]$Exe,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory)][string]$LogBase,   # writes <LogBase>.out.txt and <LogBase>.err.txt
        [int]$TimeoutSec = 1800,
        [switch]$NoCounters
    )
    $outFile = "$LogBase.out.txt"; $errFile = "$LogBase.err.txt"
    Say ("$Exe " + ($Arguments -join " "))

    $smi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    $pollFile = "$LogBase.gpu.txt"
    $poll = $null
    if ($smi) {
        $poll = Start-Process -FilePath $smi.Source -ArgumentList @("--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "500") `
                -RedirectStandardOutput $pollFile -PassThru -NoNewWindow
    }

    $sp = @{ FilePath = $Exe; ArgumentList = (Quote-Args $Arguments); RedirectStandardOutput = $outFile; RedirectStandardError = $errFile; PassThru = $true }
    if ($script:IsWin) { $sp.NoNewWindow = $true }
    $t0 = Get-Date
    $p = Start-Process @sp

    $diskBytes = 0.0; $minAvail = $null; $maxShared = $null; $timedOut = $false
    while (-not $p.WaitForExit(1000)) {
        if (-not $NoCounters) {
            $d = Get-CounterSum '\PhysicalDisk(_Total)\Disk Read Bytes/sec'; if ($null -ne $d) { $diskBytes += $d }
            $a = Get-CounterSum '\Memory\Available MBytes'; if ($null -ne $a -and ($null -eq $minAvail -or $a -lt $minAvail)) { $minAvail = $a }
            $sh = Get-CounterSum '\GPU Adapter Memory(*)\Shared Usage'; if ($null -ne $sh -and ($null -eq $maxShared -or $sh -gt $maxShared)) { $maxShared = $sh }
        }
        if (((Get-Date) - $t0).TotalSeconds -gt $TimeoutSec) { $timedOut = $true; try { $p.Kill() } catch {}; break }
    }
    $secs = ((Get-Date) - $t0).TotalSeconds
    $code = if ($timedOut) { -1 } else { $p.ExitCode }

    $peak = $null
    if ($poll) {
        try { $poll.Kill() } catch {}
        Start-Sleep -Milliseconds 300
        if (Test-Path $pollFile) {
            $vals = Get-Content $pollFile | Where-Object { $_ -match "^\s*\d+\s*$" } | ForEach-Object { [int]$_.Trim() }
            if ($vals) { $peak = ($vals | Measure-Object -Maximum).Maximum }
        }
    }
    $so = if (Test-Path $outFile) { [string](Get-Content $outFile -Raw) } else { "" }
    $se = if (Test-Path $errFile) { [string](Get-Content $errFile -Raw) } else { "" }
    $text = $so + "`n" + $se
    return [pscustomobject]@{
        ExitCode = $code; TimedOut = $timedOut; Text = $text; StdOut = $so; StdErr = $se; Seconds = [math]::Round($secs, 1)
        PeakGpuMiB = $peak
        DiskReadGiB = if ($diskBytes -gt 0) { [math]::Round($diskBytes / 1GB, 2) } else { $null }
        MinAvailMiB = $minAvail
        MaxSharedGpuMiB = if ($null -ne $maxShared) { [math]::Round($maxShared / 1MB) } else { $null }
    }
}

function Tail([string]$text, [int]$n = 6) {
    ($text -split "`r?`n" | Where-Object { $_.Trim() } | Select-Object -Last $n) -join " | "
}
