# SPDX-License-Identifier: GPL-3.0-or-later
# Launches Lichii with the frame budget tracers, then writes frame_budget.py's report for the run.
#   powershell -ExecutionPolicy Bypass -File tools\nx2\profile_run.ps1 -Exe <lichii.exe> [-Game <file>] [-Light]
# -Light leaves EDEN_NX2_GPU_PROFILE off (it times every zone, so it costs a little itself).
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)][string]$Exe,
    [string]$Game,
    [string]$OutDir,
    [switch]$Light,
    [string]$Extra = ''
)
$ErrorActionPreference = 'Stop'
$Exe = (Resolve-Path -LiteralPath $Exe).Path
$log = Join-Path ([Environment]::GetFolderPath('ApplicationData')) 'eden\log\eden_log.txt'
if (-not $OutDir) {
    $OutDir = Join-Path (Split-Path $Exe -Parent) ('profile-runs\' + (Get-Date -Format 'yyyyMMdd-HHmmss'))
}
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null

$env:EDEN_DKB_FRAME_TRACE = '1'
$env:EDEN_NX2_STUTTER_TRACE = '1'
$env:EDEN_NX2_HOSTTIMING_STATS = '1'
$env:EDEN_NX2_GUEST_PROFILE = '1'
if (-not $Light) { $env:EDEN_NX2_GPU_PROFILE = '1' }
foreach ($pair in ($Extra -split '\s+' | Where-Object { $_ })) {
    $name, $value = $pair -split '=', 2
    Set-Item -Path "Env:$name" -Value $value
}

$arguments = @()
if ($Game) { $arguments = @('-g', ('"' + (Resolve-Path -LiteralPath $Game).Path + '"')) }
Write-Host "Profiling $Exe. Play the slow scene for at least 30 s, then close Lichii."
if ($arguments) { $process = Start-Process -FilePath $Exe -ArgumentList $arguments -PassThru }
else { $process = Start-Process -FilePath $Exe -PassThru }
$process.WaitForExit()

Copy-Item -LiteralPath $log -Destination (Join-Path $OutDir 'eden_log.txt')
$python = if (Get-Command py -ErrorAction SilentlyContinue) { @('py', '-3') } else { @('python') }
$script = Join-Path $PSScriptRoot 'frame_budget.py'
$report = Join-Path $OutDir 'frame_budget.md'
$cmd = $python[0]
$rest = @($python | Select-Object -Skip 1) + @('-I', $script, (Join-Path $OutDir 'eden_log.txt'), '--windows')
& $cmd @rest | Out-File -FilePath $report -Encoding utf8
Get-Content -LiteralPath $report -TotalCount 40
Write-Host "`nReport: $report"
