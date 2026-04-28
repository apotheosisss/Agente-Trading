# setup_scheduler.ps1
# Registra el scheduler crypto en Windows Task Scheduler.
# Ejecutar UNA SOLA VEZ como administrador.
# Crypto opera 24/7 — corre todos los dias a las 08:00.

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$UvPath     = (Get-Command uv -ErrorAction SilentlyContinue).Source

if (-not $UvPath) {
    Write-Error "No se encontro 'uv' en el PATH."
    exit 1
}

$Action = New-ScheduledTaskAction `
    -Execute $UvPath `
    -Argument "run python scheduler.py" `
    -WorkingDirectory $ProjectDir

# Todos los dias a las 08:00 (mercado crypto siempre abierto)
$Trigger = New-ScheduledTaskTrigger -Daily -At "08:00"

$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

$TaskName = "TradingAgentCrypto"
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Señales diarias crypto — feature/crypto branch" `
    -RunLevel Highest

Write-Host ""
Write-Host "Tarea '$TaskName' registrada." -ForegroundColor Green
Write-Host "  Horario: todos los dias a las 08:00"
Write-Host "  Log:     $ProjectDir\data\08_reporting\daily_log\"
