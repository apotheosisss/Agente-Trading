# setup_scheduler.ps1
# Registra el scheduler diario en Windows Task Scheduler.
# Ejecutar UNA SOLA VEZ como administrador:
#   Right-click > "Run as Administrator" en PowerShell
#   .\setup_scheduler.ps1

$ProjectDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ScriptPath = Join-Path $ProjectDir "scheduler.py"
$UvPath     = (Get-Command uv -ErrorAction SilentlyContinue).Source

if (-not $UvPath) {
    Write-Error "No se encontro 'uv' en el PATH. Instala uv primero: https://docs.astral.sh/uv/"
    exit 1
}

# Accion: ejecutar "uv run python scheduler.py" desde el directorio del proyecto
$Action = New-ScheduledTaskAction `
    -Execute $UvPath `
    -Argument "run python scheduler.py" `
    -WorkingDirectory $ProjectDir

# Trigger: cada dia de lunes a viernes a las 16:00 (cierre mercado NY = 16:00 EST)
# Ajusta la hora segun tu zona horaria (Chile = UTC-4, entonces 16:00 NY = 17:00 Chile aprox)
$Trigger = New-ScheduledTaskTrigger `
    -Weekly `
    -DaysOfWeek Monday, Tuesday, Wednesday, Thursday, Friday `
    -At "17:00"

# Configuracion
$Settings = New-ScheduledTaskSettingsSet `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 10) `
    -StartWhenAvailable `
    -RunOnlyIfNetworkAvailable

# Registrar tarea (requiere permisos de administrador)
$TaskName = "TradingAgentScheduler"

# Eliminar tarea anterior si existe
Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false -ErrorAction SilentlyContinue

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Description "Genera señales de trading diarias con el pipeline de feature/polymarket" `
    -RunLevel Highest

Write-Host ""
Write-Host "Tarea '$TaskName' registrada correctamente." -ForegroundColor Green
Write-Host "  Directorio: $ProjectDir"
Write-Host "  Horario:    Lunes-Viernes a las 17:00"
Write-Host "  Log diario: $ProjectDir\data\08_reporting\daily_log\"
Write-Host ""
Write-Host "Para ejecutar manualmente ahora:"
Write-Host "  uv run python scheduler.py"
Write-Host ""
Write-Host "Para verificar estado de la tarea:"
Write-Host "  Get-ScheduledTask -TaskName '$TaskName' | Get-ScheduledTaskInfo"
