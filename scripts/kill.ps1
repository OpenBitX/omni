[CmdletBinding()]
param(
  [Parameter(Position = 0)]
  [ValidateRange(1, 65535)]
  [int] $Port = 3000
)

$ErrorActionPreference = "Stop"

$port = $Port
$pids = @()

try {
  $pids = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction Stop |
    Select-Object -ExpandProperty OwningProcess -Unique
} catch {
  Write-Host "未发现端口 $port 的监听进程。"
  exit 0
}

if (-not $pids -or $pids.Count -eq 0) {
  Write-Host "未发现端口 $port 的监听进程。"
  exit 0
}

Write-Host "将结束占用端口 $port 的进程 PID: $($pids -join ', ')"

foreach ($procId in $pids) {
  try {
    $procName = $null
    try { $procName = (Get-Process -Id $procId -ErrorAction Stop).ProcessName } catch {}
    Stop-Process -Id $procId -Force -ErrorAction Stop
    if ($procName) {
      Write-Host "已结束 PID $procId ($procName)"
    } else {
      Write-Host "已结束 PID $procId"
    }
  } catch {
    Write-Host "结束 PID $procId 失败：$($_.Exception.Message)"
  }
}