param(
  [string]$TraceFile = "${PSScriptRoot}\..\..\data\processed\traces\all_traces.json",
  [string]$OutDir = "${PSScriptRoot}\..\..\data\processed\traces\all_traces",
  [switch]$ClearAfter
)

$TraceFile = (Resolve-Path -LiteralPath $TraceFile).Path

$resolvedOutDir = Resolve-Path -LiteralPath $OutDir -ErrorAction SilentlyContinue
if ($null -ne $resolvedOutDir) {
  $OutDir = $resolvedOutDir.Path
} else {
  # Create and then resolve
  $created = New-Item -ItemType Directory -Force -Path $OutDir
  $OutDir = (Resolve-Path -LiteralPath $created.FullName).Path
}

# Ensure output directory exists
if (!(Test-Path -LiteralPath $OutDir)) {
  New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
}

if (!(Test-Path -LiteralPath $TraceFile)) {
  throw "Trace file not found: $TraceFile"
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$dest = Join-Path $OutDir "all_traces_$ts.json"

$len = (Get-Item -LiteralPath $TraceFile).Length
if ($len -eq 0) {
  Write-Host "Trace file is empty, nothing to archive: $TraceFile"
  exit 0
}

Copy-Item -LiteralPath $TraceFile -Destination $dest -Force
Write-Host "Archived $TraceFile ($len bytes) -> $dest"

function Truncate-FileWithRetry {
  param(
    [Parameter(Mandatory = $true)][string]$Path,
    [int]$Retries = 10,
    [int]$DelayMs = 200
  )

  for ($i = 0; $i -lt $Retries; $i++) {
    try {
      $fs = [System.IO.File]::Open($Path, [System.IO.FileMode]::Open, [System.IO.FileAccess]::Write, [System.IO.FileShare]::ReadWrite)
      try {
        $fs.SetLength(0)
      } finally {
        $fs.Dispose()
      }
      return
    } catch {
      Start-Sleep -Milliseconds $DelayMs
    }
  }

  throw "Failed to truncate file after $Retries attempts: $Path"
}

if ($ClearAfter) {
  # Truncate content but keep the file so the bind mount stays valid.
  Truncate-FileWithRetry -Path $TraceFile
  Write-Host "Cleared $TraceFile"
}
