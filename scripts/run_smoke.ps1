$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "PYTHON_NOT_FOUND: $Python"
}
Push-Location $ProjectRoot
try {
    & $Python -m finops_r.cli run --config 'configs\smoke.json'
    if ($LASTEXITCODE -ne 0) { throw "FINOPSR_SMOKE_FAILED: $LASTEXITCODE" }
}
finally {
    Pop-Location
}
