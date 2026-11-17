param([switch]$UseLock)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv/Scripts/python.exe'
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $pythonPath)) {
        & py -3.10 -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Nao foi possivel criar o ambiente Python 3.10.' }
    }
    & $pythonPath -c 'import sys; assert sys.version_info[:2] == (3, 10), "Use Python 3.10 neste perfil."'
    if ($LASTEXITCODE -ne 0) { throw 'Versao do Python incorreta.' }
    & $pythonPath -m pip install 'pip==25.3'
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar pip.' }
    $cachedTorch = Join-Path $projectRoot '.cache/wheels/torch-2.5.1+cu121-cp310-cp310-win_amd64.whl'
    if (Test-Path -LiteralPath $cachedTorch) {
        $expectedHash = '9b22d6d98aa56f9317902dec0e066814a6edba1aada90110ceea2bb0678df22f'
        $actualHash = (Get-FileHash -LiteralPath $cachedTorch -Algorithm SHA256).Hash.ToLowerInvariant()
        if ($actualHash -ne $expectedHash) { throw 'SHA-256 da wheel em cache difere do indice oficial.' }
        & $pythonPath -m pip install $cachedTorch
    } else {
        & $pythonPath -m pip install 'torch==2.5.1+cu121' --index-url https://download.pytorch.org/whl/cu121
    }
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar PyTorch CUDA.' }
    $requirements = if ($UseLock) { 'requirements-windows.lock.txt' } else { 'requirements.txt' }
    & $pythonPath -m pip install -r $requirements
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar dependencias.' }
    & $pythonPath -m pip install --no-deps -e .
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar o projeto.' }
    & $pythonPath -m pip check
    if ($LASTEXITCODE -ne 0) { throw 'Dependencias incompativeis.' }
    Write-Host 'Ambiente pronto. Execute o diagnostico e o smoke test descritos no README.'
} finally {
    Pop-Location
}

