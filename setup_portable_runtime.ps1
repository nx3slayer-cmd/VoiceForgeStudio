# ===========================================================================
# VoiceForge Studio — Zero-Install Portable Runtime Provisioner (Windows 11)
# ===========================================================================
$ErrorActionPreference = "Stop"
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
Set-Location $ScriptDir

$PythonVer = "3.11.9"
$PythonZipName = "python-$PythonVer-embed-amd64.zip"
$PythonUrl = "https://www.python.org/ftp/python/$PythonVer/$PythonZipName"
$RuntimeDir = Join-Path $ScriptDir "runtime"

Write-Host "===========================================================" -ForegroundColor Cyan
Write-Host "  Provisioning Embedded Python ($PythonVer) + CUDA Runtime  " -ForegroundColor Cyan
Write-Host "===========================================================" -ForegroundColor Cyan

if (-not (Test-Path $RuntimeDir)) {
    New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
}

$PythonExe = Join-Path $RuntimeDir "python.exe"
if (-not (Test-Path $PythonExe)) {
    $ZipDest = Join-Path $RuntimeDir $PythonZipName
    Write-Host "`n[1/6] Downloading Python $PythonVer Embeddable Package..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $PythonUrl -OutFile $ZipDest -UseBasicParsing

    Write-Host "[2/6] Extracting to $RuntimeDir..." -ForegroundColor Yellow
    Expand-Archive -Path $ZipDest -DestinationPath $RuntimeDir -Force
    Remove-Item $ZipDest -Force

    # Patch python311._pth to enable site-packages & pip
    $PthFile = Get-ChildItem -Path $RuntimeDir -Filter "*._pth" | Select-Object -First 1
    if ($PthFile) {
        Write-Host "[3/6] Patching $($PthFile.Name) to enable 'import site'..." -ForegroundColor Yellow
        $Content = Get-Content $PthFile.FullName
        $NewContent = @()
        foreach ($Line in $Content) {
            if ($Line -match "^#import site") {
                $NewContent += "import site"
            } else {
                $NewContent += $Line
            }
        }
        $NewContent += "."
        $NewContent += "Lib"
        $NewContent += "Lib\site-packages"
        $NewContent | Set-Content $PthFile.FullName -Encoding Ascii
    }

    # Install pip
    Write-Host "[4/6] Bootstrapping pip into isolated runtime..." -ForegroundColor Yellow
    $GetPipUrl = "https://bootstrap.pypa.io/get-pip.py"
    $GetPipPy = Join-Path $RuntimeDir "get-pip.py"
    Invoke-WebRequest -Uri $GetPipUrl -OutFile $GetPipPy -UseBasicParsing
    & $PythonExe $GetPipPy --no-warn-script-location
    Remove-Item $GetPipPy -Force
} else {
    Write-Host "✓ Embedded Python detected at $PythonExe" -ForegroundColor Green
}

# 5. Install PyTorch with CUDA 12.1 and Backend Libraries
Write-Host "`n[5/7] Installing CUDA-enabled PyTorch & Framework Dependencies..." -ForegroundColor Yellow

$PipArgs = @(
    "-m", "pip", "install", "--upgrade",
    "pip", "setuptools", "wheel", "--no-warn-script-location"
)
& $PythonExe $PipArgs

Write-Host "Installing PyTorch (CUDA 12.1)..." -ForegroundColor Yellow
& $PythonExe -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --no-warn-script-location

Write-Host "Installing Core Dependencies..." -ForegroundColor Yellow
$Packages = @(
    "fastapi>=0.111.0",
    "uvicorn[standard]>=0.30.0",
    "soundfile>=0.12.1",
    "numpy<2.0.0",
    "scipy",
    "playwright>=1.44.0",
    "huggingface_hub",
    "kokoro>=0.3.4",
    "websockets"
)
& $PythonExe -m pip install $Packages --no-warn-script-location

Write-Host "Installing Zero-Shot Cloning Engine (Chatterbox-Turbo)..." -ForegroundColor Yellow
& $PythonExe -m pip install chatterbox-tts --no-deps --no-warn-script-location
& $PythonExe -m pip install conformer diffusers omegaconf "librosa<1.0.0" audioread "ml-dtypes==0.5.1" s3tokenizer pyloudnorm pykakasi --no-warn-script-location

# 6. Pre-download Chatterbox-Turbo model weights (No Token Required)
Write-Host "`n[6/7] Downloading Zero-Shot Cloning Model Weights..." -ForegroundColor Yellow
$ModelDownloadScript = @"
from huggingface_hub import snapshot_download
from pathlib import Path
dest = Path('pretrained_models/chatterbox-turbo')
dest.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id='ResembleAI/chatterbox-turbo', local_dir=str(dest), token=False)
print('✓ Chatterbox-Turbo model downloaded.')
"@
& $PythonExe -c $ModelDownloadScript

# 7. Install Portable Playwright Chromium
Write-Host "`n[7/7] Installing Isolated Playwright Chromium..." -ForegroundColor Yellow
$BrowsersDir = Join-Path $RuntimeDir "playwright-browsers"
$env:PLAYWRIGHT_BROWSERS_PATH = $BrowsersDir
& $PythonExe -m playwright install chromium

Write-Host "`n===========================================================" -ForegroundColor Green
Write-Host "✓ Windows Portable Runtime Setup Complete!" -ForegroundColor Green
Write-Host "===========================================================" -ForegroundColor Green
