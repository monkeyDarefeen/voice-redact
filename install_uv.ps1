# ============================================================
# Windows CPU install using uv (no GPU/CUDA) -- Gemma edition
# For laptops/desktops without a GPU.
# Usage: pwsh -ExecutionPolicy Bypass -File install_uv.ps1 -ProjectDir ".\"
# ============================================================

param(
    [Parameter(Mandatory=$true)]
    [string]$ProjectDir
)

if (-not (Test-Path $ProjectDir)) {
    New-Item -ItemType Directory -Force -Path $ProjectDir | Out-Null
}

$ProjectDir = (Resolve-Path $ProjectDir).Path
Set-Location $ProjectDir

# ----------------------------------------------------------
# 0. Install uv if not present
# ----------------------------------------------------------
Write-Host "=== [0/7] Installing uv ==="
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "  Installing uv..."
    powershell -ExecutionPolicy Bypass -c "irm https://astral.sh/uv/install.ps1 | iex"
    $env:PATH = "$env:USERPROFILE\.cargo\bin;$env:PATH"
}

# ----------------------------------------------------------
# 1. Ensure Python 3.12 is available, create venv
# ----------------------------------------------------------
Write-Host "=== [1/7] Creating virtual environment ==="
if (-not (Get-Command python3.12 -ErrorAction SilentlyContinue)) {
    Write-Host "  Python 3.12 not on PATH, installing via uv..."
    uv python install 3.12
}
uv venv .venv --python 3.12
& .venv\Scripts\activate.ps1

# ----------------------------------------------------------
# 2. Install PyTorch (CPU) + openai
# ----------------------------------------------------------
Write-Host "=== [2/7] Installing PyTorch (CPU) + openai ==="
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
uv pip install openai requests

# ----------------------------------------------------------
# 3. Install OmniVoice
# ----------------------------------------------------------
Write-Host "=== [3/7] Installing OmniVoice ==="
uv pip install omnivoice

# Re-pin torch versions (omnivoice may downgrade them, breaking torchvision)
uv pip install --force-reinstall torch torchaudio torchvision --index-url https://download.pytorch.org/whl/cpu

# ----------------------------------------------------------
# 4. Build llama.cpp (CPU, no CUDA)
# ----------------------------------------------------------
Write-Host "=== [4/7] Building llama.cpp ==="
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    Write-Host "  Installing cmake via winget..."
    winget install --id Kitware.CMake --accept-source-agreements --accept-package-agreements
    $env:PATH = "C:\Program Files\CMake\bin;$env:PATH"
}

if (-not (Test-Path "llama.cpp\build\bin\llama-cli.exe")) {
    git clone https://github.com/ggml-org/llama.cpp
    cmake llama.cpp -B llama.cpp/build -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=OFF
    cmake --build llama.cpp/build --config Release --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
    Copy-Item "llama.cpp\build\bin\llama-*" -Destination "llama.cpp\" -Recurse
} else {
    Write-Host "  llama.cpp binaries already built, skipping."
}

# ----------------------------------------------------------
# 5. Download Gemma model
# ----------------------------------------------------------
Write-Host "=== [5/7] Downloading Gemma model ==="
if (-not (Test-Path "unsloth\gemma-4-E2B-it-GGUF\gemma-4-E2B-it-UD-Q8_K_XL.gguf")) {
    uv pip install huggingface-hub
    hf download unsloth/gemma-4-E2B-it-GGUF `
        --local-dir unsloth/gemma-4-E2B-it-GGUF `
        --include "*mmproj-BF16*" `
        --include "*UD-Q8_K_XL*"
} else {
    Write-Host "  Gemma model files already present, skipping."
}

# ----------------------------------------------------------
# 6. Build CrispASR (CPU, no CUDA)
# ----------------------------------------------------------
Write-Host "=== [6/7] Building CrispASR ==="

if (-not (Test-Path "CrispASR\build\bin\crispasr.exe")) {
    git clone https://github.com/CrispStrobe/CrispASR
    Push-Location CrispASR
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF
    cmake --build build --target crispasr
    Pop-Location
} else {
    Write-Host "  CrispASR binary already built, skipping."
}

# ----------------------------------------------------------
# 7. Download GGUF model
# ----------------------------------------------------------
Write-Host "=== [7/7] Downloading Qwen3 ASR GGUF model ==="
if (-not (Test-Path "qwen3-asr-1.7b-f16.gguf")) {
    Invoke-WebRequest -Uri "https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/main/qwen3-asr-1.7b-f16.gguf" -OutFile "qwen3-asr-1.7b-f16.gguf"
} else {
    Write-Host "  GGUF model already present, skipping."
}

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
Write-Host ""
Write-Host "=== Installation complete ==="
Write-Host "Project directory: $ProjectDir"
Write-Host ""
Write-Host "To run the pipeline:"
Write-Host "  (The pipelines auto-start the Gemma server on port 8080.)"
Write-Host "  cd $ProjectDir"
Write-Host "  .venv\Scripts\activate.ps1"
Write-Host "  python pipeline.py --input-audio your_audio.wav --output-audio output.wav"
Write-Host "  python batch_pipeline.py --input-folder test --output-folder output"
