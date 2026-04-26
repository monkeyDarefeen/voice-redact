#!/bin/bash
set -e

# ============================================================
# Lightweight install using uv (no conda needed) -- Gemma edition
# Replaces the ONNX privacy filter with Gemma via llama.cpp.
# Usage: bash install_uv.sh <PROJECT_DIR>
#   PROJECT_DIR can be an existing directory (e.g. ./test)
# ============================================================

if [ -z "$1" ]; then
    echo "Usage: bash install_uv.sh <PROJECT_DIR>"
    echo "  e.g. bash install_uv.sh ./"
    echo "       bash install_uv.sh /app/datasets/local_test/anonVT/test"
    exit 1
fi

PROJECT_DIR="$(cd "$1" 2>/dev/null && pwd || { mkdir -p "$1" && cd "$1" && pwd; })"
cd "$PROJECT_DIR"

# ----------------------------------------------------------
# 0. Install uv if not present
# ----------------------------------------------------------
echo "=== [0/6] Installing uv ==="
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # Source to get uv in PATH for this session
    export PATH="$HOME/.local/bin:$PATH"
fi

# ----------------------------------------------------------
# 1. Ensure Python 3.12 is available, create venv
# ----------------------------------------------------------
echo "=== [1/6] Creating virtual environment ==="
if ! command -v python3.12 &>/dev/null; then
    echo "  Python 3.12 not on PATH, installing via uv..."
    uv python install 3.12
fi
uv venv .venv --python 3.12
source .venv/bin/activate

# ----------------------------------------------------------
# 2. Install PyTorch (CUDA 12.8) + openai
# ----------------------------------------------------------
echo "=== [2/6] Installing PyTorch + openai ==="
uv pip install \
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install openai requests

# ----------------------------------------------------------
# 3. Install OmniVoice
# ----------------------------------------------------------
echo "=== [3/6] Installing OmniVoice ==="
uv pip install omnivoice

# Re-pin torch versions (omnivoice may downgrade them, breaking torchvision)
uv pip install --force-reinstall torch torchaudio torchvision \
    --index-url https://download.pytorch.org/whl/cu128

# ----------------------------------------------------------
# 4. Build llama.cpp (needed to serve Gemma)
# ----------------------------------------------------------
echo "=== [4/7] Building llama.cpp ==="
if ! command -v cmake &> /dev/null; then
    echo "  Installing cmake and build tools..."
    apt-get update && apt-get install -y cmake build-essential
fi

if [ ! -f llama.cpp/build/bin/llama-cli ]; then
    git clone https://github.com/ggml-org/llama.cpp
    cmake llama.cpp -B llama.cpp/build \
        -DBUILD_SHARED_LIBS=OFF -DGGML_CUDA=ON
    cmake --build llama.cpp/build --config Release -j$(nproc) --clean-first \
        --target llama-cli llama-mtmd-cli llama-server llama-gguf-split
    cp llama.cpp/build/bin/llama-* llama.cpp
else
    echo "  llama.cpp binaries already built, skipping."
fi

# ----------------------------------------------------------
# 5. Download Gemma model
# ----------------------------------------------------------
echo "=== [5/7] Downloading Gemma model ==="
if [ ! -f unsloth/gemma-4-E2B-it-GGUF/gemma-4-E2B-it-UD-Q8_K_XL.gguf ]; then
    uv pip install huggingface-hub
    hf download unsloth/gemma-4-E2B-it-GGUF \
        --local-dir unsloth/gemma-4-E2B-it-GGUF \
        --include "*mmproj-BF16*" \
        --include "*UD-Q8_K_XL*"
else
    echo "  Gemma model files already present, skipping."
fi

# ----------------------------------------------------------
# 6. Build CrispASR (C++ binary, no Python deps)
# ----------------------------------------------------------
echo "=== [6/7] Building CrispASR ==="

if [ ! -f CrispASR/build/bin/crispasr ]; then
    git clone https://github.com/CrispStrobe/CrispASR
    cd CrispASR
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON
    cmake --build build -j$(nproc) --target crispasr
    cd ..
else
    echo "  CrispASR binary already built, skipping."
fi

# ----------------------------------------------------------
# 7. Download GGUF model
# ----------------------------------------------------------
echo "=== [7/7] Downloading Qwen3 ASR GGUF model ==="
if [ ! -f qwen3-asr-1.7b-f16.gguf ]; then
    wget https://huggingface.co/cstr/qwen3-asr-1.7b-GGUF/resolve/main/qwen3-asr-1.7b-f16.gguf
else
    echo "  GGUF model already present, skipping."
fi

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo ""
echo "=== Installation complete ==="
echo "Project directory: $PROJECT_DIR"
echo ""
echo "To run the pipeline:"
echo "  (The pipelines auto-start the Gemma server on port 8080.)"
echo "  cd $PROJECT_DIR"
echo "  source .venv/bin/activate"
echo "  python pipeline.py --input-audio your_audio.wav --output-audio output.wav"
echo "  python batch_pipeline.py --input-folder test --output-folder output"
