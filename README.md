# voice-redact

Transcribe audio, redact PII with Gemma 4, then regenerate speech with a clean voice.

**Pipeline flow:** `Audio → Transcribe (Qwen3-TTS) → Redact (Gemma 4) → Voice Clone (OmniVoice) → Clean Audio`

**voice-redact** takes your audio recordings, strips out personally identifiable information (names, phone numbers, addresses), and regenerates natural-sounding speech with the sensitive parts removed. The output keeps the speaker's voice but zero private data survives — making it safe to share.

## Prerequisites

### Linux (GPU)

- NVIDIA GPU with CUDA support
- CUDA toolkit installed

### macOS / Windows (CPU)

- No GPU required
- **macOS:** Homebrew (installed automatically if missing)
- **Windows:** PowerShell, Visual Studio Build Tools (Desktop C++ workload)

## Installation

Choose the script for your platform:

**Linux with GPU:**
```bash
bash install_uv.sh /path/to/project/dir
```
Installs PyTorch with CUDA 12.8 and builds llama.cpp / CrispASR with GPU acceleration.

**macOS (CPU):**
```bash
bash install_macos_cpu.sh /path/to/project/dir
```

**Windows (CPU):**
```powershell
pwsh -ExecutionPolicy Bypass -File install_uv.ps1 -ProjectDir ".\"
```

All scripts will:
1. Install `uv` and create a Python 3.12 virtual environment
2. Install PyTorch + OpenAI Python SDK
3. Install OmniVoice (TTS)
4. Build llama.cpp (serves Gemma via OpenAI-compatible API)
5. Download the Gemma 4 2B GGUF model
6. Build CrispASR (speech-to-text)
7. Download the Qwen3 ASR GGUF model

### Windows: After Installation

Before running the pipeline, update `config.json` to use `.exe` extensions for the native binary paths:

```json
{
  "paths": {
    "crispasr_bin": "CrispASR/build/bin/crispasr.exe",
    "llama_server_bin": "llama.cpp/build/bin/llama-server.exe"
  }
}
```

The default config uses Linux-style paths (no `.exe`). On Windows, `subprocess.run` will fail with `FileNotFoundError: [WinError 2]` if the `.exe` extension is missing. Adjust the paths in `config.json` to match where the install script actually built the binaries — they may vary depending on your CMake generator and build directory layout.

## Usage

### Single-file pipeline

```bash
# Linux / macOS
source .venv/bin/activate
python pipeline.py --input-audio test/id1.wav --output-audio output/id1.wav
```

```powershell
# Windows
.venv\Scripts\activate.ps1
python pipeline.py --input-audio test/id1.wav --output-audio output/id1.wav
```

The pipeline will:
1. Transcribe the audio with CrispASR
2. Auto-start the Gemma server on port 8080 (if not already running)
3. Send the transcript to Gemma for PII redaction and synthetic text generation
4. Generate voice-cloned audio using the synthetic text
5. Save a `<filename>_transcript.json` alongside the output WAV with all three text variants

### Batch pipeline

```bash
# Linux / macOS
source .venv/bin/activate
python batch_pipeline.py --input-folder test --output-folder output
# or
uv run batch_pipeline.py --input-folder test --output-folder output
```

```powershell
# Windows
.venv\Scripts\activate.ps1
python batch_pipeline.py --input-folder test --output-folder output
```

The pipeline will:
1. Start a CrispASR server (port 8081), transcribe all files, then stop it
2. Start the Gemma server (port 8080), redact all transcripts, then stop it
3. Load OmniVoice once, generate audio for all files, then unload
4. Save `transcripts.json` in the output folder with all entries keyed by filename

### Shell script for batch processing

```bash
bash run_folder_pipeline.sh
```

Runs the single-file pipeline for each `.wav` in `test/`, outputting to `output/`. The Gemma server starts on the first file and stays warm for subsequent files.

## Output JSON format

**Single-file** (`output/id1_transcript.json`):
```json
{
  "id1": {
    "original_text": "Call John Smith at 852-599-21...",
    "redacted_text": "Call [NAME] at [PHONE]...",
    "synthetic_text": "Call Alice Vance at five five five...",
    "language": "en"
  }
}
```

**Batch** (`output/transcripts.json`):
```json
{
  "id1.wav": {
    "original_text": "...",
    "redacted_text": "...",
    "synthetic_text": "...",
    "language": "en"
  },
  "id2.wav": { ... }
}
```

## Command-line options

| Option | Description | Default |
|---|---|---|
| `--input-audio` | Path to input WAV file | (required, single) |
| `--output-audio` | Path for output WAV file | (required, single) |
| `--input-folder` | Folder containing audio files | (required, batch) |
| `--output-folder` | Folder for output files | (required, batch) |
| `--language` | Language code (e.g. `en`, `zh`) | auto-detect |
| `--gemma-url` | Gemma server URL | `http://localhost:8080/v1` |

## Architecture

| Step | Component | Port | How it's managed |
|---|---|---|---|
| 1. Transcribe | CrispASR (C++ binary) | 8081 (batch server) | Auto-started/stopped |
| 2. Redact | Gemma 4 via llama-server | 8080 | Auto-started/stopped |
| 3. Voice Clone | OmniVoice (in-process) | — | Loaded/unloaded in-process |

The Gemma server is started before Step 2 and stopped after Step 3 (batch) or left running (single-file, for reuse across files in `run_folder_pipeline.sh`).

## Configuration

All tunable parameters are in JSON config files at the project root — no code edits needed.

| File | What it controls |
|---|---|
| `config.json` | Model paths, ports, ASR backend, Gemma inference params (temperature, max_tokens, grammar), TTS model/device/dtype/sample_rate, voice generation mode |
| `prompts.json` | Gemma system prompt and user message template |

To change models, ports, or generation parameters, edit `config.json`. To change the PII redaction instructions, edit `prompts.json`.

### Voice Generation Modes

Set `tts.voice_mode` in `config.json` to choose how Step 3 generates audio:

| Mode | `voice_mode` value | Behavior |
|---|---|---|
| **Original** (default) | `"original"` | Clones the original speaker's voice using their input audio as reference |
| **Reference** | `"reference"` | Uses a different reference audio + reference text (set `tts.reference.ref_audio_path` and `tts.reference.ref_text`) |
| **Random** | `"random"` | Uses OmniVoice Voice Design — picks random speaker attributes (gender, age, pitch, style, accent/dialect) each time. Edit `tts.random.attributes` to control the pool |

**Mode resolution:** If `voice_mode` is set to a string, it takes priority. If null/missing, the validator looks at which sub-section (`original`, `reference`, `random`) has `enabled` non-null. Exactly one mode must be active.

### Config Validation

Before running the pipeline, validate all config files:

```bash
python validate_config.py
```

This checks required keys, types, voice mode consistency, file existence, and prompt structure. Exits 0 on success, 1 with errors listed.

## Key files

| File | Purpose |
|---|---|
| `pipeline.py` | Single-file orchestrator |
| `batch_pipeline.py` | Batch orchestrator |
| `transcribe_step.py` | Step 1: CrispASR transcription |
| `redact_step.py` | Step 2: Gemma PII redaction |
| `voice_clone_step.py` | Step 3: OmniVoice generation |
| `load_config.py` | Shared config loader (used by all scripts) |
| `validate_config.py` | Config validation script |
| `config.json` | Model paths, ports, and inference parameters |
| `prompts.json` | Gemma system/user prompts |
| `install_uv.sh` | Linux (GPU) setup script |
| `install_macos_cpu.sh` | macOS (CPU) setup script |
| `install_uv.ps1` | Windows (CPU) setup script |
| `run_folder_pipeline.sh` | Shell loop for single-file pipeline |

## Special Thanks and References:
https://github.com/ggml-org/llama.cpp
https://github.com/CrispStrobe/CrispASR
https://unsloth.ai/docs/models/gemma-4
https://huggingface.co/cstr

