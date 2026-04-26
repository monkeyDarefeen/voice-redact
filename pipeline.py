"""Orchestrator: Transcribe (CrispASR) -> Redact (Gemma) -> Voice Clone (OmniVoice).

Step 1 runs as a C++ binary (no Python env needed).
Steps 2-3 run in the active Python environment (uv venv).
Auto-starts llama-server for Gemma if not already running.

Auto-detects: if running inside a venv or .venv exists alongside this script, uses it.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_config import load_config, resolve_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _server_ready(port):
    """Check if the llama-server is responding on the given port."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _start_llama_server(port=None):
    """Start llama-server if not already running. Returns (process, started) tuple."""
    cfg = load_config()
    if port is None:
        port = cfg["ports"]["gemma"]
    llama_server_bin = resolve_path("paths.llama_server_bin")
    gemma_model = resolve_path("paths.gemma_model")

    if _server_ready(port):
        print(f"  Gemma server already running on port {port}")
        return None, False

    if not os.path.exists(llama_server_bin):
        print(f"ERROR: llama-server not found at {llama_server_bin}", file=sys.stderr)
        print("  Run: bash install_uv.sh <PROJECT_DIR>", file=sys.stderr)
        sys.exit(1)

    print(f"  Starting Gemma server on port {port} ...")
    proc = subprocess.Popen(
        [llama_server_bin, "--model", gemma_model, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    # Wait for server to be ready
    for _ in range(120):
        if _server_ready(port):
            print("  Gemma server ready.\n")
            return proc, True
        time.sleep(1)

    print("ERROR: Gemma server failed to start.", file=sys.stderr)
    proc.kill()
    sys.exit(1)


def _stop_llama_server(proc, port=8080):
    """Stop the llama-server process."""
    if proc is not None:
        print(f"  Stopping Gemma server ...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  Gemma server stopped.")


def _resolve_python():
    """Return the Python interpreter for steps 2-3."""
    # If running inside a venv, use that interpreter
    if hasattr(sys, "base_prefix") and sys.prefix != sys.base_prefix:
        return sys.executable
    # If a .venv exists next to this script, use it
    venv_python = os.path.join(SCRIPT_DIR, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        return venv_python
    # Fallback to current Python
    return sys.executable


def run_step(python_exe: str, script: str, args: list[str]) -> subprocess.CompletedProcess:
    cmd = [python_exe, script] + args
    print(f"  CMD: {' '.join(cmd)}")
    return subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser(
        description="Transcribe -> Redact -> Voice Clone pipeline"
    )
    parser.add_argument("--input-audio", required=True, help="Path to input WAV file")
    parser.add_argument("--output-audio", required=True, help="Path for output WAV file")
    parser.add_argument("--language", default=None, help="Language code (e.g. en, zh). None = auto-detect")
    args = parser.parse_args()

    step_python = _resolve_python()
    print(f"Using Python: {step_python}\n")

    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp:
        tmp_path = tmp.name
    with tempfile.NamedTemporaryFile(suffix=".json", mode="w", encoding="utf-8", delete=False) as tmp2:
        tmp2_path = tmp2.name

    cfg = load_config()
    gemma_port = cfg["ports"]["gemma"]
    gemma_proc = None
    gemma_started = False

    try:
        # Step 1: Transcribe with CrispASR (no Python env needed)
        run_step(
            sys.executable,
            os.path.join(SCRIPT_DIR, "transcribe_step.py"),
            ["--input-audio", args.input_audio, "--output-json", tmp_path]
            + (["--language", args.language] if args.language else []),
        )

        with open(tmp_path, encoding="utf-8") as f:
            transcript = json.load(f)
        print(f"Raw transcript: {transcript['text'][:100]}...\n")

        # Start Gemma server if needed
        gemma_proc, gemma_started = _start_llama_server(gemma_port)

        # Step 2: Redact with Gemma
        run_step(
            step_python,
            os.path.join(SCRIPT_DIR, "redact_step.py"),
            ["--input-json", tmp_path, "--output-json", tmp2_path],
        )

        with open(tmp2_path, encoding="utf-8") as f:
            redacted = json.load(f)
        print(f"Synthetic text: {redacted['text'][:100]}...\n")

        # Step 3: Voice Clone
        run_step(
            step_python,
            os.path.join(SCRIPT_DIR, "voice_clone_step.py"),
            ["--input-json", tmp2_path, "--ref-audio", args.input_audio,
             "--output-audio", args.output_audio],
        )

        # Save transcript JSON with all three text variants
        wav_name = os.path.splitext(os.path.basename(args.input_audio))[0]
        transcript_out = {
            wav_name: {
                "original_text": transcript["text"],
                "redacted_text": redacted.get("redacted"),
                "synthetic_text": redacted.get("text"),
                "language": redacted.get("language"),
            }
        }
        output_dir = os.path.dirname(args.output_audio)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        transcript_path = os.path.join(output_dir, wav_name + "_transcript.json")
        with open(transcript_path, "w", encoding="utf-8") as f:
            json.dump(transcript_out, f, indent=2)
        print(f"Transcript saved to {transcript_path}")
        print(f"\nPipeline complete. Output: {args.output_audio}")

    finally:
        _stop_llama_server(gemma_proc, port=gemma_port)
        for p in (tmp_path, tmp2_path):
            if os.path.exists(p):
                os.unlink(p)


if __name__ == "__main__":
    main()
