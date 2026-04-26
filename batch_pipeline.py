"""Batch pipeline: Transcribe all -> Redact all (Gemma) -> Voice Clone all.

Loads each model once, processes all files, then unloads.
Step 1 uses CrispASR HTTP server (model loaded once, port from config to avoid Gemma conflict).
Step 2 auto-starts llama-server for Gemma.
Steps 2-3 load models in-process.
"""
import argparse
import gc
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_config import load_config, load_prompts, resolve_path, resolve_voice_mode, build_random_instruct

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def _gemma_server_ready(port):
    """Check if the Gemma server is responding."""
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def _start_gemma_server(cfg):
    """Start llama-server for Gemma if not already running. Returns the process."""
    port = cfg["ports"]["gemma"]
    llama_server_bin = resolve_path("paths.llama_server_bin")
    gemma_model = resolve_path("paths.gemma_model")

    if _gemma_server_ready(port):
        print(f"  Gemma server already running on port {port}")
        return None

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
    for _ in range(120):
        if _gemma_server_ready(port):
            print("  Gemma server ready.\n")
            return proc
        time.sleep(1)

    print("ERROR: Gemma server failed to start.", file=sys.stderr)
    proc.kill()
    sys.exit(1)


def _stop_gemma_server(proc):
    """Stop the Gemma server process."""
    if proc is not None:
        print("  Stopping Gemma server ...")
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("  Gemma server stopped.\n")


def get_audio_files(folder):
    """Return sorted list of audio files in folder."""
    extensions = {".wav", ".mp3", ".flac", ".ogg"}
    files = []
    for f in sorted(os.listdir(folder)):
        if os.path.splitext(f)[1].lower() in extensions:
            files.append(os.path.join(folder, f))
    return files


# ------------------------------------------------------------------
# Step 1: Transcribe all audios via CrispASR server
# ------------------------------------------------------------------
def transcribe_all(audio_files, output_dir, language=None):
    """Start CrispASR server, transcribe all files, stop server."""
    cfg = load_config()
    crispasr_bin = resolve_path("paths.crispasr_bin")
    model_path = resolve_path("paths.asr_model")
    port = cfg["ports"]["crispasr_server"]

    print(f"[1/3] Transcribing {len(audio_files)} audio files with CrispASR server ...")

    server = subprocess.Popen(
        [crispasr_bin, "--server", "--backend", cfg["asr"]["backend"], "-m", model_path, "--port", str(port)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    base_url = f"http://127.0.0.1:{port}"
    # Wait for server to be ready
    for attempt in range(60):
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1) as resp:
                if resp.status == 200:
                    break
        except Exception:
            pass
        time.sleep(1)
    else:
        print("  ERROR: CrispASR server failed to start.", file=sys.stderr)
        server.kill()
        sys.exit(1)

    import requests
    transcripts = []
    os.makedirs(output_dir, exist_ok=True)

    for i, audio_path in enumerate(audio_files, 1):
        fname = os.path.basename(audio_path)
        print(f"  [{i}/{len(audio_files)}] Transcribing {fname} ...")

        data = {}
        if language:
            data["language"] = language

        with open(audio_path, "rb") as f:
            resp = requests.post(
                f"{base_url}/v1/audio/transcriptions",
                files={"file": (fname, f, "audio/wav")},
                data=data,
            )

        result = resp.json()
        text = result.get("text", "").strip()
        lang = result.get("language", None)

        entry = {"filename": fname, "text": text, "language": lang}
        transcripts.append(entry)
        print(f"    {text[:100]}...")

    # Save intermediate transcripts
    transcript_path = os.path.join(output_dir, "transcripts.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(transcripts, f, indent=2)
    print(f"  Transcripts saved to {transcript_path}")

    # Stop server
    server.send_signal(signal.SIGTERM)
    server.wait(timeout=10)
    print("  CrispASR server stopped.\n")

    return transcripts


# ------------------------------------------------------------------
# Step 2: Redact all transcripts with Gemma
# ------------------------------------------------------------------
def redact_all(transcripts, output_dir, gemma_url="http://localhost:8080/v1"):
    """Use Gemma via OpenAI-compatible API to redact all transcripts."""
    import re

    from openai import OpenAI

    cfg = load_config()
    prompts = load_prompts()
    gemma = cfg["gemma"]

    print("[2/3] Redacting sensitive entities with Gemma ...")

    def _parse_output(raw_content):
        clean_text = re.sub(
            r"<\|channel>thought\n.*?\n<channel|>", "", raw_content, flags=re.DOTALL
        ).strip()
        json_match = re.search(r"\{.*\}", clean_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                return {"error": "Invalid JSON format", "raw": clean_text}
        return {"error": "No JSON found", "raw": clean_text}

    client = OpenAI(base_url=gemma_url, api_key=gemma["api_key"])

    for i, entry in enumerate(transcripts, 1):
        response = client.chat.completions.create(
            model=gemma["model_name"],
            messages=[
                {"role": "system", "content": prompts["gemma_system"]},
                {
                    "role": "user",
                    "content": prompts["gemma_user_template"].format(text=entry["text"]),
                },
            ],
            temperature=gemma["temperature"],
            max_tokens=gemma["max_tokens"],
            extra_body={
                "top_k": gemma["top_k"],
                "min_p": gemma["min_p"],
                "grammar": gemma["grammar"],
            },
        )

        result = _parse_output(response.choices[0].message.content)

        if "error" in result:
            print(f"  [{i}/{len(transcripts)}] {entry['filename']}: Gemma error: {result['error']}")
            entry["redacted_text"] = entry["text"]
            entry["synthetic_text"] = entry["text"]
        else:
            entry["synthetic_text"] = result.get("synthetic", entry["text"])
            entry["redacted_text"] = result.get("redacted", entry["text"])
            print(f"  [{i}/{len(transcripts)}] {entry['filename']}: {entry['synthetic_text'][:80]}...")

    print("  Gemma redaction complete.\n")

    return transcripts


# ------------------------------------------------------------------
# Step 3: Voice Clone all redacted transcripts
# ------------------------------------------------------------------
def voice_clone_all(transcripts, audio_folder, output_dir):
    """Load OmniVoice once, generate audio for all redacted transcripts."""
    cfg = load_config()
    tts = cfg["tts"]
    voice_mode = resolve_voice_mode(cfg)

    print(f"[3/3] Generating audio with OmniVoice (mode={voice_mode}) ...")

    import torch
    import soundfile as sf
    from omnivoice import OmniVoice

    model = OmniVoice.from_pretrained(
        tts["model_name"],
        device_map=tts["device"],
        dtype=getattr(torch, tts["dtype"]),
    )

    os.makedirs(output_dir, exist_ok=True)

    # Pre-resolve reference audio path for reference mode
    ref_audio_path = None
    ref_text = None
    if voice_mode == "reference":
        ref_cfg = tts["reference"]
        ref_audio_path = ref_cfg["ref_audio_path"]
        if not os.path.isabs(ref_audio_path):
            ref_audio_path = os.path.join(SCRIPT_DIR, ref_audio_path)
        ref_text = ref_cfg["ref_text"]

    for i, entry in enumerate(transcripts, 1):
        out_path = os.path.join(output_dir, entry["filename"])

        print(f"  [{i}/{len(transcripts)}] Generating {out_path} ...")

        if voice_mode == "original":
            audio = model.generate(
                text=entry["synthetic_text"],
                ref_audio=os.path.join(audio_folder, entry["filename"]),
                ref_text=entry["text"],
            )
        elif voice_mode == "reference":
            audio = model.generate(
                text=entry["synthetic_text"],
                ref_audio=ref_audio_path,
                ref_text=ref_text,
            )
        elif voice_mode == "random":
            instruct = build_random_instruct(cfg)
            audio = model.generate(
                text=entry["synthetic_text"],
                instruct=instruct,
            )
            print(f"    Voice design instruct: {instruct}")

        sf.write(out_path, audio[0], tts["sample_rate"])
        print(f"    Done.")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  OmniVoice unloaded.\n")

    # Save consolidated transcript JSON keyed by filename
    result = {}
    for entry in transcripts:
        result[entry["filename"]] = {
            "original_text": entry["text"],
            "redacted_text": entry["redacted_text"],
            "synthetic_text": entry["synthetic_text"],
            "language": entry["language"],
        }

    transcript_path = os.path.join(output_dir, "transcripts.json")
    with open(transcript_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"  Consolidated transcripts saved to {transcript_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Batch pipeline: Transcribe all -> Redact all (Gemma) -> Voice Clone all"
    )
    parser.add_argument("--input-folder", required=True, help="Folder containing audio files")
    parser.add_argument("--output-folder", required=True, help="Folder for output files")
    parser.add_argument("--language", default=None, help="Language code (e.g. en, zh). None = auto-detect")
    parser.add_argument(
        "--gemma-url",
        default="http://localhost:8080/v1",
        help="Base URL of the llama.cpp OpenAI-compatible server running Gemma",
    )
    args = parser.parse_args()

    audio_files = get_audio_files(args.input_folder)
    if not audio_files:
        print(f"ERROR: No audio files found in {args.input_folder}", file=sys.stderr)
        sys.exit(1)

    print(f"Found {len(audio_files)} audio files:\n")
    for f in audio_files:
        print(f"  - {os.path.basename(f)}")
    print()

    # Step 1: Transcribe all
    transcripts = transcribe_all(audio_files, args.output_folder, language=args.language)

    # Start Gemma server
    cfg = load_config()
    gemma_proc = _start_gemma_server(cfg)

    try:
        # Step 2: Redact all with Gemma
        transcripts = redact_all(transcripts, args.output_folder, gemma_url=args.gemma_url)

        # Step 3: Voice Clone all
        voice_clone_all(transcripts, args.input_folder, args.output_folder)
    finally:
        _stop_gemma_server(gemma_proc)

    print(f"Pipeline complete. Output: {args.output_folder}")


if __name__ == "__main__":
    main()
