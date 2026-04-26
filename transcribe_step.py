"""Step 1: Transcribe audio with CrispASR (C++ binary, no Python env needed)."""
import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_config import load_config, resolve_path

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    cfg = load_config()
    crispasr_bin = resolve_path("paths.crispasr_bin")
    model_path = resolve_path("paths.asr_model")

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-audio", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()

    print("[1/3] Transcribing audio with CrispASR ...")
    cmd = [
        crispasr_bin,
        "--backend", cfg["asr"]["backend"],
        "-m", model_path,
        "-f", args.input_audio,
    ]
    if args.language:
        cmd += ["-l", args.language]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  CrispASR error:\n{result.stderr}", file=sys.stderr)
        sys.exit(1)

    text = result.stdout.strip()
    stderr_text = result.stderr

    lang = None
    for line in stderr_text.splitlines():
        if "detected language:" in line:
            lang = line.split("detected language:")[-1].strip()
            break

    print(f"  Language: {lang}")
    print(f"  Transcript: {text[:200]}...")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump({"text": text, "language": lang}, f)

    print("  Done.\n")


if __name__ == "__main__":
    main()
