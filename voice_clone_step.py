"""Step 3: Generate voice-cloned audio with OmniVoice. Runs in pipeline_privacy env."""
import argparse
import gc
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_config import load_config, resolve_path, resolve_voice_mode, build_random_instruct

import torch
import soundfile as sf
from omnivoice import OmniVoice


def main():
    cfg = load_config()
    tts = cfg["tts"]
    voice_mode = resolve_voice_mode(cfg)

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--ref-audio", required=True)
    parser.add_argument("--output-audio", required=True)
    args = parser.parse_args()

    print(f"[3/3] Generating audio with OmniVoice (mode={voice_mode}) ...")
    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    model = OmniVoice.from_pretrained(
        tts["model_name"],
        device_map=tts["device"],
        dtype=getattr(torch, tts["dtype"]),
    )

    if voice_mode == "original":
        audio = model.generate(
            text=data["text"],
            ref_audio=args.ref_audio,
            ref_text=data["raw_text"],
        )
        print(f"  Using original reference audio: {args.ref_audio}")

    elif voice_mode == "reference":
        ref_cfg = tts["reference"]
        ref_audio_path = ref_cfg["ref_audio_path"]
        if not os.path.isabs(ref_audio_path):
            ref_audio_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ref_audio_path)
        audio = model.generate(
            text=data["text"],
            ref_audio=ref_audio_path,
            ref_text=ref_cfg["ref_text"],
        )
        print(f"  Using reference audio: {ref_audio_path}")

    elif voice_mode == "random":
        instruct = build_random_instruct(cfg)
        audio = model.generate(
            text=data["text"],
            instruct=instruct,
        )
        print(f"  Voice design instruct: {instruct}")

    else:
        print(f"ERROR: Unknown voice mode '{voice_mode}'", file=sys.stderr)
        sys.exit(1)

    sf.write(args.output_audio, audio[0], tts["sample_rate"])
    print(f"  Output written to: {args.output_audio}")

    del model
    gc.collect()
    torch.cuda.empty_cache()
    print("  OmniVoice unloaded.")


if __name__ == "__main__":
    main()
