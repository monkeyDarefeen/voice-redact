"""Validate config.json and prompts.json at startup.

Exit 0 if all checks pass, exit 1 with the first error found.
Usage: python validate_config.py
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_config import resolve_path, resolve_voice_mode

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(PROJECT_DIR, "config.json")
PROMPTS_PATH = os.path.join(PROJECT_DIR, "prompts.json")

errors = []
warnings = []


def check(condition: bool, msg: str, is_warning: bool = False):
    if not condition:
        if is_warning:
            warnings.append(msg)
        else:
            errors.append(msg)


def main():
    cfg = None
    prompts = None

    # Load config
    print("Validating configuration files...\n")

    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        print("[OK] config.json is valid JSON")
    except FileNotFoundError:
        print(f"[FAIL] config.json not found at {CONFIG_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[FAIL] config.json has invalid JSON: {e}")
        sys.exit(1)

    # Load prompts
    try:
        with open(PROMPTS_PATH, encoding="utf-8") as f:
            prompts = json.load(f)
        print("[OK] prompts.json is valid JSON")
    except FileNotFoundError:
        print(f"[FAIL] prompts.json not found at {PROMPTS_PATH}")
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"[FAIL] prompts.json has invalid JSON: {e}")
        sys.exit(1)

    # --- config.json: required top-level sections ---
    for section in ("paths", "ports", "asr", "gemma", "tts"):
        check(section in cfg, f"Missing top-level section: '{section}'")

    # --- paths section ---
    if "paths" in cfg:
        for key in ("crispasr_bin", "asr_model", "llama_server_bin", "gemma_model"):
            check(key in cfg["paths"], f"paths.{key} is missing")
            if key in cfg["paths"]:
                abs_path = resolve_path(f"paths.{key}")
                check(os.path.exists(abs_path), f"paths.{key} does not exist: {abs_path}", is_warning=True)

    # --- ports section ---
    if "ports" in cfg:
        for key in ("gemma", "crispasr_server"):
            check(key in cfg["ports"], f"ports.{key} is missing")
            if key in cfg["ports"]:
                check(isinstance(cfg["ports"][key], int), f"ports.{key} must be an integer")

    # --- asr section ---
    if "asr" in cfg:
        check("backend" in cfg["asr"], "asr.backend is missing")
        if "backend" in cfg["asr"]:
            check(cfg["asr"]["backend"] in ("qwen3",), f"asr.backend '{cfg['asr']['backend']}' is unsupported")

    # --- gemma section ---
    if "gemma" in cfg:
        gemma = cfg["gemma"]
        for key in ("model_name", "api_key"):
            check(key in gemma and gemma[key], f"gemma.{key} is missing or empty")

        check("temperature" in gemma, "gemma.temperature is missing")
        if "temperature" in gemma:
            check(isinstance(gemma["temperature"], (int, float)), "gemma.temperature must be a number")

        check("max_tokens" in gemma, "gemma.max_tokens is missing")
        if "max_tokens" in gemma:
            check(isinstance(gemma["max_tokens"], int), "gemma.max_tokens must be an integer")

        for key in ("top_k", "min_p"):
            check(key in gemma, f"gemma.{key} is missing")

        check("grammar" in gemma and gemma["grammar"], "gemma.grammar is missing or empty")

    # --- tts section ---
    if "tts" in cfg:
        tts = cfg["tts"]
        for key in ("model_name", "device", "dtype", "sample_rate"):
            check(key in tts and tts[key], f"tts.{key} is missing or empty")

        if "sample_rate" in tts:
            check(isinstance(tts["sample_rate"], int), "tts.sample_rate must be an integer")

        # Voice mode validation
        vm = resolve_voice_mode(cfg)
        print(f"[OK] Resolved voice mode: '{vm}'")

        if vm == "reference":
            ref = tts.get("reference", {})
            check(ref.get("ref_audio_path"), "tts.reference.ref_audio_path is missing (required for reference mode)")
            if ref.get("ref_audio_path"):
                ref_path = ref["ref_audio_path"]
                if not os.path.isabs(ref_path):
                    ref_path = os.path.join(PROJECT_DIR, ref_path)
                check(os.path.exists(ref_path), f"tts.reference.ref_audio_path does not exist: {ref_path}")
            check(ref.get("ref_text"), "tts.reference.ref_text is missing (required for reference mode)")

        if vm == "random":
            rand_cfg = tts.get("random", {})
            attrs = rand_cfg.get("attributes", {})
            non_empty = [k for k, v in attrs.items() if isinstance(v, list) and len(v) > 0]
            check(len(non_empty) > 0, "tts.random.attributes must have at least one non-empty category list")
            whisper_val = rand_cfg.get("whisper")
            if whisper_val is not None:
                check(isinstance(whisper_val, bool), "tts.random.whisper must be a boolean (true/false) or null")

        # Warn if multiple sub-sections have enabled non-null
        enabled_modes = []
        for mode in ("original", "reference", "random"):
            section = tts.get(mode, {})
            if section.get("enabled") is not None:
                enabled_modes.append(mode)
        if len(enabled_modes) > 1 and not tts.get("voice_mode"):
            warnings.append(f"Multiple voice sub-sections have 'enabled' set: {enabled_modes}. Use voice_mode to disambiguate.")

    # --- prompts.json checks ---
    check("gemma_system" in prompts, "prompts.json is missing 'gemma_system'")
    check("gemma_user_template" in prompts, "prompts.json is missing 'gemma_user_template'")
    if "gemma_user_template" in prompts:
        check("{text}" in prompts["gemma_user_template"], "prompts.json 'gemma_user_template' must contain {{text}} placeholder")

    # --- Print results ---
    print()
    if warnings:
        for w in warnings:
            print(f"[WARN] {w}")
        print()

    if errors:
        for e in errors:
            print(f"[FAIL] {e}")
        print(f"\n{len(errors)} error(s) found. Fix config.json / prompts.json before running the pipeline.")
        sys.exit(1)
    else:
        print("All checks passed.")
        sys.exit(0)


if __name__ == "__main__":
    main()
