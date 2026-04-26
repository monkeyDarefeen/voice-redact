"""Shared configuration loader for the speech-privacy pipeline."""
import json
import os
import random

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
PROMPTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "prompts.json")

_config = None
_prompts = None


def load_config():
    global _config
    if _config is None:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            _config = json.load(f)
    return _config


def load_prompts():
    global _prompts
    if _prompts is None:
        with open(PROMPTS_PATH, encoding="utf-8") as f:
            _prompts = json.load(f)
    return _prompts


def resolve_path(key: str) -> str:
    """Resolve a relative path from config to an absolute path relative to the project root."""
    cfg = load_config()
    parts = key.split(".")
    value = cfg
    for p in parts:
        value = value[p]
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), value)


def resolve_voice_mode(cfg=None):
    """Determine the active voice generation mode.

    Priority:
    1. If tts.voice_mode is a string, use it.
    2. Otherwise, find the sub-section with 'enabled' non-null.
    """
    if cfg is None:
        cfg = load_config()
    tts = cfg["tts"]

    # Direct voice_mode field takes priority
    vm = tts.get("voice_mode")
    if vm in ("original", "random", "reference"):
        return vm

    # Fallback: whichever sub-section has enabled non-null
    for mode in ("original", "reference", "random"):
        section = tts.get(mode, {})
        if section.get("enabled") is not None:
            return mode

    return "original"  # ultimate default


def build_random_instruct(cfg=None):
    """Build a random voice design instruct string from the config's attribute lists.

    For each category, 50/50 chance of including it. If none included, fallback to gender.
    If tts.random.style is non-null, it is appended to the result.
    """
    if cfg is None:
        cfg = load_config()
    rand_cfg = cfg["tts"]["random"]
    attributes = rand_cfg["attributes"]
    selected = []

    for category, options in attributes.items():
        if options and random.random() < 0.5:
            selected.append(random.choice(options))

    # Fallback: always pick a gender if nothing else selected
    if not selected:
        selected.append(random.choice(attributes.get("gender", ["female"])))

    # Append whisper if set
    if rand_cfg.get("whisper"):
        selected.append("whisper")

    return ", ".join(selected)
