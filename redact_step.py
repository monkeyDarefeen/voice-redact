"""Step 2: Redact sensitive entities using Gemma via local llama.cpp server."""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_config import load_config, load_prompts

from openai import OpenAI


def _parse_output(raw_content: str):
    """Extract the JSON payload while ignoring the internal <|channel>thought block."""
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


def main():
    cfg = load_config()
    prompts = load_prompts()

    parser = argparse.ArgumentParser()
    parser.add_argument("--input-json", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument(
        "--gemma-url",
        default="http://localhost:8080/v1",
        help="Base URL of the llama.cpp OpenAI-compatible server",
    )
    args = parser.parse_args()

    print("[2/3] Redacting sensitive entities with Gemma ...")
    with open(args.input_json, encoding="utf-8") as f:
        data = json.load(f)

    gemma = cfg["gemma"]
    client = OpenAI(base_url=args.gemma_url, api_key=gemma["api_key"])

    response = client.chat.completions.create(
        model=gemma["model_name"],
        messages=[
            {"role": "system", "content": prompts["gemma_system"]},
            {
                "role": "user",
                "content": prompts["gemma_user_template"].format(text=data["text"]),
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
        print(f"  Gemma error: {result['error']}", file=sys.stderr)
        print(f"  Raw output: {result.get('raw', '')}", file=sys.stderr)
        sys.exit(1)

    synthetic_text = result.get("synthetic", data["text"])
    redacted_text = result.get("redacted", data["text"])
    print(f"  Redacted: {redacted_text[:200]}...")
    print(f"  Synthetic: {synthetic_text[:200]}...")

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "text": synthetic_text,
                "raw_text": data["text"],
                "language": data.get("language"),
                "redacted": redacted_text,
            },
            f,
        )

    print("  Done.")


if __name__ == "__main__":
    main()
