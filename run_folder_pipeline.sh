#!/bin/bash

# Input and output directories
INPUT_DIR="test"
OUTPUT_DIR="output"

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Loop through all .wav files in the input directory
for file in "$INPUT_DIR"/*.wav; do
    # Get just the filename (without path)
    filename=$(basename "$file")

    # Run the command
    uv run pipeline.py \
        --input-audio "$file" \
        --output-audio "$OUTPUT_DIR/$filename"
done
