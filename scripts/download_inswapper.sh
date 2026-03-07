#!/bin/bash
# Download inswapper_128 model from HuggingFace

set -e

MODELS_DIR="$(dirname "$0")/../models"

echo "Downloading inswapper_128.onnx from HuggingFace..."
echo "Target directory: $MODELS_DIR"

mkdir -p "$MODELS_DIR"

MODEL_FILE="inswapper_128.onnx"

if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "- $MODEL_FILE already exists, skipping"
else
    echo "Downloading $MODEL_FILE..."
    curl -L "https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128_fp16.onnx?download=true" \
         -o "$MODELS_DIR/$MODEL_FILE"

    echo "- $MODEL_FILE downloaded"
fi

echo ""
echo "Inswapper model downloaded successfully!"
echo "Location: $MODELS_DIR/$MODEL_FILE"
