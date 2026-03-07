#!/bin/bash
# Download buffalo_l models from HuggingFace

set -e

MODELS_DIR="$(dirname "$0")/../models"
BUFFALO_DIR="$MODELS_DIR/buffalo_l"

echo "Downloading buffalo_l models from HuggingFace..."
echo "Target directory: $BUFFALO_DIR"

mkdir -p "$BUFFALO_DIR"

# Base URL for HuggingFace
BASE_URL="https://huggingface.co/immich-app/buffalo_l/resolve/main"

# det_10g.onnx - YOLOv8n face detector
# w600k_r50.onnx - ArcFace ResNet50 recognition
MODELS=(
    "det_10g.onnx"
    "w600k_r50.onnx"
)

for model in "${MODELS[@]}"; do
    if [ -f "$BUFFALO_DIR/$model" ]; then
        echo "- $model already exists, skipping"
    else
        echo "Downloading $model..."
        curl -L "$BASE_URL/$model" -o "$BUFFALO_DIR/$model"
        echo "- $model downloaded"
    fi
done

echo ""
echo "Buffalo_l models downloaded successfully!"
echo "Location: $BUFFALO_DIR"
