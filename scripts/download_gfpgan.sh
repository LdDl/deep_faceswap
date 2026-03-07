#!/bin/bash
# Download GFPGAN model for face enhancement

set -e

MODELS_DIR="$(dirname "$0")/../models"

echo "Downloading GFPGANv1.4.pth for face enhancement..."
echo "Target directory: $MODELS_DIR"

mkdir -p "$MODELS_DIR"

MODEL_FILE="GFPGANv1.4.pth"

if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "- $MODEL_FILE already exists, skipping"
else
    echo "Downloading $MODEL_FILE..."
    curl -L "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth" \
         -o "$MODELS_DIR/$MODEL_FILE"

    echo "- $MODEL_FILE downloaded"
fi

echo ""
echo "GFPGAN model downloaded successfully!"
echo "Location: $MODELS_DIR/$MODEL_FILE"
echo ""
echo "Note: This model needs to be converted to ONNX format before use."
echo "Conversion script will be provided in future updates."
