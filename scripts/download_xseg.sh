#!/bin/bash
# Download XSeg model for foreground preservation

set -e

MODELS_DIR="$(dirname "$0")/../models"

echo "Downloading xseg_3.onnx from FaceFusion..."
echo "Target directory: $MODELS_DIR"

mkdir -p "$MODELS_DIR"

MODEL_FILE="xseg_3.onnx"

if [ -f "$MODELS_DIR/$MODEL_FILE" ]; then
    echo "- $MODEL_FILE already exists, skipping"
else
    echo "Downloading $MODEL_FILE..."
    curl -L "https://github.com/facefusion/facefusion-assets/releases/download/models-3.2.0/xseg_3.onnx" \
         -o "$MODELS_DIR/$MODEL_FILE"

    echo "- $MODEL_FILE downloaded"
fi

echo ""
echo "XSeg model downloaded successfully!"
echo "Location: $MODELS_DIR/$MODEL_FILE"
