#!/bin/bash
# Download buffalo_l models from InsightFace

set -e

MODELS_DIR="$(dirname "$0")/../models"
BUFFALO_DIR="$MODELS_DIR/buffalo_l"
TMP_DIR="/tmp/buffalo_l_download_$$"

echo "Downloading buffalo_l models from InsightFace..."
echo "Target directory: $BUFFALO_DIR"

mkdir -p "$BUFFALO_DIR"
mkdir -p "$TMP_DIR"

# det_10g.onnx - YOLOv8n face detector
# w600k_r50.onnx - ArcFace ResNet50 recognition
# 2d106det.onnx - 106-point facial landmark detector for mouth mask feature
# Check if models already exist
if [ -f "$BUFFALO_DIR/det_10g.onnx" ] && [ -f "$BUFFALO_DIR/w600k_r50.onnx" ] && [ -f "$BUFFALO_DIR/2d106det.onnx" ]; then
    echo "Models already exist, skipping download"
    exit 0
fi

# Download buffalo_l.zip from InsightFace GitHub releases
echo "Downloading buffalo_l.zip..."
curl -L "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip" \
    -o "$TMP_DIR/buffalo_l.zip"

# Extract only needed models
echo "Extracting models..."
unzip -j "$TMP_DIR/buffalo_l.zip" "det_10g.onnx" "w600k_r50.onnx" "2d106det.onnx" -d "$BUFFALO_DIR"

# Cleanup
rm -rf "$TMP_DIR"

echo ""
echo "Buffalo_l models downloaded successfully!"
echo "Location: $BUFFALO_DIR"
