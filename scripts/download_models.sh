#!/bin/bash
# Download all required models

set -e

SCRIPT_DIR="$(dirname "$0")"

echo "Downloading models..."
echo ""

bash "$SCRIPT_DIR/download_buffalo_l.sh"
echo ""

bash "$SCRIPT_DIR/download_inswapper.sh"
echo ""

echo "All models downloaded."
echo ""
echo "Build and run:"
echo "cd .. && \\"
echo "cargo build --release && \\"
echo "./target/release/deep-faceswap swap -s source.jpg -t target.jpg -o output.jpg"
