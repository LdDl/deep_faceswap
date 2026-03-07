# Convert GFPGANv1.4 to ONNX

Strictly taken from: https://github.com/clibdev/GFPGAN-onnxruntime-demo?tab=readme-ov-file

This script converts GFPGANv1.4 PyTorch model to ONNX format for use with ONNX Runtime.

## Requirements

Python 3.14+

## Usage

```bash
python3 -m venv venv_torch2onnx
source venv_torch2onnx/bin/activate
pip install -r ../requirements-torch2onnx.txt
python torch2onnx.py \
  --src_model_path $GFPGAN_TORCH_WEIGHTS_INPUT_FILE \
  --dst_model_path $GFPGAN_ONNX_WEIGHTS_OUTPUT_FILE \
  --opset 17
```

Example (opset defaults is 17. I did test it with 11 as well, but I prefer to stick with the latest compatible):

```bash
python torch2onnx.py \
  --src_model_path ../../models/GFPGANv1.4.pth \
  --dst_model_path ../../models/GFPGANv1.4.onnx
```

To use with a different opset version:

```bash
python torch2onnx.py \
  --src_model_path ../../models/GFPGANv1.4.pth \
  --dst_model_path ../../models/GFPGANv1.4.onnx \
  --opset 11
```

## How it works

The script:
1. Loads the GFPGANReconsitution architecture (standalone, no external dependencies)
2. Loads weights from PyTorch checkpoint (supports `params_ema` and `params` keys)
3. Transforms checkpoint key names to match the model:
   - Replaces dots with 'dot' in StyleGAN decoder keys
   - Restores `.weight` and `.bias` for module parameters
   - Keeps `dotweight` and `dotbias` for direct parameters
4. Skips noise keys (not needed for inference)
5. Exports to ONNX with opset version 17 (configurable)

Expected output:
- Missing keys: 0
- Unexpected keys: 14 (toRGB layers not used in inference)
- Skipped: 31 (noise parameters)

The resulting ONNX model:
- Input: `[1, 3, 512, 512]` tensor (RGB image, normalized to [-1, 1])
- Output: `[1, 3, 512, 512]` tensor (enhanced RGB image, range [-1, 1])

## Notes

I am using the standalone GFPGANReconsitution architecture, no GFPGAN library required. I've decided to pick this way because of no GFPGAN for Python 3.14 (07.03.2026).
