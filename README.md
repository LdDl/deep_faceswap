# Deep FaceSwap

Rust implementation of face swapping. Basically a port of [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)

## Table of contents
- [Work in progress](#work-in-progress)
- [Quick start](#quick-start)
- [CLI usage](#cli-usage)
- [CUDA support](#cuda-support)
- [Project structure](#project-structure)
- [License](#license)

## Work in progress

Current:
- Basic face swap between two images (single source + single target)
- Face detection using YOLOv8n from buffalo_l package
- Face recognition using ArcFace w600k_r50
- Face swapping using inswapper_128
- CUDA acceleration support via ONNX Runtime

Planned:
- Swap faces (single source + target, 2 photos)
- Add face enhancement
- Add mouth mask
- Video processing (single source + target)
- Multiple faces (not available in CLI)
- Online video (low priority)

## Quick start

Clone the repository and navigate to the project directory:
```bash
git clone git@github.com:LdDl/deep-faceswap.git --depth 1
cd deep-faceswap
```

### 1. Download models

```bash
cd scripts
./download_models.sh
```

This will download:
- `buffalo_l/det_10g.onnx` - Face detector (YOLOv8n)
- `buffalo_l/w600k_r50.onnx` - Face recognizer (ArcFace ResNet50)
- `inswapper_128.onnx` - Face swapper

#### Optional: Face enhancement model

If you want to use face enhancement (GFPGAN - https://github.com/TencentARC/GFPGAN), download and convert the model. I've tested it with `GFPGANv1.4.pth` only. Newer versions may not work properly, but if you tested and it worked, please let me know.

1. Download the PyTorch model:

```bash
cd scripts
./download_gfpgan.sh
```

This will download:
- `GFPGANv1.4.pth` - Face enhancement model (PyTorch format, 333MB)

2. Convert to ONNX format:

```bash
cd scripts/python
python3 -m venv venv_torch2onnx
source venv_torch2onnx/bin/activate
pip install -r requirements-torch2onnx.txt
python torch2onnx/torch2onnx.py --src_model_path ../../models/GFPGANv1.4.pth --dst_model_path ../../models/GFPGANv1.4.onnx
deactivate
# Clean up virtual environment if you don't need it anymore
rm -rf venv_torch2onnx
```

This will create:
- `GFPGANv1.4.onnx` - Converted model ready for ONNX inference

Note 1: I've used Python 3.14.2. If you have a different version, it may work, but I haven't tested it.

Note 2: The conversion process will install PyTorch CPU-only version, which is around 200MB. This is a one-time operation, and after conversion, you can remove the virtual environment if you want. GPU version of PyTorch is not required for conversion.

### 2. Build

CPU-only build:
```bash
cargo build --release
```

Build with CUDA support:
```bash
cargo build --release --features cuda
```

### 3. Run

```bash
./target/release/deep-faceswap-cli swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg
```

The tool will:
1. Load models (detector, recognizer, swapper)
2. Detect faces in source and target images
3. Extract source face embedding
4. Align target face
5. Swap faces
6. Paste result back to target image
7. Save output

## CLI usage

### Basic swap

```bash
deep-faceswap-cli swap \
  --source <source> \
  --target <target> \
  --output <output>
```

### Custom model paths

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg \
  --detector models/buffalo_l/det_10g.onnx \
  --recognizer models/buffalo_l/w600k_r50.onnx \
  --swapper models/inswapper_128.onnx
```

### Requirements

- Rust toolchain. My setup is:
  - cargo 1.93.0 (083ac5135 2025-12-15)
  - rustc 1.93.0 (254b59607 2026-01-19)

- Downloaded models
- Source and target images.
- CUDA and cuDNN for GPU acceleration (optional, but recommended for better performance)

## CUDA support

To use CUDA acceleration:

1. Install CUDA Toolkit and cuDNN. I've tested only with my current setup which is:
- CUDA 13.1
- cuDNN 9.18.1.3.-1.1
- RTX 3060

2. Build with CUDA feature:
   ```bash
   cargo build --release --features cuda
   ```

When built with the cuda feature, the tool will use CUDA for inference. If CUDA is not available at runtime, it will fall back to CPU.

**Interactive Face Selection**: If multiple faces are detected, the CLI will save face crops to `./tmp/face_crops/` and prompt you to select which face to use.

Current status for face selection: only max score face is automatically selected. Interactive selection is planned for future.
## Project structure

@todo

## License

@todo
