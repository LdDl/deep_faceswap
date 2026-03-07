# Deep FaceSwap

Rust implementation of face swapping. Basically a port of [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)

## Table of contents
- [Work in progress](#work-in-progress)
- [Quick start](#quick-start)
- [CLI usage](#cli-usage)
- [Project structure](#project-structure)
- [License](#license)

## Work in progress

Basic face swap between two images (single source + single target):
- Workspace structure (lib + CLI)
- Download scripts for models
- Face detection (YOLOv8n)
- Face recognition (ArcFace)
- Face swapping (inswapper_128)
- CLI interface

## Quick start

### 1. Download models

```bash
cd scripts
./download_models.sh
```

This will download:
- `buffalo_l/det_10g.onnx` - Face detector (YOLOv8n)
- `buffalo_l/w600k_r50.onnx` - Face recognizer (ArcFace ResNet50)
- `inswapper_128.onnx` - Face swapper

### 2. Build

```bash
cargo build --release
```

### 3. Run

```bash
./target/release/deep-faceswap swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg
```

## CLI usage

### Basic swap

```bash
deep-faceswap swap -s <source> -t <target> -o <output>
```

**Interactive Face Selection**: If multiple faces are detected, the CLI will save face crops to `./tmp/face_crops/` and prompt you to select which face to use.

## Project structure

@todo

## License

@todo
