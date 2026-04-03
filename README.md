# Deep FaceSwap

Rust implementation of face swapping. Basically a port of [Deep-Live-Cam](https://github.com/hacksider/Deep-Live-Cam)

## Table of contents
- [Work in progress](#work-in-progress)
- [Quick start](#quick-start)
- [CLI usage](#cli-usage)
- [Showcase](#showcase)
- [Web UI and REST API](#web-ui-and-rest-api)
- [CUDA support](#cuda-support)
- [Project structure](#project-structure)
- [License](#license)

## Work in progress

Current:
- Face swap for images (single source + single target)
- Face swap for video (single source + target video, frame-by-frame)
- Face detection using YOLOv8n from buffalo_l package
- Face recognition using ArcFace w600k_r50
- Face swapping using inswapper_128
- Face enhancement using GFPGAN (optional)
- Mouth mask to preserve target's mouth expression (optional)
- Multi-face support with interactive mapping (images and video)
- CUDA acceleration support via ONNX Runtime
- ROI-based paste_back for high-resolution images (~50-100x faster on 4K, ~3.7x end-to-end on 720p video)
- REST API server (actix-web) with interactive API docs
- Web UI (SvelteKit + Tailwind CSS) for image and video face swapping

Planned:
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
- `buffalo_l/2d106det.onnx` - 106-point facial landmark detector (for mouth mask feature). Note that even if you don't plan to use mouth mask feature, it will be downloaded still because it's not a big file and I simply don't want to maintain separate script for it currently.
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

### With face enhancement

Add `--enhance` flag to improve the quality of the swapped face:

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg \
  --enhance
```

This will use GFPGAN to enhance facial details after swapping. Make sure you have downloaded and converted the GFPGAN model first.

Obviously inference time will increase significantly when using enhancement.

### With mouth mask

Add `--mouth-mask` flag to preserve the target's mouth expression. This is useful when the target has an open mouth (laughing, talking) but the source has a closed mouth:

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg \
  --mouth-mask
```

The mouth mask uses a separate 106-point landmark detector (`2d106det.onnx`) to identify the mouth region, then blends the original mouth back into the swapped result.

Can be combined with enhancement:

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg \
  --mouth-mask \
  --enhance
```

### Multi-face support

Add `--multi-face` flag to swap multiple faces with interactive mapping:

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg \
  --multi-face
```

#### Multiple source images

You can provide multiple source images as a comma-separated list. All faces from all source images will be aggregated and available for mapping:

```bash
deep-faceswap-cli swap \
  --source img1.jpg,img2.jpg,img3.jpg \
  --target group_photo.jpg \
  --output output.jpg \
  --multi-face
```

Face crops will be named with the source filename prefix (e.g., `img1_face_0.jpg`, `img2_face_0.jpg`) to help identify which image each face came from.

If a source image contains no faces, a warning is logged and processing continues with the remaining images.

#### Interactive face selection

When multiple faces are detected, the CLI provides interactive face selection:

- **1:1 case** (1 source, 1 target): No prompt, swaps automatically
- **1:N case** (1 source, N targets): Prompts to select which target faces to swap (enter indices like `0,1` or `all`)
- **N:1 case** (N sources, 1 target): Prompts to select which source face to use
- **N:N case** (N sources, M targets): Prompts for full mapping in format `S:T,S:T` (e.g., `0:1,1:0`)

Face crops are saved to `./tmp/face_crops/{source,target}/` for visual inspection before swapping.

Without `--multi-face` flag, only the highest-score face from each image is used (backward compatible behavior).

Multi-face mode works with `--enhance` and `--mouth-mask` flags - all swapped faces will be enhanced/masked accordingly.

### Video processing

When the target is a video file, the tool automatically switches to video mode:

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target video.mp4 \
  --output output.mp4
```

The video pipeline:
1. Extracts all frames from the target video
2. Scans frames for faces and builds clusters using K-means
3. Processes each frame: detects face, matches to cluster, swaps
4. Re-encodes the result with the original audio track

All image flags work with video too (`--enhance`, `--mouth-mask`, `--multi-face`).

Without `--multi-face`, only one face is swapped per frame (the one closest to the source embedding). If the video contains multiple people and you want to swap more than one face, use `--multi-face`; then the tool will scan the video, cluster all detected faces, show you representative crops, and ask which source should map to which cluster.

You can specify a custom temporary directory for extracted frames:

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target video.mp4 \
  --output output.mp4 \
  --video-tmp-dir /path/to/tmp
```

### Custom model paths

```bash
deep-faceswap-cli swap \
  --source source.jpg \
  --target target.jpg \
  --output output.jpg \
  --detector models/buffalo_l/det_10g.onnx \
  --recognizer models/buffalo_l/w600k_r50.onnx \
  --swapper models/inswapper_128.onnx \
  --enhance \
  --enhancer models/GFPGANv1.4.onnx \
  --mouth-mask \
  --landmark-model models/buffalo_l/2d106det.onnx
```

### Requirements

- Rust toolchain. My setup is:
  - cargo 1.93.0 (083ac5135 2025-12-15)
  - rustc 1.93.0 (254b59607 2026-01-19)

- Downloaded models
- Source and target images.
- FFmpeg libraries (libavcodec, libavformat, etc.) for video processing
- Node.js and npm for building the Web UI frontend (I've tested it with Node.js v22)
- CUDA and cuDNN for GPU acceleration (optional, but recommended for better performance)

## Web UI and REST API

The project includes a web-based interface built with SvelteKit + Tailwind CSS, served by an actix-web REST API server. It provides the same capabilities as the CLI but with a visual interface for face mapping and progress tracking.

### Showcase

| Image swap (one to one) |
:---:
https://github.com/user-attachments/assets/09568c8e-ab18-4c1a-9163-492692338619

| Image swap (one to many) |
:---:
https://github.com/user-attachments/assets/4fa7a32a-6d4f-409a-b60c-82817262b286

| Image swap (many to many) |
:---:
https://github.com/user-attachments/assets/82d1d32b-e6bb-4456-ae7b-d88389870831

| Video swap (many to many) |
:---:
https://github.com/user-attachments/assets/f9492d17-eee8-47f5-84db-78ccdc57fbe9

> Multi-face mapping, async processing with progress, result preview. The processing stage is sped up ~100x in this recording, actual processing time depends on video length and hardware.

### Build

Build the frontend and API server (with CUDA by default):

```bash
make frontend
make api
```

Or build everything at once (frontend + API server + CLI):

```bash
make
```

For CPU-only build (without CUDA):

```bash
make CUDA=0
# make api CUDA=0
# make cli CUDA=0
```

### Run

```bash
./target/release/deep-faceswap-api \
  --port 36000 \
  --ui-dir ./frontend/build \
  --detector models/buffalo_l/det_10g.onnx \
  --recognizer models/buffalo_l/w600k_r50.onnx \
  --swapper models/inswapper_128.onnx \
  --enhancer models/GFPGANv1.4.onnx \
  --landmark-model models/buffalo_l/2d106det.onnx
```

Then open `http://localhost:36000` in your browser.

The `--enhancer` and `--landmark-model` flags are optional. Without them, the enhance and mouth mask features will not be available in the UI.

There is also a `make run` shortcut that builds everything and starts the server with default model paths.

### API server options

| Flag | Default | Description |
|---|---|---|
| `--host` | `0.0.0.0` | Host to bind to |
| `--port` | `36000` | Port to listen on |
| `--ui-dir` | (none) | Path to SvelteKit build directory |
| `--tmp-dir` | `./tmp/api_sessions` | Base directory for temporary files (frames, crops) |
| `--detector` | `models/buffalo_l/det_10g.onnx` | Face detection model |
| `--recognizer` | `models/buffalo_l/w600k_r50.onnx` | Face recognition model |
| `--swapper` | `models/inswapper_128.onnx` | Face swapper model |
| `--enhancer` | (none) | GFPGAN enhancement model |
| `--landmark-model` | (none) | 106-point landmark model for mouth mask |
| `--allowed-dir` | (all) | Comma-separated directories the file browser can access |
| `-v, --verbose` | `1` | Verbose level: 0 (errors), 1 (main), 2 (details), 3 (all) |

### API documentation

When the server is running, interactive API docs are available at `http://localhost:36000/api/docs`.

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

## Project structure

@todo

## License

@todo
