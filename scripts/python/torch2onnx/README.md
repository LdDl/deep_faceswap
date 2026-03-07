# torch2onnx

Strictly taken from: https://github.com/clibdev/GFPGAN-onnxruntime-demo?tab=readme-ov-file

Usage:
```bash
python3 -m venv venv_torch2onnx
source venv_torch2onnx/bin/activate
pip install -r ../requirements-torch2onnx.txt
python torch2onnx.py --src_model_path $GFPGAN_TORCH_WEIGHTS_INPUT_FILE --dst_model_path $GFPGAN_ONNX_WEIGHTS_OUTPUT_FILE
deactivate
# Clean up virtual environment if you don't need it anymore
rm -rf venv_torch2onnx
```