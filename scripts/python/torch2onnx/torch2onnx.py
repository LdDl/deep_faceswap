#!/usr/bin/env python3
import torch
from collections import OrderedDict
import argparse
from GFPGANReconsitution import GFPGAN

parser = argparse.ArgumentParser("ONNX converter")
parser.add_argument('--src_model_path', type=str, default=None, help='src model path')
parser.add_argument('--dst_model_path', type=str, default=None, help='dst model path')
parser.add_argument('--opset', type=int, default=17, help='ONNX opset version (default: 17)')
args = parser.parse_args()

model_path = args.src_model_path
onnx_model_path = args.dst_model_path

print("Initializing GFPGAN model...")
model = GFPGAN()
model_keys = set(model.state_dict().keys())

x = torch.rand(1, 3, 512, 512)

print(f"Loading checkpoint from {model_path}...")
checkpoint = torch.load(model_path, weights_only=False)
if 'params_ema' in checkpoint:
    state_dict = checkpoint['params_ema']
elif 'params' in checkpoint:
    state_dict = checkpoint['params']
else:
    state_dict = checkpoint

new_state_dict = OrderedDict()

for k, v in state_dict.items():
    if "stylegan_decoder" in k:
        k_all_dots = k.replace('.', 'dot')

        if k_all_dots in model_keys:
            new_state_dict[k_all_dots] = v
        else:
            k_restored = k_all_dots.replace('dotweight', '.weight').replace('dotbias', '.bias')
            if k_restored in model_keys:
                new_state_dict[k_restored] = v
    else:
        new_state_dict[k] = v

model.load_state_dict(new_state_dict, strict=False)
model.eval()

print(f"Exporting to ONNX (opset {args.opset})...")
torch.onnx.export(
    model,
    x,
    onnx_model_path,
    export_params=True,
    opset_version=args.opset,
    do_constant_folding=True,
    input_names=['input'],
    output_names=['output'],
    dynamo=False
)

print(f"ONNX model saved to {onnx_model_path}")
