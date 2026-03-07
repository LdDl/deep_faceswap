import torch
from collections import OrderedDict
import argparse
from GFPGANReconsitution import GFPGAN

parser = argparse.ArgumentParser("ONNX converter")
parser.add_argument('--src_model_path', type=str, default=None, help='src model path')
parser.add_argument('--dst_model_path', type=str, default=None, help='dst model path')
args = parser.parse_args()

# device = torch.device('cuda')
model_path = args.src_model_path
onnx_model_path = args.dst_model_path

model = GFPGAN()  # .cuda()

x = torch.rand(1, 3, 512, 512)  # .cuda()

checkpoint = torch.load(model_path, weights_only=False)
print(f"Checkpoint keys: {checkpoint.keys()}")

if 'params_ema' in checkpoint:
    state_dict = checkpoint['params_ema']
elif 'params' in checkpoint:
    state_dict = checkpoint['params']
else:
    state_dict = checkpoint

print(f"State dict has {len(state_dict)} keys")
print(f"First 5 keys: {list(state_dict.keys())[:5]}")

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    # stylegan_decoderdotto_rgbsdot1dotmodulated_convdotbias
    if "stylegan_decoder" in k:
        k_modified = k.replace('.', 'dot')
        k_modified = k_modified.replace('dotweight', '.weight')
        k_modified = k_modified.replace('dotbias', '.bias')
        new_state_dict[k_modified] = v
    else:
        new_state_dict[k] = v

print(f"New state dict has {len(new_state_dict)} keys")

result = model.load_state_dict(new_state_dict, strict=False)
print(f"Missing keys: {len(result.missing_keys)}")
print(f"Unexpected keys: {len(result.unexpected_keys)}")
if result.missing_keys:
    print(f"First 5 missing: {result.missing_keys[:5]}")

model.eval()

with torch.no_grad():
    torch.onnx.export(
        model,
        x,
        onnx_model_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamo=False
    )

print(f"ONNX model saved to {onnx_model_path}")
