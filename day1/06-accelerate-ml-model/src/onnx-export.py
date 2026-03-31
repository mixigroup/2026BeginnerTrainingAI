# ref: https://huggingface.co/facebook/sam-vit-base
#
# SAM Image Encoder を ONNX にエクスポートするスクリプト
#
# 使い方:
#   uv run python src/onnx-export.py
#
import torch
from transformers import SamModel

model = SamModel.from_pretrained("facebook/sam-vit-base")
model.eval()

encoder = model.vision_encoder
dummy_input = torch.randn(1, 3, 1024, 1024)

with torch.no_grad():
    torch.onnx.export(
        encoder,
        dummy_input,
        "sam-vit-b-encoder.onnx",
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        opset_version=18,
        do_constant_folding=True,
    )

print("ONNX エクスポート完了: sam-vit-b-encoder.onnx")
