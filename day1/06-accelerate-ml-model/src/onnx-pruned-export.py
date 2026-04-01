# プルーニング済み SAM Image Encoder を ONNX にエクスポートするスクリプト
#
# 使い方:
#   uv run python src/onnx-pruned-export.py
#
# 事前に pruning.py を実行して sam-vit-b-encoder-pruned.pt を生成しておくこと。
#
import torch
from transformers import SamModel

MODEL_ID = "facebook/sam-vit-base"
PRUNED_WEIGHTS = "sam-vit-b-encoder-pruned.pt"
OUTPUT_ONNX = "sam-vit-b-encoder-pruned.onnx"

# モデルをロードし、プルーニング済み重みを適用
sam = SamModel.from_pretrained(MODEL_ID)
sam.eval()
encoder = sam.vision_encoder
encoder.load_state_dict(torch.load(PRUNED_WEIGHTS, weights_only=True))

# ONNX にエクスポート
dummy_input = torch.randn(1, 3, 1024, 1024)
with torch.no_grad():
    torch.onnx.export(
        encoder,
        dummy_input,
        OUTPUT_ONNX,
        input_names=["pixel_values"],
        output_names=["image_embeddings"],
        opset_version=18,
        do_constant_folding=True,
    )

print(f"プルーニング済み ONNX エクスポート完了: {OUTPUT_ONNX}")
