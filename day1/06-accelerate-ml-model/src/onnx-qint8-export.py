# SAM Image Encoder の ONNX モデルを INT8 量子化するスクリプト
#
# 使い方:
#   uv run python src/onnx-qint8-export.py
#
from onnxruntime.quantization import QuantType, quantize_dynamic

model_fp32 = "sam-vit-b-encoder.onnx"
model_quant = "sam-vit-b-encoder-quantized.onnx"

quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QInt8,
)

print(f"量子化完了: {model_quant}")
