from onnxruntime.quantization import quantize_dynamic, QuantType

model_fp32 = "yolo26m-pose.onnx"
model_quant = "yolo26m-pose-quantized.onnx"
quantized_model = quantize_dynamic(
    model_fp32,
    model_quant,
    weight_type=QuantType.QInt8,
)
