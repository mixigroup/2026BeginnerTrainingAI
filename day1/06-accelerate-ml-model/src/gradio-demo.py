import os
import time

import gradio as gr
import numpy as np
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

# Pre-load all available models at startup
MODEL_FILES = {
    "PyTorch (.pt)": "yolo26m-pose.pt",
    # "ONNX (.onnx)": "yolo26m-pose.onnx",
    # "ONNX qint8 (.onnx)": "yolo26m-pose-quantized.onnx",
    # "Pruned ONNX (.onnx)": "yolo26m-pose-pruned.onnx",
    # export 済みモデルを追加する場合はここにパスを追加
    # "LiteRT (.tflite)": "yolo26m-pose.tflite",
}

loaded_models: dict[str, YOLO] = {}
for _name, _path in MODEL_FILES.items():
    # .pt is downloaded automatically; .onnx must be exported beforehand
    if _path.endswith(".pt") or os.path.exists(_path):
        loaded_models[_name] = YOLO(_path)

available = list(loaded_models.keys())


def predict(image: np.ndarray | None, model_name: str) -> tuple[np.ndarray | None, str]:
    """Run pose estimation with the selected model and return annotated image + perf stats."""
    if image is None:
        return None, ""

    model = loaded_models.get(model_name)
    if model is None:
        return None, f"Model '{model_name}' is not loaded."

    t_start = time.perf_counter()
    results = model(image, verbose=False)
    t_total_ms = (time.perf_counter() - t_start) * 1000

    annotated = results[0].plot()[:, :, ::-1]  # BGR -> RGB

    # results[0].speed: {"preprocess": ms, "inference": ms, "postprocess": ms}
    speed = results[0].speed
    fps = 1000.0 / t_total_ms if t_total_ms > 0 else 0.0

    perf = (
        f"FPS        : {fps:.1f}\n"
        f"Total      : {t_total_ms:.1f} ms\n"
        f"Preprocess : {speed['preprocess']:.1f} ms\n"
        f"Inference  : {speed['inference']:.1f} ms\n"
        f"Postprocess: {speed['postprocess']:.1f} ms"
    )

    return annotated, perf


with gr.Blocks() as demo:
    gr.Markdown("## YOLO26m Pose Estimation — Real-time Webcam")

    model_radio = gr.Radio(
        choices=available,
        value=available[0],
        label="Model",
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(
                sources="webcam",
                streaming=True,
                type="numpy",
                label="Webcam Input",
            )
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Pose Estimation")
            perf_text = gr.Textbox(label="Performance", lines=5)

    input_image.stream(
        predict,
        inputs=[input_image, model_radio],
        outputs=[output_image, perf_text],
        stream_every=0.033,  # request a new frame every ~33 ms (target 30 fps)
        concurrency_limit=1,  # process one frame at a time to avoid queue buildup
    )

if __name__ == "__main__":
    demo.launch()
