import os
import time

import cv2
import gradio as gr
import numpy as np
from ultralytics import YOLO  # type: ignore[reportPrivateImportUsage]

# Pre-load all available models at startup
MODEL_FILES = {
    "PyTorch (.pt)": "yolov8m-pose.pt",
    "ONNX (.onnx)": "yolov8m-pose.onnx",
    "ONNX qint8 (.onnx)": "yolov8m-pose-quantized.onnx",
    "Pruned ONNX (.onnx)": "yolov8m-pose-pruned.onnx",
    # export 済みモデルを追加する場合はここにパスを追加
    # "LiteRT (.tflite)": "yolov8m-pose.tflite",
}

loaded_models: dict[str, YOLO] = {}
for _name, _path in MODEL_FILES.items():
    # .pt is downloaded automatically; .onnx must be exported beforehand
    if _path.endswith(".pt") or os.path.exists(_path):
        loaded_models[_name] = YOLO(_path)

available = list(loaded_models.keys())
if not available:
    raise RuntimeError(
        "利用可能なモデルが見つかりません。"
        "yolov8m-pose.pt をダウンロードするか、ONNX ファイルをエクスポートしてください。"
    )


def predict_frame(
    frame: np.ndarray | None, model_name: str
) -> tuple[np.ndarray | None, str]:
    """Webカメラから受け取った1フレームを解析して結果を返す。"""
    if frame is None:
        return None, ""

    model = loaded_models.get(model_name)
    if model is None:
        return frame, f"Model '{model_name}' is not loaded."

    t_start = time.perf_counter()
    results = model(frame, verbose=False)
    t_total_ms = (time.perf_counter() - t_start) * 1000

    # results[0].plot() は BGR で返すので RGB に変換
    annotated = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)

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
    gr.Markdown("## YOLOv8m Pose Estimation — Webカメラ ストリーム解析")
    gr.Markdown("Webカメラを起動すると、リアルタイムでポーズ推定が実行されます。")

    model_radio = gr.Radio(
        choices=available,
        value=available[0],
        label="Model",
    )

    with gr.Row():
        with gr.Column():
            webcam_input = gr.Image(
                sources=["webcam"], type="numpy", label="Webカメラ入力"
            )
        with gr.Column():
            output_image = gr.Image(type="numpy", label="Pose Estimation")
            perf_text = gr.Textbox(label="Performance", lines=5)

    webcam_input.stream(
        predict_frame,
        inputs=[webcam_input, model_radio],
        outputs=[output_image, perf_text],
        stream_every=0.1,
        concurrency_limit=1,
    )

if __name__ == "__main__":
    demo.launch()
