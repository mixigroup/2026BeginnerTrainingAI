# ref: https://docs.ultralytics.com/ja/tasks/pose/
#
from ultralytics import YOLO

# Load yolo26m-pose model (downloads automatically if not cached)
model = YOLO("yolo26m-pose.pt")

# Export to ONNX
model.export(
    format="onnx",
    imgsz=640,  # 入力画像サイズ
    opset=17,  # ONNX opset バージョン
    simplify=True,  # onnxslim でモデルグラフを最適化
    dynamic=False,  # 固定バッチサイズ（動的バッチにする場合は True）
    end2end=True,  # エンドツーエンドモデルとしてエクスポート（モデルのみの場合はデフォルト False）
    nms=True,  # NMS をモデルに組み込んでエクスポート（モデルのみの場合はデフォルト False）
    # 高速化手法
    # half=True,  # FP16 モデルとしてエクスポート（フル精度の場合はデフォルト False）
)
