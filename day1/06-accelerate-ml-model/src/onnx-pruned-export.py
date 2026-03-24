# ref: https://docs.ultralytics.com/ja/tasks/pose/
#
# プルーニング済みモデルを ONNX にエクスポートするスクリプト
#
# 使い方:
#   uv run python src/onnx-pruned-export.py
#
from ultralytics import YOLO

# プルーニング済みモデルをロード
model = YOLO("yolo26m-pose-pruned.pt")

# ONNX にエクスポート
model.export(
    format="onnx",
    imgsz=640,  # 入力画像サイズ
    opset=17,  # ONNX opset バージョン
    simplify=True,  # onnxslim でモデルグラフを最適化
    dynamic=False,  # 固定バッチサイズ
    end2end=True,  # エンドツーエンドモデルとしてエクスポート
    nms=True,  # NMS をモデルに組み込んでエクスポート
)
