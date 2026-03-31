"""L1 Unstructured Pruning を YOLOv8 Pose モデルに適用するスクリプト。

torch.nn.utils.prune を使い、Conv2d / Linear 層の重みを 30% ゼロ化する。
追加の依存ライブラリは不要（PyTorch 標準機能のみ）。
"""

import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO

# --- 設定 ---
INPUT_MODEL = "yolov8m-pose.pt"
OUTPUT_MODEL = "yolov8m-pose-pruned.pt"
PRUNE_AMOUNT = 0.30  # 30% の重みをゼロにする


def calc_sparsity(model: torch.nn.Module) -> float:
    """モデル全体のスパース率（ゼロの重みの割合）を計算する。"""
    total = 0
    zeros = 0
    for param in model.parameters():
        total += param.numel()
        zeros += (param == 0).sum().item()
    return zeros / total if total > 0 else 0.0


def apply_pruning(model: torch.nn.Module, amount: float) -> None:
    """Conv2d と Linear 層に L1 Unstructured Pruning を適用する。"""
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            prune.l1_unstructured(module, name="weight", amount=amount)


def remove_pruning_reparametrization(model: torch.nn.Module) -> None:
    """プルーニングの再パラメータ化を除去し、マスクを重みに永続化する。"""
    for module in model.modules():
        if isinstance(module, (torch.nn.Conv2d, torch.nn.Linear)):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # プルーニングが適用されていない層はスキップ


def main() -> None:
    # モデルのロード
    yolo = YOLO(INPUT_MODEL)
    model = yolo.model

    # プルーニング前のスパース率
    sparsity_before = calc_sparsity(model)
    print(f"プルーニング前のスパース率: {sparsity_before:.2%}")

    # L1 Unstructured Pruning を適用
    apply_pruning(model, amount=PRUNE_AMOUNT)

    # 再パラメータ化を除去（weight_orig + weight_mask → weight に統合）
    remove_pruning_reparametrization(model)

    # プルーニング後のスパース率（remove 後に計算しないと weight_orig が返るため 0% になる）
    sparsity_after = calc_sparsity(model)
    print(f"プルーニング後のスパース率: {sparsity_after:.2%}")

    # プルーニング済みモデルを保存
    yolo.save(OUTPUT_MODEL)
    print(f"プルーニング済みモデルを保存しました: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
