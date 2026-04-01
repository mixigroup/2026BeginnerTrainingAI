"""L1 Unstructured Pruning を SAM Image Encoder に適用するスクリプト。

torch.nn.utils.prune を使い、Linear 層の重みを 30% ゼロ化する。
ViT では Linear 層（Attention QKV 射影・MLP）がパラメータの大部分を占めるため、
CNN の Conv2d とは異なり Linear 層をプルーニングの対象とする。
"""

import torch
import torch.nn.utils.prune as prune
from transformers import SamModel

# --- 設定 ---
MODEL_ID = "facebook/sam-vit-base"
OUTPUT_MODEL = "sam-vit-b-encoder-pruned.pt"
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
    """Linear 層に L1 Unstructured Pruning を適用する。"""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            prune.l1_unstructured(module, name="weight", amount=amount)


def remove_pruning_reparametrization(model: torch.nn.Module) -> None:
    """プルーニングの再パラメータ化を除去し、マスクを重みに永続化する。"""
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            try:
                prune.remove(module, "weight")
            except ValueError:
                pass  # プルーニングが適用されていない層はスキップ


def main() -> None:
    # モデルのロード
    sam = SamModel.from_pretrained(MODEL_ID)
    sam.eval()
    encoder = sam.vision_encoder

    # プルーニング前のスパース率
    sparsity_before = calc_sparsity(encoder)
    print(f"プルーニング前のスパース率: {sparsity_before:.2%}")

    # L1 Unstructured Pruning を適用
    apply_pruning(encoder, amount=PRUNE_AMOUNT)

    # 再パラメータ化を除去（weight_orig + weight_mask → weight に統合）
    remove_pruning_reparametrization(encoder)

    # プルーニング後のスパース率
    sparsity_after = calc_sparsity(encoder)
    print(f"プルーニング後のスパース率: {sparsity_after:.2%}")

    # プルーニング済みモデルを保存
    torch.save(encoder.state_dict(), OUTPUT_MODEL)
    print(f"プルーニング済みモデルを保存しました: {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
