"""PyTorch モデル定義（Iris 分類用の全結合ネットワーク）

注意: このクラスは 05-model-trainig/src/model.py の FCNet と同一の定義である必要がある。
学習スクリプト（05）で保存した state_dict をこちらでロードするため、
構造が一致しないとロードに失敗する。
"""

import torch
import torch.nn as nn


class FCNet(nn.Module):
    """全結合ニューラルネットワーク

    Architecture: Input → Hidden layers (ReLU) → Output (logits)

    Args:
        input_dim: 入力特徴量の数
        hidden_dims: 各隠れ層のユニット数のリスト
        num_classes: 出力クラス数
        dropout_rate: ドロップアウト率（0.0 = ドロップアウトなし）
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dims: list[int] | None = None,
        num_classes: int = 3,
        dropout_rate: float = 0.0,
    ) -> None:
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [64, 32]

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(nn.ReLU())
            if dropout_rate > 0.0:
                layers.append(nn.Dropout(p=dropout_rate))
            in_dim = h_dim
        layers.append(nn.Linear(in_dim, num_classes))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """順伝播。raw logits を返す。"""
        return self.net(x)
