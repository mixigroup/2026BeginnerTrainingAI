# 05 - MLモデル学習 ハンズオン

スライド day1 3章「MLモデル学習」に対応するハンズオンです。
PyTorch を使ってモデルの学習・過学習・転移学習を実践します。

## ハンズオン一覧

| ノートブック | テーマ | データセット |
|---|---|---|
| `notebook_01_iris.py` | Iris 分類（全結合NN） | Iris（scikit-learn） |
| `notebook_02_overfitting.py` | 過学習と対策 | Iris（少量サブセット） |
| `notebook_03_transfer.py` | CNN 転移学習 | CIFAR-10（torchvision） |

## セットアップ

```bash
cd day1/05-model-trainig
uv sync
```

## 実行方法

```bash
# ハンズオン1: Iris 分類
uv run marimo edit notebook_01_iris.py

# ハンズオン2: 過学習と対策
uv run marimo edit notebook_02_overfitting.py

# ハンズオン3: CNN 転移学習
uv run marimo edit notebook_03_transfer.py
```

## ディレクトリ構成

```
05-model-trainig/
├── notebook_01_iris.py        # Iris 分類（全結合NN）
├── notebook_02_overfitting.py # 過学習と対策
├── notebook_03_transfer.py    # CNN 転移学習
├── src/
│   ├── dataset.py             # データロード・前処理
│   ├── model.py               # モデル定義（FCNet, ResNet18）
│   └── evaluate.py            # 学習ループ・可視化
├── pyproject.toml
└── README.md
```

## 各ハンズオンの概要

### notebook_01_iris.py - Iris 分類

- Iris データセット（150 サンプル、4 特徴量、3 クラス）を全結合 NN で分類
- `nn.Module` でモデル定義、手動学習ループの基礎を学ぶ
- 学習曲線・混同行列で結果を評価

**PyTorch キーコンセプト:**
```python
model = FCNet(input_dim=4, hidden_dims=[64, 32], num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(epochs):
    logits = model(X_batch)   # Forward
    loss = criterion(logits, y_batch)
    loss.backward()           # Backward
    optimizer.step()          # Update
    optimizer.zero_grad()
```

---

### notebook_02_overfitting.py - 過学習と対策

- データ削減（30 サンプル）+ 大型モデル（512×3 層）で過学習を再現
- **Early Stopping**: `val_loss` が改善しなければ学習を早期終了
- **Dropout + Weight Decay**: 正則化で汎化性能を改善
- 3 アプローチの学習曲線を比較

---

### notebook_03_transfer.py - CNN 転移学習

- CIFAR-10（32×32 カラー画像、10 クラス）を ResNet18 で分類
- **Phase 1**: backbone を凍結してヘッドのみ学習（高速）
- **Phase 2**: backbone を解凍して小さい lr で fine-tuning
- 凍結あり/なしの精度・学習曲線を比較

**転移学習の流れ:**
```python
model = ResNet18TransferModel(num_classes=10, pretrained=True)

# Phase 1: head-only
model.freeze_backbone()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

# Phase 2: fine-tuning
model.unfreeze_backbone()
optimizer = optim.Adam(model.parameters(), lr=1e-4)  # small lr!
```

---

## 依存パッケージ

- `torch` / `torchvision` - モデル定義・学習・データ読み込み
- `scikit-learn` - Iris データセット・前処理
- `matplotlib` / `seaborn` - 可視化
- `marimo` - インタラクティブノートブック
