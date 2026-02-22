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

## ノートブックなしで実行する方法

marimo が使えない環境でも、`src/` モジュールを直接使って実行できます。

### ハンズオン1: Iris 分類（スタンドアロンスクリプト）

```bash
uv run python train_iris.py
```

実行すると `learning_curves_iris.png` と `confusion_matrix_iris.png` が生成されます。

### Python スクリプトから直接使う

```python
import sys
sys.path.insert(0, ".")  # src/ を import できるようにする

import torch
import torch.nn as nn
import torch.optim as optim

from src.dataset import load_iris_dataloaders
from src.model import FCNet
from src.evaluate import train_model, evaluate, plot_learning_curves

# データ読み込み
train_loader, val_loader, test_loader, class_names = load_iris_dataloaders(batch_size=16)

# モデル定義
model = FCNet(input_dim=4, hidden_dims=[64, 32], num_classes=3)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 学習
history = train_model(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    criterion=criterion,
    optimizer=optimizer,
    epochs=100,
    device=device,
)

# 評価
test_loss, test_acc = evaluate(model, test_loader, criterion, device)
print(f"Test Accuracy: {test_acc*100:.1f}%")

# 学習曲線を保存
fig = plot_learning_curves(history)
fig.savefig("learning_curves.png")
```

### 過学習デモ（スクリプト例）

```python
from src.dataset import load_iris_dataloaders
from src.model import OversizedFCNet
from src.evaluate import train_model, plot_learning_curves

# 少量データ（30 サンプル）で過学習させる
train_loader, val_loader, _, _ = load_iris_dataloaders(subset_size=30)

model = OversizedFCNet(input_dim=4, num_classes=3)
# ... 学習 ...
```

### 転移学習（スクリプト例）

```python
from src.dataset import load_cifar10_dataloaders
from src.model import ResNet18TransferModel
from src.evaluate import train_model

train_loader, test_loader, class_names = load_cifar10_dataloaders()

model = ResNet18TransferModel(num_classes=10, pretrained=True)

# Phase 1: ヘッドのみ
model.freeze_backbone()
# ... 学習 ...

# Phase 2: Fine-tuning
model.unfreeze_backbone()
# ... 小さい lr で学習 ...
```

---

## 依存パッケージ

- `torch` / `torchvision` - モデル定義・学習・データ読み込み
- `scikit-learn` - Iris データセット・前処理
- `matplotlib` / `seaborn` - 可視化
- `marimo` - インタラクティブノートブック
