"""01-simple-model-inference 用モデルの学習スクリプト

PyTorch FCNet と LightGBM を Iris データで学習し、
../01-simple-model-inference/models/ に保存する。

Usage:
    cd day1/05-model-trainig
    uv run python src/train_models_for_01.py
"""

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import FCNet

MODELS_DIR = (
    Path(__file__).resolve().parent.parent
    / ".."
    / "01-simple-model-inference"
    / "models"
)

# ハイパーパラメータ
RANDOM_STATE = 42
TEST_RATIO = 0.2
VAL_RATIO = 0.2
BATCH_SIZE = 16
HIDDEN_DIMS = [64, 32]
LEARNING_RATE = 0.01
EPOCHS = 100


def prepare_data():
    """Iris データの読み込み・分割・正規化を行う。"""
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target

    # test を先に分割
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_RATIO, random_state=RANDOM_STATE, stratify=y
    )

    # 残りを train / val に分割
    val_ratio_adj = VAL_RATIO / (1.0 - TEST_RATIO)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp,
        y_temp,
        test_size=val_ratio_adj,
        random_state=RANDOM_STATE,
        stratify=y_temp,
    )

    # StandardScaler を train データで fit
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train).astype(np.float32)
    X_val_scaled = scaler.transform(X_val).astype(np.float32)
    X_test_scaled = scaler.transform(X_test).astype(np.float32)

    return (
        X_train_scaled,
        X_val_scaled,
        X_test_scaled,
        y_train,
        y_val,
        y_test,
        scaler,
        list(iris.target_names),
        list(iris.feature_names),
    )


def train_nn(X_train, y_train, X_val, y_val):
    """PyTorch FCNet を学習し、best val_loss のモデルを返す。"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = FCNet(input_dim=4, hidden_dims=HIDDEN_DIMS, num_classes=3)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train).long()),
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val).long()),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(1, EPOCHS + 1):
        # --- Train ---
        model.train()
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()

        # --- Validate ---
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                val_loss += criterion(logits, y_batch).item() * len(y_batch)
                correct += (logits.argmax(1) == y_batch).sum().item()
                total += len(y_batch)

        val_loss /= total
        val_acc = correct / total

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{EPOCHS} | "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    model.load_state_dict(best_state)
    model.cpu()
    return model


def train_lgbm(X_train, y_train, X_val, y_val):
    """LightGBM モデルを学習して返す。"""
    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=["train", "valid"],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(10),
        ],
    )
    return model


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== データ準備 ===")
    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        scaler,
        target_names,
        feature_names,
    ) = prepare_data()
    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    # スケーラーパラメータを保存
    scaler_path = MODELS_DIR / "scaler.json"
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "feature_names": feature_names,
        "target_names": target_names,
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f, indent=2, ensure_ascii=False)
    print(f"Saved: {scaler_path}")

    # --- NN 学習 ---
    print("\n=== Neural Network 学習 ===")
    nn_model = train_nn(X_train, y_train, X_val, y_val)

    # テスト評価
    nn_model.eval()
    with torch.no_grad():
        preds = nn_model(torch.from_numpy(X_test)).argmax(1).numpy()
        acc = (preds == y_test).mean()
        print(f"NN Test Accuracy: {acc * 100:.1f}%")

    # NN モデル保存
    nn_path = MODELS_DIR / "iris_nn.pt"
    torch.save(
        {
            "model_config": {
                "input_dim": 4,
                "hidden_dims": HIDDEN_DIMS,
                "num_classes": 3,
            },
            "state_dict": nn_model.state_dict(),
        },
        nn_path,
    )
    print(f"Saved: {nn_path}")

    # --- LightGBM 学習 ---
    print("\n=== LightGBM 学習 ===")
    lgbm_model = train_lgbm(X_train, y_train, X_val, y_val)

    # テスト評価
    lgbm_proba = lgbm_model.predict(X_test, num_iteration=lgbm_model.best_iteration)
    lgbm_preds = np.argmax(lgbm_proba, axis=1)
    lgbm_acc = (lgbm_preds == y_test).mean()
    print(f"LightGBM Test Accuracy: {lgbm_acc * 100:.1f}%")

    # LightGBM モデル保存
    lgbm_path = MODELS_DIR / "iris_lgbm.txt"
    lgbm_model.save_model(str(lgbm_path))
    print(f"Saved: {lgbm_path}")

    print("\n=== 完了 ===")
    print(f"モデル保存先: {MODELS_DIR.resolve()}")


if __name__ == "__main__":
    main()
