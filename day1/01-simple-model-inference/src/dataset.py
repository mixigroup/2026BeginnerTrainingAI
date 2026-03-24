"""Iris 推論ハンズオン用データセットユーティリティ

事前学習で保存したスケーラーパラメータを使って、
学習時と同一の前処理を再現する。
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def load_scaler_params() -> dict:
    """保存済みの StandardScaler パラメータ（mean, scale）を読み込む。"""
    with open(MODELS_DIR / "scaler.json") as f:
        return json.load(f)


def load_iris_data() -> tuple[pd.DataFrame, object]:
    """Iris データセットを DataFrame として返す（EDA 用）。"""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data["target"] = iris.target
    data["species"] = [iris.target_names[t] for t in iris.target]
    return data, iris


def normalize_features(X: np.ndarray) -> np.ndarray:
    """保存済みスケーラーで特徴量を標準化する。

    学習時と同じ mean / scale を使うことで、推論時の前処理を再現する。
    """
    params = load_scaler_params()
    mean = np.array(params["mean"], dtype=np.float32)
    scale = np.array(params["scale"], dtype=np.float32)
    return ((X - mean) / scale).astype(np.float32)


def prepare_test_data(
    test_ratio: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """学習時と同じ分割でテストデータを取得し、正規化して返す。

    Returns:
        X_test_scaled: 正規化済みテスト特徴量
        y_test: テストラベル（整数）
        target_names: クラス名リスト
    """
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target

    # 学習時と同じ分割を再現（最初に test を分離する）
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )

    X_test_scaled = normalize_features(X_test)
    return X_test_scaled, y_test, list(iris.target_names)
