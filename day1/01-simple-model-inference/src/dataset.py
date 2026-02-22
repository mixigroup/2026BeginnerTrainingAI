"""Dataset utilities for Iris classification hands-on."""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.preprocessing import normalize


def load_iris_data() -> tuple[pd.DataFrame, object]:
    """Load Iris dataset and return as DataFrame.

    Returns:
        data: DataFrame with feature columns and 'target' column
        iris: Original sklearn Bunch object (contains target_names, etc.)
    """
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    data["target"] = iris.target
    return data, iris


def shuffle_data(data: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Shuffle DataFrame rows with a fixed random seed.

    Args:
        data: Input DataFrame
        seed: Random seed for reproducibility

    Returns:
        Shuffled DataFrame
    """
    rng = np.random.default_rng(seed=seed)
    shuffled_indices = rng.permutation(len(data))
    return data.iloc[shuffled_indices]


def split_features_and_labels(data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split DataFrame into feature matrix X and label vector y.

    Args:
        data: DataFrame with 4 feature columns and 'target' column

    Returns:
        X: Feature matrix of shape (n_samples, 4)
        y: Label vector of shape (n_samples, 1)
    """
    X = data.iloc[:, :4].values
    y = data.iloc[:, 4:].values
    return X, y


def normalize_features(X: np.ndarray) -> np.ndarray:
    """Normalize feature matrix to [0, 1] range column-wise.

    Args:
        X: Feature matrix of shape (n_samples, n_features)

    Returns:
        Normalized feature matrix
    """
    return normalize(X, axis=0)


def split_dataset(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    valid_ratio: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split dataset into train / valid / test sets.

    Args:
        X: Feature matrix
        y: Label array
        train_ratio: Fraction of data to use for training (default 0.8)
        valid_ratio: Fraction of data to use for validation (default 0.1)
            Remaining fraction is used for test.

    Returns:
        X_train, X_valid, X_test, y_train, y_valid, y_test
    """
    total = len(X)
    train_len = int(train_ratio * total)
    valid_len = int(valid_ratio * total)

    X_train = X[:train_len]
    X_valid = X[train_len : train_len + valid_len]
    X_test = X[train_len + valid_len :]

    y_train = y[:train_len]
    y_valid = y[train_len : train_len + valid_len]
    y_test = y[train_len + valid_len :]

    return X_train, X_valid, X_test, y_train, y_valid, y_test
