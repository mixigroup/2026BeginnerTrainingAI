"""
Dataset utilities for model training hands-on.

Provides data loading and preprocessing for:
- Iris dataset (sklearn) for classification with fully connected NN
- CIFAR-10 dataset (torchvision) for CNN transfer learning
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def load_iris_dataloaders(
    batch_size: int = 32,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42,
    subset_size: int | None = None,
) -> tuple[DataLoader, DataLoader, DataLoader, list[str]]:
    """Load Iris dataset and return DataLoaders for train/val/test splits.

    Args:
        batch_size: Mini-batch size for DataLoader.
        val_ratio: Fraction of data to use for validation.
        test_ratio: Fraction of data to use for testing.
        random_state: Random seed for reproducibility.
        subset_size: If specified, use only this many samples (for overfitting demo).

    Returns:
        Tuple of (train_loader, val_loader, test_loader, class_names).
    """
    iris = load_iris()
    X, y = iris.data.astype(np.float32), iris.target

    if subset_size is not None:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=subset_size, replace=False)
        X, y = X[idx], y[idx]

    # First split off test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state, stratify=y
    )

    # Then split remaining into train and validation
    val_ratio_adj = val_ratio / (1.0 - test_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio_adj, random_state=random_state, stratify=y_temp
    )

    # Normalize using training statistics
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train).astype(np.float32)
    X_val = scaler.transform(X_val).astype(np.float32)
    X_test = scaler.transform(X_test).astype(np.float32)

    def _make_loader(X_arr: np.ndarray, y_arr: np.ndarray, shuffle: bool) -> DataLoader:
        dataset = TensorDataset(
            torch.from_numpy(X_arr),
            torch.from_numpy(y_arr).long(),
        )
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    train_loader = _make_loader(X_train, y_train, shuffle=True)
    val_loader = _make_loader(X_val, y_val, shuffle=False)
    test_loader = _make_loader(X_test, y_test, shuffle=False)

    return train_loader, val_loader, test_loader, list(iris.target_names)


def load_cifar10_dataloaders(
    batch_size: int = 64,
    data_root: str = "./data",
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, list[str]]:
    """Load CIFAR-10 dataset with standard augmentation and return DataLoaders.

    Args:
        batch_size: Mini-batch size for DataLoader.
        data_root: Directory to download/cache the dataset.
        num_workers: Number of subprocesses for data loading.

    Returns:
        Tuple of (train_loader, test_loader, class_names).
    """
    import torchvision
    import torchvision.transforms as transforms

    # CIFAR-10 per-channel mean/std (computed from the training split)
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2023, 0.1994, 0.2010)

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=train_transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=test_transform
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    class_names = [
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck",
    ]
    return train_loader, test_loader, class_names


def get_iris_raw() -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return raw Iris data as numpy arrays for EDA.

    Returns:
        Tuple of (X, y, feature_names, class_names).
    """
    iris = load_iris()
    return (
        iris.data.astype(np.float32),
        iris.target,
        list(iris.feature_names),
        list(iris.target_names),
    )
