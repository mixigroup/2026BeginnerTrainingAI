import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        # ハンズオン3: CNN の転移学習（Transfer Learning）

        このノートブックでは ImageNet で学習済みの **ResNet18** を CIFAR-10 の分類に適用します。

        ## 学習ゴール

        | ステップ | 内容 |
        |---|---|
        | **1. データ準備** | CIFAR-10 を DataLoader で読み込む |
        | **2. モデル準備** | pretrained ResNet18 + 新しい分類ヘッド |
        | **3. 凍結学習** | backbone を freeze → ヘッドのみ学習 |
        | **4. Fine-tuning** | backbone を unfreeze → 全体を小さい lr で学習 |
        | **5. 比較** | 凍結あり/なしの学習曲線・精度を比較 |

        ## なぜ転移学習が有効なのか？

        ```
        ImageNet (120万枚) で学習した特徴抽出器
                ↓
        CIFAR-10 (5万枚) に適用
        ```

        - 大量データで学習したエッジ・テクスチャ・形状の特徴がそのまま使える
        - 少ないデータ・少ないエポックでも高い精度を達成できる
        """
    )
    return


@app.cell(hide_code=True)
def _():
    import sys

    sys.path.insert(0, ".")

    import numpy as np
    import matplotlib.pyplot as plt
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from src.dataset import load_cifar10_dataloaders
    from src.model import ResNet18TransferModel
    from src.evaluate import (
        train_model,
        plot_learning_curves,
        compare_learning_curves,
        plot_confusion_matrix,
    )

    return (
        sys,
        np,
        plt,
        torch,
        nn,
        optim,
        load_cifar10_dataloaders,
        ResNet18TransferModel,
        train_model,
        plot_learning_curves,
        compare_learning_curves,
        plot_confusion_matrix,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Step 1: データセット準備（CIFAR-10）

        CIFAR-10: 32×32 カラー画像、10 クラス（飛行機・自動車・鳥・猫・鹿・犬・カエル・馬・船・トラック）

        - 学習データ: 50,000 枚
        - テストデータ: 10,000 枚

        ### 前処理（torchvision.transforms）

        学習時:
        - `RandomHorizontalFlip` – 左右反転でデータ拡張
        - `RandomCrop(32, padding=4)` – ランダムクロップでデータ拡張
        - `Normalize` – CIFAR-10 の平均/標準偏差で正規化

        テスト時:
        - `Normalize` のみ（拡張なし）
        """
    )
    return


@app.cell
def _(load_cifar10_dataloaders, mo):
    BATCH_SIZE = 64  # Edit: batch size

    print("Downloading CIFAR-10 (first time only)...")
    train_loader, test_loader, class_names = load_cifar10_dataloaders(
        batch_size=BATCH_SIZE,
        data_root="./data",
    )

    mo.md(
        f"""
        ### CIFAR-10 データセット

        | セット | サンプル数 |
        |---|---|
        | Train | **{len(train_loader.dataset):,}** |
        | Test  | **{len(test_loader.dataset):,}** |

        クラス: {", ".join(class_names)}
        """
    )
    return BATCH_SIZE, train_loader, test_loader, class_names


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"### サンプル画像の確認")
    return


@app.cell
def _(class_names, np, plt, train_loader):
    # Show a batch of sample images (denormalize for display)
    mean = np.array([0.4914, 0.4822, 0.4465])
    std = np.array([0.2023, 0.1994, 0.2010])

    images, labels = next(iter(train_loader))
    images_np = images[:16].numpy()
    # Denormalize: (C, H, W) -> (H, W, C)
    images_np = images_np.transpose(0, 2, 3, 1) * std + mean
    images_np = np.clip(images_np, 0, 1)

    fig_samples, axes = plt.subplots(2, 8, figsize=(16, 4))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(images_np[i])
        ax.set_title(class_names[labels[i].item()], fontsize=8)
        ax.axis("off")
    fig_samples.suptitle("CIFAR-10 Sample Images", fontsize=12)
    fig_samples.tight_layout()
    fig_samples
    return mean, std, images, labels, images_np, fig_samples, axes


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Step 2: モデル準備

        ### ResNet18 の構造

        ```
        [Conv層群 (backbone)] → [Global Average Pooling] → [Linear(512, 1000)]
                                                                    ↑ ここを差し替える
        ```

        ### 転移学習の設定

        ```python
        # ImageNet 学習済み重みで初期化
        model = ResNet18TransferModel(num_classes=10, pretrained=True)

        # backbone を凍結（勾配計算しない）
        model.freeze_backbone()

        # 新しいヘッドだけ学習される
        # model.head = nn.Linear(512, 10)  # CIFAR-10 用に差し替え済み
        ```
        """
    )
    return


@app.cell
def _(ResNet18TransferModel, mo):
    # Load pretrained ResNet18, replace head with 10-class output
    model = ResNet18TransferModel(num_classes=10, pretrained=True)

    # Freeze backbone: only head will be trained
    model.freeze_backbone()
    n_trainable_frozen = model.get_num_trainable_params()
    n_total = sum(p.numel() for p in model.parameters())

    mo.md(
        f"""
        ### パラメータ数

        | | パラメータ数 |
        |---|---|
        | 総パラメータ | **{n_total:,}** |
        | 学習可能（凍結時） | **{n_trainable_frozen:,}** （ヘッドのみ） |

        backbone を freeze すると、学習対象は **{n_trainable_frozen:,}** パラメータのみ（線形ヘッド）。
        """
    )
    return model, n_trainable_frozen, n_total


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Step 3: Phase 1 - ヘッドのみ学習（backbone 凍結）

        backbone が凍結されているため:
        - 学習が高速（少ないパラメータを更新するだけ）
        - 事前学習された特徴量をそのまま活用
        - 比較的大きい学習率を使える

        まず数エポック学習して、ヘッドの重みを調整します。
        """
    )
    return


@app.cell
def _(mo):
    HEAD_EPOCHS = 5  # Edit: epochs for head-only training
    HEAD_LR = 0.001  # Edit: learning rate for head training

    mo.md(
        f"""
        ### Phase 1 ハイパーパラメータ

        - エポック数: **{HEAD_EPOCHS}**
        - 学習率: **{HEAD_LR}**
        - 最適化: Adam（backbone 凍結 → ヘッドのみ更新）
        """
    )
    return HEAD_EPOCHS, HEAD_LR


@app.cell
def _(
    HEAD_EPOCHS,
    HEAD_LR,
    model,
    nn,
    optim,
    torch,
    train_loader,
    test_loader,
    train_model,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    criterion = nn.CrossEntropyLoss()

    # Phase 1: head-only training (backbone frozen)
    optimizer_head = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=HEAD_LR,
    )

    print("=== Phase 1: Head-only training (backbone frozen) ===")
    # NOTE: Using test_loader as val_loader for simplicity (CIFAR-10 has no official val split).
    # In production, create a separate validation split from training data to avoid data leakage.
    history_head = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer_head,
        epochs=HEAD_EPOCHS,
        device=device,
        verbose=True,
        verbose_interval=1,
    )
    return device, criterion, optimizer_head, history_head


@app.cell
def _(history_head, plot_learning_curves):
    fig_head = plot_learning_curves(history_head, title="Phase 1: Head-only Training")
    fig_head
    return (fig_head,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Step 4: Phase 2 - Fine-tuning（backbone を解凍）

        backbone を解凍して、全パラメータを学習します。

        ### 重要: 学習率を小さくする

        ```python
        model.unfreeze_backbone()
        optimizer = optim.Adam(model.parameters(), lr=1e-4)  # 小さい lr!
        ```

        **なぜ学習率を小さくするのか？**
        - backbone の学習済み重みは貴重
        - 大きい lr で更新すると、事前学習の恩恵が失われる（catastrophic forgetting）
        - 小さい lr で微調整することで、特徴量を保ちながら CIFAR-10 に適応させる
        """
    )
    return


@app.cell
def _(mo):
    FINETUNE_EPOCHS = 5  # Edit: epochs for fine-tuning
    FINETUNE_LR = 1e-4  # Edit: small learning rate for fine-tuning

    mo.md(
        f"""
        ### Phase 2 ハイパーパラメータ

        - エポック数: **{FINETUNE_EPOCHS}**
        - 学習率: **{FINETUNE_LR}** （Phase 1 より小さい）
        - 最適化: Adam（全パラメータを更新）
        """
    )
    return FINETUNE_EPOCHS, FINETUNE_LR


@app.cell
def _(
    FINETUNE_EPOCHS,
    FINETUNE_LR,
    criterion,
    device,
    model,
    optim,
    torch,
    train_loader,
    test_loader,
    train_model,
):
    # Phase 2: Unfreeze backbone for fine-tuning
    model.unfreeze_backbone()
    n_trainable_unfrozen = model.get_num_trainable_params()
    print(f"Trainable params after unfreeze: {n_trainable_unfrozen:,}")

    optimizer_ft = optim.Adam(model.parameters(), lr=FINETUNE_LR)

    print("=== Phase 2: Fine-tuning (backbone unfrozen) ===")
    # NOTE: same val/test note as Phase 1
    history_ft = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer_ft,
        epochs=FINETUNE_EPOCHS,
        device=device,
        verbose=True,
        verbose_interval=1,
    )
    return n_trainable_unfrozen, optimizer_ft, history_ft


@app.cell
def _(history_ft, plot_learning_curves):
    fig_ft = plot_learning_curves(history_ft, title="Phase 2: Fine-tuning")
    fig_ft
    return (fig_ft,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## Step 5: 結果の比較

        Phase 1（ヘッドのみ）と Phase 2（Fine-tuning）の学習曲線を比較します。
        """
    )
    return


@app.cell
def _(compare_learning_curves, history_ft, history_head):
    histories_compare = {
        "Head-only (frozen backbone)": history_head,
        "Fine-tuning (unfrozen backbone)": history_ft,
    }

    fig_cmp_acc = compare_learning_curves(histories_compare, metric="val_acc")
    fig_cmp_acc
    return (histories_compare, fig_cmp_acc)


@app.cell
def _(compare_learning_curves, histories_compare):
    fig_cmp_loss = compare_learning_curves(histories_compare, metric="val_loss")
    fig_cmp_loss
    return (fig_cmp_loss,)


@app.cell
def _(class_names, device, model, plot_confusion_matrix, test_loader):
    cm_fig = plot_confusion_matrix(
        model,
        test_loader,
        class_names,
        device=device,
        title="CIFAR-10 Confusion Matrix (after fine-tuning)",
    )
    cm_fig
    return (cm_fig,)


@app.cell(hide_code=True)
def _():
    import lightning.pytorch as pl
    from lightning.pytorch.loggers import TensorBoardLogger
    from src.lightning_model import ClassifierModule, MetricsCallback

    return pl, TensorBoardLogger, ClassifierModule, MetricsCallback


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## PyTorch Lightning 版: 転移学習

        同じ 2 フェーズ学習を Lightning で実装します。

        | 比較項目 | PyTorch（上記） | PyTorch Lightning（下記） |
        |---|---|---|
        | 学習ループ | 手動 `for epoch` | `trainer.fit()` 1行 |
        | フェーズ切り替え | optimizer を手動再作成 | `learning_rate` を更新して新 Trainer |
        | ログ記録 | `print()` | TensorBoard |

        学習曲線は TensorBoard で確認できます（`uv run tensorboard --logdir runs`）。
        """
    )
    return


@app.cell
def _(ResNet18TransferModel, ClassifierModule, HEAD_LR):
    # PyTorch 版とは別の新しいインスタンスを使用
    lit_resnet = ResNet18TransferModel(num_classes=10, pretrained=True)
    lit_resnet.freeze_backbone()
    lit_transfer_model = ClassifierModule(lit_resnet, learning_rate=HEAD_LR)
    print(f"Phase 1 学習可能パラメータ: {lit_resnet.get_num_trainable_params():,}（ヘッドのみ）")
    return lit_resnet, lit_transfer_model


@app.cell
def _(pl, TensorBoardLogger, lit_transfer_model, train_loader, test_loader, HEAD_EPOCHS):
    lit_trainer_head = pl.Trainer(
        max_epochs=HEAD_EPOCHS,
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="runs", name="cifar10_resnet_phase1"),
        enable_progress_bar=True,
        log_every_n_steps=5,
    )
    print("=== Lightning Phase 1: Head-only ===")
    # NOTE: test_loader を val_loader として使用（CIFAR-10 に公式 val split がないため）
    lit_trainer_head.fit(lit_transfer_model, train_loader, test_loader)
    print("Phase 1 完了")
    return (lit_trainer_head,)


@app.cell
def _(pl, TensorBoardLogger, lit_resnet, lit_transfer_model, train_loader, test_loader, FINETUNE_EPOCHS, FINETUNE_LR):
    # backbone を解凍して fine-tuning
    lit_resnet.unfreeze_backbone()
    lit_transfer_model.learning_rate = FINETUNE_LR  # configure_optimizers は fit() 開始時に再呼び出し
    print(f"Phase 2 学習可能パラメータ: {lit_resnet.get_num_trainable_params():,}（全パラメータ）")

    lit_trainer_ft = pl.Trainer(
        max_epochs=FINETUNE_EPOCHS,
        accelerator="auto",
        logger=TensorBoardLogger(save_dir="runs", name="cifar10_resnet_phase2"),
        enable_progress_bar=True,
        log_every_n_steps=5,
    )
    print("=== Lightning Phase 2: Fine-tuning ===")
    lit_trainer_ft.fit(lit_transfer_model, train_loader, test_loader)

    lit_ft_results = lit_trainer_ft.test(lit_transfer_model, test_loader, verbose=True)
    lit_ft_test_acc = lit_ft_results[0]["test_acc"]
    print(f"\nLightning Fine-tuning Test Accuracy: {lit_ft_test_acc * 100:.1f}%")
    return lit_trainer_ft, lit_ft_results, lit_ft_test_acc


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ---

        ## まとめ

        ### 転移学習の 2 フェーズ学習の効果

        | フェーズ | backbone | 学習率 | 目的 |
        |---|---|---|---|
        | **Phase 1（ヘッドのみ）** | 凍結 | 比較的大（1e-3）| ヘッドの初期適応 |
        | **Phase 2（Fine-tuning）** | 解凍 | 小さい（1e-4） | 全体の微調整 |

        ### PyTorch Lightning での転移学習

        - `freeze_backbone()` / `unfreeze_backbone()` は `nn.Module` の操作なので PyTorch と同じ
        - フェーズ切り替えは `learning_rate` 更新 + 新しい `Trainer` で実現
        - `configure_optimizers()` は `fit()` 開始時に毎回呼ばれるため、`learning_rate` を変えれば反映される

        ### 試してみよう

        1. `HEAD_EPOCHS` を 10 に増やして Phase 1 の精度上限を確認
        2. `FINETUNE_LR` を `1e-3` に増やすと何が起きるか確認（catastrophic forgetting）
        3. `pretrained=False` に変えてスクラッチ学習と比較してみよう
        4. 混同行列でどのクラスが間違えやすいか確認

        ### ResNet18 の代替モデル

        torchvision には他にも多くのモデルが用意されています：

        ```python
        from torchvision import models
        # ResNet50, EfficientNet, ViT など
        models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        ```
        """
    )
    return


if __name__ == "__main__":
    app.run()
