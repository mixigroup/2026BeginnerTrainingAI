import matplotlib.pyplot as plt
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger

from dataset import load_iris_dataloaders
from model import FCNet
from lightning_model import ClassifierModule, MetricsCallback
from evaluate import plot_confusion_matrix


def main() -> None:

    # ハイパーパラメータ
    BATCH_SIZE = 16
    HIDDEN_DIMS = [64, 32]
    LEARNING_RATE = 0.01
    EPOCHS = 100

    # 1. データ読み込み
    train_loader, val_loader, test_loader, class_names = load_iris_dataloaders(
        batch_size=BATCH_SIZE, val_ratio=0.2, test_ratio=0.2, random_state=42
    )
    print(
        f"Train: {len(train_loader.dataset)}, "  # pyright: ignore
        f"Val: {len(val_loader.dataset)}, "  # pyright: ignore
        f"Test: {len(test_loader.dataset)}"  # pyright: ignore
    )

    # 2. モデル構築
    #    FCNet（nn.Module）を ClassifierModule（LightningModule）でラップ
    backbone = FCNet(input_dim=4, hidden_dims=HIDDEN_DIMS, num_classes=3)
    lit_model = ClassifierModule(backbone, learning_rate=LEARNING_RATE)

    # 3. コールバック設定
    metrics_cb = MetricsCallback()  # エポックごとのメトリクス収集
    early_stopping_cb = EarlyStopping(
        monitor="val_loss",
        patience=20,
        mode="min",
        verbose=True,
    )

    # 4. TensorBoard ロガー設定
    #    runs/iris_fcnet/ 以下にログが書き出される
    #    確認方法: uv run tensorboard --logdir runs
    tb_logger = TensorBoardLogger(save_dir="runs", name="iris_fcnet")

    # 5. Trainer 設定
    #    accelerator="auto" で GPU/MPS/CPU を自動選択
    trainer = pl.Trainer(
        max_epochs=EPOCHS,
        accelerator="auto",
        callbacks=[metrics_cb, early_stopping_cb],
        logger=tb_logger,
        enable_progress_bar=True,
        enable_model_summary=True,
        log_every_n_steps=1,
    )

    # 6. 学習（fit が train + validation ループを自動で管理）
    print("Training with PyTorch Lightning...")
    trainer.fit(lit_model, train_loader, val_loader)

    # 7. テスト評価
    results = trainer.test(lit_model, test_loader, verbose=True)
    test_loss = results[0]["test_loss"]
    test_acc = results[0]["test_acc"]
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc * 100:.1f}%")

    # TensorBoard ログのパスを表示
    print(f"\nTensorBoard ログ: {tb_logger.log_dir}")
    print("可視化するには: uv run tensorboard --logdir runs")

    # 8. 混同行列の保存
    #    LightningModule は nn.Module のサブクラスなのでそのまま渡せる
    device = lit_model.device
    fig_cm = plot_confusion_matrix(lit_model, test_loader, class_names, device=device)
    fig_cm.savefig("confusion_matrix_iris.png", dpi=100, bbox_inches="tight")
    plt.close(fig_cm)
    print("Saved: confusion_matrix_iris.png")


if __name__ == "__main__":
    main()
