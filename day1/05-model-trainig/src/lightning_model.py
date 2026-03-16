"""
PyTorch Lightning モジュール定義。

FCNet / OversizedFCNet / ResNet18TransferModel などの nn.Module を
LightningModule としてラップし、pl.Trainer で学習できるようにします。

主要クラス:
- ClassifierModule: 分類タスク汎用 LightningModule ラッパー
- MetricsCallback: エポックごとのメトリクスを収集するコールバック
"""

import lightning.pytorch as pl
import torch
import torch.nn as nn

from src.evaluate import TrainingHistory


class ClassifierModule(pl.LightningModule):
    """分類タスク汎用 LightningModule。

    任意の nn.Module をラップして PyTorch Lightning の学習ループに対応させます。
    training_step / validation_step / test_step / configure_optimizers を提供します。

    Args:
        model: 分類モデル（nn.Module）。ロジットを返すこと。
        learning_rate: Adam オプティマイザの学習率。
        weight_decay: Adam オプティマイザの weight decay（L2 正則化）。
    """

    def __init__(
        self,
        model: nn.Module,
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()
        # model 以外のハイパーパラメータをチェックポイントに保存
        self.save_hyperparameters(ignore=["model"])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """モデルの forward pass（ロジットを返す）。"""
        return self.model(x)

    def _shared_step(self, batch: tuple, stage: str) -> torch.Tensor:
        """loss と accuracy を計算してログに記録する共通ステップ。"""
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # prog_bar=True で tqdm プログレスバーに表示
        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self._shared_step(batch, "train")

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "val")

    def test_step(self, batch: tuple, batch_idx: int) -> None:
        self._shared_step(batch, "test")

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Adam オプティマイザを返す。"""
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


class MetricsCallback(pl.Callback):
    """エポックごとの train/val メトリクスを収集するコールバック。

    trainer.fit() 完了後に to_training_history() で TrainingHistory に変換し、
    plot_learning_curves() などの既存の可視化関数に渡せます。

    使い方::

        metrics_cb = MetricsCallback()
        trainer = pl.Trainer(callbacks=[metrics_cb])
        trainer.fit(lit_model, train_loader, val_loader)
        history = metrics_cb.to_training_history()
    """

    def __init__(self) -> None:
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_accs: list[float] = []
        self.val_accs: list[float] = []

    def on_validation_epoch_end(
        self, trainer: pl.Trainer, pl_module: pl.LightningModule
    ) -> None:
        """バリデーション終了後（サニティチェックを除く）にメトリクスを収集。"""
        # Trainer 開始時のサニティチェックはスキップ
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        self.train_losses.append(metrics.get("train_loss", torch.tensor(0.0)).item())
        self.train_accs.append(metrics.get("train_acc", torch.tensor(0.0)).item())
        self.val_losses.append(metrics.get("val_loss", torch.tensor(0.0)).item())
        self.val_accs.append(metrics.get("val_acc", torch.tensor(0.0)).item())

    def to_training_history(self) -> TrainingHistory:
        """収集したメトリクスを TrainingHistory オブジェクトに変換。"""
        history = TrainingHistory()
        history.train_losses = list(self.train_losses)
        history.val_losses = list(self.val_losses)
        history.train_accs = list(self.train_accs)
        history.val_accs = list(self.val_accs)
        return history
