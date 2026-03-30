import marimo

__generated_with = "0.20.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ハンズオン2: 過学習と対策

    このノートブックでは、**意図的に過学習を起こし**、その対策を実践します。

    ## 学習ゴール

    | ステップ | 内容 |
    |---|---|
    | **1. 過学習を観察** | データ削減 + モデル肥大化でオーバーフィットを再現 |
    | **2. Early Stopping** | val_loss が改善しなければ学習を早期終了 |
    | **3. Dropout + Weight Decay** | 正則化で汎化性能を改善 |
    | **4. 比較** | 3 つのアプローチの学習曲線を並べて比較 |

    ## なぜ過学習が起きるのか？

    - **モデルが複雑すぎる** → 訓練データの雑音まで覚えてしまう
    - **データが少なすぎる** → 汎化のために必要なパターンが学べない
    - **エポックが多すぎる** → 訓練データに特化しすぎる

    ```
    【過学習のサイン】
    train_loss ↓↓  val_loss ↑↑  （train と val が乖離）
    train_acc  ↑↑  val_acc  停滞または低下
    ```
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim

    from src.dataset import load_iris_dataloaders
    from src.model import OversizedFCNet, FCNet
    from src.evaluate import (
        evaluate,
        train_model,
        plot_learning_curves,
        compare_learning_curves,
    )

    return (
        FCNet,
        OversizedFCNet,
        compare_learning_curves,
        evaluate,
        load_iris_dataloaders,
        nn,
        optim,
        plot_learning_curves,
        torch,
        train_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 1: 過学習を意図的に起こす

    ### 戦略
    - **データ削減**: 学習データを 30 サンプルに限定
    - **モデル肥大化**: Iris には不釣り合いな大型ネットワーク（512×3層）
    - **エポック増加**: 300 エポック学習

    → train_loss はほぼ 0 になるが、val_loss は悪化する（過学習）
    """)
    return


@app.cell
def _(load_iris_dataloaders, mo):
    # Small dataset for overfitting demo
    SUBSET_SIZE = 30  # Only 30 training samples

    train_loader_ov, val_loader_ov, test_loader_ov, class_names = load_iris_dataloaders(
        batch_size=16,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
        subset_size=SUBSET_SIZE,
    )

    n_train_ov = len(train_loader_ov.dataset)
    n_val_ov = len(val_loader_ov.dataset)

    mo.md(
        f"""
        ### データセット (過学習実験用)

        | セット | サンプル数 |
        |---|---|
        | Train  | **{n_train_ov}** （削減済み） |
        | Valid  | **{n_val_ov}** |

        30 サンプルはかなり少ない → 過学習しやすい設定です。
        """
    )
    return test_loader_ov, train_loader_ov, val_loader_ov


@app.cell
def _(
    OversizedFCNet,
    nn,
    optim,
    torch,
    train_loader_ov,
    train_model,
    val_loader_ov,
):
    EPOCHS_OV = 300  # Many epochs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ov = OversizedFCNet(input_dim=4, num_classes=3)
    criterion_ov = nn.CrossEntropyLoss()
    optimizer_ov = optim.Adam(model_ov.parameters(), lr=0.001)

    print(f"Trainable parameters: {sum(p.numel() for p in model_ov.parameters()):,}")
    print("Training oversized model (no early stopping)...")

    history_ov = train_model(
        model=model_ov,
        train_loader=train_loader_ov,
        val_loader=val_loader_ov,
        criterion=criterion_ov,
        optimizer=optimizer_ov,
        epochs=EPOCHS_OV,
        device=device,
        verbose=True,
        verbose_interval=50,
    )
    return criterion_ov, device, history_ov, model_ov


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 過学習の確認

    - `train_loss` が急速に低下している
    - `val_loss` はある時点から上昇 or 停滞している
    - この乖離が **過学習（overfitting）**
    """)
    return


@app.cell
def _(history_ov, plot_learning_curves):
    fig_ov = plot_learning_curves(
        history_ov, title="Oversized Model (No Regularization)"
    )
    fig_ov
    return


@app.cell
def _(criterion_ov, device, evaluate, mo, model_ov, test_loader_ov):
    test_loss_ov, test_acc_ov = evaluate(model_ov, test_loader_ov, criterion_ov, device)

    mo.md(
        f"""
        ### 過学習モデルのテスト評価

        | 指標 | 値 |
        |---|---|
        | Test Loss | **{test_loss_ov:.4f}** |
        | Test Accuracy | **{test_acc_ov:.2%}** |
        | 学習エポック数 | **300** |

        過学習しているため、テスト精度が低い可能性があります。
        """
    )
    return test_acc_ov, test_loss_ov


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step 2: 対策1 - Early Stopping

    `val_loss` が **`patience` エポック連続で改善しない**場合、学習を打ち切ります。

    ```python
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            break  # Early stopping!
    ```

    - **メリット**: 過学習が始まった直後に停止 → 最良の汎化性能を保持
    - **注意**: `patience` が小さすぎると学習が早く終わりすぎる
    """)
    return


@app.cell
def _(
    OversizedFCNet,
    device,
    nn,
    optim,
    train_loader_ov,
    train_model,
    val_loader_ov,
):
    PATIENCE = 20  # Edit: early stopping patience

    model_es = OversizedFCNet(input_dim=4, num_classes=3)
    criterion_es = nn.CrossEntropyLoss()
    optimizer_es = optim.Adam(model_es.parameters(), lr=0.001)

    print(f"Early stopping with patience={PATIENCE}")

    history_es = train_model(
        model=model_es,
        train_loader=train_loader_ov,
        val_loader=val_loader_ov,
        criterion=criterion_es,
        optimizer=optimizer_es,
        epochs=300,
        device=device,
        early_stopping_patience=PATIENCE,
        verbose=True,
        verbose_interval=20,
    )
    print(f"Stopped at epoch: {len(history_es.val_losses)}")
    return criterion_es, history_es, model_es


@app.cell
def _(history_es, plot_learning_curves):
    fig_es = plot_learning_curves(history_es, title="Early Stopping")
    fig_es
    return


@app.cell
def _(
    criterion_es,
    device,
    evaluate,
    history_es,
    mo,
    model_es,
    test_acc_ov,
    test_loader_ov,
    test_loss_ov,
):
    stopped_epoch = len(history_es.val_losses)
    test_loss_es, test_acc_es = evaluate(model_es, test_loader_ov, criterion_es, device)

    mo.md(
        f"""
        ### Early Stopping モデルのテスト評価

        | 指標 | 過学習モデル | Early Stopping |
        |---|---|---|
        | 停止エポック | 300 | **{stopped_epoch}** |
        | Test Loss | {test_loss_ov:.4f} | **{test_loss_es:.4f}** |
        | Test Accuracy | {test_acc_ov:.2%} | **{test_acc_es:.2%}** |

        Early Stopping により学習を早期に打ち切ることで、汎化性能が改善されているか確認しましょう。
        """
    )
    return test_acc_es, test_loss_es


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 3: 対策2 - Dropout + Weight Decay

    ### Dropout
    学習時にランダムにニューロンを **無効化**（確率 `p` で出力を 0 に）。

    ```
    [o o o o o] → [o 0 o 0 o]  # 推論時は全ニューロンを使用
    ```

    - 部分的なサブネットワークで学習 → アンサンブル効果
    - `model.train()` 時のみ有効 → `model.eval()` 時は全ニューロンが使われる

    ### Weight Decay（L2 正則化）
    損失に **パラメータのノルム**を加算 → 重みが大きくなりすぎることを防ぐ。

    ```python
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    ```
    """)
    return


@app.cell
def _(FCNet, device, nn, optim, train_loader_ov, train_model, val_loader_ov):
    # Regularized model: smaller FC net + dropout + weight decay
    DROPOUT = 0.4  # Edit: dropout probability
    WD = 1e-3  # Edit: weight decay (L2 regularization strength)

    model_reg = FCNet(
        input_dim=4,
        hidden_dims=[512, 512, 512, 256, 128],  # Same large architecture
        num_classes=3,
        dropout_rate=DROPOUT,
    )
    criterion_reg = nn.CrossEntropyLoss()
    optimizer_reg = optim.Adam(model_reg.parameters(), lr=0.001, weight_decay=WD)

    print(f"Dropout={DROPOUT}, Weight Decay={WD}")

    history_reg = train_model(
        model=model_reg,
        train_loader=train_loader_ov,
        val_loader=val_loader_ov,
        criterion=criterion_reg,
        optimizer=optimizer_reg,
        epochs=300,
        device=device,
        verbose=True,
        verbose_interval=50,
    )
    return criterion_reg, history_reg, model_reg


@app.cell
def _(history_reg, plot_learning_curves):
    fig_reg = plot_learning_curves(history_reg, title="Dropout + Weight Decay")
    fig_reg
    return


@app.cell
def _(criterion_reg, device, evaluate, mo, model_reg, test_loader_ov):
    test_loss_reg, test_acc_reg = evaluate(
        model_reg, test_loader_ov, criterion_reg, device
    )

    mo.md(
        f"""
        ### Dropout + Weight Decay モデルのテスト評価

        | 指標 | 値 |
        |---|---|
        | Test Loss | **{test_loss_reg:.4f}** |
        | Test Accuracy | **{test_acc_reg:.2%}** |

        正則化によってテスト精度が改善されているか確認しましょう。
        """
    )
    return test_acc_reg, test_loss_reg


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 4: 3 つのアプローチを比較

    同じ val_loss スケールで比較することで、各対策の効果を確認します。
    """)
    return


@app.cell
def _(compare_learning_curves, history_es, history_ov, history_reg):
    # Compare validation loss
    comparison_histories = {
        "No regularization": history_ov,
        "Early Stopping": history_es,
        "Dropout + Weight Decay": history_reg,
    }

    fig_compare_loss = compare_learning_curves(comparison_histories, metric="val_loss")
    fig_compare_loss
    return (comparison_histories,)


@app.cell
def _(compare_learning_curves, comparison_histories):
    fig_compare_acc = compare_learning_curves(comparison_histories, metric="val_acc")
    fig_compare_acc
    return


@app.cell
def _(
    history_es,
    history_ov,
    history_reg,
    mo,
    test_acc_es,
    test_acc_ov,
    test_acc_reg,
    test_loss_es,
    test_loss_ov,
    test_loss_reg,
):
    mo.md(f"""
    ### テストセットでの総合比較

    | 手法 | エポック数 | Test Loss | Test Accuracy |
    |---|---|---|---|
    | No Regularization | {len(history_ov.val_losses)} | {test_loss_ov:.4f} | {test_acc_ov:.2%} |
    | Early Stopping | {len(history_es.val_losses)} | {test_loss_es:.4f} | {test_acc_es:.2%} |
    | Dropout + Weight Decay | {len(history_reg.val_losses)} | {test_loss_reg:.4f} | {test_acc_reg:.2%} |

    テストセットは学習にもバリデーションにも使われていない **未知データ** です。
    この結果が各手法の真の汎化性能を表しています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## まとめ

    ### 各手法の特徴

    | 手法 | 概要 | 効果 | 注意点 |
    |---|---|---|---|
    | **Early Stopping** | val_loss が改善しなければ停止 | 最良の汎化点で学習終了 | patience の設定が必要 |
    | **Dropout** | 学習時にニューロンをランダム無効化 | アンサンブル効果・汎化改善 | p が大きすぎると学習が遅い |
    | **Weight Decay** | 重みが大きくなりすぎることを防ぐ | モデルの複雑さを制限 | 強すぎると underfitting |

    ### PyTorch Lightning での EarlyStopping

    手動実装と Lightning コールバックは同じロジックですが、
    Lightning 版は `patience`, `min_delta`, `mode` などのオプションが充実しています。

    ### 試してみよう

    1. `PATIENCE` を 5 や 50 に変えて Early Stopping の挙動を確認
    2. `DROPOUT` を 0.0 〜 0.5 で変えて効果の違いを観察
    3. `WD` を 0.0 や 1e-2 に変えて Weight Decay の影響を確認
    4. `SUBSET_SIZE` を 60 に増やすと過学習は軽減されるか確認

    ### 次のノートブック

    `notebook_03_transfer.py` で事前学習済み ResNet18 を使った転移学習を実践します。
    """)
    return


if __name__ == "__main__":
    app.run()
