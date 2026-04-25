import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # ハンズオン1: Iris 分類（全結合ニューラルネットワーク）

    このノートブックでは PyTorch を使って Iris データセットを分類します。

    ## 学習ゴール

    | ステップ | 内容 |
    |---|---|
    | **1. データ確認** | shape・統計・可視化（EDA） |
    | **2. 前処理** | 正規化・train/valid/test 分割・DataLoader 作成 |
    | **3. モデル構築** | `nn.Module` サブクラスで全結合 NN を定義 |
    | **4. 学習** | loss・optimizer・エポック数を設定して学習ループを回す |
    | **5. 評価** | 学習曲線・混同行列でモデルを評価 |

    ## PyTorch の基本概念

    - `nn.Module` – モデルを定義するベースクラス
    - `DataLoader` – ミニバッチ処理を自動化
    - `loss.backward()` + `optimizer.step()` – 手動の勾配更新ループ
    """)
    return


@app.cell
def _():

    import pandas as pd
    import seaborn as sns
    import torch
    import torch.nn as nn

    from src.dataset import get_iris_raw, load_iris_dataloaders
    from src.model import FCNet
    from src.evaluate import (
        train_model,
        evaluate,
        plot_learning_curves,
        plot_confusion_matrix,
    )

    return (
        FCNet,
        evaluate,
        get_iris_raw,
        load_iris_dataloaders,
        nn,
        pd,
        plot_confusion_matrix,
        plot_learning_curves,
        sns,
        torch,
        train_model,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 1: データ確認（EDA）

    Iris データセットの基本情報を確認します。
    """)
    return


@app.cell
def _(get_iris_raw, pd):
    X_raw, y_raw, feature_names, class_names = get_iris_raw()

    # Create DataFrame for EDA
    df = pd.DataFrame(X_raw, columns=feature_names)
    df["target"] = y_raw
    df["species"] = [class_names[t] for t in y_raw]
    df
    return (df,)


@app.cell(hide_code=True)
def _(df, mo):
    mo.md(f"""
    ### データの概要

    - サンプル数: **{len(df)}** 件
    - 特徴量数: **4** 列（sepal length/width, petal length/width）
    - クラス数: **3** クラス（setosa / versicolor / virginica）

    #### 統計情報
    """)
    return


@app.cell
def _(df):
    df.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 可視化（散布図行列）

    特徴量の組み合わせで、クラスの分離しやすさを確認します。
    """)
    return


@app.cell
def _(df, sns):
    pair_fig = sns.pairplot(df, hue="species", palette="tab10")
    pair_fig.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 2: 前処理

    - 標準化（StandardScaler）で各特徴量を平均0・標準偏差1に変換
    - train : valid : test = 60% : 20% : 20% に分割
    - `DataLoader` でミニバッチ処理を設定
    """)
    return


@app.cell
def _(load_iris_dataloaders, mo):
    BATCH_SIZE = 16  # Edit: mini-batch size

    train_loader, val_loader, test_loader, class_names_loader = load_iris_dataloaders(
        batch_size=BATCH_SIZE,
        val_ratio=0.2,
        test_ratio=0.2,
        random_state=42,
    )

    # Count samples
    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    n_test = len(test_loader.dataset)

    mo.md(
        f"""
        ### 分割結果

        | セット | サンプル数 |
        |---|---|
        | Train  | **{n_train}** |
        | Valid  | **{n_val}** |
        | Test   | **{n_test}** |

        - バッチサイズ: **{BATCH_SIZE}**
        - 各バッチの形状: `(batch, 4)` → `(batch,)` （特徴量 → クラスラベル）
        """
    )
    return class_names_loader, test_loader, train_loader, val_loader


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 3: モデル構築

    `src/model.py` の `FCNet` を使います。

    ### アーキテクチャ

    ```
    Input(4) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(3)
    ```

    - **Linear（全結合層）**: すべての入力ノードがすべての出力ノードに接続
    - **ReLU**: 負の値を 0 にする非線形活性化関数
    - **出力**: 3 クラスの raw logits（CrossEntropy 内部で Softmax）

    ### ハイパーパラメータ

    以下の値を変えて学習結果の違いを観察しましょう。
    """)
    return


@app.cell
def _(FCNet):
    # --- Hyperparameters: Edit these to experiment! ---
    HIDDEN_DIMS = [64, 32]  # Hidden layer sizes
    DROPOUT_RATE = 0.0  # Dropout probability (0.0 = no dropout)
    LEARNING_RATE = 0.001  # Optimizer learning rate
    EPOCHS = 100  # Number of training epochs

    model = FCNet(
        input_dim=4,
        hidden_dims=HIDDEN_DIMS,
        num_classes=3,
        dropout_rate=DROPOUT_RATE,
    )
    print(model)
    return EPOCHS, LEARNING_RATE, model


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 4: 学習

    PyTorch の手動学習ループ:

    ```python
    for epoch in range(epochs):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()  # 勾配をリセット
            logits = model(X_batch)  # 順伝播
            loss = criterion(logits, y_batch)  # 損失計算
            loss.backward()  # 逆伝播（勾配計算）
            optimizer.step()  # パラメータ更新
    ```
    """)
    return


@app.cell
def _(
    EPOCHS,
    LEARNING_RATE,
    model,
    nn,
    torch,
    train_loader,
    train_model,
    val_loader,
):
    import torch.optim as optim

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        epochs=EPOCHS,
        device=device,
        verbose=True,
        verbose_interval=20,
    )
    return device, history


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ---

    ## Step 5: 評価

    ### 学習曲線

    - `train_loss` と `val_loss` が近い → 汎化できている
    - `val_loss` が途中から上昇 → 過学習のサイン
    """)
    return


@app.cell
def _(history, plot_learning_curves):
    curve_fig = plot_learning_curves(history, title="Iris FC-Net")
    curve_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### テストデータの評価
    """)
    return


@app.cell
def _(device, evaluate, model, nn, test_loader):
    criterion_eval = nn.CrossEntropyLoss()
    test_loss, test_acc = evaluate(model, test_loader, criterion_eval, device)
    print(
        f"Test Loss: {test_loss:.4f}  |  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.1f}%)"
    )
    return (test_acc,)


@app.cell
def _(class_names_loader, device, model, plot_confusion_matrix, test_loader):
    cm_fig = plot_confusion_matrix(
        model,
        test_loader,
        class_names_loader,
        device=device,
        title="Iris - Confusion Matrix",
    )
    cm_fig
    return


@app.cell(hide_code=True)
def _(lit_test_acc, mo, test_acc):
    mo.md(f"""
    ---

    ## まとめ

    ### PyTorch vs PyTorch Lightning 精度比較

    | 手法 | Test Accuracy |
    |---|---|
    | PyTorch（手動ループ） | **{test_acc * 100:.1f}%** |
    | PyTorch Lightning | **{lit_test_acc * 100:.1f}%** |

    同じモデル・同じデータで学習しているため精度はほぼ同じです。
    Lightning は「実装コスト」を下げるための抽象化であり、精度は変わりません。

    ### 試してみよう

    1. `HIDDEN_DIMS` を `[128, 64, 32]` に増やして精度の変化を確認
    2. `LEARNING_RATE` を `0.001` や `0.1` に変えて学習曲線の違いを観察
    3. `EPOCHS` を増やして過学習が起きるか確認
    4. `DROPOUT_RATE = 0.3` を設定して過学習を抑制してみよう

    ### 次のノートブック

    `notebook_02_overfitting.py` で過学習を意図的に起こし、対策を実践します。
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
