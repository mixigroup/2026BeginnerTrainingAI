import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell
def _(mo):
    mo.md(
        r"""
        # Iris分類ハンズオン：テーブルデータのML推論

        このノートブックでは、**アヤメ（Iris）データセット**を使って、テーブルデータの機械学習の**推論**を体験します。

        ## 推論の3つの共通フェーズ

        どんなモデルでも、推論は以下の3フェーズで構成されます。

        | フェーズ | 内容 |
        |---|---|
        | **1. Preprocess** | 入力データを tensor（多次元配列）に変換・正規化 |
        | **2. Forward** | モデルに tensor を入れて、出力 tensor を得る |
        | **3. Postprocess** | 出力 tensor をタスクに適した形に変換（クラスラベルなど） |

        ## 今回扱うモデル

        - **Neural Network（全結合NN）**：PyTorch で実装
        - **LightGBM（勾配ブースティング）**：テーブルデータで広く使われる手法

        どちらも事前に学習済みのモデルをロードして、推論のみ行います。
        """
    )
    return


@app.cell
def _():
    import sys

    sys.path.insert(0, ".")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import torch
    import lightgbm as lgb

    from src.dataset import (
        load_iris_data,
        load_scaler_params,
        normalize_features,
        prepare_test_data,
    )
    from src.model import FCNet
    from src.evaluate import plot_confusion_matrix, format_classification_report

    return (
        sys,
        np,
        pd,
        plt,
        sns,
        torch,
        lgb,
        load_iris_data,
        load_scaler_params,
        normalize_features,
        prepare_test_data,
        FCNet,
        plot_confusion_matrix,
        format_classification_report,
    )


@app.cell
def _(mo):
    mo.md("## データセットの確認")
    return


@app.cell
def _(load_iris_data):
    data, iris = load_iris_data()
    data
    return data, iris


@app.cell
def _(data, mo):
    mo.md(
        f"""
        ### データの概要

        - サンプル数: **{len(data)}** 件
        - 特徴量数: **4** 列（sepal length, sepal width, petal length, petal width）
        - クラス数: **3** クラス（setosa=0, versicolor=1, virginica=2）

        #### 統計情報
        """
    )
    return


@app.cell
def _(data):
    data.describe()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ### データの可視化（EDA）

        各特徴量の組み合わせを散布図で可視化します。
        クラスごとに色分けされており、分類に有効な特徴量を確認できます。
        """
    )
    return


@app.cell
def _(data, sns):
    pair_fig = sns.pairplot(data, hue="species", palette="tab10")
    pair_fig.figure
    return (pair_fig,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 1: Preprocess（前処理）

        テーブルデータをモデルが受け取れる形式に変換します。

        ### 前処理の手順

        1. **テストデータの取得** - 学習時と同じ分割を再現
        2. **標準化（StandardScaler）** - 学習時に計算した平均・標準偏差を使って正規化
        3. **tensor 変換** - NumPy 配列を PyTorch tensor に変換（NN用）

        ### なぜ学習時と同じスケーラーを使うのか？

        学習時に「平均=5.0, 標準偏差=1.0」で正規化して学習したモデルに対して、
        推論時に異なるスケーラーで正規化すると、モデルが見たことのない分布のデータが入力され、
        正しい予測ができなくなります。
        """
    )
    return


@app.cell
def _(load_scaler_params, mo, np, prepare_test_data):
    # テストデータの取得（学習時と同じ分割）
    X_test, y_test, TARGET_NAMES = prepare_test_data()

    # スケーラーパラメータの確認
    scaler_params = load_scaler_params()

    mo.md(
        f"""
        ### 前処理結果

        - テストデータ: **{len(X_test)}** 件
        - クラス名: {TARGET_NAMES}

        #### 保存済みスケーラーパラメータ

        | 特徴量 | 平均（mean） | 標準偏差（scale） |
        |---|---|---|
        | {scaler_params["feature_names"][0]} | {scaler_params["mean"][0]:.4f} | {scaler_params["scale"][0]:.4f} |
        | {scaler_params["feature_names"][1]} | {scaler_params["mean"][1]:.4f} | {scaler_params["scale"][1]:.4f} |
        | {scaler_params["feature_names"][2]} | {scaler_params["mean"][2]:.4f} | {scaler_params["scale"][2]:.4f} |
        | {scaler_params["feature_names"][3]} | {scaler_params["mean"][3]:.4f} | {scaler_params["scale"][3]:.4f} |

        正規化後の先頭3件:
        ```
        {np.array2string(X_test[:3], precision=4)}
        ```
        """
    )
    return X_test, y_test, TARGET_NAMES, scaler_params


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 2: Forward（Neural Network の推論）

        ### モデルのロード

        事前に学習済みの PyTorch モデル（`models/iris_nn.pt`）をロードします。

        ```
        Input(4) → Linear(64) → ReLU → Linear(32) → ReLU → Linear(3)
        ```

        - **Linear（全結合層）**: すべての入力ノードがすべての出力ノードに接続
        - **ReLU**: 負の値を 0 にする非線形活性化関数
        - **出力**: 3 クラスの raw logits（Softmax 前の値）
        """
    )
    return


@app.cell
def _(FCNet, torch):
    # 保存済みモデルをロード
    checkpoint = torch.load("models/iris_nn.pt", weights_only=True)
    config = checkpoint["model_config"]

    nn_model = FCNet(
        input_dim=config["input_dim"],
        hidden_dims=config["hidden_dims"],
        num_classes=config["num_classes"],
    )
    nn_model.load_state_dict(checkpoint["state_dict"])
    nn_model.eval()  # 推論モードに切り替え

    print(f"モデル構造:\n{nn_model}")
    print(f"\nパラメータ数: {sum(p.numel() for p in nn_model.parameters()):,}")
    return checkpoint, config, nn_model


@app.cell
def _(mo):
    mo.md(
        r"""
        ### 推論の実行

        テストデータを tensor に変換してモデルに入力し、出力（logits）を得ます。

        ```python
        # NumPy → PyTorch tensor に変換
        X_tensor = torch.from_numpy(X_test)

        # 推論（勾配計算は不要なので no_grad で囲む）
        with torch.no_grad():
            logits = model(X_tensor)
        ```
        """
    )
    return


@app.cell
def _(X_test, mo, nn_model, np, torch):
    # NumPy 配列を PyTorch tensor に変換
    X_tensor = torch.from_numpy(X_test)

    # 推論実行
    with torch.no_grad():
        nn_logits = nn_model(X_tensor)

    # logits の中身を確認（先頭3件）
    mo.md(
        f"""
        ### 推論結果（logits）

        logits は各クラスに対するモデルの「確信度」を表す値です。
        まだ確率には変換されていない raw な値です。

        先頭3件の logits:
        ```
        {np.array2string(nn_logits.numpy()[:3], precision=4)}
        ```

        各行で最も大きい値のインデックスが予測クラスになります。
        """
    )
    return X_tensor, nn_logits


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 3: Postprocess（後処理・評価）

        ### logits → 予測クラス

        `argmax` で各サンプルの最大 logit のインデックスを取得 → クラスラベルに変換します。

        ```python
        y_pred = logits.argmax(dim=1).numpy()
        ```

        ### 評価指標

        - **Accuracy**: 全体の正解率
        - **Precision / Recall / F1**: クラスごとの詳細な精度評価
        - **Confusion Matrix**: どのクラスをどのクラスと間違えたかを可視化
        """
    )
    return


@app.cell
def _(TARGET_NAMES, mo, nn_logits, format_classification_report, y_test):
    # logits → 予測ラベル
    y_pred_nn = nn_logits.argmax(dim=1).numpy()

    # 評価レポート
    nn_report = format_classification_report(y_test, y_pred_nn, TARGET_NAMES)
    mo.md(f"### Neural Network 評価結果\n\n```\n{nn_report}\n```")
    return y_pred_nn, nn_report


@app.cell
def _(TARGET_NAMES, plot_confusion_matrix, y_pred_nn, y_test):
    nn_cm_fig = plot_confusion_matrix(
        y_test, y_pred_nn, TARGET_NAMES, title="Neural Network - Confusion Matrix"
    )
    nn_cm_fig
    return (nn_cm_fig,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ### Softmax で確率に変換

        logits に Softmax を適用すると、各クラスの確率（合計 = 1.0）に変換できます。

        ```python
        probabilities = torch.softmax(logits, dim=1)
        ```

        モデルがどの程度「確信」を持って予測しているかを確認できます。
        """
    )
    return


@app.cell
def _(TARGET_NAMES, mo, nn_logits, np, pd, torch):
    # Softmax で確率に変換
    nn_proba = torch.softmax(nn_logits, dim=1).numpy()

    # 先頭5件を確率テーブルで表示
    proba_df = pd.DataFrame(nn_proba[:5], columns=TARGET_NAMES)
    proba_df["predicted"] = [TARGET_NAMES[i] for i in nn_proba[:5].argmax(axis=1)]
    proba_df["confidence"] = nn_proba[:5].max(axis=1)

    mo.md(
        f"""
        先頭5件の予測確率:

        {proba_df.to_markdown(index=False, floatfmt=".4f")}
        """
    )
    return nn_proba, proba_df


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## 追加タスク: LightGBM（勾配ブースティング）

        同じ Iris データで学習済みの **LightGBM** モデルを使って推論してみます。

        ### LightGBM の特徴

        - 決定木のアンサンブル学習（勾配ブースティング）
        - テーブルデータで高性能を発揮しやすい
        - 学習が高速
        - 特徴量の重要度が確認しやすい

        ### NN との違い

        | 比較項目 | Neural Network | LightGBM |
        |---|---|---|
        | 入力 | PyTorch tensor | NumPy 配列そのまま |
        | 出力 | logits（要 softmax） | 確率（softmax 済み） |
        | モデル形式 | `.pt`（state_dict） | `.txt`（テキスト） |
        """
    )
    return


@app.cell
def _(lgb):
    # LightGBM モデルのロード
    lgbm_model = lgb.Booster(model_file="models/iris_lgbm.txt")
    print("LightGBM モデルをロードしました")
    print(f"  ブースティング回数: {lgbm_model.num_trees()}")
    print(f"  特徴量数: {lgbm_model.num_feature()}")
    return (lgbm_model,)


@app.cell
def _(X_test, lgbm_model, mo, np):
    # LightGBM の推論（NumPy 配列をそのまま入力）
    lgbm_proba = lgbm_model.predict(X_test)
    y_pred_lgbm = np.argmax(lgbm_proba, axis=1)

    mo.md(
        f"""
        ### LightGBM の推論結果

        LightGBM は直接確率を出力します（softmax 不要）。

        先頭3件の予測確率:
        ```
        {np.array2string(lgbm_proba[:3], precision=4)}
        ```
        """
    )
    return lgbm_proba, y_pred_lgbm


@app.cell
def _(TARGET_NAMES, mo, format_classification_report, y_pred_lgbm, y_test):
    lgbm_report = format_classification_report(y_test, y_pred_lgbm, TARGET_NAMES)
    mo.md(f"### LightGBM 評価結果\n\n```\n{lgbm_report}\n```")
    return (lgbm_report,)


@app.cell
def _(TARGET_NAMES, plot_confusion_matrix, y_pred_lgbm, y_test):
    lgbm_cm_fig = plot_confusion_matrix(
        y_test, y_pred_lgbm, TARGET_NAMES, title="LightGBM - Confusion Matrix"
    )
    lgbm_cm_fig
    return (lgbm_cm_fig,)


@app.cell
def _(lgb, lgbm_model, plt):
    lgb.plot_importance(lgbm_model, importance_type="split", figsize=(10, 5))
    importance_fig = plt.gcf()
    importance_fig
    return (importance_fig,)


@app.cell
def _(lgb, lgbm_model):
    # LightGBM が学習した決定木の可視化
    lgb.create_tree_digraph(lgbm_model)
    return


@app.cell
def _(mo, nn_report, lgbm_report):
    mo.md(
        """
        ---

        ## まとめ

        ### 推論の3フェーズ

        | フェーズ | NN（PyTorch） | LightGBM |
        |---|---|---|
        | **Preprocess** | NumPy → StandardScaler → tensor | NumPy → StandardScaler |
        | **Forward** | `model(tensor)` → logits | `model.predict(array)` → 確率 |
        | **Postprocess** | `argmax(logits)` → クラスラベル | `argmax(確率)` → クラスラベル |

        ### 試してみよう

        1. `logits` と `softmax(logits)` の値を比較してみよう
        2. テストデータの中で「間違えたサンプル」を特定し、その確率分布を確認してみよう
        3. NN と LightGBM で予測が異なるサンプルはあるか？
        """
    )
    return


if __name__ == "__main__":
    app.run()
