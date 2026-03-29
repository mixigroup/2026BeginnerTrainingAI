import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
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

    - **LightGBM（勾配ブースティング）**：テーブルデータで広く使われる手法

    事前に学習済みのモデルをロードして、推論のみ行います。

    > Neural Network 版は `notebook.py` を参照してください。
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    import lightgbm as lgb
    import pathlib

    from src.dataset import (
        load_iris_data,
        load_scaler_params,
        normalize_features,
        prepare_test_data,
    )
    from src.evaluate import plot_confusion_matrix, format_classification_report

    return (
        format_classification_report,
        lgb,
        load_iris_data,
        load_scaler_params,
        np,
        pathlib,
        plot_confusion_matrix,
        plt,
        prepare_test_data,
        sns,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## データセットの確認
    """)
    return


@app.cell
def _(load_iris_data):
    data, iris = load_iris_data()
    data
    return (data,)


@app.cell(hide_code=True)
def _(data, mo):
    mo.md(f"""
    ### データの概要

    - サンプル数: **{len(data)}** 件
    - 特徴量数: **4** 列（sepal length, sepal width, petal length, petal width）
    - クラス数: **3** クラス（setosa=0, versicolor=1, virginica=2）

    #### 統計情報
    """)
    return


@app.cell
def _(data):
    data.describe()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### データの可視化（EDA）

    各特徴量の組み合わせを散布図で可視化します。
    クラスごとに色分けされており、分類に有効な特徴量を確認できます。
    """)
    return


@app.cell
def _(data, sns):
    pair_fig = sns.pairplot(data, hue="species", palette="tab10")
    pair_fig.figure
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 1: Preprocess（前処理）

    テーブルデータをモデルが受け取れる形式に変換します。

    ### 前処理の手順

    1. **テストデータの取得** - 学習時と同じ分割を再現
    2. **標準化（StandardScaler）** - 学習時に計算した平均・標準偏差を使って正規化

    ### なぜ学習時と同じスケーラーを使うのか？

    学習時に「平均=5.0, 標準偏差=1.0」で正規化して学習したモデルに対して、
    推論時に異なるスケーラーで正規化すると、モデルが見たことのない分布のデータが入力され、
    正しい予測ができなくなります。
    """)
    return


@app.cell
def _(load_scaler_params, prepare_test_data):
    # テストデータの取得（学習時と同じ分割）
    X_test, y_test, TARGET_NAMES = prepare_test_data()

    # スケーラーパラメータの確認
    scaler_params = load_scaler_params()
    return TARGET_NAMES, X_test, scaler_params, y_test


@app.cell(hide_code=True)
def _(TARGET_NAMES, X_test, mo, np, scaler_params):
    mo.md(f"""
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
    """)
    return


@app.cell(hide_code=True)
def _(mo, pathlib):
    _tree_img = mo.image(src=str(pathlib.Path("img/lgbm_tree_deep.png").resolve()))

    mo.md(rf"""
    ---

    ## Phase 2: Forward（LightGBM の推論）

    ### LightGBM の特徴

    - 決定木のアンサンブル学習（勾配ブースティング）
    - テーブルデータで高性能を発揮しやすい
    - 学習が高速
    - 特徴量の重要度が確認しやすい

    #### 決定木の例（学習済みモデルの Tree 119）

    {_tree_img}

    各ノードで特徴量の値を閾値と比較し、条件を満たすか（yes/no）で分岐します。
    葉ノード（Leaf）に到達すると、そこに格納された値が予測に使われます。
    LightGBM はこのような木を **120本**（3クラス × 40ラウンド）組み合わせて最終予測を行います。

    ### NN との違い

    | 比較項目 | Neural Network | LightGBM |
    |---|---|---|
    | 入力 | PyTorch tensor | NumPy 配列そのまま |
    | 出力 | logits（要 softmax） | 確率（softmax 済み） |
    | モデル形式 | `.pt`（state_dict） | `.txt`（テキスト） |
    """)
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
def _(X_test, lgbm_model, np):
    # LightGBM の推論（NumPy 配列をそのまま入力）
    lgbm_proba = lgbm_model.predict(X_test)
    y_pred_lgbm = np.argmax(lgbm_proba, axis=1)
    return lgbm_proba, y_pred_lgbm


@app.cell(hide_code=True)
def _(lgbm_proba, mo, np):
    mo.md(f"""
    ### LightGBM の推論結果

    LightGBM は直接確率を出力します（softmax 不要）。

    先頭3件の予測確率:
    ```
    {np.array2string(lgbm_proba[:3], precision=4)}
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase 3: Postprocess（後処理・評価）

    ### 確率 → 予測クラス

    `argmax` で各サンプルの最大確率のインデックスを取得 → クラスラベルに変換します。

    ```python
    y_pred = np.argmax(probabilities, axis=1)
    ```

    ### 評価指標

    - **Accuracy**: 全体の正解率
    - **Precision / Recall / F1**: クラスごとの詳細な精度評価
    - **Confusion Matrix**: どのクラスをどのクラスと間違えたかを可視化
    """)
    return


@app.cell
def _(TARGET_NAMES, format_classification_report, mo, y_pred_lgbm, y_test):
    lgbm_report = format_classification_report(y_test, y_pred_lgbm, TARGET_NAMES)
    mo.md(f"### LightGBM 評価結果\n\n```\n{lgbm_report}\n```")
    return


@app.cell
def _(TARGET_NAMES, plot_confusion_matrix, y_pred_lgbm, y_test):
    lgbm_cm_fig = plot_confusion_matrix(
        y_test, y_pred_lgbm, TARGET_NAMES, title="LightGBM - Confusion Matrix"
    )
    lgbm_cm_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 特徴量の重要度

    LightGBM では、各特徴量がモデルの予測にどれだけ貢献しているかを確認できます。
    """)
    return


@app.cell
def _(lgb, lgbm_model, plt):
    lgb.plot_importance(lgbm_model, importance_type="split", figsize=(10, 5))
    importance_fig = plt.gcf()
    importance_fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 決定木の可視化

    LightGBM が学習した決定木を可視化します。
    `plot_tree` や `create_tree_digraph` を使うには、システムに **Graphviz** のインストールが必要です。

    ```bash
    # Ubuntu / Debian
    sudo apt install -y graphviz

    # macOS (Homebrew)
    brew install graphviz
    ```

    `trees_to_dataframe()` を使うと、木構造をDataFrameとして取得できます。
    """)
    return


@app.cell
def _(lgbm_model):
    # 木構造を DataFrame として取得
    tree_df = lgbm_model.trees_to_dataframe()
    tree_df.head(10)
    return (tree_df,)


@app.cell(hide_code=True)
def _(mo, pathlib, tree_df):
    _tdf_img = mo.image(src=str(pathlib.Path("img/lgbm_trees_to_dataframe.png").resolve()))

    mo.md(rf"""
    #### trees_to_dataframe() の概要

    - 全ノード数: **{len(tree_df)}**
    - ツリー数: **{tree_df['tree_index'].nunique()}** 本（3クラス × {tree_df['tree_index'].nunique() // 3} ラウンド）
    - 最大深さ: **{tree_df['node_depth'].max()}**

    各行が決定木の1ノードに対応し、分岐条件（`split_feature`, `threshold`）や葉の値（`value`）を確認できます。

    {_tdf_img}
    """)
    return


@app.cell
def _(lgb, lgbm_model):
    # LightGBM が学習した決定木の可視化（Graphviz が必要）
    lgb.create_tree_digraph(lgbm_model)
    return


@app.cell
def _(lgb, lgbm_model, plt):
    # plot_tree による決定木の可視化（Graphviz が必要）
    fig_tree, ax_tree = plt.subplots(figsize=(20, 10))
    lgb.plot_tree(
        lgbm_model,
        tree_index=0,
        ax=ax_tree,
        show_info=["split_gain", "internal_value", "internal_count"],
    )
    ax_tree.set_title("LightGBM Decision Tree (Tree 0)", fontsize=16)
    fig_tree
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## まとめ

    ### 推論の3フェーズ（LightGBM）

    | フェーズ | 内容 |
    |---|---|
    | **Preprocess** | NumPy → StandardScaler |
    | **Forward** | `model.predict(array)` → 確率 |
    | **Postprocess** | `argmax(確率)` → クラスラベル |

    ### 試してみよう

    1. 特徴量の重要度を確認し、どの特徴量が分類に有効か考えてみよう
    2. テストデータの中で「間違えたサンプル」を特定し、その確率分布を確認してみよう
    3. Neural Network 版（`notebook.py`）も実行して、結果を比較してみよう
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
