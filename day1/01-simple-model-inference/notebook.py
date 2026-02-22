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

        このノートブックでは、**アヤメ（Iris）データセット**を使って、テーブルデータの機械学習を体験します。

        ## 推論の3つの共通フェーズ

        どんなモデルでも、推論は以下の3フェーズで構成されます。

        | フェーズ | 内容 |
        |---|---|
        | **1. Preprocess** | 入力データを tensor（多次元配列）に変換・正規化 |
        | **2. Forward** | モデルに tensor を入れて、出力 tensor を得る |
        | **3. Postprocess** | 出力 tensor をタスクに適した形に変換（クラスラベルなど） |

        ## 今回扱うモデル

        - **Neural Network（全結合NN）**：メインのハンズオン
        - **LightGBM（勾配ブースティング）**：追加タスク
        """
    )
    return


@app.cell
def _():
    import sys
    import io

    sys.path.insert(0, ".")

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from keras import utils
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, classification_report

    from src.dataset import (
        load_iris_data,
        shuffle_data,
        split_features_and_labels,
        normalize_features,
        split_dataset,
    )
    from src.model_nn import build_model, train_model
    from src.evaluate import (
        plot_learning_curves,
        evaluate_nn,
        get_nn_predictions,
        plot_confusion_matrix,
        print_classification_report,
    )

    return (
        io,
        np,
        pd,
        plt,
        sns,
        utils,
        lgb,
        accuracy_score,
        classification_report,
        load_iris_data,
        shuffle_data,
        split_features_and_labels,
        normalize_features,
        split_dataset,
        build_model,
        train_model,
        plot_learning_curves,
        evaluate_nn,
        get_nn_predictions,
        plot_confusion_matrix,
        print_classification_report,
    )


@app.cell
def _(mo):
    mo.md("## データセットの準備")
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
def _(data, plt, sns):
    pair_fig = sns.pairplot(data, hue="target", palette="tab10")
    pair_fig.figure
    return (pair_fig,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 1: Preprocess（前処理）

        テーブルデータをモデル入力形式に変換します。

        ### 前処理の手順

        1. **シャッフル** - データの順序による偏りを除去
        2. **特徴量 / ラベルの分割** - X（特徴量）と y（ラベル）に分ける
        3. **正規化** - 各特徴量を [0, 1] スケールに揃える
        4. **train / valid / test 分割** - モデルの評価を公平に行うため
        5. **one-hot encoding** - ラベルをベクトルに変換（NNの出力に合わせる）
        """
    )
    return


@app.cell
def _(
    data,
    iris,
    mo,
    np,
    normalize_features,
    shuffle_data,
    split_dataset,
    split_features_and_labels,
    utils,
):
    # 1. Shuffle
    data_shuffled = shuffle_data(data, seed=42)

    # 2. Split features and labels
    X, y = split_features_and_labels(data_shuffled)

    # 3. Normalize
    X_norm = normalize_features(X)

    # 4. Train / valid / test split (80% / 10% / 10%)
    X_train, X_valid, X_test, y_train_raw, y_valid_raw, y_test_raw = split_dataset(
        X_norm, y, train_ratio=0.8, valid_ratio=0.1
    )

    # Integer labels (for LightGBM and evaluation)
    y_train_label = y_train_raw.flatten()
    y_valid_label = y_valid_raw.flatten()
    y_test_label = y_test_raw.flatten()

    # 5. One-hot encoding for Neural Network
    y_train_onehot = utils.to_categorical(y_train_label, num_classes=3)
    y_valid_onehot = utils.to_categorical(y_valid_label, num_classes=3)
    y_test_onehot = utils.to_categorical(y_test_label, num_classes=3)

    TARGET_NAMES = list(iris.target_names)

    mo.md(
        f"""
        ### 分割結果

        | セット | サンプル数 |
        |---|---|
        | Train | **{len(X_train)}** |
        | Valid | **{len(X_valid)}** |
        | Test  | **{len(X_test)}**  |

        one-hot encoding の例（先頭3件）:

        ```
        y_train_label: {y_train_label[:3]}
        y_train_onehot:
        {np.array2string(y_train_onehot[:3])}
        ```
        """
    )
    return (
        X_train,
        X_valid,
        X_test,
        y_train_label,
        y_valid_label,
        y_test_label,
        y_train_onehot,
        y_valid_onehot,
        y_test_onehot,
        TARGET_NAMES,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 2: Forward（Neural Networkモデルの学習）

        ### モデル構造

        ```
        Input(4) → Dense(1000, ReLU) → Dense(500, ReLU) → Dense(300, ReLU)
                → Dropout(0.2) → Dense(3, Softmax)
        ```

        - **Dense**: 全結合層。すべての入力ノードがすべての出力ノードに接続される
        - **ReLU**: 活性化関数。負の値を 0 にする非線形変換
        - **Dropout**: ランダムにニューロンを無効化して過学習を防ぐ正則化手法
        - **Softmax**: 出力を確率分布（合計 = 1）に変換

        ### ハイパーパラメータの設定

        以下の値を変えて、学習結果の変化を確認してみましょう。
        """
    )
    return


@app.cell
def _():
    # --- Hyperparameters: Edit these to experiment! ---
    HIDDEN_UNITS = [1000, 500, 300]  # Hidden layer sizes
    DROPOUT_RATE = 0.2  # Dropout probability (0.0 - 1.0)
    LEARNING_RATE = 0.001  # Adam optimizer learning rate
    EPOCHS = 100  # Number of training epochs
    BATCH_SIZE = 100  # Mini-batch size
    return BATCH_SIZE, DROPOUT_RATE, EPOCHS, HIDDEN_UNITS, LEARNING_RATE


@app.cell
def _(
    BATCH_SIZE,
    DROPOUT_RATE,
    EPOCHS,
    HIDDEN_UNITS,
    LEARNING_RATE,
    X_train,
    X_valid,
    build_model,
    io,
    mo,
    train_model,
    y_train_onehot,
    y_valid_onehot,
):
    # Build model
    nn_model = build_model(
        input_dim=4,
        hidden_units=HIDDEN_UNITS,
        dropout_rate=DROPOUT_RATE,
        learning_rate=LEARNING_RATE,
    )

    # Display model summary
    summary_buf = io.StringIO()
    nn_model.summary(print_fn=lambda x: summary_buf.write(x + "\n"))
    summary_text = summary_buf.getvalue()
    mo.md(f"```\n{summary_text}\n```")
    return nn_model, summary_buf, summary_text


@app.cell
def _(
    BATCH_SIZE,
    EPOCHS,
    X_train,
    X_valid,
    nn_model,
    train_model,
    y_train_onehot,
    y_valid_onehot,
):
    # Train model (training progress is shown below)
    nn_history = train_model(
        nn_model,
        X_train,
        y_train_onehot,
        X_valid,
        y_valid_onehot,
        batch_size=BATCH_SIZE,
        epochs=EPOCHS,
    )
    return (nn_history,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## Phase 3: Postprocess（評価）

        学習結果を確認します。

        ### 評価指標

        - **Accuracy**: 全体の正解率
        - **Precision / Recall / F1**: クラスごとの詳細な精度評価
        - **Confusion Matrix**: どのクラスをどのクラスと間違えたかを可視化
        """
    )
    return


@app.cell
def _(nn_history, plot_learning_curves):
    # Learning curves: check for overfitting
    learning_curve_fig = plot_learning_curves(nn_history)
    learning_curve_fig
    return (learning_curve_fig,)


@app.cell
def _(
    TARGET_NAMES,
    X_test,
    get_nn_predictions,
    mo,
    nn_model,
    print_classification_report,
    y_test_onehot,
):
    y_true_nn, y_pred_nn = get_nn_predictions(nn_model, X_test, y_test_onehot)
    nn_report = print_classification_report(y_true_nn, y_pred_nn, TARGET_NAMES)
    mo.md(f"### 評価結果\n\n```\n{nn_report}\n```")
    return nn_report, y_pred_nn, y_true_nn


@app.cell
def _(TARGET_NAMES, plot_confusion_matrix, y_pred_nn, y_true_nn):
    nn_cm_fig = plot_confusion_matrix(y_true_nn, y_pred_nn, TARGET_NAMES)
    nn_cm_fig
    return (nn_cm_fig,)


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## EX: 過学習させてみよう

        Neural Network は層やユニット数を大きくすると過学習しやすくなります。

        ### 試してみよう

        上の「ハイパーパラメータの設定」セルで以下を変更してみましょう：

        ```python
        HIDDEN_UNITS = [5000, 2000, 1000]  # 層を大きくする
        EPOCHS = 200                         # エポックを増やす
        DROPOUT_RATE = 0.0                   # Dropout をなくす
        ```

        学習曲線で **train loss が下がり続けるが valid loss が上がっていく** 現象（過学習）を確認してください。
        """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
        ---

        ## 追加タスク: LightGBM（勾配ブースティング）

        Neural Network の代わりに、テーブルデータでよく使われる **LightGBM** を使って学習してみます。

        ### LightGBM の特徴

        - 決定木のアンサンブル学習（勾配ブースティング）
        - テーブルデータで高性能を発揮しやすい
        - 学習が高速
        - 特徴量の重要度が確認しやすい

        ### 注意

        LightGBM はカテゴリを内部で処理するため、**one-hot ではなく整数ラベル** を使います。
        前処理で保持した `y_train_label` / `y_valid_label` / `y_test_label` をそのまま使えます。
        """
    )
    return


@app.cell
def _(X_train, X_valid, lgb, mo, y_train_label, y_valid_label):
    lgb_train_data = lgb.Dataset(X_train, label=y_train_label)
    lgb_valid_data = lgb.Dataset(X_valid, label=y_valid_label, reference=lgb_train_data)

    lgbm_params = {
        "objective": "multiclass",
        "num_class": 3,
        "metric": "multi_logloss",
        "verbosity": -1,
    }

    lgbm_evals = {}

    lgbm_model = lgb.train(
        lgbm_params,
        lgb_train_data,
        valid_sets=[lgb_train_data, lgb_valid_data],
        num_boost_round=100,
        callbacks=[
            lgb.early_stopping(stopping_rounds=10, verbose=True),
            lgb.log_evaluation(10),
            lgb.record_evaluation(lgbm_evals),
        ],
    )

    mo.md("LightGBM の学習が完了しました。")
    return lgb_train_data, lgb_valid_data, lgbm_evals, lgbm_model, lgbm_params


@app.cell
def _(lgb, lgbm_evals, plt):
    lgb.plot_metric(lgbm_evals)
    lgbm_curve_fig = plt.gcf()
    lgbm_curve_fig
    return (lgbm_curve_fig,)


@app.cell
def _(
    TARGET_NAMES,
    X_test,
    accuracy_score,
    classification_report,
    lgbm_model,
    mo,
    np,
    y_test_label,
):
    lgbm_pred_proba = lgbm_model.predict(
        X_test, num_iteration=lgbm_model.best_iteration
    )
    y_pred_lgbm = np.argmax(lgbm_pred_proba, axis=1)

    lgbm_acc = accuracy_score(y_test_label, y_pred_lgbm)
    lgbm_report = classification_report(
        y_test_label, y_pred_lgbm, target_names=TARGET_NAMES
    )

    mo.md(
        f"""
        ### LightGBM 評価結果

        ```
        Accuracy: {lgbm_acc * 100:.2f}%

        {lgbm_report}
        ```
        """
    )
    return lgbm_acc, lgbm_pred_proba, lgbm_report, y_pred_lgbm


@app.cell
def _(TARGET_NAMES, plot_confusion_matrix, y_pred_lgbm, y_test_label):
    lgbm_cm_fig = plot_confusion_matrix(y_test_label, y_pred_lgbm, TARGET_NAMES)
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
    # Display the first decision tree learned by LightGBM
    lgb.create_tree_digraph(lgbm_model)
    return


if __name__ == "__main__":
    app.run()
