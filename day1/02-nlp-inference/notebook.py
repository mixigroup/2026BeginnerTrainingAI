import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


# Cell 01: marimoインポート
@app.cell(hide_code=True)
def _():
    import marimo as mo
    return (mo,)


# Cell 02: タイトルと導入
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        # 日本語感情分析ハンズオン

        ## タスク：Sentiment Analysis（感情分析）

        - 文章全体にポジティブ / ネガティブのラベルを付ける
        - 例：「私はとっても幸せ」→ **ポジティブ**、「私はとっても不幸」→ **ネガティブ**

        **使用モデル**: `tabularisai/multilingual-sentiment-analysis`
        - 多言語対応のBERT系モデルで、文章全体の特徴を捉えて分類する

        ---

        ### このノートブックでやること

        `pipeline` の内部を3フェーズに分解して理解する：

        1. **Preprocess（前処理）** — 文章 → テンソル
        2. **Forward（推論）** — テンソル → logits
        3. **Postprocess（後処理）** — logits → ラベル
        """
    )
    return


# Cell 03: pipelineとは
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ## pipeline とは

        HuggingFace の `pipeline` は、前処理・推論・後処理を1行で実行できる便利なAPI。

        ```python
        pipe = pipeline("text-classification", model="tabularisai/multilingual-sentiment-analysis")
        pipe("あなたのことが大好きです。")
        # → [{'label': 'Very Positive', 'score': 0.96...}]
        ```

        このノートブックでは、この `pipeline` の内部を分解して理解する。
        """
    )
    return


# Cell 04: ライブラリインポート
@app.cell
def _():
    from transformers import pipeline
    from src.sentiment import (
        load_tokenizer,
        load_model,
        preprocess,
        forward,
        postprocess,
        predict,
        evaluate,
        EVAL_DATA,
    )
    return (
        EVAL_DATA,
        evaluate,
        forward,
        load_model,
        load_tokenizer,
        pipeline,
        postprocess,
        predict,
        preprocess,
    )


# Cell 05: pipelineで1行実行（比較用）
@app.cell
def _(mo, pipeline):
    MODEL_NAME = "tabularisai/multilingual-sentiment-analysis"
    pipe = pipeline("text-classification", model=MODEL_NAME)
    pipeline_result = pipe("あなたのことが大好きです。")

    mo.md(
        f"""
        ### pipeline の出力（比較用）

        ```python
        pipe("あなたのことが大好きです。")
        ```

        結果: `{pipeline_result}`

        > 以降のセルで、この pipeline の内部を手動で再現する。
        """
    )
    return MODEL_NAME, pipe, pipeline_result


# Cell 06: Phase 1 解説
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## Phase 1: Preprocess（前処理）

        ### 文章を固定長テンソルに変換する3ステップ

        1. **Tokenizerで分割**
           - 例：「このレストランは最高でした！」
           - → `["この", "レストラン", "は", "最高", "でした", "！"]`

        2. **数値IDに変換**
           - トークン → 語彙辞書で数値化 → `input_ids`

        3. **長さ調整**
           - `padding` / `max_length` / `attention_mask`
        """
    )
    return


# Cell 07: Tokenizer読み込みとトークン化の可視化
@app.cell
def _(MODEL_NAME, load_tokenizer, mo):
    tokenizer = load_tokenizer(MODEL_NAME)
    sample_text = "このレストランは最高でした！"
    tokens = tokenizer.tokenize(sample_text)

    mo.md(
        f"""
        ### トークン化の結果

        ```python
        tokenizer.tokenize("{sample_text}")
        ```

        → `{tokens}`

        **トークン数**: {len(tokens)} 個
        """
    )
    return sample_text, tokenizer, tokens


# Cell 08: テンソル化（preprocess）
@app.cell
def _(mo, preprocess, sample_text, tokenizer):
    encoded = preprocess(tokenizer, sample_text)

    mo.md(
        f"""
        ### preprocess() の結果

        ```python
        encoded = preprocess(tokenizer, "{sample_text}")
        ```

        **input_ids**（トークンを数値IDに変換）:
        ```
        {encoded["input_ids"].tolist()}
        ```

        **attention_mask**（実際のトークンは1、paddingは0）:
        ```
        {encoded["attention_mask"].tolist()}
        ```

        - テンソル長: {encoded["input_ids"].shape[1]}
        - 実トークン数（attention_mask の合計）: {encoded["attention_mask"].sum().item()}
        """
    )
    return (encoded,)


# Cell 09: Phase 2 解説
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## Phase 2: Forward（モデル計算）

        ### Sentiment Analysis の推論フロー

        1. **入力**
           - `input_ids` + `attention_mask`

        2. **モデル処理**
           - BERT系モデルで文章全体の特徴を抽出
           - [CLS]トークンの特徴量をもとに分類

        3. **出力: logits**
           - shape: `(batch_size, num_labels)`
           - 例：`(1, 5)` → 5ラベル（Very Negative / Negative / Neutral / Positive / Very Positive）
           - 各ラベルに対する生スコア（確率ではない）
        """
    )
    return


# Cell 10: モデル読み込みとforward pass
@app.cell
def _(MODEL_NAME, encoded, forward, load_model, mo):
    model = load_model(MODEL_NAME)
    logits = forward(model, encoded)

    mo.md(
        f"""
        ### forward() の結果

        ```python
        model = load_model(MODEL_NAME)
        logits = forward(model, encoded)
        ```

        **logits** shape: `{list(logits.shape)}`

        **logits の値**（生スコア）:
        ```
        {logits.tolist()}
        ```

        **ラベルの種類**: `{list(model.config.id2label.values())}`

        > logits はまだ生スコア（確率ではない）。次のPhaseで確率に変換する。
        """
    )
    return logits, model


# Cell 11: Phase 3 解説
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## Phase 3: Postprocess（後処理）

        ### logits からラベルへの変換

        1. **softmax で確率に変換**
           - logitsを確率分布に変換（合計が1になる）
           - 例：`[3.2, -1.5, 0.1, ...]` → `[0.88, 0.01, 0.05, ...]`

        2. **argmax でラベルID取得**
           - 最大確率のインデックスを選択

        3. **ラベル名にマッピング**
           - `model.config.id2label` でラベルIDを文字列に変換
        """
    )
    return


# Cell 12: softmax → argmax → ラベルマッピング（postprocess）
@app.cell
def _(logits, mo, model, postprocess):
    pred_label, pred_score, prob_dict = postprocess(model, logits)

    prob_table = "\n".join(
        [f"| {label} | {score:.4f} |" for label, score in prob_dict.items()]
    )

    mo.md(
        f"""
        ### postprocess() の結果

        ```python
        pred_label, pred_score, prob_dict = postprocess(model, logits)
        ```

        **確率分布（softmax後）**:

        | ラベル | 確率 |
        |--------|------|
        {prob_table}

        ---

        **予測結果**: **{pred_label}**（{pred_score:.1%}）

        > pipeline の出力と同じ結果になっているはず！
        """
    )
    return pred_label, pred_score, prob_dict, prob_table


# Cell 13: 評価指標の解説
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## 評価指標

        Sentiment Analysis（感情分析）では、**文章全体の分類が正しいかを評価**する。

        最も基本的な指標は **Accuracy（正解率）**：

        ```
        Accuracy = 正解数 / 全文章数
        ```

        今回は20件のサンプルデータで精度を測定する。
        モデルの5段階ラベルは以下のように2値に変換して評価する：

        | モデルのラベル | 変換後 |
        |---|---|
        | Very Positive / Positive | Positive |
        | Very Negative / Negative | Negative |
        | Neutral | Neutral |
        """
    )
    return


# Cell 14: 20件のサンプルで精度評価
@app.cell
def _(EVAL_DATA, evaluate, mo, model, tokenizer):
    results, accuracy = evaluate(tokenizer, model, EVAL_DATA)
    correct_count = sum(1 for r in results if r[5])

    rows = "\n".join([
        f"| {r[0]} | {r[1]} | {r[2]}（{r[4]:.0%}） | {r[3]} | {'✅' if r[5] else '❌'} |"
        for r in results
    ])

    mo.md(
        f"""
        ### 評価結果（{len(EVAL_DATA)}件）

        | 文章 | 正解 | モデル予測（raw） | 変換後 | 正解？ |
        |------|------|-----------------|--------|--------|
        {rows}

        ---

        **Accuracy: {accuracy:.1%}（{correct_count}/{len(EVAL_DATA)}件正解）**
        """
    )
    return accuracy, correct_count, results, rows


# Cell 15: インタラクティブデモ解説
@app.cell(hide_code=True)
def _(mo):
    mo.md(
        """
        ---

        ## インタラクティブデモ

        下のテキストボックスに文章を入力すると、リアルタイムで感情分析を実行する。

        モデルが判定する5つのラベル：
        - Very Negative / Negative / Neutral / Positive / Very Positive
        """
    )
    return


# Cell 16: UIウィジェット定義
@app.cell
def _(mo):
    text_input = mo.ui.text(
        value="あなたのことが大好きです。",
        label="分析するテキスト",
        full_width=True,
    )
    text_input
    return (text_input,)


# Cell 17: リアルタイム推論
@app.cell
def _(mo, model, predict, text_input, tokenizer):
    _input_text = text_input.value

    if _input_text.strip():
        _pred_label, _pred_score, _prob_dict = predict(tokenizer, model, _input_text)

        _prob_rows = "\n".join([
            f"| {lbl} | {score:.4f} | {'◀ 予測' if lbl == _pred_label else ''} |"
            for lbl, score in _prob_dict.items()
        ])

        mo.md(
            f"""
            ### 分析結果

            **入力**: `{_input_text}`

            **予測ラベル**: **{_pred_label}**（{_pred_score:.1%}）

            | ラベル | 確率 | |
            |--------|------|--|
            {_prob_rows}
            """
        )
    else:
        mo.md("テキストを入力してください。")
