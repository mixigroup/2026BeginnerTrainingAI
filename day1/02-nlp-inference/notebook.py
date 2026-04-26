import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 日本語感情分析ハンズオン

    ## タスク：Sentiment Analysis（感情分析）

    - 文章全体にポジティブ / ネガティブのラベルを付ける
    - 例：「私はとっても幸せ」→ **ポジティブ**、「私はとっても不幸」→ **ネガティブ**

    **使用モデル**: `koheiduck/bert-japanese-finetuned-sentiment`
    - 東北大 BERT base (日本語 Whole Word Masking) をファインチューニングした3クラス分類モデル
    - ラベル：`POSITIVE` / `NEGATIVE` / `NEUTRAL`

    ---

    ### このノートブックでやること

    `pipeline` の内部を3フェーズに分解して理解する：

    1. **Preprocess（前処理）** — 文章 → テンソル
    2. **Forward（推論）** — テンソル → logits
    3. **Postprocess（後処理）** — logits → ラベル
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## pipeline とは

    HuggingFace の `pipeline` は、前処理・推論・後処理を1行で実行できる便利なAPI。

    ```python
    pipe = pipeline("text-classification", model="koheiduck/bert-japanese-finetuned-sentiment")
    pipe("あなたのことが大好きです。")
    # → [{'label': 'POSITIVE', 'score': 0.99...}]
    ```

    このノートブックでは、この `pipeline` の内部を分解して理解する。
    """)
    return


@app.cell
def _():
    from transformers import pipeline
    from src.sentiment import (
        load_tokenizer,
        load_model,
        preprocess,
        forward,
        postprocess,
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
        preprocess,
    )


@app.cell
def _(pipeline):
    MODEL_NAME = "koheiduck/bert-japanese-finetuned-sentiment"
    pipe = pipeline("text-classification", model=MODEL_NAME)
    pipeline_result = pipe("あなたのことが大好きです。")

    print(f"結果：{pipeline_result}")
    return (MODEL_NAME,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
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
    """)
    return


@app.cell
def _(MODEL_NAME, load_tokenizer):
    tokenizer = load_tokenizer(MODEL_NAME)
    sample_text = "このレストランは最高でした！"
    tokens = tokenizer.tokenize(sample_text)

    print(f"トークン数：{len(tokens)}")
    print(f"トークン：{tokens}")
    return sample_text, tokenizer


@app.cell
def _(tokenizer):
    vocab = tokenizer.get_vocab()
    print(f"語彙辞書：{list(vocab.items())[:10]} ...") 
    return (vocab,)


@app.cell
def _(preprocess, sample_text, tokenizer):
    encoded = preprocess(tokenizer, sample_text)
    print(f"input_ids: {encoded['input_ids'].tolist()}")
    print(f"attention_mask: {encoded['attention_mask'].tolist()}")
    print(f"テンソル長: {encoded['input_ids'].shape[1]}")
    print(
        f"実トークン数（attention_mask の合計）: {encoded['attention_mask'].sum().item()}"
    )
    return (encoded,)


@app.cell
def _(encoded, vocab):
    vocab_id_to_token = {v: k for k, v in vocab.items()}
    print(f"token 逆エンコーディング結果: {[vocab_id_to_token[i] for i in encoded['input_ids'].tolist()[0]]}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
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
       - 例：`(1, 3)` → 3ラベル（NEUTRAL / NEGATIVE / POSITIVE）
       - 各ラベルに対する生スコア（確率ではない）
    """)
    return


@app.cell
def _(MODEL_NAME, encoded, forward, load_model):
    model = load_model(MODEL_NAME)
    logits = forward(model, encoded)

    print(f"logits shape: {list(logits.shape)}")
    print(f"logits の値（生スコア）: {logits.tolist()}")
    print(f"ラベルの種類: {list(model.config.id2label.values())}")
    return logits, model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Phase 3: Postprocess（後処理）

    ### logits からラベルへの変換

    1. **softmax で確率に変換**
       - logitsを確率分布に変換（合計が1になる）
       - 例：`[-1.2, -2.1, 3.8]` → `[0.01, 0.00, 0.99]`

    2. **argmax でラベルID取得**
       - 最大確率のインデックスを選択

    3. **ラベル名にマッピング**
       - `model.config.id2label` でラベルIDを文字列に変換
    """)
    return


@app.cell
def _(logits, model, postprocess):
    pred_label, pred_score, prob_dict = postprocess(model, logits)

    print("確率分布（softmax後）:")
    for label, score in prob_dict.items():
        print(f"  {label}: {score:.4f}")
    print(f"\n予測結果: {pred_label}（{pred_score:.1%}）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 評価指標

    Sentiment Analysis（感情分析）では、**文章全体の分類が正しいかを評価**する。

    最も基本的な指標は **Accuracy（正解率）**：

    ```
    Accuracy = 正解数 / 全文章数
    ```

    今回は20件のサンプルデータで精度を測定する。
    モデルの出力ラベル（大文字）は、以下のように先頭大文字の表示形式に揃えて評価する：

    | モデル出力 | 表示形式 |
    |---|---|
    | POSITIVE | Positive |
    | NEGATIVE | Negative |
    | NEUTRAL | Neutral |
    """)
    return


@app.cell
def _(EVAL_DATA, evaluate, model, tokenizer):
    results, accuracy = evaluate(tokenizer, model, EVAL_DATA)
    correct_count = sum(1 for r in results if r[5])

    print(f"評価結果（{len(EVAL_DATA)}件）:")
    for r in results:
        mark = "✅" if r[5] else "❌"
        print(f"  {mark} [{r[1]}→{r[3]}] {r[0]}  ({r[2]}, {r[4]:.0%})")
    print(f"\nAccuracy: {accuracy:.1%}（{correct_count}/{len(EVAL_DATA)}件正解）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## まとめ

    ### 推論の3フェーズ（Sentiment Analysis with BERT）

    | フェーズ | 内容 |
    |---|---|
    | **Preprocess** | 文章 → Tokenizer → `input_ids` + `attention_mask` |
    | **Forward** | `model(input_ids, attention_mask)` → logits |
    | **Postprocess** | `softmax(logits)` → `argmax` → ラベル名 |

    ### 試してみよう

    1. 自分で文章を作って `pipe(...)` に入れ、結果が直感に合うか確認してみよう
    2. 評価結果で間違えたサンプルを観察し、確率分布（softmax 後）を見てモデルがどの程度迷っていたか確認してみよう
    3. `max_length` や `padding` の設定を変えると `input_ids` がどう変わるか確認してみよう

    ### Optional 課題：Recall / Precision を計算してみよう

    Accuracy は「全体の正解率」しか見ていないため、クラスごとの偏りや誤りの傾向はわかりません。
    クラスごとに **Precision（適合率）** と **Recall（再現率）** を計算してみよう。

    - **Precision** = そのクラスと予測したもののうち、本当に正解だった割合
        - `Precision_c = TP_c / (TP_c + FP_c)`
    - **Recall** = 本当にそのクラスのもののうち、正しく予測できた割合
        - `Recall_c = TP_c / (TP_c + FN_c)`

    ヒント:

    - `evaluate(...)` の戻り値 `results` から、各サンプルの「正解ラベル」と「予測ラベル」を取り出せる
    - `Positive` / `Negative` / `Neutral` の3クラスそれぞれについて TP / FP / FN を数えてみよう
    - `sklearn.metrics.classification_report` を使えば一発で出せるが、まずは手計算で意味を掴んでみよう
    """)
    return


if __name__ == "__main__":
    app.run()
