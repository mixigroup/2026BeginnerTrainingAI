import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


# --- Part 1: 導入 ---


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # Attention メカニズムの可視化

    Attention メカニズムの数学的原理を Word2Vec で体験し、
    BERT の Multi-Head Self-Attention を可視化することで、
    Transformer の中核技術を理解する。

    ---

    ### このノートブックでやること

    1. **Attention の数学的原理** — Query, Key, Value とソフトマックスによる重み計算
    2. **Word2Vec による Attention 体験** — 単語ベクトルで Attention スコアを手計算
    3. **Temperature パラメータ** — Attention の鋭さを制御する温度パラメータ
    4. **BERT の Multi-Head Attention 可視化** — 12層 × 12ヘッド = 144 の Attention パターン
    5. **Attention パターンの解釈** — ヘッドごとの役割と層の深さによる変化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Attention の数学的原理

    Attention は「入力のどの部分に注目すべきか」を数値化する仕組み。
    3つのベクトルで構成される：

    - **Query (Q)**: 「何を探しているか」
    - **Key (K)**: 「各要素の見出し」
    - **Value (V)**: 「各要素の中身」

    ### 計算の流れ

    1. **スコア計算**: Query と各 Key の内積で関連度を測る

    $$
    \text{score}(Q, K_i) = Q \cdot K_i
    $$

    2. **ソフトマックス正規化**: スコアを確率分布に変換（合計 = 1.0）

    $$
    \alpha_i = \frac{\exp(\text{score}_i)}{\sum_j \exp(\text{score}_j)}
    $$

    3. **加重平均**: Attention 重みで Value の加重和を計算

    $$
    \text{context} = \sum_i \alpha_i \cdot V_i
    $$

    ### Transformer の Scaled Dot-Product Attention

    $$
    \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
    $$

    | | Simple Attention | Self-Attention | Multi-Head Attention |
    |---|---|---|---|
    | **Q, K, V の出所** | 外部クエリ | 同じ入力 | 同じ入力（射影済み） |
    | **ヘッド数** | 1 | 1 | h（例: 12） |
    | **用途** | 検索・検索 | 文脈理解 | Transformer |
    """)
    return


@app.cell
def _():
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from src.attention_utils import (
        compute_attention_weights,
        compute_context_vector,
        get_bert_attentions,
        load_bert_japanese,
        plot_attention_barplot,
        plot_attention_heads_grid,
        plot_attention_heatmap,
        plot_attention_summary,
        plot_cls_attention,
    )

    matplotlib.rcParams["font.family"] = "sans-serif"
    matplotlib.rcParams["font.sans-serif"] = [
        "Hiragino Sans",
        "Hiragino Kaku Gothic ProN",
        "sans-serif",
    ]

    return (
        plt,
        np,
        sns,
        compute_attention_weights,
        compute_context_vector,
        get_bert_attentions,
        load_bert_japanese,
        plot_attention_barplot,
        plot_attention_heads_grid,
        plot_attention_heatmap,
        plot_attention_summary,
        plot_cls_attention,
    )


# --- Part 2: Word2Vec による Attention の原理 ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Word2Vec による Attention の原理

    Word2Vec の事前学習済み単語ベクトルを使って、
    Attention の計算原理を体験する。

    - **Query**: 注目したいキーワード（例: "fruit"）
    - **Key**: 文中の各単語のベクトル
    - **内積 → softmax**: 関連度の高い単語に大きな重みがつく

    Google News コーパスで学習された 300 次元ベクトル（約 300 万語）を使用。
    """)
    return


@app.cell
def _(mo):
    import gensim.downloader as api

    mo.output.append(mo.md("Word2Vec モデルをロード中... (初回は ~1.6GB ダウンロード)"))
    w2v_model = api.load("word2vec-google-news-300")

    mo.output.append(
        mo.md(f"""
    ✅ Word2Vec ロード完了

    - 語彙数: **{len(w2v_model):,}** 語
    - ベクトル次元: **{w2v_model.vector_size}**
    """)
    )
    return (w2v_model,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Query-Key の類似度で Attention Weight を計算

    Word2Vec では、意味が近い単語ほどベクトルの内積が大きくなる。
    これを利用して Attention の仕組みを再現する：

    1. Query 単語のベクトルを取得
    2. 各 Key 単語との内積（ドット積）を計算 → スコア
    3. ソフトマックスで正規化 → Attention Weight
    """)
    return


@app.cell
def _(w2v_model, np, compute_attention_weights, plot_attention_barplot, mo):
    query = "fruit"
    words = ["apple", "king", "banana", "queen", "orange", "computer"]

    query_vec = w2v_model[query]
    key_vecs = np.array([w2v_model[w] for w in words])

    _weights = compute_attention_weights(query_vec, key_vecs)

    _fig = plot_attention_barplot(words, _weights, query)
    mo.output.append(mo.md(f"### Query: `{query}` に対する各単語の Attention Weight"))
    mo.output.append(mo.as_html(_fig))
    mo.output.append(
        mo.md(
            "果物に関連する単語（apple, banana, orange）に高い重みがつき、"
            "関連の薄い単語（king, queen, computer）は低い重みになる。"
        )
    )
    return query, words, query_vec, key_vecs


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Temperature パラメータ

    Attention のスコアを温度 $T$ でスケーリングすることで、重みの分布を制御できる：

    $$
    \alpha_i = \frac{\exp(\text{score}_i / T)}{\sum_j \exp(\text{score}_j / T)}
    $$

    | Temperature | 効果 |
    |---|---|
    | **低い (< 1.0)** | 分布が鋭くなる → 最も関連の高い単語に集中 |
    | **1.0** | 標準 |
    | **高い (> 1.0)** | 分布が均一に近づく → 全体を広く見る |
    """)
    return


@app.cell
def _(query_vec, key_vecs, words, compute_attention_weights, plt, sns, np, mo):
    _temperatures = [0.5, 1.0, 2.0]
    _fig, _axes = plt.subplots(1, 3, figsize=(15, 4))

    for _i, _temp in enumerate(_temperatures):
        _w = compute_attention_weights(query_vec, key_vecs, temperature=_temp)
        sns.barplot(x=list(words), y=_w, palette="viridis", ax=_axes[_i])
        _axes[_i].set_title(f"Temperature = {_temp}", fontsize=12)
        _axes[_i].set_ylabel("Weight")
        _axes[_i].set_ylim(0, np.max(_w) * 1.2)
        _axes[_i].tick_params(axis="x", rotation=30)

    _fig.suptitle("Temperature による Attention 分布の変化", fontsize=14)
    _fig.tight_layout()

    mo.output.append(mo.as_html(_fig))
    mo.output.append(
        mo.md(
            "温度が低いほど最も関連の高い単語に集中し、"
            "高いほど均一な分布になる。LLM の生成時にも同じ原理が使われる。"
        )
    )
    return


# --- Part 3: BERT の Multi-Head Attention 可視化 ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## BERT の Multi-Head Attention 可視化

    BERT（Bidirectional Encoder Representations from Transformers）は、
    **Self-Attention** を多層に重ねたモデル。

    - `bert-base-japanese-v3`: **12層 × 12ヘッド = 144 の Attention パターン**
    - 各ヘッドは異なる言語的パターンを学習する
    - 浅い層は局所的な関係、深い層は大域的な関係を捉える傾向がある

    ここでは、日本語テキストを入力し、各層・各ヘッドの Attention を
    ヒートマップとして可視化する。
    """)
    return


@app.cell
def _(load_bert_japanese, mo):
    BERT_MODEL = "cl-tohoku/bert-base-japanese-v3"
    mo.output.append(mo.md(f"BERT モデルをロード中: `{BERT_MODEL}` ..."))

    bert_model, bert_tokenizer = load_bert_japanese(BERT_MODEL)

    _n_params = sum(p.numel() for p in bert_model.parameters())
    _config = bert_model.config
    mo.output.append(
        mo.md(f"""
    ✅ BERT ロード完了

    - パラメータ数: **{_n_params / 1e6:.1f}M**
    - レイヤー数: **{_config.num_hidden_layers}**
    - ヘッド数: **{_config.num_attention_heads}**
    - 隠れ層次元: **{_config.hidden_size}**
    """)
    )
    return bert_model, bert_tokenizer, BERT_MODEL


@app.cell
def _(bert_model, bert_tokenizer, get_bert_attentions, mo):
    sentence = "今日は渋谷の会場で、AIのアテンションについて詳しく学んでいます。"
    attentions, tokens = get_bert_attentions(bert_model, bert_tokenizer, sentence)

    mo.output.append(mo.md(f"### 入力文: 「{sentence}」"))
    mo.output.append(
        mo.md(f"""
    - トークン列: `{tokens}`
    - トークン数: **{len(tokens)}**
    - Attention テンソル shape: **{list(attentions.shape)}**
      - `[{attentions.shape[0]} layers, {attentions.shape[1]} heads, {attentions.shape[2]} tokens, {attentions.shape[3]} tokens]`
    """)
    )
    return sentence, attentions, tokens


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Attention テンソルの構造

    Attention テンソルは `(layers, heads, seq_len, seq_len)` の 4 次元。

    - `attentions[l][h][i][j]` = Layer `l`, Head `h` で、トークン `i` がトークン `j` に向ける Attention の強さ
    - 各行（Query 方向）の合計は 1.0（softmax の結果）
    - ヒートマップの明るい部分 = 強い Attention

    ```
    Query (attending)    →  [CLS] 今日 は 渋谷 の 会場 で ...
                            ↓    ↓  ↓  ↓   ↓  ↓   ↓
    Key (attended to)    →  各トークンへの重み（行の合計 = 1.0）
    ```
    """)
    return


@app.cell
def _(attentions, tokens, plot_attention_heatmap, mo):
    _fig = plot_attention_heatmap(attentions, tokens, layer=0, head=0)
    mo.output.append(mo.md("### Layer 0, Head 0 の Attention パターン"))
    mo.output.append(mo.as_html(_fig))
    mo.output.append(
        mo.md(
            "行方向（Query）の各トークンが、列方向（Key）のどのトークンに注目しているかを表す。"
        )
    )
    return


@app.cell
def _(attentions, tokens, plot_attention_heads_grid, mo):
    _fig = plot_attention_heads_grid(attentions, tokens, layer=0)
    mo.output.append(mo.md("### Layer 0: 全 12 ヘッドの Attention パターン"))
    mo.output.append(mo.as_html(_fig))
    mo.output.append(
        mo.md(
            "各ヘッドが異なるパターンを捉えていることが分かる。"
            "一部のヘッドは隣接トークンに集中し、別のヘッドは特定のトークンに強く反応する。"
        )
    )
    return


@app.cell
def _(attentions, tokens, plot_attention_heatmap, plt, mo):
    _layers_to_compare = [0, 6, 11]
    _fig, _axes = plt.subplots(1, 3, figsize=(20, 6))

    for _i, _layer_idx in enumerate(_layers_to_compare):
        plot_attention_heatmap(
            attentions, tokens, layer=_layer_idx, head=0, ax=_axes[_i]
        )

    _fig.suptitle("層の深さによる Attention パターンの変化 (Head 0)", fontsize=14)
    _fig.tight_layout()

    mo.output.append(mo.as_html(_fig))
    mo.output.append(
        mo.md("""
    - **Layer 0（浅い層）**: 隣接するトークンや局所的なパターンに注目する傾向
    - **Layer 6（中間層）**: より広い範囲の関係を捉え始める
    - **Layer 11（最終層）**: 文全体の大域的な関係や、タスクに必要な情報に集中
    """)
    )
    return


@app.cell
def _(attentions, tokens, plot_attention_summary, plot_cls_attention, plt, mo):
    _fig, _axes = plt.subplots(1, 2, figsize=(16, 6))

    plot_attention_summary(attentions, tokens, ax=_axes[0])
    plot_cls_attention(attentions, tokens, ax=_axes[1])

    _fig.tight_layout()

    mo.output.append(mo.md("### 全体の Attention パターン"))
    mo.output.append(mo.as_html(_fig))
    mo.output.append(
        mo.md(
            "[CLS] トークンは文全体の情報を集約する役割を持つ。"
            "どのトークンに強く注目しているかで、モデルが文のどこを重要と判断しているかが分かる。"
        )
    )
    return


# --- Part 4: まとめ ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## まとめ

    | トピック | 学んだこと |
    |---------|-----------|
    | **Attention の原理** | Q・K の内積 → softmax → 重み付き和で文脈ベクトルを計算 |
    | **Temperature** | 温度パラメータで Attention の集中度を制御 |
    | **Word2Vec + Attention** | 事前学習済み単語ベクトルで Attention の動作を体験 |
    | **BERT Multi-Head** | 12ヘッド × 12層 = 144 パターンの Attention を可視化 |
    | **ヘッドの役割** | 位置情報、構文、意味など異なるパターンを学習 |
    | **層の深さ** | 浅い層は局所パターン、深い層は大域的な関係を捉える |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ### ハンズオン課題

    - [ ] `query` を "queen", "doctor", "computer" に変えて Attention の変化を観察する
    - [ ] `words` のリストを変えて、文脈による Attention の違いを比較する
    - [ ] Temperature を 0.1, 0.5, 1.0, 5.0 で比較し、分布の変化をグラフで確認する
    - [ ] BERT に異なる日本語文を入力し、Attention パターンの変化を観察する
    - [ ] Layer 0 と Layer 11 の同じヘッドを比較し、浅い層と深い層の違いを考察する
    - [ ] [CLS] トークンの Attention を複数文で比較し、重要語の抽出に使えるか検討する
    """)
    return


if __name__ == "__main__":
    app.run()
