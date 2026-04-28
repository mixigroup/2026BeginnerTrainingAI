import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


# --- Part 1: 導入 + モデルロード ---


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # SigLIP2 — テキストと画像のベクトル空間理解

    SigLIP2（Sigmoid Loss for Language-Image Pre-training 2）を使って、
    テキストと画像を同じベクトル空間にエンコードし、両者の距離を計算することで、
    マルチモーダルモデルの仕組みを理解する。

    ---

    ### このノートブックでやること

    1. **Contrastive Learning** — SigLIP2 の学習方法（Sigmoid vs Softmax）を理解
    2. **xm3600 データセット** — 多言語マルチモーダルデータの EDA
    3. **Embedding 取得** — 画像・日本語テキストの特徴ベクトルを取得
    4. **コサイン類似度** — 画像↔テキスト、画像↔画像、テキスト↔テキスト の距離を計測
    5. **TensorBoardX** — ベクトル空間を 3D で可視化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## SigLIP2 とは？

    SigLIP2 は Google が開発したマルチモーダルモデルで、
    **画像とテキストを同じベクトル空間に埋め込む**ことで両者の関連度を計算できる。

    ### CLIP との違い

    | | CLIP | SigLIP2 |
    |---|---|---|
    | **損失関数** | Softmax（NxN 行列全体で正規化） | Sigmoid（各ペア独立に判定） |
    | **学習** | バッチ内の全ペアを比較 | 各画像-テキストペアを独立に「一致/不一致」判定 |
    | **多言語** | 英語中心 | **35言語以上対応（日本語含む）** |
    | **効率** | 大バッチサイズが必要 | 小バッチでも安定 |

    ### Sigmoid Loss の直感的理解

    ```
    CLIP:   「この画像はN個のテキストのうち、どれに最も近い？」（多クラス分類）
    SigLIP: 「この画像とこのテキストは一致する？ Yes/No」（二値分類 × 全ペア）
    ```

    SigLIP のアプローチは各ペアを独立に判定するため、
    バッチサイズに依存しにくく、学習が安定する。
    """)
    return


@app.cell
def _():
    import torch
    import numpy as np
    import matplotlib.pyplot as plt
    from PIL import Image
    from src.siglip_utils import (
        load_siglip_model,
        encode_images,
        encode_texts,
        cosine_similarity_matrix,
        plot_similarity_heatmap,
        export_embeddings_to_tensorboard,
        decode_image,
    )

    # matplotlib で日本語フォントを使用
    import japanize_matplotlib  # noqa: F401

    return (
        torch,
        np,
        plt,
        Image,
        load_siglip_model,
        encode_images,
        encode_texts,
        cosine_similarity_matrix,
        plot_similarity_heatmap,
        export_embeddings_to_tensorboard,
        decode_image,
    )


@app.cell
def _(torch):
    MODEL_NAME = "google/siglip2-base-patch16-224"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return MODEL_NAME, DEVICE


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## モデルのロード

    `google/siglip2-base-patch16-224` をロードする。

    - **ViT-B/16**: Vision Transformer Base、パッチサイズ 16x16
    - **224px**: 入力画像解像度
    - **86M パラメータ**: 軽量で CPU でも動作可能
    """)
    return


@app.cell
def _(MODEL_NAME, load_siglip_model, mo):
    mo.output.append(mo.md(f"モデルをロード中: `{MODEL_NAME}` ..."))
    model, processor = load_siglip_model(MODEL_NAME)

    n_params = sum(p.numel() for p in model.parameters())
    mo.output.append(
        mo.md(f"""
    ✅ モデルロード完了

    - パラメータ数: **{n_params / 1e6:.1f}M**
    - 画像埋め込み次元: **{model.config.vision_config.hidden_size}**
    - テキスト埋め込み次元: **{model.config.text_config.hidden_size}**
    """)
    )
    return model, processor


# --- Part 2: xm3600 データセット + EDA ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Crossmodal-3600（xm3600）データセット

    **xm3600** は Google が公開した多言語マルチモーダルデータセット：

    - **3,600 枚** の地理的に多様な画像
    - **36 言語** の人手によるキャプション（日本語含む）
    - 翻訳ではなく、各言語のネイティブスピーカーが独立に記述
    - マルチモーダルモデルの多言語評価に最適

    ここでは日本語キャプション付きのサブセットを使ってデータの特徴を探る。
    """)
    return


@app.cell
def _():
    N_SAMPLES = 50
    return (N_SAMPLES,)


@app.cell
def _(N_SAMPLES, mo):
    from datasets import load_dataset

    mo.output.append(mo.md("xm3600 データセットをロード中..."))

    xm_dataset = load_dataset("floschne/xm3600", split="ja")
    xm_dataset = xm_dataset.select(range(min(N_SAMPLES, len(xm_dataset))))

    mo.output.append(
        mo.md(f"""
    ✅ データセットロード完了

    - サンプル数: **{len(xm_dataset)}**
    - カラム: `{list(xm_dataset.column_names)}`
    """)
    )
    return (xm_dataset,)


@app.cell
def _(xm_dataset, decode_image, mo):
    # データセットのサンプルを表示
    items = []
    for _i in range(min(6, len(xm_dataset))):
        _s = xm_dataset[_i]
        _img = decode_image(_s["image"])
        _caption = _s["captions"][0]
        items.append(
            mo.vstack(
                [
                    mo.image(_img, width=180),
                    mo.md(f"**#{_i}**: {_caption}"),
                ]
            )
        )

    mo.output.append(mo.md("### データセットサンプル"))
    mo.output.append(mo.hstack(items, wrap=True))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### EDA: データセットの探索的分析

    データをモデルに入力する前に、画像とキャプションの特徴を把握しよう。
    """)
    return


@app.cell
def _(xm_dataset, decode_image, np, plt, mo):
    # --- EDA: 画像サイズの分布 ---
    widths = []
    heights = []
    for _s in xm_dataset:
        w, h = decode_image(_s["image"]).size
        widths.append(w)
        heights.append(h)

    fig_eda, axes = plt.subplots(1, 3, figsize=(15, 4))

    # 幅の分布
    axes[0].hist(widths, bins=20, color="steelblue", edgecolor="white")
    axes[0].set_title("画像の幅の分布")
    axes[0].set_xlabel("幅 (px)")
    axes[0].set_ylabel("頻度")

    # 高さの分布
    axes[1].hist(heights, bins=20, color="coral", edgecolor="white")
    axes[1].set_title("画像の高さの分布")
    axes[1].set_xlabel("高さ (px)")
    axes[1].set_ylabel("頻度")

    # 幅 x 高さの散布図
    axes[2].scatter(widths, heights, alpha=0.6, s=30, color="mediumpurple")
    axes[2].set_title("幅 x 高さ")
    axes[2].set_xlabel("幅 (px)")
    axes[2].set_ylabel("高さ (px)")
    axes[2].set_aspect("equal")

    fig_eda.tight_layout()

    mo.output.append(mo.md("#### 画像サイズの分布"))
    mo.output.append(mo.as_html(fig_eda))
    mo.output.append(
        mo.md(f"""
    - 幅: 最小 {min(widths)}px, 最大 {max(widths)}px, 平均 {np.mean(widths):.0f}px
    - 高さ: 最小 {min(heights)}px, 最大 {max(heights)}px, 平均 {np.mean(heights):.0f}px
    """)
    )
    return


@app.cell
def _(xm_dataset, np, plt, mo):
    # --- EDA: キャプションの文字数分布 ---
    captions_all = [_s["captions"][0] for _s in xm_dataset]
    caption_lengths = [len(c) for c in captions_all]

    # キャプションあたりの候補数
    n_captions_per_sample = [len(_s["captions"]) for _s in xm_dataset]

    fig_txt, axes_txt = plt.subplots(1, 2, figsize=(12, 4))

    # 文字数分布
    axes_txt[0].hist(caption_lengths, bins=20, color="seagreen", edgecolor="white")
    axes_txt[0].set_title("キャプション文字数の分布")
    axes_txt[0].set_xlabel("文字数")
    axes_txt[0].set_ylabel("頻度")

    # キャプション候補数
    axes_txt[1].hist(
        n_captions_per_sample,
        bins=range(1, max(n_captions_per_sample) + 2),
        color="goldenrod",
        edgecolor="white",
        align="left",
    )
    axes_txt[1].set_title("1画像あたりのキャプション数")
    axes_txt[1].set_xlabel("キャプション数")
    axes_txt[1].set_ylabel("頻度")

    fig_txt.tight_layout()

    mo.output.append(mo.md("#### キャプションの特徴"))
    mo.output.append(mo.as_html(fig_txt))
    mo.output.append(
        mo.md(f"""
    - 文字数: 最小 {min(caption_lengths)}, 最大 {max(caption_lengths)}, 平均 {np.mean(caption_lengths):.1f}
    - 1画像あたりのキャプション数: 最小 {min(n_captions_per_sample)}, 最大 {max(n_captions_per_sample)}
    """)
    )

    # 最短・最長キャプションの表示
    shortest_idx = int(np.argmin(caption_lengths))
    longest_idx = int(np.argmax(caption_lengths))
    mo.output.append(
        mo.md(f"""
    **最短キャプション** (#{shortest_idx}, {caption_lengths[shortest_idx]}文字): {captions_all[shortest_idx]}

    **最長キャプション** (#{longest_idx}, {caption_lengths[longest_idx]}文字): {captions_all[longest_idx]}
    """)
    )
    return


# --- Part 3: 埋め込みとコサイン類似度 ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 画像・テキストの埋め込み

    xm3600 の画像と日本語キャプションを SigLIP2 でエンコードし、
    同じベクトル空間上の特徴ベクトルを取得する。
    """)
    return


@app.cell
def _(
    xm_dataset, model, processor, DEVICE, encode_images, encode_texts, decode_image, mo
):
    mo.output.append(mo.md("xm3600 の画像・テキストをエンコード中..."))

    xm_images = [decode_image(_s["image"]) for _s in xm_dataset]
    xm_texts = [_s["captions"][0] for _s in xm_dataset]

    xm_img_emb = encode_images(model, processor, xm_images, DEVICE)
    xm_txt_emb = encode_texts(model, processor, xm_texts, DEVICE)

    mo.output.append(
        mo.md(f"""
    ✅ エンコード完了

    - 画像埋め込み: `{xm_img_emb.shape}` （{xm_img_emb.shape[0]}枚 × {xm_img_emb.shape[1]}次元）
    - テキスト埋め込み: `{xm_txt_emb.shape}` （{xm_txt_emb.shape[0]}文 × {xm_txt_emb.shape[1]}次元）

    どちらも **同じ {xm_img_emb.shape[1]} 次元空間** に埋め込まれている！
    """)
    )
    return xm_images, xm_texts, xm_img_emb, xm_txt_emb


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## コサイン類似度

    2つのベクトル $\mathbf{a}$ と $\mathbf{b}$ のコサイン類似度は：

    $$
    \text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{|\mathbf{a}| \cdot |\mathbf{b}|}
    $$

    L2 正規化済みベクトル（$|\mathbf{a}| = |\mathbf{b}| = 1$）の場合、
    **単純な内積（ドット積）** で計算できる：

    $$
    \text{cosine\_sim}(\mathbf{a}, \mathbf{b}) = \mathbf{a} \cdot \mathbf{b}
    $$
    """)
    return


# --- Part 4: 3種の距離分析 ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 3つの距離を分析しよう

    同じベクトル空間に埋め込まれた画像とテキストの間で、3種類の類似度を見る：

    1. **画像 ↔ テキスト**: 画像とキャプションの対応関係
    2. **画像 ↔ 画像**: 似た内容の画像どうしの距離
    3. **テキスト ↔ テキスト**: 似た意味のキャプションどうしの距離
    """)
    return


@app.cell
def _(
    xm_img_emb,
    xm_txt_emb,
    xm_texts,
    cosine_similarity_matrix,
    plot_similarity_heatmap,
    mo,
):
    # 画像↔テキスト（10x10 サブセット）
    n_sub = 10
    sim_img_txt = cosine_similarity_matrix(xm_img_emb[:n_sub], xm_txt_emb[:n_sub])

    row_labels = [f"画像{_i}" for _i in range(n_sub)]
    col_labels = [_t[:15] + "..." if len(_t) > 15 else _t for _t in xm_texts[:n_sub]]

    fig_it = plot_similarity_heatmap(
        sim_img_txt,
        row_labels=row_labels,
        col_labels=col_labels,
        title="画像 ↔ テキスト 類似度（xm3600）",
    )

    mo.output.append(mo.md("### 画像 ↔ テキスト 類似度"))
    mo.output.append(mo.as_html(fig_it))
    mo.output.append(
        mo.md("対角線（画像 i とキャプション i のペア）が高い値を示すか確認しよう。")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### 画像 ↔ 画像 の類似度

    同じベクトル空間上で画像どうしの距離を見る。
    似た内容の画像（例: 同じ被写体、同じシーン）は近い位置にあるはず。
    """)
    return


@app.cell
def _(xm_img_emb, xm_texts, cosine_similarity_matrix, plot_similarity_heatmap, mo):
    # 画像↔画像（15x15 サブセット）
    n_ii = 15
    sim_img_img = cosine_similarity_matrix(xm_img_emb[:n_ii], xm_img_emb[:n_ii])

    labels_ii = [f"{_i}:{_t[:10]}" for _i, _t in enumerate(xm_texts[:n_ii])]

    fig_ii = plot_similarity_heatmap(
        sim_img_img,
        row_labels=labels_ii,
        col_labels=labels_ii,
        title="画像 ↔ 画像 類似度",
    )

    mo.output.append(mo.as_html(fig_ii))
    mo.output.append(
        mo.md(
            "対角線は自分自身との類似度（= 1.0）。オフダイアゴナルで高い値のペアに注目しよう。"
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### テキスト ↔ テキスト の類似度

    キャプション（テキスト）どうしの距離を見る。
    似た意味のキャプションは、異なる画像のものでも近い位置にあるはず。
    """)
    return


@app.cell
def _(xm_txt_emb, xm_texts, cosine_similarity_matrix, plot_similarity_heatmap, mo):
    # テキスト↔テキスト（15x15 サブセット）
    n_tt = 15
    sim_txt_txt = cosine_similarity_matrix(xm_txt_emb[:n_tt], xm_txt_emb[:n_tt])

    labels_tt = [f"{_i}:{_t[:12]}" for _i, _t in enumerate(xm_texts[:n_tt])]

    fig_tt = plot_similarity_heatmap(
        sim_txt_txt,
        row_labels=labels_tt,
        col_labels=labels_tt,
        title="テキスト ↔ テキスト 類似度",
    )

    mo.output.append(mo.as_html(fig_tt))
    mo.output.append(
        mo.md("意味的に近いキャプションのペアで類似度が高くなっているか確認しよう。")
    )
    return


# --- Part 5: 検索 ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## テキスト → 画像 検索

    日本語のクエリテキストを入力し、
    xm3600 の画像の中から最も類似度の高いものを検索する。
    """)
    return


@app.cell
def _(
    xm_img_emb,
    xm_txt_emb,
    xm_images,
    xm_texts,
    model,
    processor,
    DEVICE,
    encode_texts,
    cosine_similarity_matrix,
    np,
    mo,
):
    # テキスト → 画像検索
    query_text = "動物の写真"
    query_emb = encode_texts(model, processor, [query_text], DEVICE)

    similarities = cosine_similarity_matrix(query_emb, xm_img_emb)[0]
    top_k = 5
    top_indices = np.argsort(similarities)[::-1][:top_k]

    mo.output.append(mo.md(f"### クエリ: 「{query_text}」に最も近い画像 Top-{top_k}"))

    result_items = []
    for rank, idx in enumerate(top_indices):
        result_items.append(
            mo.vstack(
                [
                    mo.image(xm_images[idx], width=180),
                    mo.md(
                        f"**#{rank + 1}** (類似度: {similarities[idx]:.3f})\n\n{xm_texts[idx]}"
                    ),
                ]
            )
        )
    mo.output.append(mo.hstack(result_items))

    # 画像 → テキスト検索
    mo.output.append(mo.md("---"))
    mo.output.append(mo.md("### 画像 → テキスト 検索（画像0 に最も近いキャプション）"))

    sim_i2t = cosine_similarity_matrix(xm_img_emb[:1], xm_txt_emb)[0]
    top_txt_indices = np.argsort(sim_i2t)[::-1][:top_k]

    mo.output.append(mo.image(xm_images[0], width=200))
    for rank, idx in enumerate(top_txt_indices):
        mo.output.append(
            mo.md(f"**#{rank + 1}** (類似度: {sim_i2t[idx]:.3f}): {xm_texts[idx]}")
        )
    return


# --- Part 6: TensorBoardX 可視化 ---


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TensorBoardX による埋め込み可視化

    TensorBoardX の **Embedding Projector** を使うと、
    高次元の埋め込みベクトルを **PCA** や **t-SNE** で 3D/2D に射影し、
    インタラクティブに探索できる。

    - 画像とテキストがどのようにクラスタを形成するか
    - 対応する画像-テキストペアが近くにあるか
    - どんな画像/テキストが似た場所に集まるか

    を視覚的に確認できる。
    """)
    return


@app.cell
def _(
    xm_img_emb, xm_txt_emb, xm_images, xm_texts, export_embeddings_to_tensorboard, mo
):
    import os
    import shutil
    from tensorboardX import SummaryWriter

    log_dir = "runs/siglip2_embeddings"
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    writer = SummaryWriter(log_dir)

    image_labels = [f"[IMG] {_i}: {xm_texts[_i][:20]}" for _i in range(len(xm_images))]
    text_labels = [f"[TXT] {_i}: {xm_texts[_i][:20]}" for _i in range(len(xm_texts))]

    export_embeddings_to_tensorboard(
        writer=writer,
        image_embeddings=xm_img_emb,
        text_embeddings=xm_txt_emb,
        image_labels=image_labels,
        text_labels=text_labels,
        images=xm_images,
        tag="siglip2_multimodal",
    )
    writer.close()

    mo.output.append(
        mo.md(f"""
    ✅ TensorBoardX エクスポート完了

    - 出力先: `{log_dir}/`
    - 画像埋め込み: {len(xm_images)} 件
    - テキスト埋め込み: {len(xm_texts)} 件
    - 合計: {len(xm_images) + len(xm_texts)} 点
    """)
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## TensorBoard の起動方法

    ターミナルで以下を実行:

    ```bash
    uv run tensorboard --logdir=runs/siglip2_embeddings
    ```

    > ⚠️ TensorBoard は Projector のデータを再帰的に検索しないため、
    > `projector_config.pbtxt` があるディレクトリを直接指定する必要がある。

    ブラウザで `http://localhost:6006/#projector` を開く。

    ### 見るべきポイント

    - **PCA / t-SNE** を切り替えて、クラスタの形成を確認
    - `[IMG]` と `[TXT]` のラベルでフィルタリング
    - 対応する画像-テキストペアが近くに配置されているか
    - サムネイル画像をクリックして詳細を確認
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## まとめ

    | トピック | 学んだこと |
    |---------|-----------|
    | **SigLIP2** | Sigmoid 対照学習、多言語対応のデュアルエンコーダ |
    | **xm3600** | 多言語マルチモーダル評価データセット（画像サイズ・キャプション文字数の分布） |
    | **埋め込み** | 画像とテキストを同じベクトル空間にマッピング |
    | **コサイン類似度** | モダリティ間・モダリティ内の距離測定 |
    | **TensorBoardX** | 埋め込み空間の 3D インタラクティブ可視化 |

    ---

    ### ハンズオン課題

    - [ ] クエリテキストを変えて検索結果の変化を観察する
    - [ ] `N_SAMPLES` を増やして、より多くのデータで可視化する
    - [ ] 画像↔画像で最も類似度が高いペアを見つけ、共通点を考察する
    - [ ] 英語のテキストで同じ検索を試し、日本語との結果を比較する
    - [ ] TensorBoard で t-SNE の perplexity を変えてクラスタの変化を観察する
    - [ ] 同じ画像の複数キャプション（`captions` リスト）を使って埋め込みの安定性を検証する
    """)
    return


if __name__ == "__main__":
    app.run()
