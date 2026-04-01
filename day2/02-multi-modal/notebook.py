import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # CLIP — テキストと画像の関係理解

    CLIP（Contrastive Language-Image Pre-training）を使って、
    テキストと画像を同じ特徴空間にエンコードし、両者の類似度を計算することで、
    マルチモーダルモデルの仕組みを理解する。

    ---

    ### このノートブックでやること

    1. **Contrastive Learning** — CLIP の学習方法を理解
    2. **Embedding 取得** — 画像・テキストの特徴ベクトルを取得
    3. **コサイン類似度** — テキストと画像の関連度を定量化
    4. **Zero-shot 分類** — CLIP による画像分類を体験
    """)
    return


if __name__ == "__main__":
    app.run()
