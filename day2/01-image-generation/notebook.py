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
    # Stable Diffusion 画像生成ハンズオン

    拡散モデル（Diffusion Model）の基本的な仕組みを理解し、
    Stable Diffusion を使って実際に画像を生成する体験を通じて、生成 AI の動作を学ぶ。

    ---

    ### このノートブックでやること

    1. **Forward Process（拡散過程）** — 画像にノイズを段階的に加える過程を可視化
    2. **Reverse Process（逆拡散過程）** — ノイズから画像を復元する過程を理解
    3. **Stable Diffusion パイプライン** — 実際に画像を生成
    4. **プロンプトエンジニアリング** — プロンプトやパラメータの影響を観察
    """)
    return


if __name__ == "__main__":
    app.run()
