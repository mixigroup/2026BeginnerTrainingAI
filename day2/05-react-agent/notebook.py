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
    # ReAct Agent

    ReAct（Reasoning + Acting）パターンを題材に、Agent の概念とループ構造を学ぶ。
    Gemini API を使い、Thought → Action → Observation のサイクルを手動で実装する。

    ---

    ### このノートブックでやること

    1. **ReAct パターンの理解** — Thought / Action / Observation の役割
    2. **ReAct プロンプトの設計** — LLM に推論過程を明示的に出力させる
    3. **ReAct Agent の実装** — while ループで Agent ループを構築
    4. **04 との比較** — Function Calling ベース vs プロンプトベースの違い
    """)
    return


if __name__ == "__main__":
    app.run()
