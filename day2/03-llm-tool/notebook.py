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
    # Agentic Loop と Function Calling

    フレームワークの `create_agent` を使わずに、while 文で自作の Agentic Loop を実装する。
    Gemini API の Function Calling 機能を使い、LLM がツールを呼び出す仕組みを体験する。

    ---

    ### このノートブックでやること

    1. **Function Calling の基礎** — ツール関数の定義とモデルへの受け渡し
    2. **レスポンス解析** — `function_call` の抽出とツール実行
    3. **Agentic Loop** — while ループで Reasoning → Action → Observation を繰り返す
    4. **複数ツールの連鎖** — 複数のツールを組み合わせたタスク実行
    """)
    return


if __name__ == "__main__":
    app.run()
