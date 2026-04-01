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
    # LLM API の Stateless 性質理解

    LLM API は本質的に stateless（状態を持たない）であることを理解し、
    会話履歴を管理する重要性を学ぶ。Gemini API（Google GenAI SDK）を使用する。

    ---

    ### このノートブックでやること

    1. **Gemini API の基本呼び出し** — `google.genai.Client` でテキスト生成
    2. **Stateless の体験** — 2回連続で呼び出し、前の会話を覚えていないことを確認
    3. **会話履歴の管理** — `contents` リストで文脈を手動管理
    4. **System Instruction** — システムプロンプトで LLM の振る舞いを変更
    5. **マルチターン会話** — Chat セッションの実装
    """)
    return


if __name__ == "__main__":
    app.run()
