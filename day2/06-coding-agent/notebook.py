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
    # Plan-Solve Coding Agent

    Plan（計画）と Solve（実行）を分離したアーキテクチャで、バグ修正エージェントを作成する。
    LangGraph を使用して状態管理とワークフローを実装する。

    ---

    ### このノートブックでやること

    1. **Plan-Solve アーキテクチャ** — 計画と実行の分離設計
    2. **State 定義** — LangGraph の TypedDict による状態管理
    3. **ノード実装** — Plan / Solve / Verify の各ノード
    4. **グラフ構築** — StateGraph でノードとエッジを接続
    5. **Agent 実行** — サンプルバグレポートで動作確認
    """)
    return


if __name__ == "__main__":
    app.run()
