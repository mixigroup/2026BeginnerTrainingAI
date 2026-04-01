# 06. Plan-Solve Coding Agent — 実装計画

## 目的

Plan（計画）と Solve（実行）を分離したアーキテクチャで、バグ修正エージェントを作成する。
LangGraph を使用して状態管理とワークフローを実装する。

## 学習内容

- Plan-Solve アーキテクチャの設計思想
- LangGraph による状態管理（StateGraph）
- 条件分岐によるリトライ制御
- コード生成・修正の自動化

## 技術スタック

- **Google GenAI SDK** (`google-genai`) — LLM 呼び出し
- **LangGraph** — ワークフロー・状態管理
- **langchain-google-genai** — LangGraph と Gemini のブリッジ

## 依存パッケージ

```toml
dependencies = [
    "marimo>=0.20.2",
    "pyzmq>=27.1.0",
    "google-genai>=1.0.0",
    "langgraph>=0.4.0",
    "langchain-google-genai>=2.1.0",
    "pydantic>=2.0",
]
```

## アーキテクチャ

```
User Input (Bug Report)
    ↓
┌─────────────────┐
│  Plan Node      │ ← コードを読んで問題を分析、修正計画を立てる
└─────────────────┘
    ↓
┌─────────────────┐
│  Solve Node     │ ← 計画に基づいてコードを修正
└─────────────────┘
    ↓
┌─────────────────┐
│  Verify Node    │ ← 修正を検証（テスト実行）
└─────────────────┘
    ↓
  Success? ──No──→ Plan Node (再計画)
    │
   Yes
    ↓
  完了
```

## notebook.py セル構成

| # | タイプ | 内容 |
|---|--------|------|
| 1 | code | `import marimo as mo` |
| 2 | md | タイトル: Plan-Solve Coding Agent |
| 3 | md | Plan-Solve アーキテクチャの説明（上記の図を含む） |
| 4 | code | ライブラリ import（langgraph, langchain_google_genai 等） |
| 5 | md | State の定義 |
| 6 | code | `TypedDict` で `AgentState` 定義（bug_report, code, plan, solution, test_result, iteration） |
| 7 | md | Plan ノードの説明 |
| 8 | code | `plan_node(state)` — バグレポートとコードを分析し、修正計画を LLM で生成 |
| 9 | md | Solve ノードの説明 |
| 10 | code | `solve_node(state)` — 計画に基づいてコード修正を LLM で生成 |
| 11 | md | Verify ノードの説明 |
| 12 | code | `verify_node(state)` — 修正コードをテスト実行して結果を返す |
| 13 | md | グラフの構築 |
| 14 | code | `StateGraph` でノード追加、エッジ定義、条件分岐（`should_retry`） |
| 15 | md | Agent の実行 |
| 16 | code | サンプルバグレポートで `app.invoke()` を実行、結果表示 |
| 17 | md | まとめ |

## src/ ファイル構成

- `src/__init__.py`
- `src/state.py` — `AgentState` の TypedDict 定義
- `src/nodes.py` — `plan_node`, `solve_node`, `verify_node` の実装
- `src/tools.py` — コード実行ツール（`subprocess` による安全な実行）、ファイル読み込みツール

## LangGraph + Gemini の組み合わせ例

```python
from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from typing import TypedDict

llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")

class AgentState(TypedDict):
    bug_report: str
    code: str
    plan: str
    solution: str
    test_result: str
    iteration: int

def plan_node(state: AgentState) -> dict:
    prompt = f"バグレポート: {state['bug_report']}\nコード: {state['code']}\n修正計画を立ててください。"
    response = llm.invoke(prompt)
    return {"plan": response.content}

workflow = StateGraph(AgentState)
workflow.add_node("plan", plan_node)
# ... ノード・エッジ追加
app = workflow.compile()
```

## ハンズオン課題

- より詳細なテストノードを実装する
- コードレビューノードを追加する
- 複数ファイルにまたがる修正に対応させる
- 成功率の分析を行う
