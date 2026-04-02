# 05. ReAct Agent — 実装計画

## 目的

ReAct（Reasoning + Acting）パターンを題材に、Agent の概念とループ構造を学ぶ。
04 の Function Calling ベースとは異なり、プロンプトで推論過程を明示的に制御するアプローチを実装する。

## 学習内容

- ReAct パターンの基本構造（Thought → Action → Observation）
- プロンプトによる Agent の推論制御
- Function Calling ベース（04）との比較
- Agent の終了条件の設計

## 技術スタック

- **Google GenAI SDK** (`google-genai`) のみ（llama-index は使わない）

## 依存パッケージ

```toml
dependencies = [
    "marimo>=0.20.2",
    "pyzmq>=27.1.0",
    "google-genai>=1.0.0",
]
```

## notebook.py セル構成

| # | タイプ | 内容 |
|---|--------|------|
| 1 | code | `import marimo as mo` |
| 2 | md | タイトル: ReAct Agent |
| 3 | md | ReAct とは — Thought/Action/Observation の説明、論文の概要 |
| 4 | code | ライブラリ import、Client 初期化 |
| 5 | md | ReAct プロンプトの設計 |
| 6 | code | ReAct 形式のプロンプトテンプレート定義（Thought/Action/Observation の出力フォーマット指定） |
| 7 | md | ツールの定義 |
| 8 | code | Agent 用ツール関数群（検索、計算、Wikipedia 等） |
| 9 | md | ReAct Agent ループの実装 |
| 10 | code | while ループ: LLM にプロンプト送信 → Thought/Action をパース → ツール実行 → Observation を追記 → 繰り返し → Final Answer で終了 |
| 11 | md | 実行と観察 |
| 12 | code | 実際のタスクで Agent を実行し、推論過程を表示 |
| 13 | md | 04（Function Calling）との比較、ReAct の利点・課題 |

## src/ ファイル構成

- `src/__init__.py`
- `src/agent.py` — ReAct プロンプトテンプレート、出力パーサー、ツールレジストリヘルパー

## 04 との差分

| 観点 | 04 (Function Calling) | 05 (ReAct) |
|------|----------------------|-------------|
| ツール呼び出しの判断 | モデルの組み込み機能 | プロンプトで制御 |
| 推論過程 | ブラックボックス | Thought で明示的に出力 |
| ツール定義 | JSON Schema or Python 関数 | プロンプト内にテキスト記述 |
| エラー耐性 | 高い（構造化出力） | パース失敗のリスクあり |

## ReAct プロンプトの例

```
あなたは以下のツールを使ってタスクを解決するアシスタントです。

利用可能なツール:
- search(query): Web検索を行う
- calculate(expression): 数式を計算する

以下のフォーマットで回答してください:

Thought: [何を考えているか]
Action: [ツール名(引数)]
Observation: [ツールの実行結果（システムが埋める）]
... (必要な回数だけ繰り返す)
Thought: [最終的な判断]
Final Answer: [最終回答]
```

## ハンズオン課題

- 新しいツールを追加して ReAct Agent を拡張する
- 推論過程（Thought）の質をプロンプト改善で高める
- 04 の Function Calling 版と同じタスクを実行し、結果を比較する
- Agent が無限ループに陥るケースとその対策を考える
