# 03. LLM API の Stateless 性質 — 実装計画

## 目的

LLM API は本質的に stateless（状態を持たない）であることを理解し、会話履歴を管理する重要性を学ぶ。

## 学習内容

- LLM API の stateless な特性
- messages パラメータによる文脈の管理
- System Instruction の役割
- マルチターン会話の実装

## 技術スタック

- **Google GenAI SDK** (`google-genai`) — Gemini API を呼び出す
- OpenAI SDK は使用しない

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
| 2 | md | タイトル: LLM API の Stateless 性質理解 |
| 3 | md | Gemini API の概要・セットアップ説明 |
| 4 | code | `google.genai.Client` の初期化（環境変数 `GOOGLE_API_KEY` から取得） |
| 5 | md | 最初の API 呼び出し |
| 6 | code | `client.models.generate_content(model="gemini-2.0-flash", contents="...")` で基本呼び出し |
| 7 | md | Stateless の体験（名前を覚えていない） |
| 8 | code | ケース1: 「名前は太郎」→「名前は？」を別々のリクエストで送る → 覚えていない |
| 9 | md | メッセージ履歴の管理 |
| 10 | code | ケース2: `contents` リストに会話履歴を蓄積して送る → 覚えている |
| 11 | md | System Instruction の効果 |
| 12 | code | `config=GenerateContentConfig(system_instruction="...")` で振る舞い変更 |
| 13 | md | マルチターン会話 |
| 14 | code | `client.chats.create()` で Chat セッションを使った会話実装 |
| 15 | md | まとめ |

## src/ ファイル構成

- `src/__init__.py`
- `src/chat.py` — 会話履歴のフォーマット・表示ヘルパー関数

## Google GenAI SDK の主要 API

```python
from google import genai
from google.genai.types import GenerateContentConfig, Content, Part

client = genai.Client(api_key="...")

# 基本呼び出し
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="こんにちは"
)
print(response.text)

# System Instruction 付き
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="...",
    config=GenerateContentConfig(
        system_instruction="あなたは関西弁で話すアシスタントです"
    )
)

# マルチターン Chat
chat = client.chats.create(model="gemini-2.0-flash")
response = chat.send_message("私の名前は太郎です")
response = chat.send_message("私の名前は何ですか？")
```

## ハンズオン課題

- 長い会話を続けて context window の制限を体験する
- System Instruction で LLM の振る舞いを変えてみる
- 会話履歴の要約戦略を考える
