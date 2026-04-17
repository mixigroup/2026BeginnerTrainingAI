# 02. LLM API の Stateless 性質

LLM API は本質的に stateless（状態を持たない）であることを理解し、会話履歴を管理する重要性を学ぶ。

- **モデル**: [Gemini 2.5 Flash](https://ai.google.dev/gemini-api/docs/models#gemini-2.5-flash) — Google の高速な大規模言語モデル
- **SDK**: [Google GenAI SDK](https://github.com/googleapis/python-genai) (`google-genai`)
- **認証**: Vertex AI (ADC) — Workbench のサービスアカウント認証を自動使用

---

## 学習内容

1. **Gemini API の基本呼び出し** — `google.genai.Client` でテキスト生成
2. **Stateless の体験** — 2回連続で呼び出し、前の会話を覚えていないことを確認
3. **会話履歴の管理** — `contents` リストで文脈を手動管理
4. **System Instruction** — システムプロンプトで LLM の振る舞いを変更
5. **マルチターン会話** — Chat セッションで履歴管理を自動化

---

## ディレクトリ構造

```
02-llm-api/
├── src/
│   ├── __init__.py
│   └── chat.py        # 会話履歴のフォーマット・表示ヘルパー
├── notebook.py         # marimo ノートブック（インタラクティブ版）
├── pyproject.toml      # 依存パッケージ定義
└── README.md
```

### src/chat.py の主な関数

| 関数 | 役割 |
|------|------|
| `format_contents(contents)` | Content リストを Markdown 形式に整形 |
| `format_chat_history(chat)` | Chat セッションの履歴を Markdown 形式に整形 |

---

## 環境セットアップ

```bash
# 依存パッケージをインストール（初回のみ）
uv sync
```

---

## 実行方法

```bash
uv run marimo edit notebook.py
```

ブラウザが自動的に開きます。上から順にセルを実行してください。

---

## CLI で試す

Workbench 上では ADC（サービスアカウント認証）が自動で使われるため、API キーの設定は不要です。

### 1. Client の初期化

```python
from google import genai
from google.genai.types import Content, GenerateContentConfig, Part

client = genai.Client(
    vertexai=True,
    project="hr-mixi",
    location="asia-northeast1",
)
MODEL_NAME = "gemini-2.5-flash"
```

### 2. 基本的な API 呼び出し

```python
response = client.models.generate_content(
    model=MODEL_NAME,
    contents="日本で一番高い山は何ですか？",
)
print(response.text)
```

### 3. Stateless の体験

```python
# リクエスト 1: 名前を伝える
r1 = client.models.generate_content(
    model=MODEL_NAME,
    contents="私の名前は太郎です。よろしくお願いします。",
)
print("リクエスト1:", r1.text)

# リクエスト 2: 名前を聞く（別リクエスト = 前の情報なし）
r2 = client.models.generate_content(
    model=MODEL_NAME,
    contents="私の名前は何ですか？",
)
print("リクエスト2:", r2.text)
# → LLM は前のリクエストの内容を覚えていない！
```

### 4. 会話履歴の管理

```python
history = [
    Content(role="user", parts=[Part(text="私の名前は太郎です。よろしくお願いします。")]),
    Content(role="model", parts=[Part(text="太郎さん、よろしくお願いします！")]),
    Content(role="user", parts=[Part(text="私の名前は何ですか？")]),
]

response = client.models.generate_content(
    model=MODEL_NAME,
    contents=history,
)
print(response.text)
# → 会話履歴を含めることで文脈を維持できる！
```

### 5. System Instruction

```python
config = GenerateContentConfig(
    system_instruction="あなたは関西弁で話すアシスタントです。すべての返答を関西弁で行ってください。",
)

response = client.models.generate_content(
    model=MODEL_NAME,
    contents="日本で一番高い山は何ですか？",
    config=config,
)
print(response.text)
```

### 6. マルチターン会話（Chat セッション）

```python
chat = client.chats.create(model=MODEL_NAME)

r1 = chat.send_message("私の名前は太郎です。好きな食べ物はラーメンです。")
print("Model:", r1.text)

r2 = chat.send_message("私の名前と好きな食べ物は何ですか？")
print("Model:", r2.text)
# → SDK が自動的に会話履歴を管理してくれる
```

---

## ノートブックの構成

| セクション | 内容 |
|------------|------|
| **Gemini API の概要** | SDK の紹介・Vertex AI セットアップ |
| **最初の API 呼び出し** | `generate_content` で基本的なテキスト生成 |
| **Stateless の体験** | 2回の別リクエストで「覚えていない」ことを確認 |
| **メッセージ履歴の管理** | `contents` リストに会話履歴を手動で蓄積 |
| **System Instruction** | システムプロンプトで関西弁アシスタントを実現 |
| **マルチターン会話** | `client.chats.create()` で履歴管理を自動化 |
| **まとめ** | 学習内容の振り返りとハンズオン課題 |
