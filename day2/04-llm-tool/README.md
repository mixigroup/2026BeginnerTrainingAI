# 04. Function Calling

LLM がツールを呼び出す仕組みを体験する。フレームワークを使わずに while 文でツール呼び出しのループを自作する。

- **モデル**: [Gemini 3 Flash](https://ai.google.dev/gemini-api/docs/models) — Google の最新高速モデル
- **SDK**: [Google GenAI SDK](https://github.com/googleapis/python-genai) (`google-genai`)
- **認証**: Vertex AI (ADC) — Workbench のサービスアカウント認証を自動使用

---

## 学習内容

1. **Function Calling の基礎** — Python 関数をツールとして渡し、自動生成スキーマを確認する
2. **1往復の実行** — Function Call → ツール実行 → 結果返送 → 最終回答
3. **ループの実装** — while ループでツール呼び出しを繰り返す
4. **並列 Function Calling** — 複数ツールを1ターンで同時呼び出し

---

## ディレクトリ構造

```
04-llm-tool/
├── notebook.py         # marimo ノートブック（インタラクティブ版）
├── pyproject.toml      # 依存パッケージ定義
└── README.md
```

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

## Function Calling とは

LLM 単体では「現在の天気」「リアルタイムの情報」を知ることができません。
**Function Calling** は、LLM が「どのツールをどの引数で呼ぶべきか」を判断し、
その結果を受け取って最終回答を生成する仕組みです。

```
ユーザー: 「大阪の天気は？」
       ↓
LLM:  「get_weather("大阪") を呼んでください」  ← Function Call
       ↓
アプリ: get_weather("大阪") を実行 → {"weather": "曇り", ...}
       ↓
LLM:  「大阪は曇りで、気温22度です」            ← 最終回答
```

**重要**: LLM は実際にツールを実行しません。どのツールを呼ぶかを指示するだけです。実行はアプリケーション側が行います。

### ユースケース

- **データ取得** — LLM が持てないリアルタイム情報を外部から取得する（例: 現在の天気、通貨換算、社内 DB の検索）
- **アクション実行** — 外部システムを操作する（例: フォーム送信、アプリ状態の更新、メール送信）

---

## CLI で試す

### 1. Client の初期化

```python
from google import genai
from google.genai import types

client = genai.Client(
    vertexai=True,
    project="hr-mixi",
    location="global",
)
MODEL_NAME = "gemini-3-flash-preview"
```

### 2. ツール関数の定義

```python
def get_weather(location: str) -> dict:
    """指定された都市の現在の天気情報を取得する。"""
    data = {"東京": {"weather": "晴れ", "temperature_celsius": 25}}
    return {"location": location, **data.get(location, {})}
```

### 3. Python 関数を直接渡す（自動スキーマ生成）

```python
import json

# SDK が自動生成するスキーマを確認
decl = types.FunctionDeclaration.from_callable(callable=get_weather, client=client)
print(json.dumps(decl.model_dump(exclude_none=True), ensure_ascii=False, indent=2))

# ツールを渡して呼び出し
response = client.models.generate_content(
    model=MODEL_NAME,
    contents="大阪の天気は？",
    config=types.GenerateContentConfig(
        tools=[get_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    ),
)
fc = response.function_calls[0]
print(f"name: {fc.name}, args: {dict(fc.args)}")
```

### 4. 手動1往復

```python
user_content = types.Content(role="user", parts=[types.Part.from_text(text="大阪の天気は？")])

# ① generate_content → function_call が返る
response1 = client.models.generate_content(
    model=MODEL_NAME,
    contents=[user_content],
    config=types.GenerateContentConfig(
        tools=[get_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    ),
)
fc = response1.function_calls[0]

# ② ツールを実行
result = get_weather(**dict(fc.args))

# ③ FunctionResponse を返送 → 最終回答
fn_response_content = types.Content(
    role="tool",
    parts=[types.Part.from_function_response(name=fc.name, response={"result": result})],
)
response2 = client.models.generate_content(
    model=MODEL_NAME,
    contents=[user_content, response1.candidates[0].content, fn_response_content],
    config=types.GenerateContentConfig(
        tools=[get_weather],
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    ),
)
print(response2.text)
```

### 5. while ループ（ツール連鎖）

```python
def get_current_location() -> dict:
    """ユーザーの現在地を取得する。"""
    return {"city": "東京"}

TOOL_MAP = {"get_current_location": get_current_location, "get_weather": get_weather}

contents = [types.Content(role="user", parts=[types.Part.from_text(text="現在地の天気を教えて")])]
config = types.GenerateContentConfig(
    tools=[get_current_location, get_weather],
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)

while True:
    response = client.models.generate_content(model=MODEL_NAME, contents=contents, config=config)
    if not response.function_calls:
        print(response.text)
        break
    fn_parts = []
    for fc in response.function_calls:
        fn = TOOL_MAP[fc.name]
        result = fn(**dict(fc.args))
        print(f"  {fc.name}({dict(fc.args)}) → {result}")
        fn_parts.append(types.Part.from_function_response(name=fc.name, response={"result": result}))
    contents.append(response.candidates[0].content)
    contents.append(types.Content(role="tool", parts=fn_parts))
```

---

## ノートブックの構成

| セクション | 内容 |
|------------|------|
| **Function Calling とは** | 仕組みの概要・ユースケース |
| **ツール関数の定義** | `get_current_location`, `get_weather` のモック実装 |
| **Python 関数を直接渡す** | 自動スキーマ生成の確認 + 1往復の実行 |
| **ループの実装** | while ループで `get_current_location` → `get_weather` を連鎖 |
| **並列 Function Calling** | 1ターンで複数の Function Call を同時処理 |
| **まとめ** | 学習内容の振り返りとハンズオン課題 |

---

## ハンズオン課題

- [ ] 新しくツールを作って Function Calling を試してみよう
