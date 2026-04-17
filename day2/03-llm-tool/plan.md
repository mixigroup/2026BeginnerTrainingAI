# 03. Agentic Loop と Function Calling — 実装計画

## 目的

フレームワークの `create_agent` を使わずに、while 文で自作の Agentic Loop を実装する。
Gemini API の Function Calling 機能を使い、LLM がツールを呼び出す仕組みを体験する。

## 学習内容

- Function Calling の仕組み
- ツールの定義と実行
- Agentic Loop の基本構造（Reasoning → Action → Observation）
- 複数ツールの連鎖実行

## 技術スタック

- **Google GenAI SDK** (`google-genai`) — Function Calling 対応

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
| 2 | md | タイトル: Agentic Loop と Function Calling |
| 3 | md | Function Calling の概要（LLM がツールの呼び出しを判断する仕組み） |
| 4 | code | ライブラリ import、Client 初期化 |
| 5 | md | ツール関数の定義 |
| 6 | code | `get_weather(location)`, `search_web(query)` 等のモック関数定義 |
| 7 | md | ツールをモデルに渡す |
| 8 | code | `generate_content(tools=[get_weather, search_web])` で Python 関数を直接渡す |
| 9 | md | Function Call レスポンスの解析 |
| 10 | code | レスポンスの `function_call` を取得し、ツール実行、結果を返送 |
| 11 | md | Agentic Loop の実装 |
| 12 | code | **while ループ**で繰り返し: LLM呼び出し → function_call チェック → ツール実行 → 結果返送 → テキスト応答で終了 |
| 13 | md | 複数ツールの連鎖 |
| 14 | code | 「東京の天気を調べて、その情報をもとにおすすめの服装を教えて」等の複合タスク |
| 15 | md | まとめ |

## src/ ファイル構成

- `src/__init__.py`
- `src/tools.py` — モックツール関数の定義集（天気取得、Web検索、計算機など）

## Google GenAI SDK の Function Calling API

```python
from google import genai

client = genai.Client(api_key="...")

def get_weather(location: str) -> str:
    """指定された場所の天気情報を取得する"""
    return f"{location}の天気: 晴れ、気温25度"

# Python 関数を直接 tools に渡せる（自動スキーマ変換）
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="東京の天気を教えて",
    config=GenerateContentConfig(
        tools=[get_weather],
    )
)

# function_call の取得
for part in response.candidates[0].content.parts:
    if part.function_call:
        name = part.function_call.name
        args = part.function_call.args
```

## ハンズオン課題

- 新しいツールを追加してみる（例: 計算機、ファイル読み込み）
- エラーハンドリングを追加する
- ツールの実行結果をログに記録する
- 複数のツールを組み合わせた複雑なタスクを実行する
