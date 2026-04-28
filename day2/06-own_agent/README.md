# 06. Own Agent - カスタマイズ用ベースライン ReAct エージェント

参加者が独自のツールやプロンプトを追加してカスタマイズできる ReAct エージェントベースラインです。

## 学習目標

このモジュールを通じて、以下を学びます：

- **エージェントのツール追加** — 独自の機能をツールとして実装
- **LLM パラメータの調整** — temperature や model の変更による出力の違い
- **プロダクション品質のコード** — 型ヒント、docstring、モジュール構造
- **非同期処理** — async/await を使った ReActAgent の実行
- **拡張可能な設計** — カスタマイズしやすいクリーンなアーキテクチャ

## ディレクトリ構成

```
06-own_agent/
├── src/
│   ├── __init__.py         # パッケージエクスポート
│   ├── tools.py            # ツール定義（ベースライン: multiply, add）
│   └── agent.py            # エージェント作成ロジック
├── main.py                 # エントリポイント
├── pyproject.toml          # プロジェクト設定
├── README.md               # このファイル
└── .python-version         # Python 3.12
```

## セットアップ

### 1. ディレクトリ移動

```bash
cd day2/06-own_agent
```

### 2. 依存パッケージのインストール

```bash
uv sync
```

以下のパッケージがインストールされます：

- `google-genai` — Gemini API SDK
- `llama-index-core` — ReActAgent フレームワーク
- `llama-index-llms-google-genai` — Gemini LLM 統合
- `nest-asyncio` — 非同期サポート

## 実行方法

### 基本実行

```bash
uv run main.py
```

**出力例:**
```
Query: What is 20 + (2 * 4)? Calculate step by step.
----------------------------------------------------------------------

Answer: 28
```

## アーキテクチャ

### src/tools.py

ツール定義モジュール。ベースラインとして `multiply` と `add` が含まれています。

**主要な関数:**
- `multiply(a: int, b: int) -> int` — 2つの整数を掛け算
- `add(a: int, b: int) -> int` — 2つの整数を足し算
- `get_tools() -> list[FunctionTool]` — ツールリストを返す

### src/agent.py

エージェント作成と LLM 設定を管理するモジュール。

**主要な関数:**
- `create_llm(model, project, location, temperature) -> LLM` — LLM を作成
- `create_agent(tools, llm, verbose) -> ReActAgent` — エージェントを作成

### main.py

エントリポイント。エージェントを初期化してサンプルクエリを実行します。

## 開発の方針

このエージェントは、以下の3つの方向でカスタマイズできます：

| カスタマイズ項目 | 触るファイル | 効果 |
|---------------|------------|------|
| **Tool開発** | `src/tools.py` | エージェントに新しい機能を追加（計算、API呼び出しなど） |
| **Prompt Tuning** | `main.py` | エージェントの動作や口調を変更 |
| **LLMパラメータ調整** | `main.py` | 出力の性質や使用モデルを変更 |

### 1. Tool開発 — 新しい機能を追加する

**触るファイル**: `src/tools.py`

エージェントに新しい能力を与えるには、ツールを追加します。ツールは「LLMが呼び出せる関数」です。

#### Step 1: 関数を定義する

`src/tools.py` に新しい関数を追加します。**重要なポイント**:
- docstring を必ず書く（LLMがこれを読んでツールの用途を理解します）
- 型ヒントを付ける
- 戻り値は文字列または数値にする

```python
def subtract(a: int, b: int) -> int:
    """2つの整数を引き算して結果を返す。
    
    Args:
        a: 引かれる整数
        b: 引く整数
        
    Returns:
        a - b の結果
    """
    return a - b
```

#### Step 2: ツールをLLMに登録する

`src/tools.py` の `get_tools()` 関数の戻り値リストに追加します。これがツールの登録手続きです。

```python
def get_tools() -> list[FunctionTool]:
    """ツールを作成して返す。
    
    Returns:
        FunctionTool インスタンスのリスト
    """
    return [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
        FunctionTool.from_defaults(fn=subtract),  # ← 追加
    ]
```

`FunctionTool.from_defaults(fn=関数名)` が、関数を LLM が呼び出せるツールに変換します。この時、関数の docstring が自動的にツールの説明として使われます。

#### Step 3: 動作確認

```bash
uv run main.py
```

`main.py` のクエリを変更して新しいツールをテスト:

```python
query = "What is 100 - 42?"  # subtract ツールが使われるはず
```

### 2. Prompt Tuning — エージェントの動作を変更する

**触るファイル**: `main.py`

エージェントの推論方法や口調を変えるには、システムプロンプトをカスタマイズします。

#### プロンプトを確認する

エージェントが使用しているプロンプトを確認できます:

```python
import asyncio
from src.agent import create_agent, create_llm
from src.tools import get_tools

async def main() -> None:
    llm = create_llm()
    tools = get_tools()
    agent = create_agent(tools=tools, llm=llm)
    
    # プロンプトを確認
    prompt_dict = agent.get_prompts()
    for key, prompt in prompt_dict.items():
        print(f"Prompt Key: {key}")
        print(f"Template:\n{prompt.template}")
        print("-" * 70)

if __name__ == "__main__":
    asyncio.run(main())
```

これを実行すると、ReActAgent が使用している Thought/Action/Observation フォーマットの指示が表示されます。

#### カスタムプロンプトを設定する

`agent.update_prompts()` でプロンプトをカスタマイズできます:

```python
import asyncio
from llama_index.core import PromptTemplate
from src.agent import create_agent, create_llm
from src.tools import get_tools

async def main() -> None:
    llm = create_llm()
    tools = get_tools()
    agent = create_agent(tools=tools, llm=llm)
    
    # カスタムプロンプトを作成
    custom_prompt = PromptTemplate(
        """あなたは親切で丁寧な計算アシスタントです。
計算の各ステップを必ず明示的に説明してください。

## 利用可能なツール

{tool_desc}

ツール名: {tool_names}

## 出力フォーマット

```
Thought: [何をすべきか考える]
Action: [ツール名]
Action Input: [JSON形式の入力]
```

ツールの実行結果:
```
Observation: [ツールの結果]
```

十分な情報が得られたら:
```
Thought: もうツールは必要ありません
Answer: [最終回答を日本語で]
```

## 会話履歴

"""
    )
    
    # プロンプトを設定
    agent.update_prompts({"react_header": custom_prompt})
    
    # クエリ実行
    query = "What is 20 + (2 * 4)?"
    handler = agent.run(query)
    response = await handler
    print(f"Answer: {response.response.content}")

if __name__ == "__main__":
    asyncio.run(main())
```

**重要な注意点:**
- プロンプトキーは `"react_header"` を使用
- **必須テンプレート変数**: `{tool_desc}` と `{tool_names}` の2つ
  - `{tool_desc}`: ツールの詳細説明が自動挿入される
  - `{tool_names}`: ツール名のリストが自動挿入される
- これら以外の変数（`{input}` など）は使用できない

**プロンプトで変更できること:**
- エージェントのキャラクター・口調（関西弁、丁寧語など）
- 回答言語（英語 → 日本語）
- 推論の詳細度（簡潔 vs 詳細）
- 出力フォーマット（箇条書き、JSON など）
- 特定のドメイン知識の追加

### 3. LLMパラメータの調整 — 出力の性質を変える

**触るファイル**: `main.py`

#### Temperature を変更する

`temperature` は出力のランダム性を制御します:
- `0.0` = 決定的
- `0.7` = バランス
- `1.0` = 多様

```python
llm = create_llm(
    model="gemini-2.5-flash",
    temperature=0.7,  # デフォルト 0.0 から変更
)
```

#### モデルを変更する

異なる Gemini モデルを試す:

```python
# より高性能なモデル
llm = create_llm(model="gemini-2.5-pro")
```

## 応用例

### エージェントをインタラクティブにする

`main.py` の `main()` 関数を以下のように変更:

```python
import asyncio
from src.agent import create_agent, create_llm
from src.tools import get_tools

async def main() -> None:
    """インタラクティブな ReAct エージェントを実行する。"""
    llm = create_llm()
    tools = get_tools()
    agent = create_agent(tools=tools, llm=llm)

    print("ReAct Agent (type 'quit' to exit)")
    print("-" * 70)

    while True:
        query = input("\nQuery: ")
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        try:
            handler = agent.run(query)
            response = await handler
            print(f"Answer: {response.response.content}")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())
```

実行:
```bash
uv run main.py
```

### 詳細な推論過程を表示する

`verbose=True` に設定:

```python
agent = create_agent(tools=tools, llm=llm, verbose=True)
```

## その他の拡張アイデア

### 1. 計算機能の拡張

```python
import math

def power(a: float, b: float) -> float:
    """a の b 乗を計算する。"""
    return a ** b

def sqrt(a: float) -> float:
    """平方根を計算する。"""
    return math.sqrt(a)

def factorial(n: int) -> int:
    """階乗を計算する。"""
    return math.factorial(n)
```

### 2. 時刻/日付ツール

```python
from datetime import datetime, timedelta

def current_time() -> str:
    """現在時刻を返す。"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def days_until(target_date: str) -> int:
    """指定日までの日数を計算する（YYYY-MM-DD形式）。"""
    target = datetime.strptime(target_date, "%Y-%m-%d")
    delta = target - datetime.now()
    return delta.days
```

### 3. ファイルシステムツール

```python
import os
from pathlib import Path

def list_files(directory: str = ".") -> str:
    """ディレクトリ内のファイルリストを返す。"""
    files = os.listdir(directory)
    return ", ".join(files)

def read_file(filepath: str) -> str:
    """ファイルの内容を読み込む。"""
    return Path(filepath).read_text()
```

### 4. 単位変換ツール

```python
def celsius_to_fahrenheit(celsius: float) -> float:
    """摂氏を華氏に変換する。"""
    return (celsius * 9/5) + 32

def km_to_miles(km: float) -> float:
    """キロメートルをマイルに変換する。"""
    return km * 0.621371
```

## 技術スタック

| コンポーネント | 技術 |
|--------------|------|
| **LLM** | Gemini 2.5 Flash (Vertex AI) |
| **フレームワーク** | llama-index ReActAgent |
| **ツール** | FunctionTool |
| **言語** | Python 3.12 |

## トラブルシューティング

### ImportError: cannot import name 'create_agent'

`src/__init__.py` が正しく配置されているか確認してください。

```bash
ls -la src/
```

### Vertex AI 認証エラー

GCP 認証を確認:
```bash
gcloud auth application-default login
```

プロジェクト ID が正しいか確認:
```python
llm = create_llm(project="your-project-id")
```

### Trouble shoot

`verbose=True` で推論過程を確認し、ツールの戻り値が適切か検証してください。

## 参考リンク

- **05-react-agent** — 教育版の ReAct エージェント（詳細な説明付き）
- **llama-index 公式ドキュメント** — https://docs.llamaindex.ai/
- **ReActAgent リファレンス** — https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/
- **Gemini API** — https://ai.google.dev/gemini-api/docs

## ライセンス

このチュートリアルは教育目的で作成されています。
