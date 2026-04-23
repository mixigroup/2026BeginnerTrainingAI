# 04. ReAct Agent with llama-index

ReAct（Reasoning + Acting）パターンを llama-index フレームワークで実装し、Gemini on Vertex AI で実行します。

**参考元**: [Claude Cookbook - Third-party llamaindex react agent](https://platform.claude.com/cookbook/third-party-llamaindex-react-agent)  
上記を Vertex AI 向けに編集（Anthropic Claude → Google Gemini）

## 学習内容

- **ReActパターンの理解** — Thought → Action → Observation のサイクル
- **llama-indexフレームワークの基本** — エージェント構築の実践的アプローチ
- **FunctionToolによるツール定義** — Python関数を簡単にツール化
- **QueryEngineToolによるRAGエージェント** — 決算書からの情報抽出
- **03（Function Calling）との比較** — 異なるアプローチの長所・短所

## ReActとは

ReAct（Reasoning + Acting）は、LLMが**推論**と**行動**を交互に繰り返しながらタスクを解決するパターンです。

### 基本サイクル

```
Thought: [何をすべきか考える]
  ↓
Action: [ツール名(引数)]
  ↓
Observation: [ツールの実行結果]
  ↓
Thought: [次に何をすべきか考える]
  ...（繰り返し）
  ↓
Answer: [最終回答]
```

### 論文

Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"
- arXiv: https://arxiv.org/abs/2210.03629

## ディレクトリ構成

```
04-react-agent/
├── notebook.py           # Marimo ノートブック（メイン教材）
├── pyproject.toml        # プロジェクト設定・依存関係
├── plan.md               # 実装計画
├── README.md             # このファイル
├── .python-version       # Python 3.12
└── src/
    ├── __init__.py
    └── agent.py          # ヘルパー関数（ツール定義、フォーマッタなど）
```

## セットアップ

### 1. ディレクトリ移動

```bash
cd day2/04-react-agent
```

### 2. 依存パッケージのインストール

```bash
uv sync
```

以下のパッケージがインストールされます：

- `marimo` — インタラクティブノートブック
- `google-genai` — Gemini API SDK
- `llama-index-core` — llama-indexコアライブラリ
- `llama-index-llms-gemini` — Gemini LLM統合
- `llama-index-embeddings-gemini` — Gemini Embeddings統合

## 実行

### Marimo ノートブックを起動

```bash
uv run marimo edit notebook.py
```

ブラウザで `http://localhost:2718` が自動的に開きます。

### セル実行

1. 上から順にセルを実行
2. Section 1（計算ツール）は数秒で完了
3. Section 2（RAG）は初回実行時にPDFダウンロード＋インデックス構築で数分かかります

## データ

### 自動ダウンロード

Uber/Lyft 2021年 10K SEC filings（決算報告書）は、ノートブック内で自動的にダウンロードされます。

- **Uber 2021 10K**: `data/10k/uber_2021.pdf` (~1.8MB)
- **Lyft 2021 10K**: `data/10k/lyft_2021.pdf` (~1.4MB)

### データソース

GitHub: llama_index公式サンプルデータ
- https://github.com/run-llama/llama_index/tree/main/docs/examples/data/10k

## 技術スタック

| コンポーネント | 技術 |
|--------------|------|
| **ノートブック** | Marimo (Python-based) |
| **LLM** | Gemini 2.5 Flash (Vertex AI) |
| **Embeddings** | Gemini text-embedding-004 |
| **Agentフレームワーク** | llama-index ReActAgent |
| **ツール** | FunctionTool, QueryEngineTool |
| **ベクトルストア** | VectorStoreIndex (in-memory) |

## 実装例

### 計算ツールの定義

```python
from llama_index.core.tools import FunctionTool

def multiply(a: int, b: int) -> int:
    """Multiply two integers and returns the result integer"""
    return a * b

multiply_tool = FunctionTool.from_defaults(fn=multiply)
```

### ReActエージェントの作成

```python
from llama_index.core.agent import ReActAgent

agent = ReActAgent.from_tools(
    [multiply_tool, add_tool],
    llm=llm,
    verbose=True
)

response = agent.chat("What is 20+(2*4)?")
```

### QueryEngineツールの作成

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.tools import QueryEngineTool, ToolMetadata

docs = SimpleDirectoryReader(input_files=["data.pdf"]).load_data()
index = VectorStoreIndex.from_documents(docs)
engine = index.as_query_engine(similarity_top_k=3)

tool = QueryEngineTool(
    query_engine=engine,
    metadata=ToolMetadata(
        name="data_query",
        description="Provides information from data.pdf"
    )
)
```

## ハンズオン課題

### 基礎

- [ ] 新しい計算ツール（subtract, divide）を追加し、四則演算エージェントを作成
- [ ] `verbose=True` の出力を観察し、Thought/Action/Observation の流れを理解
- [ ] `agent.get_prompts()` で ReAct system prompt を確認

### 応用

- [ ] QueryEngineの `similarity_top_k` を 1, 3, 5 に変更して精度を比較
- [ ] 03（Function Calling）と同じタスクを実行し、Thought の有無を比較
- [ ] Wikipedia検索ツールなど、外部APIを呼ぶツールを追加

### 発展

- [ ] `agent.reset()` でチャット履歴をクリアし、マルチターン会話の挙動を確認
- [ ] エージェントが無限ループに陥るケースを見つけ、対策を考える
- [ ] カスタムプロンプトで Thought の詳細度を調整（`agent.update_prompts()`）
- [ ] ストリーミングレスポンス（`agent.stream_chat()`）を試す

## トラブルシューティング

### PDFダウンロード失敗

```python
# 手動ダウンロード
!wget https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf -O data/10k/uber_2021.pdf
```

### Embedding API エラー

Gemini Embedding API のレート制限に達した場合：

```python
# Settings.chunk_size を大きくして埋め込み回数を削減
Settings.chunk_size = 1024
```

### メモリ不足

大きなPDFでインデックス構築時にメモリ不足になる場合：

```python
# ページ数を制限して読み込み
from llama_index.core import Document

docs = SimpleDirectoryReader(input_files=["data.pdf"]).load_data()
docs = docs[:100]  # 最初の100ページのみ
```

## 03（Function Calling）との比較

| 観点 | 03 (Function Calling) | 04 (ReAct) |
|------|----------------------|------------|
| ツール呼び出し判断 | モデルの組み込み機能 | プロンプトで制御 |
| 推論過程 | ブラックボックス | Thought で明示的 |
| 実装複雑度 | シンプル | やや複雑 |
| デバッグ性 | 低い | 高い（思考が見える） |
| RAG統合 | 手動実装 | QueryEngineTool で簡単 |
| レイテンシ | 速い | やや遅い（複数回呼び出し） |

## 参考リンク

- **llama-index 公式ドキュメント**: https://docs.llamaindex.ai/
- **ReActAgent リファレンス**: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/
- **Gemini API**: https://ai.google.dev/gemini-api/docs
- **Vertex AI**: https://cloud.google.com/vertex-ai/docs

## ライセンス

このチュートリアルは教育目的で作成されています。
Uber/Lyft 10K データは SEC（米国証券取引委員会）公開データです。
