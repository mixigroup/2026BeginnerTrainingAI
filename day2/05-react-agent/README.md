# 05. ReAct Agent with llama-index

ReAct（Reasoning + Acting）パターンを llama-index フレームワークで実装し、Gemini on Vertex AI で実行します。

**参考元**: [Claude Cookbook - Third-party llamaindex react agent](https://platform.claude.com/cookbook/third-party-llamaindex-react-agent)  
上記を Vertex AI 向けに編集（Anthropic Claude → Google Gemini）

## 学習内容

- **ReActパターンの理解** — Thought → Action → Observation のサイクル
- **llama-indexフレームワークの基本** — エージェント構築の実践的アプローチ
- **FunctionToolによるツール定義** — Python関数を簡単にツール化
- **QueryEngineToolによるRAGエージェント** — 決算書からの情報抽出
- **VectorStoreIndexの仕組み** — チャンキング・埋め込み・検索の流れ

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
05-react-agent/
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
cd day2/05-react-agent
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

## ハンズオン課題

### 基礎

- [ ] 新しい計算ツール（subtract, divide）を追加し、四則演算エージェントを作成
- [ ] `verbose=True` の出力を観察し、Thought/Action/Observation の流れを理解
- [ ] `agent.get_prompts()` で ReAct system prompt を確認

### 応用

- [ ] QueryEngineの `similarity_top_k` を 1, 3, 5 に変更して精度を比較
- [ ] `Settings.chunk_size` を変更してチャンク分割の粒度を調整し、検索精度への影響を確認
- [ ] Wikipedia検索ツールなど、外部APIを呼ぶツールを追加

### 発展

- [ ] `agent.reset()` でチャット履歴をクリアし、マルチターン会話の挙動を確認
- [ ] エージェントが無限ループに陥るケースを見つけ、対策を考える
- [ ] カスタムプロンプトで Thought の詳細度を調整（`agent.update_prompts()`）

## VectorStoreIndex のチャンキング詳細

### 内部処理の3ステップ

`VectorStoreIndex.from_documents()` は以下の処理を行います：

#### 1. Document → Node 変換（チャンキング）

テキストを小さな「チャンク」に分割します。

```python
# デフォルト設定
Settings.chunk_size = 512      # 512文字ごとに分割
Settings.chunk_overlap = 200   # 前後200文字オーバーラップ
```

**使用されるスプリッター:**
- デフォルト: `SentenceSplitter` — 文の途中で切らない
- 代替: `TokenTextSplitter` — トークン数ベースで分割

**チャンクサイズの影響:**
- **小さい（256）**: 粒度細かい、文脈狭い、検索精度高い、埋め込み回数多い
- **大きい（1024）**: 粒度粗い、文脈広い、検索精度やや下がる、埋め込み回数少ない

#### 2. Embedding 生成

各チャンクを埋め込みベクトルに変換します。

```python
embed_model = GoogleGenAIEmbedding(
    model_name="text-embedding-004",  # 768次元ベクトル
    vertexai_config={"project": "...", "location": "..."}
)
```

**545ページのPDFの場合:**
- 約2,130,000文字
- chunk_size=512 → 約6,200回のEmbedding API呼び出し
- chunk_size=1024 → 約3,100回（半減）

#### 3. VectorStore への保存

ベクトルをインメモリストアに保存します。

```python
# デフォルトは SimpleVectorStore（インメモリ）
index = VectorStoreIndex.from_documents(docs, embed_model=embed_model)

# クエリ時に類似チャンクを検索
engine = index.as_query_engine(similarity_top_k=3)  # 上位3チャンクを取得
```

### カスタマイズ例

#### チャンクサイズを変更

```python
from llama_index.core import Settings

Settings.chunk_size = 256       # 小さく→精度向上
Settings.chunk_overlap = 50     # オーバーラップも調整
```

#### カスタムスプリッターを使用

```python
from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter

# 文ベース
splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
nodes = splitter.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes, embed_model=embed_model)

# トークンベース
token_splitter = TokenTextSplitter(chunk_size=128, chunk_overlap=20)
nodes = token_splitter.get_nodes_from_documents(docs)
index = VectorStoreIndex(nodes, embed_model=embed_model)
```

#### 検索時のtop_kを変更

```python
# より多くのチャンクを取得
engine = index.as_query_engine(similarity_top_k=5)

# 少なくして高速化
engine = index.as_query_engine(similarity_top_k=1)
```

### パラメータのトレードオフ

| 設定 | 小さい値 | 大きい値 |
|------|---------|---------|
| **chunk_size** | 粒度細、文脈狭、API多 | 粒度粗、文脈広、API少 |
| **chunk_overlap** | 境界で情報欠損リスク | 冗長性高、ノード数増 |
| **similarity_top_k** | 高速、見落としリスク | 精度高、遅延増 |

### 推奨設定（10K決算書のような長文）

```python
Settings.chunk_size = 512
Settings.chunk_overlap = 50
engine = index.as_query_engine(similarity_top_k=3)
```

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

## 参考リンク

- **llama-index 公式ドキュメント**: https://docs.llamaindex.ai/
- **ReActAgent リファレンス**: https://docs.llamaindex.ai/en/stable/module_guides/deploying/agents/
- **Gemini API**: https://ai.google.dev/gemini-api/docs
- **Vertex AI**: https://cloud.google.com/vertex-ai/docs

## ライセンス

このチュートリアルは教育目的で作成されています。
Uber/Lyft 10K データは SEC（米国証券取引委員会）公開データです。
