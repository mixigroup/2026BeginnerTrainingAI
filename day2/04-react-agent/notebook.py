import marimo

__generated_with = "0.23.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # ReAct Agent with llama-index

    ReAct（Reasoning + Acting）パターンを llama-index フレームワークで実装し、Gemini on Vertex AI で実行します。

    **参考元**: [Claude Cookbook - Third-party llamaindex react agent](https://platform.claude.com/cookbook/third-party-llamaindex-react-agent)
    上記を Vertex AI 向けに編集（Anthropic Claude → Google Gemini）

    ---

    ### このノートブックでやること

    1. **ReAct パターンの理解** — Thought / Action / Observation のサイクル
    2. **llama-index フレームワークの基本** — エージェント構築の実践的アプローチ
    3. **FunctionTool による計算ツール** — 簡単な関数をツール化
    4. **QueryEngineTool による RAG** — Uber/Lyft 10K 決算書からの情報抽出
    5. **03 との比較** — Function Calling ベース vs ReAct パターン
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ReAct とは

    **ReAct (Reasoning + Acting)** は、LLM が**推論（Thought）**と**行動（Action）**を交互に繰り返しながらタスクを解決するパターンです。

    ### 基本サイクル

    ```
    Thought: [何をすべきか考える]
      ↓
    Action: [ツールを実行する]
      ↓
    Observation: [ツールの結果を確認]
      ↓
    Thought: [次に何をすべきか考える]
      ...（繰り返し）
      ↓
    Answer: [最終回答]
    ```

    ### 論文の概要

    Yao et al. (2022) "ReAct: Synergizing Reasoning and Acting in Language Models"

    - 推論過程を明示的に出力させることで、透明性と解釈性が向上
    - エラー時にどこで失敗したかを追跡可能
    - Few-shot プロンプティングで高い性能を発揮
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## llama-index とは

    **llama-index** は、LLM アプリケーション構築のためのフレームワークです。

    ### 主な機能

    - **データ読み込み** — PDF, Web, DB など多様なソースに対応
    - **インデックス構築** — ベクトルストア、グラフ、キーワード検索など
    - **エージェント** — ReActAgent, OpenAIAgent など複数のエージェントタイプ
    - **ツール** — FunctionTool, QueryEngineTool などツールの抽象化

    ### アーキテクチャ

    ```
    ユーザークエリ
       ↓
    ReActAgent (llama-index)
       ↓
    ツール選択 & 実行
       ├─ FunctionTool (計算など)
       └─ QueryEngineTool (RAG)
           └─ VectorStoreIndex (Gemini Embeddings)
       ↓
    最終回答
    ```
    """)
    return


@app.cell
def _():
    # llama-parse is async-first, running the async code in a notebook requires nest_asyncio
    import nest_asyncio

    nest_asyncio.apply()

    from llama_index.core import Settings
    from llama_index.llms.google_genai import GoogleGenAI
    from llama_index.embeddings.google_genai import GoogleGenAIEmbedding

    return GoogleGenAI, GoogleGenAIEmbedding, Settings


@app.cell
def _(GoogleGenAI, GoogleGenAIEmbedding, Settings):
    _vertexai_config = {
        "project": "hr-mixi",
        "location": "asia-northeast1",
    }

    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        vertexai_config=_vertexai_config,
        temperature=0.0,
    )

    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        vertexai_config=_vertexai_config,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512
    return llm, embed_model


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 1: ReAct Agent over Calculator Tools

    まずは簡単な計算ツール（multiply, add）を使った ReAct Agent を構築します。
    """)
    return


@app.cell
def _():
    from llama_index.core.tools import FunctionTool

    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)
    return add_tool, multiply_tool


@app.cell
def _(add_tool, llm, multiply_tool):
    from llama_index.core.agent.workflow import ReActAgent

    calc_agent = ReActAgent(tools=[multiply_tool, add_tool], llm=llm, verbose=True)
    return (calc_agent,)


@app.cell
async def _(calc_agent, mo):
    calc_response = await calc_agent.run("What is 20+(2*4)? Calculate step by step")

    mo.md(f"""
    ### 計算結果

    **Query:** "What is 20+(2*4)? Calculate step by step"

    **Response:**

    {calc_response.response.content}
    """)
    return


@app.cell
def _(calc_agent, mo):
    prompt_dict = calc_agent.get_prompts()

    system_prompt = None
    for k, v in prompt_dict.items():
        system_prompt = v.template
        break

    mo.md(f"""
    ### ReAct System Prompt

    llama-index の ReActAgent が内部で使用しているプロンプトを確認できます。

    ```
    {system_prompt[:800] if system_prompt else "No system prompt found"}
    ...
    ```

    このプロンプトが、LLM に Thought/Action/Observation のフォーマットで出力させる指示を与えています。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Section 2: ReAct Agent over QueryEngine (RAG) Tools

    次に、Uber/Lyft の 10K 決算書を読み込み、QueryEngine ツールを使った RAG エージェントを構築します。
    """)
    return


@app.cell
def _(mo):
    import urllib.request
    import os

    # Download PDFs
    os.makedirs("data/10k", exist_ok=True)

    files_to_download = [
        (
            "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/uber_2021.pdf",
            "data/10k/uber_2021.pdf",
        ),
        (
            "https://raw.githubusercontent.com/run-llama/llama_index/main/docs/examples/data/10k/lyft_2021.pdf",
            "data/10k/lyft_2021.pdf",
        ),
    ]

    for url, filepath in files_to_download:
        if not os.path.exists(filepath):
            urllib.request.urlretrieve(url, filepath)
            print(f"Downloaded: {filepath}")
        else:
            print(f"Already exists: {filepath}")

    mo.md("""
    ### データダウンロード完了

    Uber/Lyft 2021 年 10K 決算書（PDF）をダウンロードしました。
    """)
    return


@app.cell
def _(embed_model):
    from llama_index.core import SimpleDirectoryReader, VectorStoreIndex

    lyft_docs = SimpleDirectoryReader(
        input_files=["./data/10k/lyft_2021.pdf"]
    ).load_data()
    uber_docs = SimpleDirectoryReader(
        input_files=["./data/10k/uber_2021.pdf"]
    ).load_data()

    lyft_index = VectorStoreIndex.from_documents(lyft_docs, embed_model=embed_model)
    uber_index = VectorStoreIndex.from_documents(uber_docs, embed_model=embed_model)
    return lyft_index, uber_index


@app.cell
def _(lyft_index, mo, uber_index):
    from llama_index.core.tools import QueryEngineTool, ToolMetadata

    # Create QueryEngines
    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)

    # Create QueryEngine Tools
    query_engine_tools = [
        QueryEngineTool(
            query_engine=lyft_engine,
            metadata=ToolMetadata(
                name="lyft_10k",
                description="Provides information about Lyft financials for year 2021. Use a detailed plain text question as input to the tool.",
            ),
        ),
        QueryEngineTool(
            query_engine=uber_engine,
            metadata=ToolMetadata(
                name="uber_10k",
                description="Provides information about Uber financials for year 2021. Use a detailed plain text question as input to the tool.",
            ),
        ),
    ]

    mo.md("""
    ### QueryEngine Tools 作成完了

    - **lyft_10k**: Lyft 2021 年決算情報
    - **uber_10k**: Uber 2021 年決算情報

    各ツールは VectorStoreIndex をベースにした QueryEngine を持ち、similarity_top_k=3 で関連文書を検索します。
    """)
    return (query_engine_tools,)


@app.cell
def _(llm, query_engine_tools):
    from llama_index.core.agent.workflow import ReActAgent as ReActAgent2

    rag_agent = ReActAgent2(tools=query_engine_tools, llm=llm, verbose=True)
    return (rag_agent,)


@app.cell
async def _(mo, rag_agent):
    rag_response1 = await rag_agent.run("What was Lyft's revenue growth in 2021?")

    mo.md(f"""
    ### Query 1: Lyft の売上成長率

    **Query:** "What was Lyft's revenue growth in 2021?"

    **Response:**

    {rag_response1.response.content}
    """)
    return


@app.cell
async def _(mo, rag_agent):
    rag_response2 = await rag_agent.run(
        "Compare and contrast the revenue growth of Uber and Lyft in 2021, then give an analysis"
    )

    mo.md(f"""
    ### Query 2: Uber vs Lyft 比較分析

    **Query:** "Compare and contrast the revenue growth of Uber and Lyft in 2021, then give an analysis"

    **Response:**

    {rag_response2.response.content}
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Comparison: Function Calling (03) vs ReAct (04)

    | 観点 | 03 (Function Calling) | 04 (ReAct with llama-index) |
    |------|----------------------|------------------------------|
    | **ツール呼び出し判断** | モデルの組み込み機能 | プロンプトで制御（llama-index が抽象化） |
    | **推論過程** | ブラックボックス | Thought で明示的に出力 |
    | **ツール定義** | JSON Schema or Python 関数 | FunctionTool でラップ |
    | **エラー耐性** | 高い（構造化出力） | パース失敗のリスク（フレームワークが緩和） |
    | **透明性** | 低い | 高い（思考過程が見える） |
    | **実装複雑度** | シンプル | やや複雑（フレームワーク学習コスト） |
    | **RAG 統合** | 手動実装必要 | QueryEngineTool で簡単 |
    | **スケーラビリティ** | カスタムコードが増える | フレームワークの機能活用 |

    ### ReAct の利点

    1. **透明性**: 推論過程（Thought）が明示的に見える
    2. **デバッグ性**: どこで失敗したか追跡しやすい
    3. **教育的価値**: LLM がどう考えているかを理解できる
    4. **柔軟性**: 複雑なマルチステップタスクに対応しやすい
    5. **フレームワーク統合**: llama-index で RAG やツール管理が容易

    ### ReAct の課題

    1. **トークン消費**: Thought が長くなると context が肥大化
    2. **レイテンシ**: 複数回の LLM 呼び出しが必要
    3. **無限ループリスク**: 終了条件を適切に設計しないと停止しない
    4. **プロンプト依存**: フォーマットが崩れると動作不良
    5. **コスト**: Function Calling より API 呼び出し回数が多い傾向

    ### どちらを選ぶべきか

    - **Function Calling (03)**: シンプルなツール呼び出し、本番環境の高速性重視
    - **ReAct (04)**: 複雑な推論が必要、デバッグ性・透明性重視、RAG 統合が必要

    実践では、タスクの性質に応じて使い分けることが重要です。
    """)
    return


if __name__ == "__main__":
    app.run()
