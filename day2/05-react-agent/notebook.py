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
    5. **VectorStoreIndex の仕組み** — チャンキング・埋め込み・検索の流れ
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

    calc_agent = ReActAgent(tools=[multiply_tool, add_tool], llm=llm, verbose=False)
    return (calc_agent,)


@app.cell
async def _(calc_agent, mo):
    _query = "What is 20+(2*4)? Calculate step by step"
    _handler = calc_agent.run(_query)

    # Collect ReAct reasoning steps
    _steps = []
    _step_num = 1

    async for _event in _handler.stream_events():
        # Capture LLM response (Thought + Action)
        if hasattr(_event, "response") and hasattr(_event.response, "content"):
            _content = _event.response.content
            if _content and ("Thought:" in _content or "Answer:" in _content):
                _steps.append(f"**Step {_step_num}:**\n```\n{_content}\n```")

        # Capture tool execution result (Observation)
        if hasattr(_event, "tool_output"):
            _result = (
                _event.tool_output.blocks[0].text
                if _event.tool_output.blocks
                else str(_event.tool_output.raw_output)
            )
            _steps.append(f"```\nObservation: {_result}\n```")
            _step_num += 1

    calc_response = await _handler

    mo.md(f"""
    ### 計算結果

    **Query:** "{_query}"

    #### ReAct Agent の推論過程

    {chr(10).join(_steps)}

    #### 最終回答

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

    このプロンプトが、LLM に Thought/Action/Observation のフォーマットで出力させる指示を与えています。

    ------------------------- system prompt -------------------------------

    {system_prompt if system_prompt else "No system prompt found"}
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### PDF から テキストへの変換

    #### データサイズ
    - **Lyft 2021 10K**: 238 ページ（約 863,000 文字）
    - **Uber 2021 10K**: 307 ページ（約 1,267,000 文字）
    - **合計**: 545 ページ（約 2,131,000 文字）

    次のセルで、これらの PDF を `SimpleDirectoryReader` でパースし、テキストとして読み込みます。
    内部では `pypdf` と `llama-index-readers-file` を使って PDF からテキストを抽出しています。
    抽出されたテキストは `Document` オブジェクトとして保存され、後続の VectorStoreIndex 構築に使用されます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### VectorStoreIndex の内部処理

    `VectorStoreIndex.from_documents()` は以下の3ステップで動作します：

    #### Step 1: Document → Node 変換（チャンキング）

    - `Settings.chunk_size = 512` で設定したサイズでテキストを分割
    - デフォルトでは `SentenceSplitter` が使われる（文の途中で切らない）
    - `chunk_overlap` でチャンク間のオーバーラップを設定（デフォルト 200 文字）

    #### Step 2: Embedding 生成

    - 各チャンク（ノード）に対して埋め込みベクトルを生成
    - `text-embedding-004` を使用 → 768 次元ベクトル

    #### Step 3: VectorStore への保存

    - デフォルトでは `SimpleVectorStore`（インメモリ）
    - クエリ時に `similarity_top_k=3` で類似チャンクを検索

    #### カスタマイズ可能な設定

    ```python
    # チャンクサイズを変更
    Settings.chunk_size = 256  # 小さく→粒度細かい、文脈狭い

    # カスタムスプリッター
    from llama_index.core.node_parser import SentenceSplitter
    splitter = SentenceSplitter(chunk_size=512, chunk_overlap=50)
    nodes = splitter.get_nodes_from_documents(docs)
    index = VectorStoreIndex(nodes, embed_model=embed_model)

    # 検索時の top_k を変更
    engine = index.as_query_engine(similarity_top_k=5)  # デフォルト 2
    ```

    詳細は README.md の「VectorStoreIndex のチャンキング詳細」セクションを参照してください。
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

    rag_agent = ReActAgent2(tools=query_engine_tools, llm=llm, verbose=False)
    return (rag_agent,)


@app.cell
async def _(mo, rag_agent):
    _query1 = "What was Lyft's revenue growth in 2021?"
    _handler1 = rag_agent.run(_query1)

    # Collect ReAct reasoning steps
    _steps1 = []
    _step_num1 = 1

    async for _event in _handler1.stream_events():
        # Capture LLM response (Thought + Action)
        if hasattr(_event, "response") and hasattr(_event.response, "content"):
            _content = _event.response.content
            if _content and ("Thought:" in _content or "Answer:" in _content):
                _steps1.append(f"**Step {_step_num1}:**\n```\n{_content}\n```")

        # Capture tool execution result (Observation)
        if hasattr(_event, "tool_output"):
            _result = (
                _event.tool_output.blocks[0].text
                if _event.tool_output.blocks
                else str(_event.tool_output.raw_output)
            )
            _steps1.append(f"```\nObservation: {_result}\n```")
            _step_num1 += 1

    rag_response1 = await _handler1

    mo.md(f"""
    ### Query 1: Lyft の売上成長率

    **Query:** "{_query1}"

    #### ReAct Agent の推論過程

    {chr(10).join(_steps1)}

    #### 最終回答

    {rag_response1.response.content}
    """)
    return


@app.cell
async def _(mo, rag_agent):
    _query2 = "Compare and contrast the revenue growth of Uber and Lyft in 2021, then give an analysis"
    _handler2 = rag_agent.run(_query2)

    # Collect ReAct reasoning steps
    _steps2 = []
    _step_num2 = 1

    async for _event in _handler2.stream_events():
        # Capture LLM response (Thought + Action)
        if hasattr(_event, "response") and hasattr(_event.response, "content"):
            _content = _event.response.content
            if _content and ("Thought:" in _content or "Answer:" in _content):
                _steps2.append(f"**Step {_step_num2}:**\n```\n{_content}\n```")

        # Capture tool execution result (Observation)
        if hasattr(_event, "tool_output"):
            _result = (
                _event.tool_output.blocks[0].text
                if _event.tool_output.blocks
                else str(_event.tool_output.raw_output)
            )
            _steps2.append(f"```\nObservation: {_result}\n```")
            _step_num2 += 1

    rag_response2 = await _handler2

    mo.md(f"""
    ### Query 2: Uber vs Lyft 比較分析

    **Query:** "{_query2}"

    #### ReAct Agent の推論過程

    {chr(10).join(_steps2)}

    #### 最終回答

    {rag_response2.response.content}
    """)
    return


if __name__ == "__main__":
    app.run()
