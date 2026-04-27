"""
ReAct Agent with llama-index - Script Version

ReAct（Reasoning + Acting）パターンを llama-index フレームワークで実装し、
Gemini on Vertex AI で実行します。

このスクリプトは notebook.py と同じ処理を行います。
"""

import asyncio
import os
import urllib.request

import nest_asyncio

nest_asyncio.apply()

from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.tools import FunctionTool, QueryEngineTool, ToolMetadata
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.google_genai import GoogleGenAI


def print_section(title):
    """Print a section header."""
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70 + "\n")


def setup_llm_and_embeddings():
    """Setup LLM and embeddings with Vertex AI."""
    print_section("Setup: LLM and Embeddings")

    vertexai_config = {
        "project": "hr-mixi",
        "location": "asia-northeast1",
    }

    llm = GoogleGenAI(
        model="gemini-2.5-flash",
        vertexai_config=vertexai_config,
        temperature=0.0,
    )

    embed_model = GoogleGenAIEmbedding(
        model_name="text-embedding-004",
        vertexai_config=vertexai_config,
    )

    Settings.llm = llm
    Settings.embed_model = embed_model
    Settings.chunk_size = 512

    print("✓ LLM: gemini-2.5-flash")
    print("✓ Embeddings: text-embedding-004")
    print("✓ Chunk size: 512")

    return llm, embed_model


async def section1_calculator_agent(llm):
    """Section 1: ReAct Agent over Calculator Tools."""
    print_section("Section 1: ReAct Agent over Calculator Tools")

    def multiply(a: int, b: int) -> int:
        """Multiply two integers and returns the result integer"""
        return a * b

    def add(a: int, b: int) -> int:
        """Add two integers and returns the result integer"""
        return a + b

    multiply_tool = FunctionTool.from_defaults(fn=multiply)
    add_tool = FunctionTool.from_defaults(fn=add)

    calc_agent = ReActAgent(tools=[multiply_tool, add_tool], llm=llm, verbose=True)

    print("計算クエリ: What is 20+(2*4)? Calculate step by step\n")
    calc_response = await calc_agent.run("What is 20+(2*4)? Calculate step by step")

    print("\n" + "-" * 70)
    print("計算結果:")
    print(calc_response.response.content)
    print("-" * 70)

    # Show ReAct System Prompt
    print("\n" + "-" * 70)
    print("ReAct System Prompt")
    print("-" * 70)
    print("\nllama-index の ReActAgent が内部で使用しているプロンプトを確認できます。")
    print("このプロンプトが、LLM に Thought/Action/Observation のフォーマットで出力させる指示を与えています。\n")
    print("-" * 70 + " system prompt " + "-" * 70)
    prompt_dict = calc_agent.get_prompts()
    for k, v in prompt_dict.items():
        print(v.template)
        break
    print("-" * 154)


async def section2_rag_agent(llm, embed_model):
    """Section 2: ReAct Agent over QueryEngine (RAG) Tools."""
    print_section("Section 2: ReAct Agent over QueryEngine (RAG) Tools")

    # Download PDFs
    print("データダウンロード中...")
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
            print(f"  Downloaded: {filepath}")
        else:
            print(f"  Already exists: {filepath}")

    # Load documents
    print("\n" + "-" * 70)
    print("PDF から テキストへの変換")
    print("-" * 70)
    print("\nドキュメント読み込み中...")
    lyft_docs = SimpleDirectoryReader(input_files=["./data/10k/lyft_2021.pdf"]).load_data()
    uber_docs = SimpleDirectoryReader(input_files=["./data/10k/uber_2021.pdf"]).load_data()

    lyft_text_size = sum(len(doc.text) for doc in lyft_docs)
    uber_text_size = sum(len(doc.text) for doc in uber_docs)

    print(f"  Lyft: {len(lyft_docs)} pages ({lyft_text_size:,} 文字)")
    print(f"  Uber: {len(uber_docs)} pages ({uber_text_size:,} 文字)")
    print(f"  合計: {len(lyft_docs) + len(uber_docs)} pages ({lyft_text_size + uber_text_size:,} 文字)")
    print("\nSimpleDirectoryReader が pypdf を使って PDF からテキストを抽出します。")
    print("抽出されたテキストは Document オブジェクトとして保存されます。")

    # Build indices
    print("\n" + "-" * 70)
    print("VectorStoreIndex の内部処理")
    print("-" * 70)
    print("\nVectorStoreIndex.from_documents() の 3 ステップ:")
    print("  1. Document → Node 変換（チャンキング）")
    print("     - Settings.chunk_size=512 でテキストを分割")
    print("     - SentenceSplitter で文境界を意識して分割")
    print("  2. Embedding 生成")
    print("     - 各チャンクに対して text-embedding-004 で埋め込みベクトル生成")
    print("  3. VectorStore への保存")
    print("     - SimpleVectorStore（インメモリ）に保存")
    print("\nインデックス構築中...")
    lyft_index = VectorStoreIndex.from_documents(lyft_docs, embed_model=embed_model)
    uber_index = VectorStoreIndex.from_documents(uber_docs, embed_model=embed_model)
    print("  ✓ Lyft index built")
    print("  ✓ Uber index built")

    # Create QueryEngine Tools
    lyft_engine = lyft_index.as_query_engine(similarity_top_k=3)
    uber_engine = uber_index.as_query_engine(similarity_top_k=3)

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

    rag_agent = ReActAgent(tools=query_engine_tools, llm=llm, verbose=True)

    # Query 1: Lyft's revenue growth
    print("\n" + "-" * 70)
    print("Query 1: Lyft の売上成長率")
    print("-" * 70)
    print("Query: What was Lyft's revenue growth in 2021?\n")

    rag_response1 = await rag_agent.run("What was Lyft's revenue growth in 2021?")

    print("\n" + "-" * 70)
    print("Response:")
    print(rag_response1.response.content)
    print("-" * 70)

    # Query 2: Uber vs Lyft comparison
    print("\n" + "-" * 70)
    print("Query 2: Uber vs Lyft 比較分析")
    print("-" * 70)
    print("Query: Compare and contrast the revenue growth of Uber and Lyft in 2021\n")

    rag_response2 = await rag_agent.run(
        "Compare and contrast the revenue growth of Uber and Lyft in 2021, then give an analysis"
    )

    print("\n" + "-" * 70)
    print("Response:")
    print(rag_response2.response.content)
    print("-" * 70)




async def main():
    """Main function."""
    print("\n" + "=" * 70)
    print("ReAct Agent with llama-index")
    print("=" * 70)
    print("\nReAct（Reasoning + Acting）パターンを llama-index フレームワークで実装")
    print("Gemini on Vertex AI で実行します\n")

    # Setup
    llm, embed_model = setup_llm_and_embeddings()

    # Section 1: Calculator Agent
    await section1_calculator_agent(llm)

    # Section 2: RAG Agent
    await section2_rag_agent(llm, embed_model)

    print("\n" + "=" * 70)
    print("完了")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.get_event_loop().run_until_complete(main())
