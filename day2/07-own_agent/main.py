"""ReAct エージェントのメインエントリポイント。

実行方法: uv run main.py
"""

from __future__ import annotations

import asyncio

import nest_asyncio

nest_asyncio.apply()

from src.agent import create_agent, create_llm  # noqa: E402
from src.tools import get_tools  # noqa: E402


async def main() -> None:
    """サンプルクエリで ReAct エージェントを実行する。"""
    # LLM を初期化
    llm = create_llm()

    # ツールを取得
    tools = get_tools()

    # エージェントを作成
    agent = create_agent(tools=tools, llm=llm, verbose=False)

    # サンプルクエリを実行
    query = "What is 20 + (2 * 4)? Calculate step by step."
    print(f"Query: {query}")
    print("-" * 70)

    # エージェントを実行
    handler = agent.run(query)
    response = await handler

    # 結果を出力
    print(f"\nAnswer: {response.response.content}")


if __name__ == "__main__":
    asyncio.run(main())
