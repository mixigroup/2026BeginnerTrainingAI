"""エージェントの作成と設定。

このモジュールは、設定可能な LLM 設定で ReActAgent の初期化を処理します。
"""

from __future__ import annotations

from llama_index.core import Settings
from llama_index.core.agent.workflow import ReActAgent
from llama_index.core.llms import LLM
from llama_index.core.tools import BaseTool
from llama_index.llms.google_genai import GoogleGenAI


def create_llm(
    model: str = "gemini-2.5-flash",
    project: str = "hr-mixi",
    location: str = "asia-northeast1",
    temperature: float = 0.0,
) -> LLM:
    """LLM を作成して設定する。

    Args:
        model: Gemini モデル名
        project: GCP プロジェクト ID
        location: GCP リージョン
        temperature: サンプリング温度（0.0 = 決定的）

    Returns:
        設定済み LLM インスタンス
    """
    llm = GoogleGenAI(
        model=model,
        vertexai_config={"project": project, "location": location},
        temperature=temperature,
    )

    Settings.llm = llm
    return llm


def create_agent(tools: list[BaseTool], llm: LLM, verbose: bool = False) -> ReActAgent:
    """指定されたツールと LLM で ReActAgent を作成する。

    Args:
        tools: エージェントが使用するツールのリスト
        llm: 言語モデルインスタンス
        verbose: 詳細な推論ステップを出力するかどうか

    Returns:
        設定済み ReActAgent インスタンス
    """
    return ReActAgent(tools=tools, llm=llm, verbose=verbose)
