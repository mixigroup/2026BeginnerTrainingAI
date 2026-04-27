"""ReAct エージェント用のツール定義。

このモジュールにはベースラインの計算ツールが含まれています。
参加者はここにカスタムツールを追加できます。
"""

from __future__ import annotations

from llama_index.core.tools import FunctionTool


def multiply(a: int, b: int) -> int:
    """2つの整数を掛け算して結果を返す。

    Args:
        a: 1つ目の整数
        b: 2つ目の整数

    Returns:
        a と b の積
    """
    return a * b


def add(a: int, b: int) -> int:
    """2つの整数を足し算して結果を返す。

    Args:
        a: 1つ目の整数
        b: 2つ目の整数

    Returns:
        a と b の和
    """
    return a + b


def get_tools() -> list[FunctionTool]:
    """ツールを作成して返す。

    Returns:
        FunctionTool インスタンスのリスト
    """
    return [
        FunctionTool.from_defaults(fn=multiply),
        FunctionTool.from_defaults(fn=add),
    ]
