"""会話履歴のフォーマット・表示ヘルパー関数。"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from google.genai.types import Content

_ROLE_LABEL = {
    "user": "User",
    "model": "Model",
}


def format_contents(contents: list[Content]) -> str:
    """Content リストを Markdown 形式の文字列に変換する。

    Args:
        contents: google.genai.types.Content のリスト

    Returns:
        mo.md() に渡せる Markdown 文字列
    """
    lines: list[str] = []
    for content in contents:
        label = _ROLE_LABEL.get(content.role, content.role)
        text = " ".join(part.text for part in content.parts if part.text)
        lines.append(f"**{label}**: {text}")
    return "\n\n".join(lines)


def format_chat_history(chat: object) -> str:
    """Chat セッションの履歴を Markdown 形式の文字列に変換する。

    Args:
        chat: client.chats.create() で生成された ChatSession オブジェクト

    Returns:
        mo.md() に渡せる Markdown 文字列
    """
    return format_contents(chat.get_history())
