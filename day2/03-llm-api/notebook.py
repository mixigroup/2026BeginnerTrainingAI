import marimo

__generated_with = "0.20.2"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # LLM API の Stateless 性質理解

    LLM API は本質的に stateless（状態を持たない）であることを理解し、
    会話履歴を管理する重要性を学ぶ。Gemini API（Google GenAI SDK）を使用する。

    ---

    ### このノートブックでやること

    1. **Gemini API の基本呼び出し** — `google.genai.Client` でテキスト生成
    2. **Stateless の体験** — 2回連続で呼び出し、前の会話を覚えていないことを確認
    3. **会話履歴の管理** — `contents` リストで文脈を手動管理
    4. **System Instruction** — システムプロンプトで LLM の振る舞いを変更
    5. **マルチターン会話** — Chat セッションの実装
    """)
    return


# ---------------------------------------------------------------------------
# Cell 3: Gemini API の概要・セットアップ
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Gemini API の概要

    [Gemini](https://ai.google.dev/) は Google が提供する大規模言語モデルです。
    Python からは **Google GenAI SDK** (`google-genai`) を使って呼び出します。

    今回は **Vertex AI** 経由で Gemini を使います。
    Workbench 上では VM のサービスアカウント認証（ADC）が自動で使われるため、
    API キーの設定は不要です。

    ```python
    from google import genai

    # Vertex AI バックエンドを使用（ADC 認証）
    client = genai.Client(
        vertexai=True,
        project="hr-mixi",
        location="asia-northeast1",
    )
    ```
    """)
    return


# ---------------------------------------------------------------------------
# Cell 4: Client の初期化
# ---------------------------------------------------------------------------
@app.cell
def _():
    from google import genai
    from google.genai.types import Content, GenerateContentConfig, Part

    from src.chat import format_chat_history, format_contents

    client = genai.Client(
        vertexai=True,
        project="hr-mixi",
        location="asia-northeast1",
    )
    MODEL_NAME = "gemini-2.0-flash"

    return (
        Content,
        GenerateContentConfig,
        MODEL_NAME,
        Part,
        client,
        format_chat_history,
        format_contents,
    )


# ---------------------------------------------------------------------------
# Cell 5: 最初の API 呼び出し（説明）
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 最初の API 呼び出し

    `client.models.generate_content()` にモデル名とプロンプト（文字列）を渡すだけで、
    テキスト生成ができます。
    """)
    return


# ---------------------------------------------------------------------------
# Cell 6: 基本的な generate_content 呼び出し
# ---------------------------------------------------------------------------
@app.cell
def _(MODEL_NAME, client, mo):
    _response = client.models.generate_content(
        model=MODEL_NAME,
        contents="日本で一番高い山は何ですか？",
    )

    mo.md(f"""
    **質問**: 「日本で一番高い山は何ですか？」

    **レスポンス**:
    > {_response.text}
    """)
    return


# ---------------------------------------------------------------------------
# Cell 7: Stateless の説明
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Stateless の体験

    LLM API は **stateless** です。
    つまり、各リクエストは完全に独立しており、前のリクエストの内容を覚えていません。

    ```
    リクエスト1: 「名前は太郎です」 → レスポンス1（覚えた風の返事）
    リクエスト2: 「名前は何？」     → レスポンス2（知らない）
    ```

    2つのリクエストの間に **共有される状態は一切ありません**。
    次のセルで実際に確かめてみましょう。
    """)
    return


# ---------------------------------------------------------------------------
# Cell 8: Stateless の体験（2回の別リクエスト）
# ---------------------------------------------------------------------------
@app.cell
def _(MODEL_NAME, client, mo):
    # リクエスト 1: 名前を伝える
    _response1 = client.models.generate_content(
        model=MODEL_NAME,
        contents="私の名前は太郎です。よろしくお願いします。",
    )

    # リクエスト 2: 名前を聞く（別リクエスト = 前の情報なし）
    _response2 = client.models.generate_content(
        model=MODEL_NAME,
        contents="私の名前は何ですか？",
    )

    mo.md(f"""
    **リクエスト1**: 「私の名前は太郎です。よろしくお願いします。」
    > {_response1.text}

    ---

    **リクエスト2**: 「私の名前は何ですか？」
    > {_response2.text}

    → **LLM は前のリクエストの内容を覚えていない！** これが stateless の特性です。
    """)
    return


# ---------------------------------------------------------------------------
# Cell 9: メッセージ履歴管理の説明
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## メッセージ履歴の管理

    会話の文脈を維持するには、**過去のやり取りをすべて `contents` に含めて送信** します。

    ```python
    contents = [
        Content(role="user",  parts=[Part(text="名前は太郎です")]),
        Content(role="model", parts=[Part(text="太郎さん、よろしく！")]),
        Content(role="user",  parts=[Part(text="名前は何？")]),  # ← 今回の質問
    ]
    ```

    API は受け取った `contents` 全体を「会話の履歴」として解釈し、
    最後のメッセージに対する返答を生成します。
    """)
    return


# ---------------------------------------------------------------------------
# Cell 10: 会話履歴を contents に蓄積して送る
# ---------------------------------------------------------------------------
@app.cell
def _(Content, MODEL_NAME, Part, client, format_contents, mo):
    _history = [
        Content(
            role="user",
            parts=[Part(text="私の名前は太郎です。よろしくお願いします。")],
        ),
        Content(
            role="model",
            parts=[
                Part(
                    text="太郎さん、よろしくお願いします！何かお手伝いできることはありますか？"
                )
            ],
        ),
        Content(
            role="user",
            parts=[Part(text="私の名前は何ですか？")],
        ),
    ]

    _response = client.models.generate_content(
        model=MODEL_NAME,
        contents=_history,
    )

    mo.md(f"""
    **送信した会話履歴**:

    {format_contents(_history)}

    ---

    **レスポンス**:
    > {_response.text}

    → 会話履歴を含めることで、**文脈を維持** できます！
    """)
    return


# ---------------------------------------------------------------------------
# Cell 11: System Instruction の説明
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## System Instruction

    **System Instruction**（システムプロンプト）は、LLM の振る舞いを制御する特別な指示です。
    ユーザーのメッセージとは別に、モデルの「役割」や「制約」を定義できます。

    ```python
    config = GenerateContentConfig(
        system_instruction="あなたは関西弁で話すアシスタントです"
    )
    ```

    - ペルソナの設定（キャラクター、専門家など）
    - 出力フォーマットの指定（JSON、箇条書きなど）
    - 言語や口調の制御
    """)
    return


# ---------------------------------------------------------------------------
# Cell 12: System Instruction の実験
# ---------------------------------------------------------------------------
@app.cell
def _(GenerateContentConfig, MODEL_NAME, client, mo):
    _config = GenerateContentConfig(
        system_instruction="あなたは関西弁で話すアシスタントです。すべての返答を関西弁で行ってください。",
    )

    _response = client.models.generate_content(
        model=MODEL_NAME,
        contents="日本で一番高い山は何ですか？",
        config=_config,
    )

    mo.md(f"""
    **System Instruction**: 「あなたは関西弁で話すアシスタントです」

    **質問**: 「日本で一番高い山は何ですか？」

    **レスポンス**:
    > {_response.text}
    """)
    return


# ---------------------------------------------------------------------------
# Cell 13: マルチターン会話の説明
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## マルチターン会話（Chat セッション）

    `client.chats.create()` を使うと、**SDK が自動的に会話履歴を管理** してくれます。
    手動で `contents` リストを組み立てる必要がなくなります。

    ```python
    chat = client.chats.create(model="gemini-2.0-flash")
    chat.send_message("名前は太郎です")   # 内部で履歴を蓄積
    chat.send_message("名前は何？")       # 過去の履歴も自動送信
    ```

    内部的には Cell 10 と同じことをしていますが、履歴管理が自動化されています。
    """)
    return


# ---------------------------------------------------------------------------
# Cell 14: Chat セッションの実装
# ---------------------------------------------------------------------------
@app.cell
def _(MODEL_NAME, client, format_chat_history, mo):
    _chat = client.chats.create(model=MODEL_NAME)

    _chat.send_message("私の名前は太郎です。好きな食べ物はラーメンです。")
    _chat.send_message("私の名前と好きな食べ物は何ですか？")

    mo.md(f"""
    **Chat セッション（自動で履歴管理）**

    {format_chat_history(_chat)}

    → `client.chats.create()` が内部で会話履歴を管理してくれるため、
    明示的に `contents` を組み立てる必要がありません。
    """)
    return


# ---------------------------------------------------------------------------
# Cell 15: まとめ
# ---------------------------------------------------------------------------
@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## まとめ

    | 概念 | ポイント |
    |------|----------|
    | **Stateless** | LLM API は各リクエストが独立。前の会話を覚えていない |
    | **会話履歴** | `contents` リストに過去のやり取りを含めることで文脈を維持 |
    | **System Instruction** | モデルの振る舞い（ペルソナ・制約）を制御する特別な指示 |
    | **Chat セッション** | `client.chats.create()` で履歴管理を SDK に委譲 |

    ---

    ### ハンズオン課題

    - 長い会話を続けて **context window の制限** を体験してみよう
    - System Instruction でさまざまな振る舞いを試してみよう
    - 会話履歴が長くなったとき、どう **要約** すればよいか考えてみよう
    """)
    return


if __name__ == "__main__":
    app.run()
