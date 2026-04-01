import marimo

__generated_with = "0.21.1"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Python 環境入門 - インタラクティブノートブック

    このノートブックでは **marimo** の基本的な使い方を体験します。
    https://marimo.io/

    ## marimo とは？

    marimo は Python 用の**リアクティブノートブック**です。
    従来の Jupyter Notebook とは異なり、セル間の依存関係を自動で解析し、
    変数が変更されると依存するセルが**自動で再実行**されます。

    ### marimo の主な特徴

    - **リアクティブ実行**: セルを実行すると、依存するセルが自動で再実行されます
    - **インタラクティブ UI**: スライダーや入力欄などの UI コンポーネントが使えます
    - **Pure Python**: ファイル形式は `.py` なので git との相性が抜群です（出力は保存されません）
    - **再現性**: 実行順序がデータフローグラフで決まるため、常に一貫した結果が得られます
    - **アプリとしても動作**: `marimo run` コマンドでノートブックをWebアプリとして公開できます
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## キーボードショートカット

    ### 実行系

    | ショートカット | 動作 |
    |---|---|
    | `Ctrl/Cmd + Enter` | 現在のセルを実行 |
    | `Shift + Enter` | 現在のセルを実行して下に新セルを追加 |
    | `Ctrl/Cmd + Shift + R` | 変更されたセルをすべて実行 |

    ### セル操作

    | ショートカット | 動作 |
    |---|---|
    | `Ctrl/Cmd + Shift + O` | 上にセルを追加 |
    | `Ctrl/Cmd + Shift + P` | 下にセルを追加 |
    | `Shift + Backspace` | セルを削除 |
    | `Ctrl/Cmd + /` | コメントのオン/オフ |
    | `Ctrl/Cmd + B` | セルのコードフォーマット（ruff が必要） |
    | `Ctrl/Cmd + K` | コマンドパレットを開く |

    ### セッションリスタート

    実行順序の乱れや状態がおかしくなった場合は、左サイドバーの **「Restart」** ボタンでセッションを初期化できます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## リアクティブ実行とデータフロー

    marimo のリアクティブ実行は、以下のシンプルなルールに基づいています：

    > **セルが実行されると、そのセルが定義するグローバル変数を参照している他のすべてのセルが自動的に実行される**

    marimo はセルのコードを（実行せずに）解析して、各セルの**参照（refs）** と**定義（defs）** を特定します：

    - **参照（refs）**: セルが読み取るが定義しないグローバル変数
    - **定義（defs）**: セルが定義するグローバル変数

    ### 重要なルール

    1. **グローバル変数名は一意でなければならない**: 同じ変数名を複数のセルで定義できません
    2. **実行順序はセルの並び順ではない**: データフローグラフによって決まります
    3. **セルを削除すると変数も削除される**: プログラムの状態が常にコードと一致します
    4. **循環参照は禁止**: セル間の循環依存はエラーになります
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### アンダースコア変数はセルにローカル

    アンダースコア (`_`) で始まる変数はセルの「プライベート変数」です。
    複数のセルで同じアンダースコア付き変数名を定義でき、他のセルからは参照できません。

    ```python
    # セル1
    _temp = 42  # このセルでのみ有効

    # セル2
    _temp = 99  # エラーにならない（別のプライベート変数）
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 自動実行を無効にする

    ノートブックのフッターで「On Cell Change」を「lazy」に変更すると、自動実行を無効にできます。

    lazy モードでは、セルを実行した後、その子孫セルは自動実行されず「stale（古い）」とマークされます。
    高コストな計算を含むノートブックで便利です。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## リアクティブ実行のデモ

    以下のセルで `changed` の値を `True` に変更して実行してみてください。
    下のセルが**自動的に更新**されます。
    """)
    return


@app.cell
def _():
    # この値を True に変更して実行してみてください
    changed = False
    return (changed,)


@app.cell
def _(changed, mo):
    mo.md(
        f"""
        **変更を検知しました！** `changed` の値は `{changed}` です。

        `changed` の値を更新すると、この変数を参照しているこのセルが
        **自動的に再実行**されました。これがリアクティブ実行です。
        """
        if changed
        else """
        上のセルで `changed` を `True` に変更し、`Ctrl/Cmd + Enter` で実行してください。
        """
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 1. 基本的な変数と計算
    """)
    return


@app.cell
def _():
    # Basic variables and operations
    name = "Python"
    version = 3.12
    message = f"Hello from {name} {version}!"
    print(message)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 2. リストの操作
    """)
    return


@app.cell
def _():
    # List operations
    fruits = ["apple", "banana", "cherry", "date", "elderberry"]

    # List comprehension: filter fruits with more than 5 characters
    long_fruits = [f for f in fruits if len(f) > 5]

    print("All fruits:", fruits)
    print("Long name fruits:", long_fruits)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 3. FizzBuzz
    """)
    return


@app.function
def fizzbuzz(n: int) -> list[str]:
    """1 から n までの FizzBuzz 結果を返す。"""
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


@app.cell
def _():
    result = fizzbuzz(20)
    print(result)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 4. インタラクティブ UI

    marimo には豊富な UI コンポーネントが `marimo.ui` モジュールに用意されています。
    UI 要素を操作すると、その要素を参照しているセルが**自動で再実行**されます。

    ### 基本的な UI 要素

    | 要素 | 説明 |
    |---|---|
    | `mo.ui.slider` | スライダー |
    | `mo.ui.text` | テキスト入力 |
    | `mo.ui.number` | 数値入力 |
    | `mo.ui.checkbox` | チェックボックス |
    | `mo.ui.dropdown` | ドロップダウン |
    | `mo.ui.radio` | ラジオボタン |
    | `mo.ui.switch` | スイッチ |
    | `mo.ui.date` | 日付選択 |
    | `mo.ui.file` | ファイルアップロード |
    | `mo.ui.text_area` | テキストエリア |
    | `mo.ui.button` | ボタン |
    | `mo.ui.table` | テーブル |

    スライダーを動かすと、下のセルが**自動で再計算**されます。
    """)
    return


@app.cell
def _(mo):
    # Slider: change the value and the cell below will auto-update
    n_slider = mo.ui.slider(start=1, stop=30, value=15, label="FizzBuzz の上限")
    n_slider
    return (n_slider,)


@app.cell
def _(mo, n_slider):
    # This cell auto-reruns when n_slider changes
    fb_result = fizzbuzz(n_slider.value)
    mo.md(f"**1 〜 {n_slider.value} の FizzBuzz:**\n\n" + ", ".join(fb_result))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 5. テキスト入力

    入力欄に名前を入れると、挨拶が変わります。
    """)
    return


@app.cell
def _(mo):
    name_input = mo.ui.text(placeholder="名前を入力...", label="あなたの名前")
    name_input
    return (name_input,)


@app.cell
def _(mo, name_input):
    display_name = name_input.value if name_input.value else "World"
    mo.md(f"# Hello, {display_name}!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 6. さまざまな UI 要素を試す

    ドロップダウン、チェックボックス、数値入力を組み合わせた例です。
    """)
    return


@app.cell
def _(mo):
    color_dropdown = mo.ui.dropdown(
        ["赤", "青", "緑", "黄"],
        value="青",
        label="好きな色",
    )
    is_bold = mo.ui.checkbox(label="太字にする")
    repeat_count = mo.ui.number(start=1, stop=10, value=3, label="繰り返し回数")
    mo.hstack([color_dropdown, is_bold, repeat_count], justify="start", gap=1)
    return color_dropdown, is_bold, repeat_count


@app.cell
def _(color_dropdown, is_bold, mo, repeat_count):
    _text = f"{color_dropdown.value} " * repeat_count.value
    mo.md(f"**{_text}**" if is_bold.value else _text)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 複合 UI 要素

    `mo.ui.array` や `mo.ui.dictionary` を使うと、複数の UI 要素をまとめて管理できます。
    個別の UI 要素と異なり、リアクティブに動作します。
    """)
    return


@app.cell
def _(mo):

    user_info = mo.ui.dictionary(
        {
            "名前": mo.ui.text(placeholder="名前を入力..."),
            "年齢": mo.ui.number(start=0, stop=120, value=25),
            "職種": mo.ui.dropdown(["エンジニア", "デザイナー", "PM", "その他"]),
        }
    )
    user_info
    return (user_info,)


@app.cell
def _(mo, user_info):
    mo.md(f"""
    **入力された情報:**

    | 項目 | 値 |
    |---|---|
    | 名前 | {user_info.value["名前"] or "未入力"} |
    | 年齢 | {user_info.value["年齢"]} |
    | 職種 | {user_info.value["職種"] or "未選択"} |
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 7. marimo のコマンドラインツール

    marimo はコマンドラインツールとしても使えます。

    ### ノートブックの作成・編集

    ```bash
    # ノートブックサーバーを起動（新規作成・既存編集）
    marimo edit

    # 特定のファイルを編集
    marimo edit notebook.py
    ```

    ### アプリとして実行

    ```bash
    # ノートブックを読み取り専用のWebアプリとして提供（コードは非表示）
    marimo run notebook.py
    ```

    ### Jupyter ノートブックの変換

    ```bash
    # Jupyter (.ipynb) から marimo (.py) に変換
    marimo convert your_notebook.ipynb > your_app.py
    ```

    ### チュートリアル

    marimo には組み込みチュートリアルがあります：

    ```bash
    marimo tutorial intro          # 基本的な使い方
    marimo tutorial dataflow       # リアクティブ実行の詳細
    marimo tutorial ui             # UI 要素の使い方
    marimo tutorial markdown       # マークダウンの書き方
    marimo tutorial plots          # プロットの描画
    marimo tutorial sql            # SQL の使い方
    marimo tutorial layout         # レイアウトの構成
    marimo tutorial fileformat     # ファイル形式について
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 8. marimo のファイル形式

    marimo ノートブックは**Pure Python ファイル**です。

    - 出力はファイルに保存されません（JSON の差分に悩まされることはありません）
    - `git diff` で最小限の差分が表示されます
    - お好みのフォーマッター（black, ruff）でフォーマットできます
    - Python スクリプトとして直接実行可能（UI 要素はデフォルト値を使用）
    - 他のモジュールからインポート可能
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 9. Python 環境の確認

    使用中の Python バージョンやインストール済みパッケージを確認します。
    """)
    return


@app.cell
def _():
    import sys

    print(f"Python version: {sys.version}")
    print(f"Python path: {sys.executable}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 10. シェルコマンドの実行

    Python の `subprocess` モジュールを使うと、シェルコマンドをコードから実行できます。
    """)
    return


@app.cell
def _():
    import subprocess

    # Run shell command and capture output
    result_cmd = subprocess.run(
        ["pip", "list"],
        capture_output=True,
        text=True,
    )
    print(result_cmd.stdout)
    return


@app.cell
def _():
    # ここに自由にコードを書いて試してください
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
