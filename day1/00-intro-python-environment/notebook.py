import marimo

__generated_with = "0.20.1"
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

    ## marimo の特徴

    - セルを実行すると、依存するセルが**自動で再実行**されます
    - スライダーや入力欄などの UI コンポーネントが使えます
    - ファイル形式は `.py` なので git との相性が抜群です
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
    | `Ctrl/Cmd + Shift + Enter` | 上に新セルを追加して実行 |
    | `Ctrl/Cmd + Shift + R` | 変更されたセルをすべて実行 |

    ### セル操作

    | ショートカット | 動作 |
    |---|---|
    | `Ctrl/Cmd + Shift + O` | 上にセルを追加 |
    | `Ctrl/Cmd + Shift + P` | 下にセルを追加 |
    | `Shift + Backspace` | セルを削除 |
    | `Ctrl/Cmd + /` | コメントのオン/オフ |

    ### セッションリスタート

    実行順序の乱れや状態がおかしくなった場合は、左サイドバーの **「Restart」** ボタンでセッションを初期化できます。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## marimoのリアクティブセルについて

    marimo では変数が変わると、それに依存するセルが**自動で再実行**されます。

    > **重要:** marimo では同じ変数名を複数のセルで定義できません。
    > 1つの変数は1つのセルで定義し、他のセルへは関数の引数として渡します。
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## 1. 基本的な変数と計算
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
    ## 2. リストの操作
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
    ## 3. FizzBuzz
    """)
    return


@app.function
def fizzbuzz(n: int) -> list[str]:
    """Return FizzBuzz results from 1 to n."""
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
    ## 4. インタラクティブ UI

    marimo では UI コンポーネントを使えます。
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
    ## 5. テキスト入力

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
    ## 6. Python 環境の確認

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
    ### シェルコマンドの実行

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
    return (subprocess,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. GPU の確認と解放

    本講義ではモデルの学習に GPU を使用します。
    `nvidia-smi` コマンドで GPU の状況を確認できます。

    > **重要:** notebook ごとに GPU を占有するため、使い終わったら
    > 左サイドバーの **「Restart」** でセッションを終了し GPU を解放してください。
    """)
    return


@app.cell
def _(subprocess):
    # Check GPU status
    gpu_result = subprocess.run(
        ["nvidia-smi"],
        capture_output=True,
        text=True,
    )
    if gpu_result.returncode == 0:
        print(gpu_result.stdout)
    else:
        print("nvidia-smi not available (no GPU or driver not installed)")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
