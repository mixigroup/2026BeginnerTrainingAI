# 2026BeginnerTrainingAI

機械学習・生成 AI の新卒研修（2 日間）のハンズオン教材リポジトリです。

## 研修の全体像

研修は Day1 / Day2 の 2 日構成です。各日は番号付きの独立した marimo notebook ハンズオンで構成され、上から順に進めていきます。

### Day1: 機械学習の推論・学習・デプロイ (`day1/`)

| #  | ディレクトリ                  | テーマ                                            |
| -- | ----------------------------- | ------------------------------------------------- |
| 00 | `00-intro-python-environment` | Python 環境（uv / marimo）の基本操作              |
| 01 | `01-simple-model-inference`   | テーブルデータで ML 推論の 3 フェーズを体験       |
| 02 | `02-nlp-inference`            | テキストを使った NLP モデルの推論                 |
| 03 | `03-vision-inference`         | 画像を使った Object Detection の推論              |
| 04 | `04-audio-inference`          | 音声を使った推論                                    |
| 05 | `05-model-trainig`            | PyTorch でモデル学習・過学習・転移学習            |
| 06 | `06-accelerate-ml-model`      | ONNX エクスポート・INT8 量子化でモデル高速化      |
| 07 | `07-model-deploy`             | FastAPI + カスタムコンテナで Vertex AI にデプロイ |

### Day2: 生成 AI と LLM エージェント (`day2/`)

| #  | ディレクトリ       | テーマ                   |
| -- | ------------------ | ------------------------ |
| 01 | `01-multi-modal`   | マルチモーダル           |
| 02 | `02-llm-api`       | LLM API の基本           |
| 03 | `03-llm-tool`      | LLM の Tool Use          |
| 04 | `04-react-agent`   | ReAct エージェント       |
| 05 | `05-coding-agent`  | コーディングエージェント |

## 研修環境

本研修は、以下のどちらかの環境で実施できます。使いやすい方を選んでください。

1. **ローカル環境**
2. **GCP Workbench インスタンス（T4 GPU 環境）**

どちらを選んでも、[共通セットアップ](#共通セットアップ) と [ハンズオンの進め方](#ハンズオンの進め方) の手順は同じです。

## Workbench インスタンスのセットアップ

### 1. Workbench インスタンスを起動

[GCP の hr-mixi プロジェクトの workbench ホーム](https://console.cloud.google.com/dataproc/workbench/instances?hl=ja&project=hr-mixi&referrer=search)から利用いただけます。

インスタンス名はメールアドレスの `.`を`-` に置き換えて自動生成されています（例: `taro.yamada@mixi.co.jp` → `taro-yamada`）。

> [!NOTE]
> つけ忘れ防止のため、インスタンスは使っていないと数時間で切れるようになっています。
> もしインスタンスが落ちてしまった場合、[GCPコンソール](https://console.cloud.google.com/vertex-ai/workbench/instances?project=hr-mixi) から再起動してください

### 2. SSH接続

1. key 生成
    初回だけ鍵登録をしないといけないため、以下のコマンドを実行してください

    ```bash
    gcloud compute ssh --project hr-mixi --zone asia-northeast1-a <インスタンス名> --tunnel-through-iap
    ```

2. `~/.ssh/config` に以下を追加

    ```
    Host workbench
        HostName <インスタンス名>
        User <OS Loginユーザー名>
        IdentityFile ~/.ssh/google_compute_engine
        ProxyCommand gcloud compute start-iap-tunnel %h %p --listen-on-stdin --project=hr-mixi --zone=asia-northeast1-a
    ```

    - **インスタンス名**: メールアドレスの `@` より前の `.` を `-` に置換（例: `taro.yamada@mixi.co.jp` → `taro-yamada`）
    - **OS Loginユーザー名**: メールアドレスの `.` と `@` を `_` に置換（例: `taro.yamada@mixi.co.jp` → `taro_yamada_mixi_co_jp`）

3. VS Code からリモート接続

    VS Code の Remote - SSH 拡張機能を使って、Workbench に直接接続できます。

    1. VS Code に [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) 拡張機能をインストール
    2. VS Code のコマンドパレット（`Cmd+Shift+P`）→ `Remote-SSH: Connect to Host...` → `workbench` を選択

    接続後は VS Code のターミナルやエディタからリモートのファイルを直接編集できます。

### 3. GitHub CLI のインストール

1. Workbench からリポジトリを clone するために、GitHub CLI をインストールします。

    参考：<https://github.com/cli/cli/blob/trunk/docs/install_linux.md#debian>

    ```bash
    (type -p wget >/dev/null || (sudo apt update && sudo apt install wget -y)) \
     && sudo mkdir -p -m 755 /etc/apt/keyrings \
     && out=$(mktemp) && wget -nv -O$out https://cli.github.com/packages/githubcli-archive-keyring.gpg \
     && cat $out | sudo tee /etc/apt/keyrings/githubcli-archive-keyring.gpg > /dev/null \
     && sudo chmod go+r /etc/apt/keyrings/githubcli-archive-keyring.gpg \
     && sudo mkdir -p -m 755 /etc/apt/sources.list.d \
     && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | sudo tee /etc/apt/sources.list.d/github-cli.list > /dev/null \
     && sudo apt update \
     && sudo apt install gh -y
    ```

2. GitHub アカウントにログイン:

    ```bash
    gh auth login
    ```

### (任意) 1Passwordを使ったセットアップ

1PasswordでSSHキーの管理・エージェント転送を行う場合は、以下のドキュメントを参考にセットアップしてください。
ssh key setup: [Get started with 1Password for SSH \| 1Password Developer](https://developer.1password.com/docs/ssh/get-started/)
agent forward: [1Password Developer](https://developer.1password.com/docs/ssh/agent/forwarding)

`~/.ssh/config`の今回追加したworkbenchに以下を追加
```
    ForwardAgent yes
```

サーバーでのGithub認証の確認

```shell
ssh-add -L
ssh -T git@github.com
```

## 共通セットアップ

Workbench / ローカルどちらの環境でも、以下のセットアップを 1 回だけ実施します。

### 研修リポジトリを clone

```bash
git clone git@github.com:mixigroup/2026BeginnerTrainingAI.git
```

### uv のインストール

本研修では Python のパッケージ管理に [uv](https://docs.astral.sh/uv/) を使用します。

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール後、シェルを再起動するか `source ~/.local/bin/env` を実行してパスを通してください。

```bash
# インストール確認
uv --version
```

### 依存関係のインストール

macOSについては動作確認を行っていませんが、以下のツールのインストールとLightGBMのためにlibompのインストールが必要になる可能性があります。

#### Graphviz

`day1/01-simple-model-inference/notebook_lgbm.py` でGraphvizを使用します。

<https://graphviz.org/download/>

Debian/Ubuntu:
```shell
sudo apt install graphviz
```

#### FFmpeg

`day1/04-audio-inference/notebook.py` で FFmpeg を使用します。

Debian/Ubuntu:
```shell
sudo apt install ffmpeg
```

### 環境の確認

```bash
# GPU が認識されているか確認（Workbench インスタンスのみ）
nvidia-smi

# Python 環境の確認
uv python list
```

## ハンズオンの進め方

各ハンズオンは独立した uv プロジェクト（`day<N>/<NN>-*/pyproject.toml`）として構成されています。**必ず対象ハンズオンのディレクトリに移動してから** `uv sync` → `uv run marimo edit` の順に実行してください。

```bash
cd day1/00-intro-python-environment   # 例: Day1 の最初のハンズオン

# 依存パッケージのインストール（各ハンズオンごとに初回のみ）
uv sync --frozen  # 再現性のために frozen オプションを推奨

# marimo notebook を起動（ブラウザで対話的に実行）
uv run marimo edit notebook.py
```

### (推奨)VS Code での marimo notebook 実行

VS Codeの[marimo拡張機能](https://marketplace.visualstudio.com/items?itemName=marimo-team.vscode-marimo)をインストールすると、VS Code 内で marimo notebook を直接開いて実行できます。
右上のmarimoのアイコンをクリックするとNotebookが起動します。
Notebookの右上のSelect KernelやVS Codeの右下からPythonの環境を選択することができます。
基本的に自動でPythonの環境が検知されるように`.vscode/settings.json`に書いているつもりですが、もし環境が見つからない場合は<https://code.visualstudio.com/docs/python/environments>を参考にしてください。

[marimo](https://marimo.io/) は `.py` ファイルで動くリアクティブ Notebook です。ブラウザが自動で開き、セルの変数を変更すると依存する関連セルが自動で再実行されます。
