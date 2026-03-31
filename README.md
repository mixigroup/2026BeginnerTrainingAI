# 2026BeginnerTrainingAI

## Workbenchへのアクセス方法

### SSH接続

IAP（Identity-Aware Proxy）トンネル経由でSSH接続します。

```bash
gcloud compute ssh --project hr-mixi --zone asia-northeast1-a <インスタンス名> --tunnel-through-iap
```

インスタンス名はメールアドレスから自動生成されます（例: `taro.yamada@mixi.co.jp` → `taro-yamada`）。

### VS Code からリモート接続

VS Code の Remote - SSH 拡張機能を使って、Workbench に直接接続できます。

1. VS Code に [Remote - SSH](https://marketplace.visualstudio.com/items?itemName=ms-vscode-remote.remote-ssh) 拡張機能をインストール
2. 初回のみ、`gcloud compute ssh` で一度接続して SSH 鍵を生成する（`~/.ssh/google_compute_engine` が作成される）:

```bash
gcloud compute ssh --project hr-mixi --zone asia-northeast1-a <インスタンス名> --tunnel-through-iap
# 接続できたら exit で抜ける
```

1. `~/.ssh/config` に以下を追加:

```
Host workbench
    HostName <インスタンス名>
    User <OS Loginユーザー名>
    IdentityFile ~/.ssh/google_compute_engine
    ProxyCommand gcloud compute start-iap-tunnel %h %p --listen-on-stdin --project=hr-mixi --zone=asia-northeast1-a
```

- **インスタンス名**: メールアドレスの `@` より前の `.` を `-` に置換（例: `taro.yamada@mixi.co.jp` → `taro-yamada`）
- **OS Loginユーザー名**: メールアドレスの `.` と `@` を `_` に置換（例: `taro.yamada@mixi.co.jp` → `taro_yamada_mixi_co_jp`）

1. VS Code のコマンドパレット（`Cmd+Shift+P`）→ `Remote-SSH: Connect to Host...` → `workbench` を選択

接続後は VS Code のターミナルやエディタからリモートのファイルを直接編集できます。

### 作業の進め方

本研修では Python のパッケージ管理に [uv](https://docs.astral.sh/uv/) を使用します。

#### uv のインストール

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

インストール後、シェルを再起動するか `source ~/.local/bin/env` を実行してパスを通してください。

```bash
# インストール確認
uv --version
```

#### 環境の確認

```bash
# GPU が認識されているか確認
nvidia-smi

# Python環境の確認
uv python list
```

### 注意事項

- 3時間（10800秒）CPUが使われない場合、自動でインスタンスが停止します
- 停止したインスタンスは [GCPコンソール](https://console.cloud.google.com/vertex-ai/workbench/instances?project=hr-mixi) から再起動できます
