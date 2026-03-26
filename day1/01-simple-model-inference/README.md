# 01. テーブルデータの推論ハンズオン

アヤメ（Iris）データセットを使った分類タスクで、**事前学習済みモデル**を使って推論を体験します。

- **Neural Network（全結合NN）**：PyTorch で実装
- **LightGBM（勾配ブースティング）**：テーブルデータで広く使われる手法

## ハンズオンの目的

MLモデルの推論は、どのモデルでも共通の **3フェーズ** で構成されます。

```
1. Preprocess  → 入力データを tensor に変換・正規化
2. Forward     → モデルに tensor を入力して出力を得る
3. Postprocess → 出力 tensor をクラスラベル等に変換
```

このノートブックでこの流れを実際に体験します。

## ディレクトリ構造

```
01-simple-model-inference/
├── notebook.py          # marimo ノートブック（ハンズオン本体）
├── pyproject.toml       # 依存パッケージ定義
├── README.md
├── models/
│   ├── iris_nn.pt       # 学習済み PyTorch モデル（FCNet）
│   ├── iris_lgbm.txt    # 学習済み LightGBM モデル
│   └── scaler.json      # 標準化パラメータ（mean / scale）
├── src/
│   ├── dataset.py       # データ読み込み・前処理ユーティリティ
│   ├── model.py         # PyTorch モデル定義（FCNet）
│   └── evaluate.py      # 評価ユーティリティ（混同行列など）
└── images/              # ノートブック内で使用する説明画像
```

## 動作環境

> **注意**: このハンズオンは **GCP Workbench（Linux）** 上での実行を想定しています。
> macOS では LightGBM が依存する `libomp`（OpenMP ランタイム）の問題でそのままでは動作しません。

## 環境セットアップ

```bash
# 依存パッケージをインストール
uv sync
```

## 起動方法

```bash
# marimo ノートブックを起動
uv run marimo edit notebook.py
```

ブラウザが自動的に開き、ノートブックが表示されます。

## ノートブックの構成

| セクション | 内容 |
|---|---|
| **データセットの確認** | Iris データのロードと EDA（探索的データ解析） |
| **Phase 1: Preprocess** | 標準化（保存済みスケーラー使用）・tensor 変換 |
| **Phase 2: Forward** | 学習済みモデルをロードして推論を実行 |
| **Phase 3: Postprocess** | logits → クラスラベル変換・Confusion Matrix・Classification Report |
| **追加タスク: LightGBM** | 同じデータで LightGBM の推論を実行し、NN と比較 |

## モデルの再学習

`models/` 内のモデルは `05-model-trainig` で学習されたものです。再学習するには：

```bash
cd ../05-model-trainig
uv sync
uv run python src/train_models_for_01.py
```

## 評価指標

| 指標 | 説明 |
|---|---|
| **Accuracy** | 全体の正解率。クラスバランスが良い場合に有効 |
| **Precision** | 陽性と予測した中で実際に陽性の割合 |
| **Recall** | 実際に陽性のうち陽性と予測できた割合 |
| **F1** | Precision と Recall の調和平均 |
| **Confusion Matrix** | どのクラスをどのクラスと間違えたかを可視化 |
