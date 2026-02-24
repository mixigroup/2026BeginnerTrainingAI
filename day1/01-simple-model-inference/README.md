# 01. テーブルデータの推論ハンズオン

アヤメ（Iris）データセットを使った分類タスクで、以下の2つのモデルを比較します。

- **Neural Network（全結合NN）**：メインのハンズオン
- **LightGBM（勾配ブースティング）**：追加タスク

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
├── src/
│   ├── dataset.py       # データ読み込み・前処理・分割ユーティリティ
│   ├── model_nn.py      # Neural Network モデルの定義・学習
│   └── evaluate.py      # 評価ユーティリティ（学習曲線・混同行列など）
└── images/              # ノートブック内で使用する説明画像
```

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
| **データセットの準備** | Iris データのロードと EDA（探索的データ解析） |
| **Phase 1: Preprocess** | シャッフル・正規化・train/valid/test 分割・one-hot encoding |
| **Phase 2: Forward** | Neural Network の定義と学習（ハイパーパラメータを変更して実験） |
| **Phase 3: Postprocess** | 学習曲線・Accuracy・Confusion Matrix・Classification Report |
| **EX: 過学習** | パラメータを変えて意図的に過学習を起こし、学習曲線を観察 |
| **追加タスク: LightGBM** | 同じデータで LightGBM を学習し、NN と比較 |

## ハイパーパラメータの実験

`notebook.py` の「ハイパーパラメータの設定」セルで、以下の値を変更して実験してみてください。

```python
HIDDEN_UNITS = [1000, 500, 300]  # 隠れ層のユニット数
DROPOUT_RATE = 0.2               # ドロップアウト率（0.0〜1.0）
LEARNING_RATE = 0.001            # 学習率
EPOCHS = 100                     # 学習エポック数
BATCH_SIZE = 100                 # ミニバッチサイズ
```

値を変えると、marimo がリアクティブに再実行し、結果を自動更新します。

### 過学習を起こすための設定例

```python
HIDDEN_UNITS = [5000, 2000, 1000]
EPOCHS = 200
DROPOUT_RATE = 0.0
```

学習曲線で train loss が下がり続け、valid loss が上がっていく現象（過学習）を確認してください。

## 評価指標

| 指標 | 説明 |
|---|---|
| **Accuracy** | 全体の正解率。クラスバランスが良い場合に有効 |
| **Precision** | 陽性と予測した中で実際に陽性の割合 |
| **Recall** | 実際に陽性のうち陽性と予測できた割合 |
| **F1** | Precision と Recall の調和平均 |
| **Confusion Matrix** | どのクラスをどのクラスと間違えたかを可視化 |
