# 03-vision-inference：画像認識モデルの推論ハンズオン

Object Detection（物体検出）モデルを使って、推論の3フェーズ（Preprocess → Forward → Postprocess）を体験するハンズオンです。

## 目的

- 画像を入力として受け取る ML モデルの推論フローを理解する
- Preprocess / Forward / Postprocess の各フェーズで「何が起きているか」を観察する
- HuggingFace `transformers` ライブラリの基本的な使い方を習得する

## 使用モデル

**`facebook/detr-resnet-50`**（DETR: Detection Transformer）

- Transformer ベースの Object Detection モデル
- COCO データセット（80クラス）で学習済み
- 猫・犬・人・車など 80 種類の物体を検出可能

---

## 実行方法

ブラウザ上でインタラクティブに各フェーズを確認できます。

```bash
cd day1/03-vision-inference
uv run marimo edit notebook.py
```

ブラウザが開き、ノートブックを上から順に実行できます。

---

## 学習内容

### Phase 1: Preprocess（前処理）

生の画像データをモデルが受け付ける形式に変換します。

| 処理 | 内容 |
|---|---|
| 画像の読み込み | URL またはローカルパスから PIL Image として読み込む |
| リサイズ | 短辺が 800px になるようにアスペクト比を保って拡縮 |
| 正規化 | ImageNet の mean/std（`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`）で標準化 |
| テンソル変換 | PIL Image → PyTorch Tensor（形状: `[1, 3, H, W]`） |

> **ポイント**：正規化後の値は 0〜255 ではなく負の値も含む範囲になります。

### Phase 2: Forward（推論）

前処理済みの tensor をモデルに入力し、生の出力（logits）を得ます。

DETR は Transformer ベースのモデルで、100 個のオブジェクトクエリを使って同時に複数の物体を検出します。

| 出力テンソル | 形状 | 内容 |
|---|---|---|
| `logits` | `[1, 100, 92]` | 各クエリのクラス スコア（softmax 前）。92 = 80クラス + "no object" + padding |
| `pred_boxes` | `[1, 100, 4]` | 各クエリの bounding box（cx, cy, w, h）。値は画像サイズに対して正規化された [0, 1] |

### Phase 3: Postprocess（後処理）

生の出力 tensor から人間が理解できる検出結果に変換します。

| 処理 | 内容 |
|---|---|
| Softmax | logits を確率（0〜1）に変換 |
| 閾値フィルタリング | confidence スコアが閾値未満の検出を除外 |
| 座標変換 | 正規化 [0, 1] の相対座標 → 画像上のピクセル座標（Pascal VOC 形式） |
| ラベル変換 | クラス id → クラス名（"cat"、"dog" など） |

---

## ディレクトリ構成

```
03-vision-inference/
├── README.md           # このファイル
├── notebook.py         # marimoノートブック（インタラクティブなハンズオン）
├── pyproject.toml      # 依存ライブラリ定義
└── src/
    ├── preprocess.py   # load_image / get_processor / preprocess_image
    ├── inference.py    # load_model / run_inference
    ├── postprocess.py  # postprocess_results / visualize_results
    └── utils.py        # download_sample_image / build_label_map
```

---

## 評価指標：IoU と mAP

Object Detection の精度は **IoU（Intersection over Union）** で測ります。

- **IoU** = 予測 bbox と正解 bbox の重なり度合い（0〜1）
- **AP50**: IoU ≥ 0.5 で「正解」とみなす（検出漏れを見る指標）
- **AP75**: IoU ≥ 0.75 で「正解」とみなす（位置精度を見る指標）
- **mAP@[.50:.95]**: 10段階の IoU 閾値で平均した総合評価（COCO標準）

---

## 試してみよう

### パラメータを変える

ノートブック内の変数を変更して、結果の違いを観察できます。

| パラメータ | デフォルト値 | 変更の効果 |
|---|---|---|
| `CONFIDENCE_THRESHOLD` | `0.9` | 小さくすると検出数が増える（ノイズも増える） |
| `IMAGE_URL` | COCO サンプル画像 | 好きな画像 URL やローカルパスに変えて試せる |
| `MODEL_NAME` | `facebook/detr-resnet-50` | `facebook/detr-resnet-101` に変えると精度が上がる |

### 確認ポイント

- **Preprocess 後**：`pixel_values` の値域が正規化によって変化していることを確認
- **Forward 後**：100 クエリすべてに値があることを確認（後処理前は全クエリが何かを検出しようとしている）
- **Postprocess 後**：閾値を `0.5` → `0.9` と変えると検出数がどう変わるか観察
