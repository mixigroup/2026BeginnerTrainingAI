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

### 方法 1：marimo ノートブック（推奨）

ブラウザ上でインタラクティブに各フェーズを確認できます。

```bash
cd day1/03-vision-inference
uv run marimo run notebook.py
```

ブラウザが開き、ノートブックを上から順に実行できます。

---

### 方法 2：スクリプト単体実行（marimo 不要）

`run.py` を使ってコマンドラインから直接実行できます。

```bash
cd day1/03-vision-inference

# デフォルト設定で実行（結果は result.png に保存）
uv run python run.py

# オプションを指定して実行
uv run python run.py \
  --image http://images.cocodataset.org/val2017/000000039769.jpg \
  --threshold 0.7 \
  --output result.png
```

**実行結果の例：**

```
[Phase 1] Preprocess
  Loading image: http://images.cocodataset.org/val2017/000000039769.jpg
  Image size: 640 x 480 px
  Loading processor: facebook/detr-resnet-50
  pixel_values shape : (1, 3, 800, 1066)
  pixel_values range : [-2.118, 2.640]

[Phase 2] Forward
  Loading model: facebook/detr-resnet-50
  Parameters: 41,540,424
  Running inference ...
  logits shape    : (1, 100, 92)
  pred_boxes shape: (1, 100, 4)

[Phase 3] Postprocess  (threshold=0.9)
  Detected 4 object(s)
    cat: score=0.998  box=(1, 54, 314, 474)
    cat: score=0.994  box=(341, 21, 638, 476)
    remote: score=0.994  box=(334, 74, 368, 188)
    remote: score=0.992  box=(39, 70, 75, 177)

[Result] Saving visualization -> result.png
Done.
```

---

### 方法 3：Python インタープリタで1行ずつ実行

`src/` 内の各モジュールを直接 import して使えます。

```bash
cd day1/03-vision-inference
uv run python3
```

```python
from src.preprocess import get_processor, preprocess_image
from src.inference import load_model, run_inference
from src.postprocess import postprocess_results, visualize_results
from src.utils import download_sample_image, build_label_map

# Phase 1: Preprocess
image = download_sample_image("http://images.cocodataset.org/val2017/000000039769.jpg")
processor = get_processor("facebook/detr-resnet-50")
inputs = preprocess_image(image, processor)

# pixel_values の中身を確認
pv = inputs["pixel_values"]
print(pv.shape)   # torch.Size([1, 3, 800, 1066])
print(pv.min(), pv.max())   # 正規化後の値域

# Phase 2: Forward
model = load_model("facebook/detr-resnet-50")
outputs = run_inference(model, inputs)
print(outputs.logits.shape)     # torch.Size([1, 100, 92])
print(outputs.pred_boxes.shape) # torch.Size([1, 100, 4])

# Phase 3: Postprocess
label_map = build_label_map(model)
detections = postprocess_results(outputs, processor,
                                 image_size=(image.width, image.height),
                                 threshold=0.9)
print(detections)

# 可視化して保存
fig = visualize_results(image, detections, label_names=label_map)
fig.savefig("result.png")
```

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
├── run.py              # スクリプト単体実行用エントリポイント
├── pyproject.toml      # 依存ライブラリ定義
└── src/
    ├── preprocess.py   # load_image / get_processor / preprocess_image
    ├── inference.py    # load_model / run_inference
    ├── postprocess.py  # postprocess_results / visualize_results
    └── utils.py        # download_sample_image / build_label_map
```

---

## 試してみよう

### パラメータを変える

| パラメータ | デフォルト値 | 変更の効果 |
|---|---|---|
| `--threshold` / `CONFIDENCE_THRESHOLD` | `0.9` | 小さくすると検出数が増える（ノイズも増える） |
| `--image` / `IMAGE_URL` | COCO サンプル画像 | 好きな画像 URL やローカルパスに変えて試せる |
| `--model` / `MODEL_NAME` | `facebook/detr-resnet-50` | `facebook/detr-resnet-101` に変えると精度が上がる |

### 確認ポイント

- **Preprocess 後**：`pixel_values` の値域が正規化によって変化していることを確認
- **Forward 後**：100 クエリすべてに値があることを確認（後処理前は全クエリが何かを検出しようとしている）
- **Postprocess 後**：閾値を `0.5` → `0.9` と変えると検出数がどう変わるか観察
