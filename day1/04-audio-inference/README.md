# 04-audio-inference

**音声認識（ASR: Automatic Speech Recognition）ハンズオン**

Whisper を使って、日本語音声をテキストに書き起こす推論パイプラインを3フェーズに分解して理解する。

---

## タスク

- **モデル**: `openai/whisper-small`（日本語対応・軽量）
- **入力**: 日本語音声データ（16kHz）
- **出力**: 書き起こしテキスト（日本語）
- **データセット**: [google/fleurs](https://huggingface.co/datasets/google/fleurs)（日本語サブセット）

---

## 構成

```
04-audio-inference/
├── notebook.py          # marimo ノートブック（メインのハンズオン）
├── src/
│   ├── __init__.py
│   ├── preprocess.py    # 音声の前処理（リサンプリング・mel spectrogram変換）
│   ├── model.py         # モデルのロードと推論（Whisper encoder-decoder）
│   ├── postprocess.py   # token ids → テキストへのデコード
│   └── evaluate.py      # WER / CER の計算
├── pyproject.toml
└── README.md
```

---

## 3フェーズの概要

### Phase 1: Preprocess（前処理）

音声データを Whisper が処理できる形式に変換する。

1. データセットから日本語音声を取得
2. サンプリングレートを 16kHz にリサンプリング
3. `AutoFeatureExtractor` で log-mel spectrogram に変換
   - shape: `(1, 80, 3000)` ← 80 mel 帯域 × 30 秒分

### Phase 2: Forward（推論）

encoder-decoder 構造で音声特徴量からトークン列を生成する。

- **Encoder（CNN + Transformer）**: log-mel spectrogram → encoder hidden states
  - shape: `(1, 1500, hidden_size)`
- **Decoder（Transformer + 自己回帰）**: encoder hidden states → token ids
  - shape: `(1, seq_len)`

### Phase 3: Postprocess（後処理）

token ids をテキスト文字列にデコードする。

- `AutoProcessor.batch_decode()` でトークン ids → テキスト
- 特殊トークン（`<|startoftranscript|>` など）を除去

---

## 評価指標

### WER（Word Error Rate）

英語などの分かち書き言語で標準的な指標。

```
WER = (S + D + I) / N
```

### CER（Character Error Rate）

日本語に適した指標。文字レベルで WER と同様に計算する。

```
正解: 今日は晴れです      （8文字）
予測: 今日は晴れでした    （9文字）
CER = 編集距離 2 / 8 = 25%
```

---

## セットアップ

```bash
uv sync
```

## 実行

```bash
uv run marimo run notebook.py
# または編集モード
uv run marimo edit notebook.py
```

---

## 参考リンク

- [Automatic Speech Recognition - HuggingFace](https://huggingface.co/docs/transformers/ja/tasks/asr)
- [openai/whisper-small - HuggingFace](https://huggingface.co/openai/whisper-small)
- [WER の解説 - Hugging Face evaluate](https://huggingface.co/spaces/evaluate-metric/wer)
- [jiwer ドキュメント](https://jiwer.readthedocs.io/en/latest/)
