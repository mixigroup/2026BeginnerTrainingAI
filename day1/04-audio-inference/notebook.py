import marimo

__generated_with = "0.20.1"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    # 音声認識（ASR）ハンズオン

    ## タスク：Automatic Speech Recognition（自動音声認識）

    - 音声データからテキストを書き起こす
    - 日本語音声を入力して日本語テキストを出力
    - 例：「今日の天気は晴れです」という音声 → `"今日の天気は晴れです"`

    **使用モデル**: `openai/whisper-small`
    - OpenAI が開発した多言語対応の ASR モデル
    - encoder-decoder 構造（音声 → 内部表現 → テキスト）

    ---

    ### このノートブックでやること

    `pipeline` の内部を3フェーズに分解して理解する：

    1. **Preprocess（前処理）** — 音声 → log-mel spectrogram
    2. **Forward（推論）** — spectrogram → token ids（encoder + decoder）
    3. **Postprocess（後処理）** — token ids → テキスト
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## pipeline とは

    HuggingFace の `pipeline` は、前処理・推論・後処理を1行で実行できる便利なAPI。

    ```python
    pipe = pipeline("automatic-speech-recognition", model="openai/whisper-small")
    pipe(audio_array)
    # → {"text": "今日の天気は晴れです"}
    ```

    このノートブックでは、この `pipeline` の内部を分解して理解する。
    """)
    return


@app.cell
def _():
    from transformers import pipeline

    from src.preprocess import (
        load_sample_audio,
        load_feature_extractor,
        resample_audio,
        extract_features,
        visualize_mel_spectrogram,
        TARGET_SAMPLING_RATE,
    )
    from src.model import load_model, encode_audio, generate_tokens
    from src.postprocess import load_processor, decode_tokens
    from src.evaluate import compute_cer, compute_wer, evaluate_batch

    return (
        TARGET_SAMPLING_RATE,
        compute_cer,
        compute_wer,
        decode_tokens,
        encode_audio,
        evaluate_batch,
        extract_features,
        generate_tokens,
        load_feature_extractor,
        load_model,
        load_processor,
        load_sample_audio,
        pipeline,
        resample_audio,
        visualize_mel_spectrogram,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### pipeline で1行実行（まず動かしてみる）

    最初に `pipeline` を使って、音声認識が動くことを確認する。
    """)
    return


@app.cell
def _(load_sample_audio, mo):
    # サンプル音声をロード（JSUT 日本語データセット）
    audio_array, sampling_rate, reference_text = load_sample_audio(index=0)

    mo.md(
        f"""
        音声の長さ: {len(audio_array) / sampling_rate:.1f} 秒\n
        サンプリングレート: {sampling_rate} Hz\n
        正解テキスト: {reference_text}
        """
    )
    return audio_array, reference_text, sampling_rate


@app.cell
def _(audio_array, pipeline, sampling_rate):
    MODEL_NAME = "openai/whisper-small"

    # pipeline で1行実行
    pipe = pipeline(
        "automatic-speech-recognition",
        model=MODEL_NAME,
        generate_kwargs={"language": "japanese"},
    )
    result = pipe({"array": audio_array, "sampling_rate": sampling_rate})
    print(f"\npipeline の出力: {result['text']}")
    return (MODEL_NAME,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Phase 1: Preprocess（前処理）

    ### 音声を log-mel spectrogram に変換する3ステップ

    1. **リサンプリング**
       - 元の音声: 任意のサンプリングレート（例: 48kHz）
       - Whisper が期待するレートに変換: → 16kHz

    2. **log-mel spectrogram に変換**
       - 時間軸 × 周波数軸の2次元表現に変換
       - 80 mel 帯域 × 3000 フレーム（= 30秒分）

    3. **テンソル化**
       - shape: `(1, 80, 3000)` → モデルへの入力
    """)
    return


@app.cell
def _(
    MODEL_NAME,
    TARGET_SAMPLING_RATE,
    audio_array,
    load_feature_extractor,
    resample_audio,
    sampling_rate,
):
    # Step 1: リサンプリング
    print(f"元のサンプリングレート: {sampling_rate} Hz")
    resampled_audio = resample_audio(
        audio_array, orig_sr=sampling_rate, target_sr=TARGET_SAMPLING_RATE
    )
    print(f"変換後のサンプリングレート: {TARGET_SAMPLING_RATE} Hz")
    print(f"元のサンプル数: {len(audio_array)}")
    print(f"変換後のサンプル数: {len(resampled_audio)}")

    # Step 2 & 3: Feature Extractor で log-mel spectrogram に変換
    feature_extractor = load_feature_extractor(MODEL_NAME)
    print(f"\nFeature Extractor: {type(feature_extractor).__name__}")
    print(f"期待するサンプリングレート: {feature_extractor.sampling_rate} Hz")
    print(f"mel 帯域数: {feature_extractor.feature_size}")
    return feature_extractor, resampled_audio


@app.cell
def _(
    TARGET_SAMPLING_RATE,
    extract_features,
    feature_extractor,
    resampled_audio,
):
    inputs = extract_features(feature_extractor, resampled_audio, TARGET_SAMPLING_RATE)
    input_features = inputs["input_features"]

    print(f"input_features shape: {tuple(input_features.shape)}")
    print("  → (batch_size, mel_bands, time_frames)")
    print(
        f"  → ({input_features.shape[0]}, {input_features.shape[1]}, {input_features.shape[2]})"
    )
    print("  → 80 mel帯域 × 3000フレーム（30秒分）")
    print(f"dtype: {input_features.dtype}")
    return input_features, inputs


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### mel spectrogram を可視化

    log-mel spectrogram を画像として表示してみよう。
    音声の特徴が「形」として見えるはず！

    - **横軸**: 時間（左→右）
    - **縦軸**: 周波数（低→高）
    - **色**: 振幅（明るいほど強い）
    """)
    return


@app.cell
def _(input_features, reference_text, visualize_mel_spectrogram):
    fig = visualize_mel_spectrogram(
        input_features,
        title=f"Log-Mel Spectrogram\n（音声: 「{reference_text}」）",
    )
    fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Phase 2: Forward（モデル計算）

    ### Whisper の encoder-decoder 構造

    ```
    入力音声
        ↓
    [log-mel spectrogram] shape: (1, 80, 3000)
        ↓
    [Encoder（CNN + Transformer）]
        ↓
    [encoder hidden states] shape: (1, 1500, hidden_size)
        ↓
    [Decoder（Transformer + 自己回帰生成）]
        ↓
    [token ids] shape: (1, seq_len)
    ```

    - **Encoder**: 音声特徴量を内部表現に変換（並列処理）
    - **Decoder**: 内部表現からトークンを1つずつ生成（自己回帰）
    """)
    return


@app.cell
def _(MODEL_NAME, encode_audio, inputs, load_model):
    # モデルのロード
    model = load_model(MODEL_NAME)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"モデルクラス: {type(model).__name__}")
    print(f"パラメータ数: {num_params:,}")

    # Encoder フェーズ
    print("\n--- Encoder ---")
    encoder_outputs = encode_audio(model, inputs["input_features"])
    hidden_states = encoder_outputs.last_hidden_state
    print(f"encoder hidden states shape: {tuple(hidden_states.shape)}")
    print(f"  → (batch_size=1, seq_len=1500, hidden_size={hidden_states.shape[-1]})")
    return (model,)


@app.cell
def _(generate_tokens, inputs, model):
    # Decoder フェーズ（自己回帰生成）
    print("--- Decoder（自己回帰生成）---")
    predicted_ids = generate_tokens(model, inputs, language="japanese")
    print(f"token ids shape: {tuple(predicted_ids.shape)}")
    print(f"token ids: {predicted_ids.tolist()}")
    print(f"生成トークン数: {predicted_ids.shape[1]}")
    return (predicted_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## Phase 3: Postprocess（後処理）

    ### token ids からテキストへの変換

    1. **token id → サブワードに変換**
       - 語彙辞書（vocabulary）でIDを文字列に変換

    2. **特殊トークンを除去**
       - `<|startoftranscript|>`, `<|ja|>`, `<|endoftext|>` などを削除

    3. **テキストに結合**
       - サブワードを連結して最終的な文字列を生成
    """)
    return


@app.cell
def _(MODEL_NAME, decode_tokens, load_processor, predicted_ids):
    processor = load_processor(MODEL_NAME)

    # 特殊トークンを含む raw デコード結果（教育用）
    raw_decoded = processor.tokenizer.decode(predicted_ids[0])
    print(f"raw decode（特殊トークンあり）: {raw_decoded}")

    # 特殊トークンを除去した最終テキスト
    transcription = decode_tokens(processor, predicted_ids)
    print(f"\n最終テキスト: {transcription}")
    return processor, transcription


@app.cell
def _(reference_text, transcription):
    print("=== 結果の確認 ===")
    print(f"正解テキスト: {reference_text}")
    print(f"予測テキスト: {transcription}")
    print("\npipeline の出力と同じ結果になっているはず！")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 評価指標

    音声認識（ASR）では、書き起こしたテキストの正確さを評価する。

    ### WER（Word Error Rate）

    英語などの分かち書き言語で標準的な指標。単語単位で誤りをカウントする。

    ```
    WER = (S + D + I) / N
      S: 置換（substitution）  ← 間違ったワードに変わった
      D: 削除（deletion）      ← ワードが消えた
      I: 挿入（insertion）     ← 余分なワードが入った
      N: 正解の単語数
    ```

    ### CER（Character Error Rate）

    日本語に適した指標。文字レベルで WER と同様に計算する。

    ```
    正解: 今日は晴れです      （8文字）
    予測: 今日は晴れでした    （9文字）
    CER = 編集距離 2 / 8 = 25%
    ```

    → 日本語は単語の境界が不明確なため、CER の方が適している
    """)
    return


@app.cell
def _(compute_cer, compute_wer, reference_text, transcription):
    # レベル1（直感的）: 1件の正誤判定
    cer = compute_cer(reference_text, transcription)
    wer = compute_wer(reference_text, transcription)

    print(f"正解: {reference_text}")
    print(f"予測: {transcription}")
    print(f"CER: {cer:.1%}")
    print(f"WER: {wer:.1%}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### レベル2（定量的）：複数サンプルで CER を測定

    複数の音声サンプルで推論を実行し、全体の CER を計算する。
    """)
    return


@app.cell
def _(
    TARGET_SAMPLING_RATE,
    decode_tokens,
    evaluate_batch,
    extract_features,
    feature_extractor,
    generate_tokens,
    model,
    processor,
    resample_audio,
):
    from datasets import load_dataset

    NUM_EVAL_SAMPLES = 5

    print("データセットをロード中...")
    # データセットを一度だけロードしてキャッシュする（毎ループ再ロードを避ける）
    eval_dataset = load_dataset(
        "japanese-asr/ja_asr.jsut_basic5000",
        split=f"test[:{NUM_EVAL_SAMPLES}]",
    )

    print(f"評価中（{NUM_EVAL_SAMPLES}件）...")

    eval_references = []
    eval_hypotheses = []

    for i, sample in enumerate(eval_dataset):
        audio_i = sample["audio"]["array"]
        sr_i = sample["audio"]["sampling_rate"]
        ref_i = sample["transcription"]

        resampled_i = resample_audio(
            audio_i, orig_sr=sr_i, target_sr=TARGET_SAMPLING_RATE
        )
        inputs_i = extract_features(
            feature_extractor, resampled_i, TARGET_SAMPLING_RATE
        )
        ids_i = generate_tokens(model, inputs_i, language="japanese")
        hyp_i = decode_tokens(processor, ids_i)
        eval_references.append(ref_i)
        eval_hypotheses.append(hyp_i)
        print(f"  [{i + 1}/{NUM_EVAL_SAMPLES}] 完了")

    results, avg_wer, avg_cer = evaluate_batch(eval_references, eval_hypotheses)

    print(f"\n{'正解テキスト':<25} {'予測テキスト':<25} {'CER':>6}")
    print("-" * 65)
    for r in results:
        print(f"{r['reference'][:22]:<22} {r['hypothesis'][:22]:<22} {r['cer']:.1%}")

    print(f"\n平均 CER: {avg_cer:.1%}（{NUM_EVAL_SAMPLES}件）")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ---

    ## 発展課題

    1. **モデルを変えてみよう**
       - `openai/whisper-tiny` → より小さく速い
       - `openai/whisper-medium` → より高精度
       - CER の変化を比較してみよう

    2. **別の言語を試してみよう**
       - `language="english"` で英語音声を認識
       - `language=None` で自動言語検出

    3. **長い音声を試してみよう**
       - Whisper は最大 30秒の音声を1回で処理できる
       - それより長い音声はどう処理する？

    ---

    ## 参考リンク

    - [Automatic Speech Recognition - HuggingFace](https://huggingface.co/docs/transformers/ja/tasks/asr)
    - [openai/whisper-small - HuggingFace](https://huggingface.co/openai/whisper-small)
    - [WER の解説 - Hugging Face evaluate](https://huggingface.co/spaces/evaluate-metric/wer)
    - [jiwer ドキュメント](https://jiwer.readthedocs.io/en/latest/)
    - [Audio classification - HuggingFace](https://huggingface.co/docs/transformers/ja/tasks/audio_classification)
    """)
    return


if __name__ == "__main__":
    app.run()
