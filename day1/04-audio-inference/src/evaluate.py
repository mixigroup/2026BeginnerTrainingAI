"""WER / CER の計算（音声認識の評価指標）"""

from __future__ import annotations

import jiwer


def compute_wer(reference: str, hypothesis: str) -> float:
    """WER（Word Error Rate）を計算する。

    英語などの分かち書き言語で標準的な指標。
    単語レベルで置換・削除・挿入の誤りをカウントする。

    WER = (S + D + I) / N
      S: 置換（substitution）
      D: 削除（deletion）
      I: 挿入（insertion）
      N: 正解の単語数

    Args:
        reference: 正解テキスト
        hypothesis: モデルの予測テキスト

    Returns:
        WER の値（0.0 〜 ∞）、完全一致なら 0.0
    """
    return jiwer.wer(reference, hypothesis)


def compute_cer(reference: str, hypothesis: str) -> float:
    """CER（Character Error Rate）を計算する。

    日本語に適した指標。文字レベルで WER と同様に計算する。

    例:
        正解: 今日は晴れです    （7文字）
        予測: 今日は晴れでした  （8文字）
        CER = 編集距離 2 / 7 ≈ 28.6%

    Args:
        reference: 正解テキスト
        hypothesis: モデルの予測テキスト

    Returns:
        CER の値（0.0 〜 ∞）、完全一致なら 0.0
    """
    return jiwer.cer(reference, hypothesis)


def evaluate_batch(
    references: list[str],
    hypotheses: list[str],
) -> tuple[list[dict], float, float]:
    """複数サンプルを評価して WER / CER を計算する。

    Args:
        references: 正解テキストのリスト
        hypotheses: モデルの予測テキストのリスト

    Returns:
        results: 各サンプルの評価結果リスト
                 各要素: {"reference": str, "hypothesis": str, "cer": float, "correct": bool}
        avg_wer: 全サンプルの平均 WER
        avg_cer: 全サンプルの平均 CER
    """
    if len(references) != len(hypotheses):
        raise ValueError(
            f"references と hypotheses の長さが異なります: "
            f"{len(references)} != {len(hypotheses)}"
        )

    results = []
    for ref, hyp in zip(references, hypotheses):
        cer = compute_cer(ref, hyp)
        is_correct = cer == 0.0
        results.append(
            {
                "reference": ref,
                "hypothesis": hyp,
                "cer": cer,
                "correct": is_correct,
            }
        )

    avg_wer = compute_wer(references, hypotheses)
    avg_cer = compute_cer(references, hypotheses)
    return results, avg_wer, avg_cer
