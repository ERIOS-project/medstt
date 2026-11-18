"""Compute error rates and term recall for ASR predictions."""

from __future__ import annotations

from collections.abc import Sequence

import jiwer

from medstt.text import contains_term, normalize_for_scoring


def text_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Return normalized WER/CER and raw WER, ignoring empty references."""
    pairs = [
        (p, r) for p, r in zip(predictions, references, strict=True) if normalize_for_scoring(r)
    ]
    if not pairs:
        return {"wer": float("nan"), "cer": float("nan"), "wer_raw": float("nan")}
    preds, refs = zip(*pairs, strict=True)
    preds_norm = [normalize_for_scoring(p) for p in preds]
    refs_norm = [normalize_for_scoring(r) for r in refs]
    return {
        "wer": jiwer.wer(refs_norm, preds_norm),
        "cer": jiwer.cer(refs_norm, preds_norm),
        "wer_raw": jiwer.wer([r.strip() for r in refs], [p.strip() for p in preds]),
    }


def word_errors(prediction: str, reference: str) -> tuple[int, int]:
    """Return the word errors (substitutions, deletions, insertions) and reference words.

    Both texts are normalized as in :func:`text_metrics`, so that summing the
    errors and words of many utterances gives the same corpus WER.
    """
    ref = normalize_for_scoring(reference)
    if not ref:
        return 0, 0
    hyp = normalize_for_scoring(prediction)
    out = jiwer.process_words(ref, hyp)
    return out.substitutions + out.deletions + out.insertions, len(ref.split())


def term_hits(predictions: Sequence[str], terms: Sequence[str]) -> list[bool]:
    """Tell, for each prediction, whether it contains its term (without case or accents)."""
    return [contains_term(p, t) for p, t in zip(predictions, terms, strict=True)]
