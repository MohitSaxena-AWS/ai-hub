"""Calibrate the duplicate-detection similarity threshold against labeled data.

Runs the configured similarity backend over the labeled justification pairs in
``eval/dataset.py``, sweeps the decision threshold, and reports
precision/recall/F1 at each cut-off — so the threshold is chosen with *evidence*
instead of being asserted.

Why this exists: the duplicate service turns a continuous similarity score into a
binary "duplicate?" decision with a single threshold. Lexical-overlap and
embedding-cosine scores live on different scales, so each backend needs its own
cut-off. This script measures where that cut-off should sit and what it costs
(false positives that annoy engineers vs. missed duplicates).

Usage:

    # Dependency-free lexical backend (no ML install needed):
    python -m eval.calibrate --backend lexical

    # Local embedding backend (requires the 'embeddings' extra):
    python -m eval.calibrate --backend embedding
    python -m eval.calibrate --backend embedding --model BAAI/bge-small-en-v1.5

    # Pick the threshold that keeps precision >= 0.90 (FPs are costly here):
    python -m eval.calibrate --backend embedding --min-precision 0.90

The scoring path mirrors production exactly: text is normalised with the same
``normalize_value`` used by the duplicate service before it is scored.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from app.core.hashing import normalize_value
from app.core.similarity import (
    LexicalSimilarity,
    LocalEmbeddingSimilarity,
    SimilarityProvider,
)
from eval.dataset import PAIRS


@dataclass
class Metrics:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 1.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 1.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


def _build_backend(name: str, model: str) -> SimilarityProvider:
    if name == "lexical":
        return LexicalSimilarity()
    if name == "embedding":
        return LocalEmbeddingSimilarity(model)
    raise SystemExit(f"unknown backend: {name!r} (use 'lexical' or 'embedding')")


def score_pairs(backend: SimilarityProvider) -> list[tuple[float, bool]]:
    """Return (similarity_score, is_duplicate) for every labeled pair."""

    scored: list[tuple[float, bool]] = []
    for a, b, is_dup in PAIRS:
        score = backend.similarity(normalize_value(a), normalize_value(b))
        scored.append((score, is_dup))
    return scored


def metrics_at(scored: list[tuple[float, bool]], threshold: float) -> Metrics:
    tp = fp = fn = tn = 0
    for score, is_dup in scored:
        predicted = score >= threshold
        if predicted and is_dup:
            tp += 1
        elif predicted and not is_dup:
            fp += 1
        elif not predicted and is_dup:
            fn += 1
        else:
            tn += 1
    return Metrics(threshold, tp, fp, fn, tn)


def separation(scored: list[tuple[float, bool]]) -> tuple[float, float]:
    """Mean score of duplicate pairs vs. non-duplicate pairs.

    A backend is only useful for this task if duplicates score clearly higher
    than non-duplicates; this is a quick, threshold-free sanity signal.
    """

    pos = [s for s, dup in scored if dup]
    neg = [s for s, dup in scored if not dup]
    return (sum(pos) / len(pos) if pos else 0.0, sum(neg) / len(neg) if neg else 0.0)


def sweep(scored: list[tuple[float, bool]], step: float = 0.05) -> list[Metrics]:
    thresholds = [round(i * step, 4) for i in range(int(1 / step) + 1)]
    return [metrics_at(scored, t) for t in thresholds]


def best_f1(grid: list[Metrics]) -> Metrics:
    return max(grid, key=lambda m: (m.f1, m.precision))


def best_at_precision(grid: list[Metrics], min_precision: float) -> Metrics | None:
    """Highest-recall threshold whose precision meets the floor."""

    eligible = [m for m in grid if m.precision >= min_precision]
    return max(eligible, key=lambda m: (m.recall, m.f1)) if eligible else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="lexical", choices=["lexical", "embedding"])
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument(
        "--min-precision",
        type=float,
        default=0.90,
        help="Precision floor for the recommended threshold (FPs are costly).",
    )
    args = parser.parse_args()

    backend = _build_backend(args.backend, args.model)
    scored = score_pairs(backend)
    grid = sweep(scored, args.step)

    pos = sum(1 for _, dup in scored if dup)
    neg = len(scored) - pos
    mean_pos, mean_neg = separation(scored)

    print(f"Backend            : {args.backend}", end="")
    print(f" ({args.model})" if args.backend == "embedding" else "")
    print(f"Labeled pairs      : {len(scored)}  ({pos} duplicates, {neg} non-duplicates)")
    print(f"Mean score (dup)   : {mean_pos:.3f}")
    print(f"Mean score (non)   : {mean_neg:.3f}")
    print(f"Separation         : {mean_pos - mean_neg:+.3f}")
    print(f"Backend default    : {backend.default_threshold:.3f}")
    print()

    print(f"{'thresh':>7} {'prec':>6} {'recall':>7} {'f1':>6}   TP FP FN TN")
    for m in grid:
        print(
            f"{m.threshold:>7.2f} {m.precision:>6.2f} {m.recall:>7.2f} {m.f1:>6.2f}"
            f"   {m.tp:>2} {m.fp:>2} {m.fn:>2} {m.tn:>2}"
        )
    print()

    bf1 = best_f1(grid)
    print(
        f"Best F1            : threshold={bf1.threshold:.2f}  "
        f"P={bf1.precision:.2f} R={bf1.recall:.2f} F1={bf1.f1:.2f}"
    )
    bp = best_at_precision(grid, args.min_precision)
    if bp is not None:
        print(
            f"Best @P>={args.min_precision:.2f}    : threshold={bp.threshold:.2f}  "
            f"P={bp.precision:.2f} R={bp.recall:.2f} F1={bp.f1:.2f}"
        )
    else:
        print(f"Best @P>={args.min_precision:.2f}    : no threshold meets the precision floor")

    at_default = metrics_at(scored, backend.default_threshold)
    print(
        f"At backend default : threshold={at_default.threshold:.2f}  "
        f"P={at_default.precision:.2f} R={at_default.recall:.2f} F1={at_default.f1:.2f}"
    )


if __name__ == "__main__":
    main()
