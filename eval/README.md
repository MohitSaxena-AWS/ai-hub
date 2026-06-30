# Duplicate-detection calibration

The duplicate service turns a continuous similarity score into a binary
"duplicate?" decision with a single threshold (`DUPLICATE_SIMILARITY_THRESHOLD`,
or each backend's tuned default). Lexical-overlap and embedding-cosine scores
live on different scales, so each backend needs its own cut-off — and that cut-off
should be **measured**, not guessed.

This folder holds a small labelled set of justification pairs and a script that
sweeps the threshold and reports precision / recall / F1, so the chosen value is
backed by evidence and its cost (false positives vs. missed duplicates) is
visible.

## Run it

```bash
# Dependency-free lexical backend (no ML install):
python -m eval.calibrate --backend lexical

# Local embedding backend (needs the embeddings extra):
pip install -e ".[embeddings]"
python -m eval.calibrate --backend embedding

# Pick the highest-recall threshold that still keeps precision >= 0.90:
python -m eval.calibrate --backend embedding --min-precision 0.90
```

## What the seed set shows today

The dataset (`dataset.py`, 28 pairs) is deliberately **hard**: the negatives are
*same-category* requests that share boilerplate vocabulary
(`"provision a new Kafka cluster for the payments platform"` vs
`"...for the fraud-detection platform"`) — exactly the false-positive risk that
matters for a bank.

| Backend | Mean(dup) | Mean(non) | Separation | At default | Best F1 |
|---------|-----------|-----------|------------|------------|---------|
| `lexical` (Jaccard) | 0.36 | 0.61 | **−0.25** | thr 0.35 → P=0.37 R=0.50 | F1 0.67 only at thr 0 |
| `embedding` (bge-small) | 0.93 | 0.84 | +0.08 | thr **0.90** → P=0.78 R=1.00 F1=0.88 | thr 0.90 |

Two findings drive design decisions:

1. **Lexical has negative separation on hard negatives.** Token overlap can't tell
   "Kafka for payments" from "Kafka for fraud" — they share almost every word.
   This confirms `lexical` is only a dependency-free *fallback* for offline runs,
   never a backend to trust for precision. (`tests/test_dedup_quality.py` encodes
   this as a characterization guard so nobody silently promotes it to default.)
2. **The embedding default was retuned 0.80 → 0.90** based on this sweep: at 0.80
   precision is only ~0.54 on same-category pairs; 0.90 maximises F1. See the
   comment on `LocalEmbeddingSimilarity.default_threshold`.

## Honest limitations / next steps

This is an *illustrative seed*, not a production benchmark — 28 pairs, authored by
one person. Before production:

- Sample real (anonymised) justifications from the live request stream.
- Get **multiple annotators** and measure inter-annotator agreement; "duplicate"
  is genuinely ambiguous at the margin.
- Grow to a few hundred pairs spanning every request_type / environment, and
  hold out a test split so the threshold isn't fit on the data it's judged by.
- Track the operating point over time (drift) and alert if precision degrades.
