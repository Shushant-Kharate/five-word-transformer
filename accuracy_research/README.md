# Accuracy Research Workspace

This directory contains accuracy-improvement research for the five-word
sentence reconstruction Transformer. The copied baseline project outside this
directory is treated as read-only reference material. The canonical project at
`C:\Users\Shushant\Desktop\5 word transformer` is never modified.

Primary metric: validation exact-match accuracy.

Protected-test rule: architecture, preprocessing, optimization, decoding, and
checkpoint selection use training and validation only. The 53,790-record test
split is evaluated only once after the final candidate is frozen.

## Experiment sequence

1. Reproduce the saved vocabulary-decoder baseline.
2. Audit split integrity, target ambiguity, repeated words, and data quality.
3. Diagnose validation errors from saved predictions.
4. Compare controlled Transformer variants, prioritizing pointer and
   permutation-based output spaces.
5. Freeze the best validation-only system and run one protected test.

