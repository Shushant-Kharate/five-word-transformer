# Architecture-only training on 20,500 targets

This is the new training path. It leaves the frozen 4,581-target experiments,
checkpoints, validation metrics, and protected test result unchanged.

The code implements the findings from the architecture investigation:

- treats the source as an unordered set without source positional encoding;
- combines learned word identity with a from-scratch character CNN;
- contextualizes the five words with a Transformer set encoder;
- scores only the 120 legal permutations instead of the full vocabulary;
- optimizes global listwise ordering loss rather than local token generation;
- includes learned unary, precedence, and adjacency evidence;
- estimates transparent structural priors from training targets only;
- uses dynamic source permutations instead of stored permutation duplication;
- preserves repeated words by source position;
- guarantees five-word validity and source-word preservation by construction;
- uses AdamW, gradient clipping, warmup, cosine learning-rate decay, and early stopping;
- selects checkpoints using validation exact match only; and
- never evaluates the protected test during training.

`prepare_data.py` creates a deterministic 16,400 / 2,050 / 2,050 target-group
split and trains a WordLevel whitespace tokenizer from the training split only.

The intended ensemble is one structured component plus candidate components
trained from seeds 42, 43, and 44. No pretrained model or external Transformer
weights are loaded.
