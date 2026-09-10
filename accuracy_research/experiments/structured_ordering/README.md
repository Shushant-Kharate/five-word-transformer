# From-Scratch Structured Ordering Transformer

This experiment uses no pretrained weights and no external data.

It treats the input as an unordered five-word set, so the encoder has no source positional encoding. Word representations combine a trainable word embedding with a character CNN trained only on the project vocabulary. A Transformer set encoder produces contextual word representations. Structured unary and pairwise precedence heads score all 120 source-position permutations, and listwise likelihood directly optimizes the globally correct ordering.

This removes four baseline mismatches:

- no nuisance source order;
- no 2,412-way vocabulary generation;
- no illegal-token label smoothing;
- no autoregressive/local greedy decision rule.

Training uses one row per unique training `(word multiset, target)` pair rather than replaying every materialized permutation. Validation selection uses the fixed validation split only. The protected test is not opened.
