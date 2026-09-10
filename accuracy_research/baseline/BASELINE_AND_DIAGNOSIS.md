# Baseline Reconstruction and Diagnosis

## Executive finding

The saved epoch-15 checkpoint reproduces exactly: **31,074 / 54,540 = 56.974697% validation exact match**, with **72.230656% word-position accuracy**, **100% valid five-word outputs**, and **100% source-word preservation**. The reproduction has zero metric drift from the saved training history.

The main limitation is not invalid decoding or sensitivity to the shuffled source order. The model usually learns one stable ordering for a five-word multiset, and that ordering is grammatically wrong for many unseen sentence groups. Across the 458 validation targets, 411 produce one identical prediction for every available source permutation. An oracle that accepts a target if *any* source permutation is decoded correctly raises exact match only to 58.97%.

## Actual task and pipeline

- Every record contains exactly five lowercase alphabetic words.
- The target is a grammatical ordering of exactly the same word multiset as the source.
- All unique source permutations are materialized. Sentence targets, not individual rows, define the train/validation/test split.
- The training split contains 433,680 rows from 3,665 source records and 3,664 unique target strings.
- Validation contains 54,540 rows from 458 target groups. Test contains 53,790 rows from 458 protected target groups.
- The tokenizer is a word-level tokenizer trained only on training inputs. The saved vocabulary has 2,412 entries including special tokens.

## Baseline model

The saved model is a conventional encoder-decoder Transformer with 697,708 parameters:

- model width 64; feed-forward width 256;
- two encoder and two decoder blocks;
- four attention heads;
- sinusoidal positions and dropout 0.1;
- full-vocabulary projection at every output step;
- teacher-forced cross-entropy with label smoothing 0.1;
- Adam at a fixed learning rate of 1e-4;
- no scheduler, gradient clipping, or weight decay.

At inference, the decoder still computes logits over the full 2,412-token vocabulary. A source constraint then selects the highest-scoring unused source word at each step. This guarantees validity and preservation, but the training loss does not directly express the five-way assignment problem used at inference.

## Training behavior

The best validation exact match occurs at epoch 15. Training continues improving while validation exact match falls to 55.53% by epoch 20 and validation loss rises, which is direct evidence of overfitting. Older plots in the copied project contain only the first three epochs and are stale; the graphs in `graphs/` are regenerated from the authoritative 20-row history.

## Data audit

- The 4,581-row base dataset has 4,580 unique target strings and one duplicated target row.
- All rows pass five-word, alphabetic, lowercase, and source/target multiset checks.
- The ordinary vocabulary contains 2,408 words. Of those, 1,091 occur in only one base sentence and 1,763 occur in at most five.
- Validation and test contain no words missing from training.
- Target strings do not overlap between splits.
- Two unordered word multisets overlap between train and validation, and one overlaps between train and test. This is a small grouping defect caused by splitting on exact target strings instead of unordered multisets; it cannot explain the low score.
- Nineteen base word multisets have conflicting target strings. The deterministic label-consistency ceiling is 99.59% on base rows and 99.56% on expanded training rows, far above the current score.

## Error analysis

- 252 validation target groups are correct for every source permutation, 188 are wrong for every permutation, and only 18 are mixed.
- A deterministic lexicographically canonical source achieves 57.21%; modal voting across all source permutations also achieves 57.21%.
- The diagnostic oracle that chooses a correct result whenever any source permutation succeeds is only 58.97%.
- Exact match declines as the rarest word becomes less represented in training: 48.45% for words appearing in one training sentence, 51.58% for 2–5, 61.65% for 6–20, and 64.44% for more than 20.
- Targets containing `-ly` adverbs score 37.50%. Targets containing pronouns score 62.77%, versus 47.05% without pronouns.
- Common failures are stable local or clause-order mistakes, such as `he can be on counted` instead of `he can be counted on` and `how is that tower tall` instead of `how tall is that tower`.

## Concrete trace

For `stop me she can not`, the model creates a 7-position encoder sequence (`SOS`, five words, `EOS`), maps it to shape 1×7×64, and autoregressively produces full-vocabulary logits of shape 1×2,412. The source constraint chooses `she`, `can`, `not`, `stop`, `me`; EOS is forced after all five source tokens are consumed. The trace confirms that inference solves a constrained assignment problem even though training optimizes unrestricted vocabulary prediction.

## Experiment direction

The highest-value next model is a from-scratch permutation-aware Transformer that scores only the five source positions and performs a global one-to-one assignment over all 120 possible position permutations. It should remove source positional encodings, replace isolated word identities with a character-aware word encoder trained only on this dataset, and optimize global permutation likelihood plus pairwise precedence auxiliaries. This removes the full-vocabulary mismatch, guarantees word preservation structurally, and introduces the morphology-sharing bias needed for rare words without using pretrained weights.

The protected test set will not be used to select among these candidates.
