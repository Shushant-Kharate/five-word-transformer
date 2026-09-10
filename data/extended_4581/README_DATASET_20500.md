# Expanded 20,500-sentence base dataset

`dataset_20500.jsonl` is the new base dataset for future experiments. The
original `dataset_4581.jsonl` remains unchanged.

Each record obeys the project rules:

- fields are exactly `input` and `target`;
- input and target each contain exactly five words;
- tokens contain lowercase ASCII letters only;
- articles `a`, `an`, and `the` are excluded;
- input is an exact permutation of target;
- targets are unique; and
- each unordered five-word multiset has exactly one target order.

The dataset contains 4,561 clean, non-conflicting records retained from the
old base plus 15,939 filtered English sentences from Tatoeba. The Tatoeba
text export is released under CC BY 2.0 FR. Attribution: [Tatoeba](https://tatoeba.org/).
Per-record Tatoeba sentence IDs are stored in
`dataset_20500_provenance.jsonl`.

The source export URL is:

`https://downloads.tatoeba.org/exports/per_language/eng/eng_sentences.tsv.bz2`

Build and selection details, hashes, and aggregate statistics are recorded in
`dataset_20500_report.json`. Run `validate_dataset_20500.py` for an independent
full-file integrity audit.

## Required next-stage rule

Split this base dataset by target group **before** generating permutations.
Never randomly split the expanded permutation rows. All permutations of one
target must remain in exactly one of train, validation, or test.

The single shuffled input stored here is only a valid base example. Training
should generate random permutations dynamically, or generate permutations
only after the group split. It is unnecessary to store all 120 permutations
for every target.
