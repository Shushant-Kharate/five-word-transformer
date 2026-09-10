# Five-Word Sentence Reconstruction Transformer

## Project objective

This project builds an encoder–decoder Transformer that receives five jumbled English words and generates the same five words in the correct order.

```text
Input:  winds leveled home nearby his
Target: winds leveled his nearby home
```

This is a sentence-reconstruction task, not language translation and not regression.

## Architecture source and attribution

The Transformer architecture and initial code organisation are adapted from:

> hkproj, `pytorch-transformer`  
> https://github.com/hkproj/pytorch-transformer

The project retains the reference implementation's from-scratch PyTorch structure:

- input embeddings scaled by `sqrt(d_model)`;
- sinusoidal positional encoding;
- custom layer normalisation;
- pre-normalised residual connections;
- scaled dot-product multi-head attention;
- encoder self-attention;
- masked decoder self-attention;
- encoder–decoder cross-attention;
- position-wise feed-forward blocks;
- vocabulary projection;
- Xavier uniform parameter initialisation;
- teacher-forced cross-entropy training; and
- greedy autoregressive decoding with source-word candidate masking.

The local data pipeline, configuration, tokenizer, metrics, model selection, test protection, and inference validation were adapted for five-word English reconstruction.

## Current status

```text
Dataset preparation:              complete
Tokenizer and dataset tensors:    complete
Transformer architecture:         complete
Training pipeline:                complete
Validation and model selection:   complete
Final-test evaluator code:        complete
Five-word inference code:         complete
Automated tests:                  31 passing
Real model training:              complete (100 epochs)
Validation review:                complete; epoch 99 confirmed
Final test result:                complete; protected outputs written
Inference demonstration:         complete
Final project report:             complete
Source-constrained decoding:      complete
Pre-training readiness:          READY
```

The user-facing decoder now applies a hard permutation constraint: at each
word position it can select only an unused token from the five-word input.
Repeated words are tracked by count, and `[EOS]` is emitted only after all five
input tokens have been used. This guarantees word preservation while leaving
the Transformer responsible for predicting their order.

The real training run completed for all 100 configured epochs. The best
validation checkpoint is epoch 99: validation loss `4.1977`, exact-match
accuracy `0.0273` (3/110), word-position accuracy `0.3109`, output-validity
rate `0.9727`, and input-word-preservation rate `0.0273`. These are validation
results from the original **unconstrained baseline**, not results produced by
the later source-constrained decoder.

The one-time held-out test evaluation has now been completed with the frozen
epoch-99 checkpoint. The final test result is: loss `3.9014`, exact-match
accuracy `0.0182` (2/110), word-position accuracy `0.3327`, output-validity
rate `0.9545`, and input-word-preservation rate `0.0182`. This protected test
result also belongs to the original unconstrained baseline. It has not been
rerun or overwritten after the decoding correction.

## Dataset

The original dataset contained 1,179 five-word pairs. Records containing a digit in either input or target were removed.

```text
Filtered records:    1,098
Training records:      878 (80%)
Validation records:    110 (10%)
Test records:          110 (10%)
```

Each JSONL record has this structure:

```json
{"input":"items enemies some contain bonus","target":"some enemies contain bonus items"}
```

Data files:

```text
data/NewDataset_fixed.jsonl
data/train_fixed.jsonl
data/validation_fixed.jsonl
data/test_fixed.jsonl
data/targets.txt
```

`targets.txt` contains all 1,098 current target sentences exactly once. The previous contraction-form target list is preserved in `data/archive/targets_before_fixed_dataset.txt`.

## Tokenizer

The project uses the reference repository's Hugging Face `WordLevel` tokenizer and whitespace pre-tokenisation.

```text
Dataset word tokens: 1,456
Special tokens:          4
Vocabulary size:     1,460
```

Special-token IDs:

```text
[UNK] = 0
[PAD] = 1
[SOS] = 2
[EOS] = 3
```

The vocabulary is created from the filtered dataset's input words. Every target is a permutation of its input, so this covers required output words without using target order. The experiment is therefore described as closed-vocabulary five-word reconstruction.

## Model configuration

| Setting | Value |
|---|---:|
| Batch size | 32 |
| Model sequence length | 7 |
| Embedding/model dimension | 64 |
| Encoder blocks | 2 |
| Decoder blocks | 2 |
| Attention heads | 4 |
| Dimension per head | 16 |
| Feed-forward dimension | 256 |
| Dropout | 0.1 |
| Vocabulary size | 1,460 |
| Learnable parameters | 513,972 |
| Adam learning rate | 0.0001 |
| Label smoothing | 0.1 |
| Maximum epochs | 100 |
| Early-stopping patience | 15 epochs |
| Seed | 42 |

The dataset sentence contains five words. The model sequence contains seven tokens:

```text
[SOS] word1 word2 word3 word4 word5 [EOS]
```

## Model flow

```text
Five jumbled words
→ shared word tokenizer
→ source embedding and positional encoding
→ two encoder blocks
→ encoded source representation
→ two autoregressive decoder blocks
→ 1,460-token vocabulary projection
→ restrict each choice to an unused input token
→ five ordered words and [EOS]
```

The source and target use the same token vocabulary but retain separate learnable source and target embedding matrices, matching the reference architecture.

## Project structure

```text
5 word transformer/
├── config.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── translate.py
├── preflight.py
├── tokenizer.json
├── requirements.txt
├── README.md
├── notebooks/
│   └── MODEL_PARAMETERS_VISUALIZATION.ipynb
├── data/
├── docs/
└── tests/
```

Runtime folders are created only when needed:

```text
weights/  epoch and best-model checkpoints
runs/     TensorBoard event logs
results/  protected final test predictions and summary
```

## Installation

The project is designed for Python 3.11.

PowerShell example:

```powershell
cd "C:\Users\Shushant\Desktop\5 word transformer"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the five direct project dependencies and their
complete pinned transitive dependency closure (36 packages total), verified
against the working Python 3.11.9 CPU environment. Python standard-library
modules such as `argparse`, `json`, `pathlib`, and `unittest` do not require
separate installation.

## Run the automated tests

```powershell
python -m unittest discover -s tests -v
```

All tests should pass before starting training.

## Explore the model parameters visually

Open this notebook in VS Code or Jupyter and run all cells from top to bottom:

```text
notebooks/MODEL_PARAMETERS_VISUALIZATION.ipynb
```

Select the project's `myenv` Python 3.11 kernel. The notebook loads the actual
epoch-99 `best_model.pt` checkpoint without changing it, lists all 68 learnable
parameter tensors, verifies the 513,972-parameter total, charts the parameter
distribution by component, displays trained-weight heatmaps, traces 2D/3D/4D
tensor shapes, and visualizes all four attention heads for a validation sample.

## Run the pre-training readiness gate

Immediately before starting the real experiment:

```powershell
python preflight.py
```

It verifies exact dependencies, frozen data/tokenizer hashes, split integrity, model size, finite forward loss, available disk space, and the absence of previous experiment outputs. The approved data hashes are stored in `data/data_manifest.json`.

## Train the model

Training requires an explicit flag:

```powershell
python train.py --train
```

For every epoch, the pipeline:

1. trains on all 878 training records;
2. calculates validation loss;
3. greedily reconstructs all 110 validation records;
4. calculates reconstruction metrics;
5. saves an epoch checkpoint;
6. updates `weights/best_model.pt` when validation improves; and
7. stops after 15 epochs without improvement or at the 100-epoch maximum.

Run without the flag to confirm readiness without training:

```powershell
python train.py
```

## Monitor training

After training creates logs:

```powershell
tensorboard --logdir runs
```

Logged values include training loss, validation loss, exact-match accuracy, word-position accuracy, output validity, and input-word preservation.

## Final test evaluation

The test evaluator loads only `weights/best_model.pt` and refuses to overwrite an existing final result.

The protected evaluation was run once after training and validation review:

```powershell
python evaluate.py --evaluate
```

It creates:

```text
results/test_predictions.jsonl
results/test_summary.json
```

Both files now exist and are the final outputs for this experiment. Do not run
the evaluator again and do not use the test result to change the architecture,
training settings, checkpoint, or decoding method.

## Reconstruct a user sentence

After training:

```powershell
python translate.py --sentence "in said it i jest"
```

Inference requires:

- exactly five whitespace-separated words;
- no digits;
- one tokenizer token per word; and
- every word to be present in the fixed project vocabulary.

The command reports the prediction, whether it ended correctly, and whether
the five input words were preserved. For every accepted input, the constrained
decoder guarantees five output words, `[EOS]`, and exact source-word multiset
preservation. The model still decides the order by comparing its learned scores
for the currently unused input words.

## Evaluation metrics

| Metric | Meaning |
|---|---|
| Cross-entropy loss | Token-prediction error; `[PAD]` is ignored. |
| Exact sentence-match accuracy | All five words and `[EOS]` must be correct. Primary metric. |
| Word-position accuracy | Fraction of target positions containing the correct word. |
| Output-validity rate | Output contains five words and ends with `[EOS]`. |
| Input-word-preservation rate | Output contains exactly the source-word multiset. |

The best checkpoint is chosen by higher validation exact-match accuracy. Lower validation loss breaks a tie.

## Main differences from the reference project

| Reference repository | This project | Reason |
|---|---|---|
| English-to-Italian translation | Jumbled-English-to-ordered-English reconstruction | Different task. |
| OPUS Books loaded online | Fixed local JSONL files | Assignment dataset already exists. |
| Separate language tokenizers | One shared English token mapping | Source and target use the same words. |
| Sequence length 350 | Sequence length 7 | Five words plus `[SOS]` and `[EOS]`. |
| `d_model=512`, 6 layers, 8 heads | `d_model=64`, 2 layers, 4 heads | Appropriate for the small dataset. |
| `d_ff=2048` | `d_ff=256` | Preserves the standard four-times expansion. |
| Batch size 8 | Batch size 32 | Short fixed sequences fit efficiently. |
| Translation metrics | Reconstruction metrics | Measures exact word ordering directly. |
| Train/validation only | Fixed train/validation/test files | Provides an unbiased final evaluation. |
| Epoch checkpoints | Epoch and best-validation checkpoints | Selects the best generalising model. |
| Immediate script training | Explicit `--train` flag | Prevents accidental long execution. |
| Full-vocabulary greedy choice | Greedy choice among unused source tokens | Prevents invented, omitted, or over-repeated words. |

## Detailed documentation

```text
docs/IMPLEMENTATION_PLAN.md
docs/PROJECT_EXPLANATION.md
docs/SECTION_1_DATA_AUDIT.md
docs/SECTION_2_TOKENIZER_AND_DATASET.md
docs/SECTION_3_TRANSFORMER_ARCHITECTURE.md
docs/SECTION_4_TRAINING_PIPELINE.md
docs/SECTION_5_VALIDATION_PIPELINE.md
docs/SECTION_6_FINAL_TEST_EVALUATION.md
docs/SECTION_7_INFERENCE_AND_DOCUMENTATION.md
docs/SECTION_8_PRETRAINING_READINESS.md
docs/SECTION_9_REAL_TRAINING.md
docs/SECTION_10_VALIDATION_REVIEW.md
docs/SECTION_11_FINAL_TEST_EVALUATION.md
docs/SECTION_12_INFERENCE_AND_FINAL_REPORT.md
docs/SOURCE_CONSTRAINED_DECODING.md
docs/FINAL_PROJECT_REPORT.md
```

These files explain what was implemented, how tensor dimensions flow, which reference code was preserved, and why every project-specific change was necessary.

## Result-reporting rule

The correct final result statement for this fixed experiment is:

> The epoch-99 checkpoint was selected using validation only and evaluated once on 110 held-out test sentences. It achieved 2 exact reconstructions (1.82%), 33.27% word-position accuracy, 95.45% output validity, and 1.82% complete input-word preservation.

This statement is the immutable result for the original unconstrained
experiment. The source-constrained inference correction was added afterward;
no new held-out test score is claimed.
