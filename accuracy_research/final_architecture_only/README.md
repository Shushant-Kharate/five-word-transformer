# Frozen architecture-only candidate

This candidate contains no pretrained Transformer, external language model, external corpus, or externally generated labels. It was selected using only the fixed training and validation splits.

The input is treated as an unordered five-word set. Three independently initialized candidate-sequence Transformers and one unary/precedence/adjacency Transformer score all 120 possible source-position permutations. Transparent position, precedence, and adjacency statistics estimated only from unique training targets provide additional fixed scores. The seven score streams are normalized and combined with the constants in `frozen_config.json`; these constants must not be changed after formal test evaluation.

The output is selected by exhaustive search over source positions. Therefore output length, source-word preservation, and duplicate-word handling are guaranteed by construction.

Frozen validation result:

- Exact match: 73.7074%
- Word-position accuracy: 84.0044%
- Original validation baseline: 56.9747%
- Absolute improvement: 16.7327 percentage points
- Learned parameters across the four checkpoints: 2,567,081

The test evaluator refuses to run a second time after `test_metrics.json` has been created.

Validation command:

```powershell
& "C:\Users\Shushant\Desktop\five_word_accuracy_gpu_env\Scripts\python.exe" accuracy_research\final_architecture_only\evaluate.py --split validation
```

One-time formal test command:

```powershell
& "C:\Users\Shushant\Desktop\five_word_accuracy_gpu_env\Scripts\python.exe" accuracy_research\final_architecture_only\evaluate.py --split test --confirm-final-test
```
