# Pointer Assignment Transformer v1

This controlled experiment replaces unrestricted vocabulary generation with the task the application actually needs: assign each of the five source positions to one of five output positions.

The model contains a Transformer encoder over the shuffled words and a Transformer decoder over five learned output-position queries. A bilinear pointer head produces a 5×5 assignment matrix. At inference, all 120 one-to-one assignments are scored and the highest-scoring permutation is returned. Consequently, every output always contains exactly the source multiset.

Selection uses validation exact match only. The protected test split is not opened by this script.

Run from the copied project root with:

```powershell
& 'C:\Users\Shushant\Desktop\five_word_accuracy_gpu_env\Scripts\python.exe' `
  'accuracy_research\experiments\pointer_assignment_v1\train.py'
```
