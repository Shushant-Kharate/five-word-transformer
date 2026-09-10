# Five-Word Sentence Reconstruction Transformer — Detailed Project Summary

## 1. Project objective

The project receives five English words in a jumbled order and reconstructs them into a meaningful five-word sentence.

Example:

```text
Input:      can i phone use your
Prediction: can i use your phone
```

This is:

- A sentence-reconstruction task.
- An encoder–decoder sequence-to-sequence problem.
- Not language translation.
- Not a regression problem.
- Not ordinary text generation, because the output must contain exactly the same five input words.

The Transformer must learn the correct order of the words.

## 2. Dataset

The current base dataset contains:

```text
4,581 five-word sentence records
```

Every record contains:

```json
{
  "input": "can i phone use your",
  "target": "can i use your phone"
}
```

The main conditions are:

- Input contains exactly five words.
- Target contains exactly five words.
- Only alphabetic words are permitted.
- No digits or punctuation.
- No articles `a`, `an`, or `the`.
- Input and target contain exactly the same words.
- Target is the meaningful sentence.
- Input is a permutation of the target.

### Permutation augmentation

For every target sentence, the program generates every possible unique ordering of its five words.

For five different words:

$$
5! = 120
$$

Therefore, a sentence such as:

```text
can i use your phone
```

produces 120 different input arrangements, all with the same target.

If words are repeated, duplicate arrangements are removed:

| Word pattern | Unique permutations |
|---|---:|
| Five different words | 120 |
| One word repeated twice | 60 |
| Two words each repeated twice | 30 |

Current distribution:

| Source-sentence type | Sentences | Permutations each |
|---|---:|---:|
| Five different words | 4,454 | 120 |
| One repeated word | 124 | 60 |
| Two repeated pairs | 3 | 30 |

This produced:

```text
542,010 total input-target records
```

The correctly ordered arrangement is included as one of the permutations.

## 3. Dataset splitting

The 4,581 base sentences were split in an 80:10:10 ratio before assigning their permutations:

| Split | Base sentences | Permutation records |
|---|---:|---:|
| Training | 3,665 | 433,680 |
| Validation | 458 | 54,540 |
| Testing | 458 | 53,790 |
| **Total** | **4,581** | **542,010** |

The split was performed by target sentence.

This means that all permutations of one target remain in the same split. For example, if:

```text
can i use your phone
```

belongs to testing, none of its 120 permutations can appear in training or validation. This prevents data leakage.

The training split contains 3,664 unique targets because one target sentence is duplicated in the original data.

## 4. Tokenizer

The project uses a WordLevel tokenizer.

Each complete word is represented by one token ID. It does not divide words into subwords or characters.

Current vocabulary:

```text
Ordinary word tokens: 2,408
Special tokens:           4
Total vocabulary:     2,412
```

Special tokens:

```text
[UNK]  Unknown word
[PAD]  Padding
[SOS]  Start of sentence
[EOS]  End of sentence
```

The tokenizer was created from the training data only. Validation and test words were not used to train the tokenizer.

Every validation and test word is nevertheless covered by the training vocabulary.

## 5. Sequence construction

Although the sentence contains five words, the Transformer sequence length is seven:

```text
[SOS] word1 word2 word3 word4 word5 [EOS]
```

Therefore:

```text
Sentence words:          5
Special tokens:          2
Encoder sequence length: 7
```

For training, three main sequences are constructed.

### Encoder input

```text
[SOS] jumbled1 jumbled2 jumbled3 jumbled4 jumbled5 [EOS]
```

Shape for one sentence:

```text
(7)
```

Shape for a batch of 32:

```text
(32, 7)
```

### Decoder input

Teacher forcing gives the decoder the correctly ordered target shifted right:

```text
[SOS] target1 target2 target3 target4 target5 [PAD]
```

Shape:

```text
(32, 7)
```

### Label

The expected output is:

```text
target1 target2 target3 target4 target5 [EOS] [PAD]
```

Shape:

```text
(32, 7)
```

The padding position is ignored while calculating loss.

## 6. Transformer configuration

The current architecture is:

| Setting | Value |
|---|---:|
| Batch size | 32 |
| Sequence length | 7 |
| Vocabulary size | 2,412 |
| Embedding/model dimension | 64 |
| Encoder blocks | 2 |
| Decoder blocks | 2 |
| Attention heads | 4 |
| Dimension per head | 16 |
| Feed-forward dimension | 256 |
| Dropout | 0.1 |
| Learning rate | 0.0001 |
| Label smoothing | 0.1 |
| Training epochs | 3 |

The dimension per attention head is:

$$
d_k = \frac{d_{model}}{h} = \frac{64}{4} = 16
$$

Each encoder and decoder block processes the complete batch. Two layers do not mean that training is repeated twice. It means the output of encoder block 1 is passed into encoder block 2. The same happens in the decoder.

## 7. Embedding and positional encoding

The input token IDs initially have shape:

```text
(32, 7)
```

The embedding converts every token into a vector containing 64 numbers:

```text
(32, 7) → (32, 7, 64)
```

The source embedding table has shape:

```text
(2412, 64)
```

The target embedding table also has shape:

```text
(2412, 64)
```

They are separate learnable embedding matrices.

Sinusoidal positional encoding has shape:

```text
(1, 7, 64)
```

It is added to the embeddings so the model knows the position of every word.

## 8. Encoder

The encoder contains two identical blocks.

Each block contains:

1. Multi-head self-attention.
2. Residual connection and layer normalization.
3. Feed-forward network.
4. Another residual connection and layer normalization.

Encoder input and output shapes remain:

```text
(32, 7, 64)
```

The encoder does not change the sequence length or model dimension. It changes the values inside the representations.

### Encoder self-attention

The encoder creates query, key and value tensors:

```text
Q = (32, 7, 64)
K = (32, 7, 64)
V = (32, 7, 64)
```

After splitting into four attention heads:

```text
(32, 4, 7, 16)
```

The dimensions mean:

```text
32 = sentences in the batch
4  = attention heads
7  = sequence positions
16 = dimensions handled by each head
```

Attention scores have shape:

```text
(32, 4, 7, 7)
```

This is a four-dimensional tensor. For every sentence and attention head, every sequence position receives a score for every other sequence position.

After combining the heads:

```text
(32, 7, 64)
```

## 9. Decoder

The decoder also contains two blocks.

Each decoder block contains:

1. Masked self-attention.
2. Encoder–decoder cross-attention.
3. Feed-forward network.
4. Residual connections and layer normalization.

Decoder input and output shape:

```text
(32, 7, 64)
```

### Masked decoder self-attention

The decoder is autoregressive. When predicting a word, it must not look at future target words.

The causal decoder mask has shape:

```text
(32, 1, 7, 7)
```

Decoder self-attention scores have shape:

```text
(32, 4, 7, 7)
```

### Cross-attention

Cross-attention connects the decoder to the encoder.

- Query comes from the decoder.
- Key comes from the encoder.
- Value comes from the encoder.

Its attention-score shape is also:

```text
(32, 4, 7, 7)
```

Cross-attention allows the decoder to examine the five jumbled input words when deciding which word belongs in the next output position.

## 10. Feed-forward network

Every encoder and decoder block contains a position-wise feed-forward network.

It expands each 64-dimensional representation to 256 dimensions:

```text
(32, 7, 64)
    ↓
(32, 7, 256)
    ↓
(32, 7, 64)
```

The first layer learns more complex features in a larger space. The second layer returns the representation to the Transformer's 64-dimensional model space.

## 11. Output projection

The decoder produces:

```text
(32, 7, 64)
```

The projection layer converts each 64-dimensional representation into scores for all 2,412 vocabulary tokens:

```text
(32, 7, 64)
    ↓
(32, 7, 2412)
```

The final tensor contains logits.

For every sequence position, the model assigns one score to every vocabulary token.

The projection weight matrix has shape:

```text
(2412, 64)
```

Its bias has shape:

```text
(2412)
```

Softmax converts the logits into probabilities:

```text
(32, 7, 2412)
```

## 12. Source-constrained decoding

The original Transformer can select any word from its complete vocabulary. That could produce words that were not present in the input.

This project adds source-constrained decoding.

At every prediction step:

1. The Transformer calculates scores for all 2,412 tokens.
2. The decoder identifies which input words remain unused.
3. All tokens outside the remaining input words are excluded.
4. The highest-scoring permitted word is selected.
5. That word is removed from the remaining-word counter.
6. `[EOS]` is produced after all five words have been used.

Repeated words are tracked using counts.

For example, if the input contains `she` twice, the decoder is allowed to select `she` twice but not three times.

This guarantees:

```text
Output contains exactly five words.
Output contains the same words as input.
No outside vocabulary word can enter the prediction.
Repeated words are preserved correctly.
```

However, it does not guarantee correct grammar. The Transformer is still responsible for deciding the order.

That is why an output can be:

```text
VALID:      True
WORDS KEPT: True
```

while still being grammatically incorrect.

## 13. Training

Training uses teacher forcing and cross-entropy loss.

For a full batch:

```text
Logits: (32, 7, 2412)
Labels: (32, 7)
```

Before cross-entropy:

```text
Flattened logits: (224, 2412)
Flattened labels: (224)
```

because:

$$
32 \times 7 = 224
$$

The loss is one scalar value:

```text
shape = ()
```

The Adam optimizer then:

1. Calculates gradients.
2. Updates every learnable parameter.
3. Repeats for the next batch.

Training batches per epoch:

```text
433,680 ÷ 32 = 13,552 full batches + 1 partial batch
Total = 13,553 batches
```

The model trained for three complete augmented epochs:

```text
13,553 × 3 = 40,659 optimizer updates
```

Three augmented epochs contain considerably more optimizer updates than the earlier 100-epoch experiment on the much smaller dataset.

## 14. Learnable parameters

The trained model contains:

```text
697,708 learnable parameters
68 learnable parameter tensors
```

Parameter distribution:

| Component | Learnable parameters |
|---|---:|
| Encoder | 99,584 |
| Decoder | 132,608 |
| Source embedding | 154,368 |
| Target embedding | 154,368 |
| Vocabulary projection | 156,780 |
| **Total** | **697,708** |

Major weight dimensions include:

```text
Source embedding:       (2412, 64)
Target embedding:       (2412, 64)
Attention matrices:     (64, 64)
Feed-forward expansion: (256, 64)
Feed-forward reduction: (64, 256)
Projection weight:      (2412, 64)
Projection bias:        (2412)
LayerNorm parameters:   (64)
```

The model has 35 two-dimensional learnable matrices and 33 one-dimensional learnable vectors.

The three- and four-dimensional objects seen during execution are intermediate tensors, not individual learnable parameter matrices.

## 15. Validation and test results

The best checkpoint was produced by the third epoch, stored internally as epoch `2` because Python counts from zero.

| Metric | Validation | Test |
|---|---:|---:|
| Records | 54,540 | 53,790 |
| Exact matches | 29,595 | 30,598 |
| Exact-match accuracy | 54.26% | 56.88% |
| Word-position accuracy | 71.36% | 73.03% |
| Valid five-word outputs | 100% | 100% |
| Input words preserved | 100% | 100% |
| Loss | 2.2513 | 2.1673 |

The 100% word-preservation result comes from source-constrained decoding.

The 56.88% test exact-match result means that approximately 57 out of every 100 test permutations were reconstructed into the exact target order.

## 16. Complete project flow

```text
4,581 meaningful five-word targets
        ↓
Generate every unique permutation
        ↓
542,010 input-target records
        ↓
Leakage-free grouped 80:10:10 split
        ↓
WordLevel tokenizer with 2,412 tokens
        ↓
Token IDs: (batch, 7)
        ↓
Embedding: (batch, 7, 64)
        ↓
Two encoder blocks
        ↓
Encoder output: (batch, 7, 64)
        ↓
Two autoregressive decoder blocks
        ↓
Decoder output: (batch, 7, 64)
        ↓
Vocabulary logits: (batch, 7, 2412)
        ↓
Restrict choices to unused input words
        ↓
Five ordered words followed by [EOS]
```

In one sentence: this project is a compact encoder–decoder Transformer that learns grammatical word ordering from every possible permutation of 4,581 five-word sentences, while constrained decoding guarantees that the output preserves exactly the words supplied in the input.
