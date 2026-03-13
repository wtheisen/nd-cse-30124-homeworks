#!/usr/bin/env python3
"""Generate lab04_solutions.ipynb — NLP & Sequence Processing lab prepping for HW05."""
import json

def md(source):
    lines = source.strip("\n").split("\n")
    return {"cell_type": "markdown", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]]}

def code(source):
    lines = source.strip("\n").split("\n")
    return {"cell_type": "code", "metadata": {}, "source": [l + "\n" for l in lines[:-1]] + [lines[-1]], "outputs": [], "execution_count": None}

cells = []

# ═══ CELL: Title ═══
cells.append(md("""# CSE 30124 - Introduction to Artificial Intelligence: Lab 04 (5 pts.)

- NETID:

This assignment covers the following topics:
- Text similarity with `collections.Counter` and cosine similarity
- Character-level tokenization (mapping text to integer sequences)
- Sequence padding with `pad_sequence`
- Embedding layers with `nn.Embedding`
- Recurrent neural networks with `nn.RNN`
- Transformers with `nn.TransformerEncoderLayer` and `nn.TransformerEncoder`

It will consist of 7 tasks:

| Task ID  | Description                                      | Points |
|----------|--------------------------------------------------|--------|
| 00       | Setup                                            | 0      |
| 01       | Text Similarity with Counter                     | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;01-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Counter and cosine similarity demo     | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;01-2     | &nbsp;&nbsp;&nbsp;&nbsp;- BoC similarity + cipher classification | 1      |
| 02       | Tokenization and Sequence Padding                | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;02-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Character tokenization demo            | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;02-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Tokenize and pad a batch               | 1      |
| 03       | nn.Embedding                                     | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;03-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Embedding layer demo                   | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;03-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Embedding lookup exercise              | 1      |
| 04       | nn.RNN                                           | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;04-1     | &nbsp;&nbsp;&nbsp;&nbsp;- RNN layer demo                         | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;04-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Build a CharRNN model                  | 1      |
| 05       | nn.TransformerEncoder                            | 1      |
| &nbsp;&nbsp;&nbsp;&nbsp;05-1     | &nbsp;&nbsp;&nbsp;&nbsp;- Transformer components demo            | 0      |
| &nbsp;&nbsp;&nbsp;&nbsp;05-2     | &nbsp;&nbsp;&nbsp;&nbsp;- Build a Transformer encoder model      | 1      |
| 06       | Generate Police Report                           | 0      |

Please complete all sections. Some questions will require written answers, while others will involve coding. Be sure to run your code cells to verify your solutions."""))

# ═══ CELL: Story Intro ═══
cells.append(md("""## *Story Progression*

Thanks to your OCR work in the last assignment, the forensics team has fully extracted the text from the kidnapping letters. But there's a problem — the text is **encoded**:

```
"v ybirq zheqrevat ze gurvfrasyblq ng gur rfgngr..."
```

It's clearly some kind of cipher, but which one? And how do we decode it? Director Bryant wants you to use AI to crack it — "treat it like a translation problem," he says. Before you can build the decoder in HW05, you need to learn the NLP tools that make sequence processing possible.

<div class="thumbnail">
    <img src="https://williamtheisen.com/nd-cse-30124-homeworks/imgs/lab04/cipher_text.png" class="img-responsive"/>
    <figcaption style="text-align: center"><b>Evidence 1: The encoded ransom note text — what cipher was used?</b></figcaption>
</div>

Detective Gaff tosses you a worn copy of "NLP for Investigators" and says: "Start with the basics — figure out what cipher this is, then we'll train a model to decode it.\""""))

# ═══ CELL: Task 00 Header ═══
cells.append(md("""## Task 00: Setup (0 pts.)
### Task 00: Code (0 pts.)"""))

# ═══ CELL: Task 00 Code ═══
cells.append(code("""import os
import math
import string
import numpy as np
import matplotlib.pyplot as plt

try:
    import google.colab
    REPO_URL = "https://github.com/nd-cse-30124-fa25/cse-30124-homeworks.git"
    REPO_NAME = "cse-30124-homeworks"
    LAB_FOLDER = "labs/lab04"

    %cd /content/
    if not os.path.exists(REPO_NAME):
        !git clone {REPO_URL}

    %cd {REPO_NAME}/{LAB_FOLDER}

    !pip install pycipher -q

except ImportError:
    print("Not running on Colab - assuming local setup.")

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

print(f"PyTorch version: {torch.__version__}")

device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device:    {device}")
print("Setup complete!")"""))

# ═══════════════════════════════════════════════════════════════
# TASK 01: Text Similarity with Counter
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 01: Text Similarity with Counter (1 pt.)
### Task 01-1: Description (0 pts.)

Before we can decode the cipher, we need to figure out **which** cipher was used. One approach: encode a known text with different ciphers, then measure which encoded version is most **similar** to our mystery text. But how do you measure text similarity?

---

#### collections.Counter

[`Counter`](https://docs.python.org/3/library/collections.html#collections.Counter) counts how often each element appears in an iterable — perfect for building frequency vectors from text:

```python
from collections import Counter

text = "hello world"
Counter(text)         # Counter({'l': 3, 'o': 2, 'h': 1, 'e': 1, ' ': 1, ...})
Counter(text.split()) # Counter({'hello': 1, 'world': 1})
```

---

#### Bag-of-Words (BoW) vs Bag-of-Characters (BoC)

Two common text representations:

| Approach | What it counts | Example: "the cat sat" |
|----------|---------------|------------------------|
| **Bag-of-Words** | Word frequencies | `{'the': 1, 'cat': 1, 'sat': 1}` |
| **Bag-of-Characters** | Character frequencies | `{'t': 3, 'a': 2, ' ': 2, ...}` |

BoW splits on whitespace. BoC counts individual characters (usually filtering to alphanumeric with `str.isalnum()`).

---

#### Cosine Similarity

[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) measures the angle between two frequency vectors — values near 1.0 mean very similar, near 0.0 means very different:

$$\\text{cosine\\_sim}(A, B) = \\frac{A \\cdot B}{\\|A\\| \\cdot \\|B\\|}$$

With `Counter` objects, the implementation is:

```python
def cosine_sim(counter1, counter2):
    # Dot product: sum of products for shared keys
    dot = sum(counter1[k] * counter2[k] for k in counter1)

    # Norms: sqrt of sum of squares
    norm1 = math.sqrt(sum(v**2 for v in counter1.values()))
    norm2 = math.sqrt(sum(v**2 for v in counter2.values()))

    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)
```

---

#### Bag-of-Words Similarity

Putting it together — BoW similarity between two texts:

```python
def bow_similarity(text1, text2):
    counter1 = Counter(text1.lower().split())
    counter2 = Counter(text2.lower().split())
    return cosine_sim(counter1, counter2)
```

### Task 01-1: Code (0 pts.)"""))

# ═══ CELL: Task 01-1 Demo Code ═══
cells.append(code("""from collections import Counter
import math

# Counter basics
text = "the cat sat on the mat"
word_counts = Counter(text.split())
char_counts = Counter(text)

print("Word counts:", dict(word_counts))
print("Char counts:", dict(char_counts))
print(f"Most common word: {word_counts.most_common(1)}")

# Cosine similarity helper
def cosine_sim(counter1, counter2):
    dot = sum(counter1[k] * counter2[k] for k in counter1)
    norm1 = math.sqrt(sum(v**2 for v in counter1.values()))
    norm2 = math.sqrt(sum(v**2 for v in counter2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)

# BoW similarity
def bow_similarity(text1, text2):
    c1 = Counter(text1.lower().split())
    c2 = Counter(text2.lower().split())
    return cosine_sim(c1, c2)

# Demo
print(f"\\nBoW sim('the cat sat', 'the cat sat'):  {bow_similarity('the cat sat', 'the cat sat'):.4f}")
print(f"BoW sim('the cat sat', 'the dog ran'):  {bow_similarity('the cat sat', 'the dog ran'):.4f}")
print(f"BoW sim('the cat sat', 'xyz abc def'):  {bow_similarity('the cat sat', 'xyz abc def'):.4f}")"""))

# ═══ CELL: Task 01-2 Exercise ═══
cells.append(md("""### Task 01-2: BoC Similarity + Cipher Classification (1 pt.)

Now implement **Bag-of-Characters** similarity and use it to classify the cipher used on the ransom note.

Steps:
1. Implement `boc_similarity(text1, text2)` — like BoW but counting **characters** (filter to alphanumeric with `str.isalnum()`, convert to lowercase)
2. Encode a test string with 4 different ciphers using `pycipher`: `Caesar(13)`, `Caesar(3)`, `Vigenere("key")`, `Affine(5, 8)`
3. Compute BoW and BoC similarity between each encoded string and the mystery text
4. Print which cipher has the highest combined similarity

### Task 01-2: Code (1 pt.)"""))

# ═══ CELL: Task 01-2 Code ═══
cells.append(code("""from pycipher import Caesar, Affine, Vigenere

# The encoded ransom note text
mystery_text = "v ybirq zheqrevat ze gurvfrasyblq ng gur rfgngr gur tnf jnf gur cresrpg zheqre jrncba vz fher vyy trg njnl jvgu vg nf jryy ubcrshyyl abobql svtherf bhg gur pbzovangvba bs gur cnqybpx cynl ba ybpxre 69 ba gur frpbaq sybbe bs phfuvat bgurejvfr v nz va erny gebhoyr"

# TODO: Implement Bag-of-Characters similarity
def boc_similarity(text1, text2):
    c1 = Counter(filter(str.isalnum, text1.lower()))
    c2 = Counter(filter(str.isalnum, text2.lower()))
    return cosine_sim(c1, c2)

# Test text to encode with different ciphers
test_text = ("Lorem ipsum dolor sit amet, consectetur adipiscing elit, "
             "sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.")
print(f"Test text: {test_text[:60]}...")

# TODO: Encode test text with each cipher
ciphers = {
    "ROT13":    Caesar(13).encipher(test_text),
    "Caesar-3": Caesar(3).encipher(test_text),
    "Vigenere":  Vigenere("key").encipher(test_text),
    "Affine":   Affine(5, 8).encipher(test_text),
}

# TODO: Compute BoW and BoC similarity for each cipher vs the mystery text
print(f"\\n{'Cipher':<12} {'BoW Sim':>8} {'BoC Sim':>8} {'Combined':>9}")
print("-" * 40)

best_cipher = None
best_score = -1

for name, encoded in ciphers.items():
    bow = bow_similarity(encoded, mystery_text)
    boc = boc_similarity(encoded, mystery_text)
    combined = bow + boc
    print(f"{name:<12} {bow:>8.4f} {boc:>8.4f} {combined:>9.4f}")
    if combined > best_score:
        best_score = combined
        best_cipher = name

print(f"\\nMost likely cipher: {best_cipher}")"""))

# ═══ CELL: Task 01-2 Expected Output ═══
cells.append(md("""### Task 01-2: Expected Output (1 pt.)
```
Test text: Lorem ipsum dolor sit amet, consectetur adipiscing ...

Cipher       BoW Sim  BoC Sim  Combined
----------------------------------------
ROT13          <highest>
Caesar-3       <lower>
Vigenere       <lower>
Affine         <lower>

Most likely cipher: ROT13
```

ROT13 should have the highest combined similarity since it preserves character frequency distributions (each letter maps to exactly one other letter with the same frequency pattern).

*Note:* In HW05, you'll implement both BoW and BoC similarity from scratch, and use them to classify the cipher on the actual ransom note text."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

"ROT13," Detective Gaff confirms. "Classic. Now we *could* just decode it directly — ROT13 is its own inverse — but Director Bryant wants an AI solution. He says we need to train a neural network that can decode *any* simple cipher, not just ROT13."

To train a neural network on text, you first need to convert characters into numbers. That process is called **tokenization**..."""))

# ═══════════════════════════════════════════════════════════════
# TASK 02: Tokenization and Sequence Padding
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 02: Tokenization and Sequence Padding (1 pt.)
### Task 02-1: Description (0 pts.)

Neural networks only understand numbers. To process text, we need to convert each character (or word) into an integer — this is **tokenization**. Then, since sentences have different lengths, we **pad** them to the same length so they can be batched together.

---

#### Character-Level Tokenization

A tokenizer maps characters to integers and back:

```python
import string

# Build vocabulary: special tokens + space + lowercase letters
vocab = ('<PAD>', '<EOS>', '<UNK>', '<SOS>', ' ') + tuple(string.ascii_lowercase)

# Create lookup dictionaries
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for i, ch in enumerate(vocab)}

# Encode text to integers
def encode(text):
    return [char2idx.get(ch, char2idx['<UNK>']) for ch in text.lower()]

# Decode integers back to text
def decode(indices):
    return ''.join(idx2char[i] for i in indices)
```

**Special tokens** have reserved indices:

| Token | Index | Purpose |
|-------|-------|---------|
| `<PAD>` | 0 | Padding (fill shorter sequences) |
| `<EOS>` | 1 | End of sequence |
| `<UNK>` | 2 | Unknown character |
| `<SOS>` | 3 | Start of sequence |

---

#### Sequence Padding with pad_sequence

[`torch.nn.utils.rnn.pad_sequence()`](https://pytorch.org/docs/stable/generated/torch.nn.utils.rnn.pad_sequence.html) pads a list of variable-length tensors to the same length:

```python
from torch.nn.utils.rnn import pad_sequence

# Three sequences of different lengths
seqs = [torch.tensor([5, 6, 7]),
        torch.tensor([8, 9]),
        torch.tensor([10, 11, 12, 13])]

# Pad to same length (default padding value = 0)
padded = pad_sequence(seqs, batch_first=True, padding_value=0)
# tensor([[ 5,  6,  7,  0],
#          [ 8,  9,  0,  0],
#          [10, 11, 12, 13]])
# Shape: (3, 4) — 3 sequences, max length 4
```

| Parameter | What it does |
|-----------|-------------|
| `batch_first=True` | Output shape: `(batch, seq_len)` instead of `(seq_len, batch)` |
| `padding_value=0` | Value used for padding (use your `<PAD>` token index) |

### Task 02-1: Code (0 pts.)"""))

# ═══ CELL: Task 02-1 Demo Code ═══
cells.append(code("""import string
from torch.nn.utils.rnn import pad_sequence

# Build vocabulary
vocab = ('<PAD>', '<EOS>', '<UNK>', '<SOS>', ' ') + tuple(string.ascii_lowercase)
char2idx = {ch: i for i, ch in enumerate(vocab)}
idx2char = {i: ch for i, ch in enumerate(vocab)}

print(f"Vocabulary size: {len(vocab)}")
print(f"First 10 entries: {list(char2idx.items())[:10]}")

# Encode/decode
def encode(text):
    return [char2idx.get(ch, char2idx['<UNK>']) for ch in text.lower()]

def decode(indices):
    return ''.join(idx2char.get(i, '?') for i in indices)

# Demo
encoded = encode("hello")
print(f"\\n'hello' -> {encoded}")
print(f"{encoded}  -> '{decode(encoded)}'")

# Padding demo
seqs = [torch.tensor(encode("hi")),
        torch.tensor(encode("hello")),
        torch.tensor(encode("hey there"))]

print(f"\\nBefore padding: lengths = {[len(s) for s in seqs]}")

padded = pad_sequence(seqs, batch_first=True, padding_value=char2idx['<PAD>'])
print(f"After padding:  shape = {padded.shape}")
print(f"Padded tensor:\\n{padded}")"""))

# ═══ CELL: Task 02-2 Exercise ═══
cells.append(md("""### Task 02-2: Tokenize and Pad a Batch (1 pt.)

Create a function that takes a list of plaintext strings, encodes each one, and produces a padded batch tensor ready for a neural network.

Steps:
1. Write `encode_batch(texts, char2idx)` that:
   - Encodes each text string using `char2idx` (lowercase, unknown chars map to `<UNK>`)
   - Appends an `<EOS>` token to each encoded sequence
   - Converts each to a `torch.long` tensor
   - Pads using `pad_sequence` with `batch_first=True` and `padding_value=0` (the `<PAD>` index)
2. Also create the corresponding ROT13-encoded targets for each text (hint: use `codecs.decode(text, 'rot_13')` to get the ROT13 version)
3. Test on the provided sample texts

### Task 02-2: Code (1 pt.)"""))

# ═══ CELL: Task 02-2 Code ═══
cells.append(code("""import codecs

# TODO: Implement encode_batch
def encode_batch(texts, char2idx):
    eos_idx = char2idx['<EOS>']
    unk_idx = char2idx['<UNK>']
    batch = []
    for text in texts:
        encoded = [char2idx.get(ch, unk_idx) for ch in text.lower()]
        encoded.append(eos_idx)  # append EOS
        batch.append(torch.tensor(encoded, dtype=torch.long))
    return pad_sequence(batch, batch_first=True, padding_value=char2idx['<PAD>'])

# Sample texts
sample_texts = [
    "hello world",
    "the quick brown fox",
    "ai is fun",
    "cipher text"
]

# TODO: Encode the plaintext
plain_batch = encode_batch(sample_texts, char2idx)
print(f"Plaintext batch shape: {plain_batch.shape}")
print(f"Plaintext batch:\\n{plain_batch}")

# TODO: Create ROT13 encoded versions
rot13_texts = [codecs.decode(t, 'rot_13') for t in sample_texts]
cipher_batch = encode_batch(rot13_texts, char2idx)
print(f"\\nROT13 texts: {rot13_texts}")
print(f"Cipher batch shape: {cipher_batch.shape}")

# Verify: decode the first sequence back
decoded_plain = decode(plain_batch[0].tolist())
decoded_cipher = decode(cipher_batch[0].tolist())
print(f"\\nDecoded plain[0]:  '{decoded_plain}'")
print(f"Decoded cipher[0]: '{decoded_cipher}'")"""))

# ═══ CELL: Task 02-2 Expected Output ═══
cells.append(md("""### Task 02-2: Expected Output (1 pt.)
```
Plaintext batch shape: torch.Size([4, 20])
Plaintext batch:
tensor([[...]])

ROT13 texts: ['uryyb jbeyq', 'gur dhvpx oebja sbk', 'nv vf sha', 'pvcure grkg']
Cipher batch shape: torch.Size([4, 20])

Decoded plain[0]:  'hello world<EOS><PAD>...'
Decoded cipher[0]: 'uryyb jbeyq<EOS><PAD>...'
```

The batch should be `(4, max_len)` where `max_len` is the length of the longest string + 1 (for `<EOS>`). Shorter sequences are padded with `<PAD>` (index 0).

*Note:* In HW05, you'll build a full `Rot13Dataset` class that generates (ciphertext, plaintext) pairs for training a seq2seq model. The tokenization pattern here is exactly what that dataset uses."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

"Good — text is now numbers," says Detective Gaff. "But right now each character is just an arbitrary integer. The number 5 isn't *meaningfully* different from 6. We need to give each character a richer representation." That's what **embedding layers** do — they map each token to a learned vector of floating-point numbers..."""))

# ═══════════════════════════════════════════════════════════════
# TASK 03: nn.Embedding
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 03: nn.Embedding (1 pt.)
### Task 03-1: Description (0 pts.)

An embedding layer is essentially a learnable lookup table. Each token index maps to a dense vector of real numbers. During training, these vectors are adjusted so that similar characters get similar vectors.

---

#### nn.Embedding

[`nn.Embedding(num_embeddings, embedding_dim)`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) creates the lookup table:

```python
embed = nn.Embedding(num_embeddings=31, embedding_dim=16)
# 31 vocabulary entries, each mapped to a 16-dimensional vector

tokens = torch.tensor([5, 10, 3])   # 3 token indices
vectors = embed(tokens)              # shape: (3, 16)
```

---

#### Input/Output Shapes

```
Input:  (batch_size, seq_len)     — integer token indices
Output: (batch_size, seq_len, embedding_dim)  — dense vectors
```

Example:
```python
embed = nn.Embedding(31, 16)
x = torch.tensor([[5, 10, 3],
                   [7,  2, 0]])    # (2, 3) — batch of 2, length 3
out = embed(x)                      # (2, 3, 16)
```

---

#### Padding Index

[`padding_idx`](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html) tells the layer to always output zeros for the padding token, and not update its embedding during training:

```python
embed = nn.Embedding(31, 16, padding_idx=0)  # index 0 = <PAD>
# embed(torch.tensor([0])) will always be a zero vector
```

---

#### How It Differs from One-Hot

| Approach | Vector size | Learned? | Example for vocab=31 |
|----------|------------|----------|---------------------|
| One-hot  | 31 (sparse) | No | `[0,0,0,0,0,1,0,...,0]` |
| Embedding | 16 (dense) | Yes | `[0.23, -0.41, 0.87, ...]` |

Embeddings are more compact and capture relationships between tokens.

### Task 03-1: Code (0 pts.)"""))

# ═══ CELL: Task 03-1 Demo Code ═══
cells.append(code("""# Create an embedding layer
vocab_size = len(vocab)  # 31
embed_dim = 16

embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
print(f"Embedding layer: {embed}")
print(f"Weight matrix shape: {embed.weight.shape}")  # (31, 16)

# Look up single tokens
token_ids = torch.tensor([5, 10, 0])  # 'a', 'f', '<PAD>'
vectors = embed(token_ids)
print(f"\\nInput: {token_ids} -> Output shape: {vectors.shape}")
print(f"<PAD> vector (should be zeros): {vectors[2]}")

# Look up a batch of sequences
batch = torch.tensor([[5, 6, 7, 0],
                      [8, 9, 0, 0]])  # (2, 4)
batch_vectors = embed(batch)
print(f"\\nBatch input: {batch.shape} -> Embedded: {batch_vectors.shape}")
# (2, 4) -> (2, 4, 16)"""))

# ═══ CELL: Task 03-2 Exercise ═══
cells.append(md("""### Task 03-2: Embedding Lookup Exercise (1 pt.)

Use `nn.Embedding` to embed the tokenized batch from Task 02.

Steps:
1. Create an `nn.Embedding` layer with `vocab_size=len(vocab)`, `embedding_dim=32`, and `padding_idx=0`
2. Pass the `plain_batch` tensor through the embedding layer
3. Print the input shape, output shape, and verify the padding positions are zero vectors
4. Compute and print the mean embedding norm for non-padding vs padding positions

### Task 03-2: Code (1 pt.)"""))

# ═══ CELL: Task 03-2 Code ═══
cells.append(code("""# TODO: Create embedding layer
embed = nn.Embedding(num_embeddings=len(vocab), embedding_dim=32, padding_idx=0)
print(f"Embedding: {embed}")

# TODO: Embed the plaintext batch from Task 02
embedded = embed(plain_batch)
print(f"\\nInput shape:  {plain_batch.shape}")
print(f"Output shape: {embedded.shape}")

# TODO: Verify padding positions are zeros
# Find where the input is <PAD> (index 0)
pad_mask = (plain_batch == 0)
pad_vectors = embedded[pad_mask]
nonpad_vectors = embedded[~pad_mask]

pad_norm = pad_vectors.norm(dim=-1).mean().item()
nonpad_norm = nonpad_vectors.norm(dim=-1).mean().item()

print(f"\\nPadding positions:     mean norm = {pad_norm:.4f} (should be 0.0)")
print(f"Non-padding positions: mean norm = {nonpad_norm:.4f} (should be > 0)")

# Show the embedding for the first character of the first sequence
first_char_idx = plain_batch[0, 0].item()
print(f"\\nFirst character: '{idx2char[first_char_idx]}' (index {first_char_idx})")
print(f"Its embedding (first 8 dims): {embedded[0, 0, :8].data}")"""))

# ═══ CELL: Task 03-2 Expected Output ═══
cells.append(md("""### Task 03-2: Expected Output (1 pt.)
```
Embedding: Embedding(31, 32, padding_idx=0)

Input shape:  torch.Size([4, 20])
Output shape: torch.Size([4, 20, 32])

Padding positions:     mean norm = 0.0000 (should be 0.0)
Non-padding positions: mean norm = <value> (should be > 0)

First character: 'h' (index <n>)
Its embedding (first 8 dims): tensor([...])
```

The output adds an embedding dimension: `(batch, seq_len)` → `(batch, seq_len, embed_dim)`. Padding positions are guaranteed to be zero vectors because of `padding_idx=0`.

*Note:* In HW05, you'll implement an `EmbeddingLayer` from scratch (it's essentially a matrix indexing operation) and also use `nn.Embedding` in the PyTorch RNN and Transformer models."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

"Each character now has a meaningful vector representation," you report. "But we still need a model that can process these vectors *in order* — the meaning of a cipher depends on the sequence." Director Bryant nods. "That's what **recurrent neural networks** are for — they read text one character at a time, building up context as they go.\""""))

# ═══════════════════════════════════════════════════════════════
# TASK 04: nn.RNN
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 04: nn.RNN (1 pt.)
### Task 04-1: Description (0 pts.)

A recurrent neural network (RNN) processes sequences one step at a time, maintaining a **hidden state** that accumulates context from previous steps. At each step, it combines the current input with the previous hidden state to produce a new hidden state.

---

#### nn.RNN

[`nn.RNN(input_size, hidden_size, batch_first=True)`](https://pytorch.org/docs/stable/generated/torch.nn.RNN.html) creates a recurrent layer:

```python
rnn = nn.RNN(input_size=32, hidden_size=64, batch_first=True)

# Input: (batch, seq_len, input_size)
x = torch.randn(4, 20, 32)    # 4 sequences, length 20, 32-dim embeddings

# Output: (all_hidden_states, final_hidden_state)
output, h_n = rnn(x)

# output shape: (4, 20, 64) — hidden state at EVERY time step
# h_n shape:    (1, 4, 64)  — hidden state at the LAST time step
```

---

#### Input/Output Shapes (with batch_first=True)

```
Input x:     (batch, seq_len, input_size)
Output:      (batch, seq_len, hidden_size)   — hidden state at each step
h_n:         (num_layers, batch, hidden_size) — final hidden state
```

---

#### The Full Pipeline: Tokens → Embeddings → RNN → Output

```
tokens:     (batch, seq_len)                    — integer indices
     ↓ nn.Embedding
embeddings: (batch, seq_len, embed_dim)          — dense vectors
     ↓ nn.RNN
hidden:     (batch, seq_len, hidden_size)        — contextual representations
     ↓ nn.Linear
logits:     (batch, seq_len, vocab_size)         — prediction for each position
```

This is exactly the architecture used for sequence-to-sequence tasks like cipher decoding.

---

#### Optional Initial Hidden State

You can pass an initial hidden state `h_0`:

```python
h_0 = torch.zeros(1, batch_size, hidden_size)  # (num_layers, batch, hidden)
output, h_n = rnn(x, h_0)
```

If omitted, `h_0` defaults to zeros.

### Task 04-1: Code (0 pts.)"""))

# ═══ CELL: Task 04-1 Demo Code ═══
cells.append(code("""# Create an RNN layer
rnn = nn.RNN(input_size=32, hidden_size=64, batch_first=True)
print(f"RNN: {rnn}")

# Fake embedded input: (batch=2, seq_len=10, embed_dim=32)
fake_input = torch.randn(2, 10, 32)

# Forward pass
output, h_n = rnn(fake_input)
print(f"\\nInput shape:  {fake_input.shape}")
print(f"Output shape: {output.shape}")    # (2, 10, 64) — all hidden states
print(f"h_n shape:    {h_n.shape}")       # (1, 2, 64) — final hidden state

# The output at the last time step equals h_n (for single-layer RNN)
print(f"\\noutput[:, -1, :] == h_n[0]: {torch.allclose(output[:, -1, :], h_n[0], atol=1e-6)}")

# Full pipeline: embed -> rnn -> linear
demo_embed = nn.Embedding(31, 32, padding_idx=0)
demo_rnn = nn.RNN(32, 64, batch_first=True)
demo_fc = nn.Linear(64, 31)

tokens = torch.tensor([[5, 6, 7, 0],
                        [8, 9, 0, 0]])     # (2, 4)
embedded = demo_embed(tokens)               # (2, 4, 32)
hidden, _ = demo_rnn(embedded)              # (2, 4, 64)
logits = demo_fc(hidden)                    # (2, 4, 31)

print(f"\\nFull pipeline: {tokens.shape} -> {embedded.shape} -> {hidden.shape} -> {logits.shape}")"""))

# ═══ CELL: Task 04-2 Exercise ═══
cells.append(md("""### Task 04-2: Build a CharRNN Model (1 pt.)

Build a simple character-level RNN model as an `nn.Module` that could decode cipher text. The model should:

- Take token indices as input: `(batch, seq_len)`
- Embed them: `nn.Embedding`
- Process with RNN: `nn.RNN`
- Project to vocab logits: `nn.Linear`
- Output: `(batch, seq_len, vocab_size)` — a prediction for each position

Architecture:
- Embedding: `vocab_size` → `embed_dim=32`, with `padding_idx=0`
- RNN: `input_size=32`, `hidden_size=64`, `batch_first=True`
- Linear: `64` → `vocab_size`

### Task 04-2: Code (1 pt.)"""))

# ═══ CELL: Task 04-2 Code ═══
cells.append(code("""# TODO: Define the CharRNN model
class CharRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, h0=None):
        emb = self.embed(x)              # (batch, seq_len, embed_dim)
        output, h_n = self.rnn(emb, h0)  # (batch, seq_len, hidden_size)
        logits = self.fc(output)          # (batch, seq_len, vocab_size)
        return logits, h_n

# TODO: Create model and test with cipher_batch from Task 02
model = CharRNN(vocab_size=len(vocab), embed_dim=32, hidden_size=64)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {total_params:,}")

# TODO: Forward pass on the cipher batch
logits, h_n = model(cipher_batch)
print(f"\\nInput shape:  {cipher_batch.shape}")
print(f"Output shape: {logits.shape}")
print(f"h_n shape:    {h_n.shape}")

# Get predicted tokens (untrained — will be random)
predictions = torch.argmax(logits, dim=-1)
decoded_pred = decode(predictions[0].tolist())
print(f"\\nPredicted (random): '{decoded_pred}'")
print(f"Actual target:      '{decode(plain_batch[0].tolist())}'")"""))

# ═══ CELL: Task 04-2 Expected Output ═══
cells.append(md("""### Task 04-2: Expected Output (1 pt.)
```
CharRNN(
  (embed): Embedding(31, 32, padding_idx=0)
  (rnn): RNN(32, 64, batch_first=True)
  (fc): Linear(in_features=64, out_features=31, bias=True)
)

Total parameters: <n>

Input shape:  torch.Size([4, 20])
Output shape: torch.Size([4, 20, 31])
h_n shape:    torch.Size([1, 4, 64])

Predicted (random): '<random characters>'
Actual target:      'hello world<EOS><PAD>...'
```

The output is `(batch, seq_len, vocab_size)` — one probability distribution over the vocabulary at each position. The predictions are random because the model hasn't been trained yet.

*Note:* In HW05, you'll build an RNN **from scratch** (implementing the recurrent block, embedding layer, and backpropagation through time) and also build this same PyTorch version with training. The architecture is identical — you're just learning to use the building blocks here."""))

# ═══ CELL: Story Progression ═══
cells.append(md("""### *Story Progression*

"The RNN can read text one character at a time," you explain. "But there's a newer architecture that's much more powerful — it can look at **all** characters simultaneously." Detective Gaff raises an eyebrow. "A Transformer?" he asks. "Those are what they use in the big language models, right?"

Exactly. Transformers use **self-attention** to process all positions in parallel, making them faster and better at capturing long-range patterns..."""))

# ═══════════════════════════════════════════════════════════════
# TASK 05: nn.TransformerEncoder
# ═══════════════════════════════════════════════════════════════
cells.append(md("""## Task 05: nn.TransformerEncoder (1 pt.)
### Task 05-1: Description (0 pts.)

Transformers process entire sequences at once using **self-attention** — each position can attend to every other position. Unlike RNNs, there's no sequential processing, which makes them much faster and better at capturing long-range dependencies.

---

#### Positional Encoding

Since Transformers process all positions in parallel, they need **positional encodings** to know the order of tokens. The standard approach uses sine and cosine waves:

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)  # even dims
        pe[:, 1::2] = torch.cos(position * div_term)  # odd dims
        self.register_buffer('pe', pe.unsqueeze(0))    # (1, max_len, d_model)

    def forward(self, x):  # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]
```

[`register_buffer`](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) stores a tensor that isn't a learnable parameter but moves with the model to GPU.

---

#### nn.TransformerEncoderLayer

[`nn.TransformerEncoderLayer`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoderLayer.html) is one block of the Transformer encoder — it contains self-attention + feed-forward:

```python
encoder_layer = nn.TransformerEncoderLayer(
    d_model=128,           # embedding dimension
    nhead=4,               # number of attention heads
    dim_feedforward=256,   # hidden dim in feed-forward network
    dropout=0.1,
    batch_first=True       # input shape: (batch, seq_len, d_model)
)
```

---

#### nn.TransformerEncoder

[`nn.TransformerEncoder`](https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder.html) stacks multiple encoder layers:

```python
encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

x = torch.randn(4, 20, 128)     # (batch, seq_len, d_model)
out = encoder(x)                  # (4, 20, 128) — same shape
```

---

#### The Full Transformer Pipeline

```
tokens:     (batch, seq_len)                    — integer indices
     ↓ nn.Embedding
embeddings: (batch, seq_len, d_model)            — dense vectors
     ↓ PositionalEncoding
positioned: (batch, seq_len, d_model)            — with position info
     ↓ nn.TransformerEncoder
encoded:    (batch, seq_len, d_model)            — contextual representations
     ↓ nn.Linear
logits:     (batch, seq_len, vocab_size)         — prediction for each position
```

---

#### Key Differences: RNN vs Transformer

| Feature | RNN | Transformer |
|---------|-----|-------------|
| Processing | Sequential (one token at a time) | Parallel (all tokens at once) |
| Context | Hidden state (limited memory) | Self-attention (full sequence) |
| Speed | Slow (can't parallelize) | Fast (fully parallelizable) |
| Long sequences | Struggles (vanishing gradient) | Handles well (direct attention) |

### Task 05-1: Code (0 pts.)"""))

# ═══ CELL: Task 05-1 Demo Code ═══
cells.append(code("""import math

# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

# Demo components
pos_enc = PositionalEncoding(d_model=128)
print(f"PositionalEncoding: pe buffer shape = {pos_enc.pe.shape}")

encoder_layer = nn.TransformerEncoderLayer(
    d_model=128, nhead=4, dim_feedforward=256,
    dropout=0.1, batch_first=True
)
print(f"\\nTransformerEncoderLayer: {encoder_layer}")

encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

# Test with fake data
fake_input = torch.randn(4, 20, 128)    # (batch, seq_len, d_model)
positioned = pos_enc(fake_input)          # add positional info
encoded = encoder(positioned)             # self-attention + FFN

print(f"\\nInput:    {fake_input.shape}")
print(f"Encoded:  {encoded.shape}")"""))

# ═══ CELL: Task 05-2 Exercise ═══
cells.append(md("""### Task 05-2: Build a Transformer Encoder Model (1 pt.)

Build an encoder-only Transformer model as an `nn.Module` for character-level sequence processing.

Architecture:
- Embedding: `vocab_size` → `d_model=128`, with `padding_idx=0`
- Positional Encoding: `d_model=128`, `max_len=512`
- Transformer Encoder: 2 layers, `nhead=4`, `dim_feedforward=256`, `dropout=0.1`, `batch_first=True`
- Linear output: `d_model=128` → `vocab_size`

The `forward` method should:
1. Embed the input tokens
2. Scale by `sqrt(d_model)` (standard practice for Transformer embeddings)
3. Add positional encoding
4. Pass through the Transformer encoder
5. Project to vocab logits with the linear layer

### Task 05-2: Code (1 pt.)"""))

# ═══ CELL: Task 05-2 Code ═══
cells.append(code("""# TODO: Define the Transformer encoder model
class CharTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout, max_len, padding_idx):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.pos_encoder = PositionalEncoding(d_model, max_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        emb = self.embedding(x) * math.sqrt(self.d_model)  # scale embeddings
        emb = self.pos_encoder(emb)                          # add position info
        encoded = self.encoder(emb)                          # self-attention
        logits = self.fc(encoded)                            # project to vocab
        return logits

# TODO: Create model and test
model = CharTransformer(
    vocab_size=len(vocab),
    d_model=128,
    nhead=4,
    num_layers=2,
    dim_feedforward=256,
    dropout=0.1,
    max_len=512,
    padding_idx=0
)
print(model)

total_params = sum(p.numel() for p in model.parameters())
print(f"\\nTotal parameters: {total_params:,}")

# TODO: Forward pass on cipher_batch
logits = model(cipher_batch)
print(f"\\nInput shape:  {cipher_batch.shape}")
print(f"Output shape: {logits.shape}")

# Get predictions (untrained)
predictions = torch.argmax(logits, dim=-1)
decoded_pred = decode(predictions[0].tolist())
print(f"\\nPredicted (random): '{decoded_pred}'")
print(f"Actual target:      '{decode(plain_batch[0].tolist())}'")"""))

# ═══ CELL: Task 05-2 Expected Output ═══
cells.append(md("""### Task 05-2: Expected Output (1 pt.)
```
CharTransformer(
  (embedding): Embedding(31, 128, padding_idx=0)
  (pos_encoder): PositionalEncoding()
  (encoder): TransformerEncoder(
    (layers): ModuleList(
      (0-1): 2 x TransformerEncoderLayer(...)
    )
  )
  (fc): Linear(in_features=128, out_features=31, bias=True)
)

Total parameters: <n>

Input shape:  torch.Size([4, 20])
Output shape: torch.Size([4, 20, 31])

Predicted (random): '<random characters>'
Actual target:      'hello world<EOS><PAD>...'
```

The Transformer has the same input/output interface as the RNN model: `(batch, seq_len)` → `(batch, seq_len, vocab_size)`. But internally it uses self-attention instead of sequential processing.

*Note:* In HW05, you'll implement a self-attention block from scratch **and** use `nn.TransformerEncoderLayer` / `nn.TransformerEncoder` to build a PyTorch Transformer. You'll train both the RNN and Transformer on ROT13 decoding and compare their performance."""))

# ═══ CELL: Final Story ═══
cells.append(md("""### *Story Progression*

You now have all the NLP tools you need for **Homework 05**:

1. **`Counter` + cosine similarity** — measuring text similarity for cipher classification
2. **Character tokenization** — converting text to integer sequences with special tokens
3. **`pad_sequence`** — batching variable-length sequences together
4. **`nn.Embedding`** — mapping token indices to dense learned vectors
5. **`nn.RNN`** — processing sequences one step at a time with hidden state
6. **`nn.TransformerEncoder`** — processing sequences in parallel with self-attention

In HW05, you'll use these tools (and build some from scratch) to decode the ransom note cipher. "Once we crack this code," says Director Bryant, "we'll finally know what the kidnapper wrote." Time to file your report!"""))

# ═══ CELL: Task 06 Header ═══
cells.append(md("""## Task 06: Generate Police Report (0 pts.)

Run the code cell below to generate a report for the Police and submit it on Canvas!

### Task 06: Code (0 pts.)"""))

# ═══ CELL: Task 06 Export Code ═══
cells.append(code("""import os, json

ASS_PATH = "nd-cse-30124-homeworks/labs"
ASS = "lab04"

try:
    from google.colab import _message, files

    repo_ipynb_path = f"/content/{ASS_PATH}/{ASS}/{ASS}.ipynb"

    nb = _message.blocking_request("get_ipynb", timeout_sec=1)["ipynb"]

    os.makedirs(os.path.dirname(repo_ipynb_path), exist_ok=True)
    with open(repo_ipynb_path, "w", encoding="utf-8") as f:
        json.dump(nb, f)

    !jupyter nbconvert --to html "{repo_ipynb_path}"
    files.download(repo_ipynb_path.replace(".ipynb", ".html"))
except:
    import subprocess

    nb_fp = os.getcwd() + f'/{ASS}.ipynb'
    print(os.getcwd())

    subprocess.run(["jupyter", "nbconvert", "--to", "html", nb_fp], check=True)
finally:
    print('[WARNING]: Unable to export notebook as .html')"""))

# ═══ Build notebook ═══
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("lab04_solutions.ipynb", "w") as f:
    json.dump(notebook, f, indent=1)

print(f"Generated notebook with {len(cells)} cells")
