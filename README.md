# nanosentencepiece

`nanosentencepiece` is a **SentencePiece-inspired** subword tokenizer library written in **modern C++20**.

It is intentionally much smaller than the real [Google SentencePiece](https://github.com/google/sentencepiece) codebase. The goal is to capture the core ideas behind reversible, raw-text subword tokenization in a way that is easy to read.

## What problem it solves

Traditional word-level tokenization breaks on unseen words and can create huge vocabularies.

`nanosentencepiece` trains a **Byte Pair Encoding (BPE)** tokenizer directly from *raw text*, then uses the learned merge rules to segment text into reusable subword pieces.

Like SentencePiece-style tokenizers, it keeps tokenization *reversible* by making whitespace explicit with a visible marker (`▁` by default), so spaces survive encode/decode.

## Why this project exists

`nanosentencepiece` is designed to be:

- **smaller-than-SentencePiece**
- focused on a polished **BPE-first v1**

## Features

### Core library
- raw-text training pipeline
- reversible whitespace handling with a SentencePiece-like marker
- deterministic BPE training
- subword encoding to pieces
- subword encoding to integer ids
- decoding from pieces or ids
- special token support:
  - `<unk>`
  - `<bos>`
  - `<eos>`
  - `<pad>`

### Model handling
- save model to an inspectable plain-text format
- load model from disk
- persist:
  - vocabulary
  - merge rules
  - special token ids
  - normalization config
  - model metadata

### Tooling
- `nsp_train`
- `nsp_encode`
- `nsp_decode`
- `nsp_inspect`

### Engineering
- modern C++20
- CMake build
- unit-style tests
- docs and architecture notes
- sample corpus files

## Build instructions

```bash
git clone <your-fork-url>
cd nanosentencepiece

cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

## Quick start

### 1. Train a small model

```bash
./build/nsp_train \
  --model build/demo.nsp \
  --vocab-size 64 \
  --min-pair-frequency 1 \
  data/tiny_corpus.txt
```

Example output:

```text
trained model: build/demo.nsp
vocabulary size: 64
merge count: 46
special ids: unk=0, bos=1, eos=2, pad=3
```

### 2. Encode text into pieces

```bash
./build/nsp_encode \
  --model build/demo.nsp \
  --text "Hello world from nanosentencepiece" \
  --output pieces \
  --bos --eos
```

Possible output:

```text
<bos> ▁hello ▁world ▁from ▁nanosentencepiece <eos>
```

### 3. Encode text into ids

```bash
./build/nsp_encode \
  --model build/demo.nsp \
  --text "Hello world from nanosentencepiece" \
  --output ids \
  --bos --eos
```

### 4. Decode ids back into text

```bash
./build/nsp_decode \
  --model build/demo.nsp \
  --ids "1 17 24 31 42 2"
```

### 5. Inspect the learned model

```bash
./build/nsp_inspect \
  --model build/demo.nsp \
  --limit 20
```

## Training pipeline overview

The v1 trainer is intentionally simple for version 1.

1. Read raw text lines from one or more files.
2. Normalize text:
   - optional ASCII lowercase
   - optional whitespace collapse
3. Replace spaces with an explicit marker (`▁`) and optionally prepend a dummy prefix marker.
4. Split the escaped text into UTF-8 codepoint-sized symbols.
5. Build the initial symbol vocabulary.
6. Count adjacent pair frequencies across the corpus.
7. Merge the most frequent pair.
8. Repeat until the target vocabulary size is reached or no merge meets the frequency threshold.
9. Build a final token-id vocabulary with special tokens first.

## Encoding overview

At inference time:

1. Normalize + escape the input text the same way as during training.
2. Split into UTF-8 codepoint-sized pieces.
3. Apply learned BPE merges **in deterministic training order**.
4. Map final pieces to vocabulary ids.
5. Use `<unk>` for pieces not found in the vocabulary.

## Decoding overview

Decoding works by:

1. converting ids back to pieces
2. skipping BOS / EOS / PAD
3. concatenating the pieces
4. restoring spaces from the `▁` marker
5. removing the dummy prefix space if enabled

Because normalization may lowercase or collapse whitespace, the round trip is best thought of as:

> **decode(encode(text)) == normalized(text)**

for supported inputs in v1.

## Model format

The model file is a simple plain-text format that is easy to inspect and debug.

It stores:

- metadata
- normalizer settings
- special tokens
- special token ids
- vocabulary entries
- ordered merge rules

Example snippet:

```text
format	NSPM
version	1
trained_vocab_size	64
normalizer.lowercase	1
normalizer.collapse_whitespace	1
normalizer.add_dummy_prefix	1
normalizer.whitespace_symbol	▁
special.unk	<unk>
...
piece	0	<unk>
piece	1	<bos>
...
merge	0	h	e	he
merge	1	he	l	hel
```

A plain-text format was chosen for v1 because it is:
- easy to inspect
- easy to diff
- easy to debug during development

## Complexity notes

### Training
This implementation recomputes adjacent pair counts after each merge.

That makes the training loop simple, deterministic, and easy to reason about, but not asymptotically optimal for very large corpora.

A rough mental model is:

- let `M` = number of merges
- let `N` = total number of symbols across the training corpus

Then the current approach is roughly $$O(n*m)$$ pair scanning, with additional vector rewrite costs when applying merges.

For a medium demo corpus, this is perfectly acceptable and keeps the implementation understandable.

### Encoding
Encoding applies merge rules in learned order, so it is roughly proportional to:

- number of merge rules
- number of current pieces in the sequence

This is a good candidate for future optimization via:
- merge-rank priority logic
- trie-based matching
- cached piece segmentation

## Testing

The test suite covers:

- normalization behavior
- reversible whitespace escaping/restoration
- deterministic BPE merge learning
- encode/decode round trips
- special token insertion
- model save/load

Run tests with:

```bash
ctest --test-dir build --output-on-failure
```

### Included
- raw-text training
- whitespace-preserving tokenization
- BPE training
- reversible decode path
- model save/load
- CLI tooling
- small, understandable architecture

### Not included in v1
- full unigram LM training
- industrial Unicode normalization
- byte fallback
- sampling-based segmentation
- optimized large-scale training data structures
- every SentencePiece training option

## Future improvements

Reasonable next steps include:

- unigram model support
- byte fallback for unseen characters
- prefix trie or rank-based encoder optimization
- corpus weighting / multi-file stats
- parallel pair counting
- benchmarking tool
- Python bindings
- richer Unicode normalization
