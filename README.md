# nanosentencepiece

`nanosentencepiece` is a compact subword tokenizer library in modern C++20 inspired by
[Google SentencePiece](https://github.com/google/sentencepiece).

- two model families: `bpe` and `unigram`
- a reusable `SentencePieceProcessor` inference facade
- model metadata and model-type-aware serialization
- optional GoogleTest-based test discovery in CMake
- a cached unigram piece index so repeated inference avoids rebuilding search state

It is still much smaller than the upstream SentencePiece project.

## What it does

`nanosentencepiece` trains subword tokenizers directly from raw text and then encodes text into:

- string pieces
- integer ids

Like SentencePiece-style tokenizers, it preserves whitespace explicitly with a visible marker
(`▁` by default), which makes decode reversible with respect to the configured normalization.

Today the library supports:

- deterministic BPE training and inference
- unigram training and Viterbi-style inference
- BOS / EOS / PAD / UNK special tokens
- model save/load in an inspectable plain-text format
- CLI tools for training, encoding, decoding, and inspection

What is working:

- immutable model ownership through `SentencePieceProcessor`
- model-type-aware metadata
- optional GoogleTest integration through CMake
- separate training and inference components
- cached unigram inference structures for repeated use

What is not here yet:

- industrial-strength Unicode normalization
- upstream `.model` protobuf compatibility
- byte fallback
- unigram pruning and richer EM variants
- subword regularization / sampling
- large-corpus optimized training data structures throughout

## Features

### Model types

- `bpe`
  - deterministic merge learning
  - deterministic merge replay during encoding
- `unigram`
  - substring seed vocabulary generation
  - hard-EM-style score refinement
  - best-path segmentation at inference time

### Core library

- raw-text training from one or more files
- reversible whitespace escaping with a SentencePiece-style marker
- encoding to pieces or ids
- decoding from pieces or ids
- `SentencePieceProcessor` facade for inference
- `Tokenizer` compatibility wrapper for the older API

### Model handling

- save model to a plain-text `.nsp` format
- load model from disk
- persist:
  - model type
  - normalizer settings
  - special token ids
  - vocabulary
  - BPE merge rules
  - unigram piece scores

### Tooling

- `nsp_train`
- `nsp_encode`
- `nsp_decode`
- `nsp_inspect`
- `examples/demo.sh`

## Build

```bash
git clone <your-fork-url>
cd nanosentencepiece

cmake -S . -B build
cmake --build build
ctest --test-dir build --output-on-failure
```

### Useful CMake options

```bash
cmake -S . -B build -DNSP_BUILD_TESTS=ON -DNSP_USE_GTEST=ON
```

- `NSP_BUILD_TESTS=ON|OFF`: build test targets
- `NSP_USE_GTEST=ON|OFF`: use GoogleTest discovery when available
- `NSP_WARNINGS_AS_ERRORS=ON|OFF`: promote warnings to errors

## Quick start

### Train a BPE model

```bash
./build/nsp_train \
  --model build/demo-bpe.nsp \
  --model-type bpe \
  --vocab-size 64 \
  --min-pair-frequency 1 \
  data/tiny_corpus.txt
```

Example output:

```text
trained model: build/demo-bpe.nsp
model type: bpe
vocabulary size: 64
merge count: 46
special ids: unk=0, bos=1, eos=2, pad=3
```

### Encode with BPE

```bash
./build/nsp_encode \
  --model build/demo-bpe.nsp \
  --text "Hello world from nanosentencepiece" \
  --output pieces \
  --bos --eos
```

Possible output:

```text
<bos> ▁hello ▁world ▁from ▁nanosentencepiece <eos>
```

### Train a unigram model

```bash
./build/nsp_train \
  --model build/demo-unigram.nsp \
  --model-type unigram \
  --vocab-size 64 \
  --min-piece-frequency 1 \
  --max-piece-length 8 \
  --unigram-iterations 4 \
  data/tiny_corpus.txt
```

Example output:

```text
trained model: build/demo-unigram.nsp
model type: unigram
vocabulary size: 64
merge count: 0
special ids: unk=0, bos=1, eos=2, pad=3
```

### Encode with unigram

```bash
./build/nsp_encode \
  --model build/demo-unigram.nsp \
  --text "Hello world from nanosentencepiece" \
  --output pieces \
  --bos --eos
```

### Decode ids or pieces

```bash
./build/nsp_decode \
  --model build/demo-bpe.nsp \
  --ids "1 17 24 31 42 2"
```

```bash
./build/nsp_decode \
  --model build/demo-bpe.nsp \
  --pieces "<bos> ▁hello ▁world <eos>"
```

### Inspect a model

```bash
./build/nsp_inspect \
  --model build/demo-bpe.nsp \
  --limit 20
```

For unigram models, `nsp_inspect` also shows stored piece scores.

## CLI reference

### `nsp_train`

```text
nsp_train --model model.nsp [--model-type bpe|unigram] --vocab-size 128
          [--min-pair-frequency 2] [--min-piece-frequency 2]
          [--max-piece-length 8] [--unigram-iterations 4]
          [--no-lowercase] [--no-collapse-whitespace] [--no-dummy-prefix]
          corpus1.txt [corpus2.txt ...]
```

Notes:

- `--min-pair-frequency` applies to BPE training.
- `--min-piece-frequency`, `--max-piece-length`, and `--unigram-iterations` apply to unigram training.
- `--model-type` defaults to `bpe`.

### `nsp_encode`

```text
nsp_encode --model model.nsp --text "hello world" [--output pieces|ids] [--bos] [--eos]
nsp_encode --model model.nsp --input text.txt [--output pieces|ids]
```

### `nsp_decode`

```text
nsp_decode --model model.nsp --ids "1 5 18 9"
nsp_decode --model model.nsp --pieces "▁he llo ▁world"
```

### `nsp_inspect`

```text
nsp_inspect --model model.nsp [--limit 32]
```

## Library usage

The main inference surface is `SentencePieceProcessor`.

```cpp
#include "nanosentencepiece/model.hpp"
#include "nanosentencepiece/processor.hpp"

using namespace nanosentencepiece;

int main() {
  const auto processor = SentencePieceProcessor::Load("build/demo-bpe.nsp");
  const auto pieces = processor.EncodeToPieces("Hello world", EncodeOptions{true, true});
  const auto ids = processor.EncodeToIds("Hello world");
  const auto text = processor.DecodeIds(ids);
}
```

`Tokenizer` still exists for compatibility, but `SentencePieceProcessor` is the intended center of gravity
for future inference work.

## Architecture overview

### Normalization

The current normalizer provides:

- optional ASCII lowercase
- optional ASCII whitespace collapse
- explicit whitespace escaping using the configured symbol
- optional dummy prefix insertion

Round-trip expectations are:

> `decode(encode(text)) == normalized(text)`

for inputs supported by the configured normalizer.

### BPE

The BPE trainer:

1. reads and normalizes raw text
2. escapes whitespace
3. splits input into UTF-8 symbols
4. interns symbols into compact ids for the training hot path
5. learns merges by adjacent pair frequency
6. stores ordered merge rules in the final model

### Unigram

The unigram trainer:

1. reads and normalizes raw text
2. escapes whitespace
3. splits input into UTF-8 symbols
4. builds a seed vocabulary from observed substrings
5. assigns initial scores from corpus counts
6. runs a hard-EM-style refinement loop
7. keeps the strongest final pieces and stores their scores

At inference time, unigram segmentation uses best-path dynamic programming.
`SentencePieceProcessor` caches the unigram piece index once per loaded model so repeated
encodes do not rebuild the search structure.

## Model format

Models are stored in a simple plain-text `.nsp` format.

Example BPE snippet:

```text
format	NSPM
version	2
model_type	bpe
trained_vocab_size	64
normalizer.lowercase	1
normalizer.collapse_whitespace	1
normalizer.add_dummy_prefix	1
normalizer.whitespace_symbol	▁
piece	0	<unk>
piece	1	<bos>
merge	0	h	e	he
```

Example unigram snippet:

```text
format	NSPM
version	2
model_type	unigram
trained_vocab_size	64
piece	0	<unk>
piece	4	▁hello
piece_score	4	-1.2345
```

Why plain text:

- easy to inspect
- easy to diff
- easy to debug while the format is still evolving

## Testing

The test suite covers:

- normalization behavior
- deterministic BPE learning
- BPE round-trip behavior
- processor/tokenizer parity
- serialization stability
- unigram training and inference
- cached unigram processor reuse

Run tests with:

```bash
ctest --test-dir build --output-on-failure
```

You can also run the example script:

```bash
bash examples/demo.sh
```

## Included today

- raw-text BPE and unigram training
- whitespace-preserving tokenization
- encode/decode for pieces and ids
- inspectable model serialization
- CLI tooling
- processor-oriented inference API
- cached unigram inference index

## Not included yet

- full SentencePiece trainer parity
- NFKC-grade Unicode normalization
- byte fallback
- stochastic segmentation / sampling
- protobuf `.model` compatibility
- Python bindings
- industrial-scale trainer optimizations across all paths

## Roadmap

High-value next steps include:

- smarter unigram pruning and seed selection
- lower-allocation unigram inference scratch buffers
- faster BPE inference structures
- large-corpus streaming and parallel training
- richer Unicode normalization
- compatibility tooling for upstream SentencePiece artifacts
- benchmark targets and perf regression tracking
