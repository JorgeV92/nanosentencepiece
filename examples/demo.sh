#!/usr/bin/env bash
set -euo pipefail

cmake -S . -B build
cmake --build build

./build/nsp_train \
  --model build/demo.nsp \
  --vocab-size 64 \
  --min-pair-frequency 1 \
  data/tiny_corpus.txt

./build/nsp_encode \
  --model build/demo.nsp \
  --text "Hello world from nanosentencepiece" \
  --output pieces \
  --bos --eos

./build/nsp_encode \
  --model build/demo.nsp \
  --text "Hello world from nanosentencepiece" \
  --output ids \
  --bos --eos

./build/nsp_inspect \
  --model build/demo.nsp \
  --limit 20
