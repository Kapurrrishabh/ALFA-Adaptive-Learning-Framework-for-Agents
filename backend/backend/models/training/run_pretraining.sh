#!/bin/bash
# Corpus, then tokenizer, then one pretraining run — in the only order that is correct.
#
# Finance normalisation and any change to the corpus both invalidate artifacts/tokenizer.json, and
# a vocabulary learned under different rules encodes the same text differently. So the tokenizer is
# rebuilt after the last collection and before training, and training happens once rather than
# twice. Arguments are passed through to train_pretrain.py (e.g. --resume).
#
# Wrap this in `caffeinate -i -m`: the run is ~11 h and an idle sleep mid-run has already cost 15
# hours once. Afterwards compare CPU time against elapsed time to confirm it never slept.
set -euo pipefail
cd "$(dirname "$0")/.."
: "${DATAFORGE_CONTACT:?set DATAFORGE_CONTACT to a real email address first}"

echo "=== rebuilding data/corpus from data/raw ==="
python3 -u dataforge/collect.py --process

echo "=== learning the vocabulary and tokenizing ==="
python3 -u scripts/prepare_pretrain.py

echo "=== pretraining ==="
python3 -u scripts/train_pretrain.py "$@"
