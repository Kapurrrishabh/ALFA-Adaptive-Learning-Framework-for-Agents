#!/bin/bash
# Grounded training set, then the conversational generator. Runs after pretraining, never before:
# the encoder is initialised from artifacts/pretrained.npz, and reading the passages is the harder
# half of this task.
#
# Wrap in `caffeinate -i -m`. Arguments pass through to train_generator.py.
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== retrieving passages and encoding pairs ==="
python3 -u scripts/prepare_generator.py

echo "=== training the generator ==="
python3 -u scripts/train_generator.py "$@"
