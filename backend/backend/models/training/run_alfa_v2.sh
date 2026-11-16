#!/bin/bash
# alfa_v2 in three steps, each run on its own and wrapped in `caffeinate -i`; one ALFA job at a time on this machine.
#   run_alfa_v2.sh build   the wide rows, the facts rows and their mix, into artifacts/alfa_v2_data
#   run_alfa_v2.sh train   the served generator's flags on the mix, keeping every checkpoint by its step
#   run_alfa_v2.sh score   each kept checkpoint on stockqa, casual and wide_unseen, which training never reads
# PASSAGE_ANSWERS=<dir> builds the facts rows from the answers written from passages_v1.jsonl, not the short answers.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
ART=backend/models/artifacts
DATA_DIR="$ART/alfa_v2_data"
WORK="$ART/alfa_v2_inputs"
FROZEN=backend/database/data/stock_qa/frozen_v1
# Provisional: the first build's stride, which gave 60,529 wide train rows beside 225,309 served; no reason was recorded
WIDE_STRIDE=34
# Two of its snapshots give 102-token evidence, over the 101 a price row may have, and prepare_advisory refuses the run
WIDE_EXCLUDED=(BAJAJHLDNG.NS.csv)

build() {
  rm -rf "$WORK/prices_wide"
  mkdir -p "$WORK/prices_wide" "$WORK/wide" "$WORK/facts"
  for f in "$PWD"/backend/database/data/prices_wide/*.csv; do
    [[ " ${WIDE_EXCLUDED[*]} " == *" $(basename "$f") "* ]] || ln -s "$f" "$WORK/prices_wide/$(basename "$f")"
  done
  ln -sf "$PWD/$ART/tokenizer.json" "$WORK/wide/tokenizer.json"
  ln -sf "$PWD/$ART/tokenizer.json" "$WORK/facts/tokenizer.json"
  $PY -u -m backend.models.training.prepare_advisory --casual --stride "$WIDE_STRIDE" \
    --prices "$WORK/prices_wide" --artifacts "$WORK/wide"
  $PY -u -m backend.models.training.prepare_stock_qa generator --frozen "$FROZEN" --artifacts "$WORK/facts" \
    ${PASSAGE_ANSWERS:+--passage-answers "$PASSAGE_ANSWERS"}
  $PY -u -m backend.models.training.prepare_alfa_v2 --wide "$WORK/wide" --wide-stride "$WIDE_STRIDE" \
    --wide-excluded "${WIDE_EXCLUDED[@]}" --facts "$WORK/facts" --frozen "$FROZEN" --out "$DATA_DIR"
}

train() {
  for needed in tokenizer.json pretrained.npz alfa_v2_train_source.npy alfa_v2_validation_source.npy alfa_v2_unseen_source.npy; do
    [ -f "$DATA_DIR/$needed" ] || { echo "missing $DATA_DIR/$needed; run $0 build first" >&2; exit 1; }
  done
  # advisory_combined's flags (session d7469cf9), its defaults spelled out; nothing served is written
  $PY -u -m backend.models.training.train_generator --artifacts "$DATA_DIR" --dataset alfa_v2 --run-name alfa_v2 \
    --extra-split unseen --freeze-encoder --epochs 1 --batch-size 8 --learning-rate 1e-4 \
    --log-every 50 --eval-every 2000 --eval-batches 25 --checkpoint-every 2000 --keep-checkpoints \
    > "$DATA_DIR/alfa_v2.log" 2>&1
}

score() {
  compgen -G "$DATA_DIR/alfa_v2.step*.npz" > /dev/null || { echo "no step checkpoints in $DATA_DIR; run $0 train first" >&2; exit 1; }
  # oldest first, which is step order: macOS's sort -V puts step9000 after step12000
  for checkpoint in $(ls -tr "$DATA_DIR"/alfa_v2.step*.npz); do
    for split in stockqa casual wide_unseen; do
      echo "=== $(basename "$checkpoint") on $split ==="
      $PY -u -m backend.models.training.check_answers --artifacts "$DATA_DIR" --dataset alfa_v2 \
        --checkpoint "$(basename "$checkpoint")" --split "$split" --rows 200 --show 0
    done
  done > "$DATA_DIR/alfa_v2.scores.log" 2>&1
}

case "${1:-}" in
  build|train|score) "$1" ;;
  *) echo "usage: $0 build|train|score" >&2; exit 2 ;;
esac
