#!/bin/bash
# alfa_v3's inputs, run on its own and wrapped in `caffeinate -i`; one ALFA job at a time on this machine.
#   run_alfa_v3.sh inputs   the served 101's rows and the wide rows, rebuilt from prices into artifacts/alfa_v3_inputs
#   run_alfa_v3.sh build    alfa_v3 from those, the facts passage answers and the v3 template answers, into alfa_v3_data
#   run_alfa_v3.sh train A|B [N]   phase 2 of run A or B in the background from its phase-1 checkpoint, or resumed from
#                                  its kept step-N checkpoint, logged to phase2.log
#   run_alfa_v3.sh score [OUT]     the newest checkpoint of each run and v2's step36000 on the same held-out rows, into
#                                  OUT, by default alfa_v3_data/scores.md
# ARTIFACTS' own advisory_casual arrays are left alone: they were built before the trend words were checked.
set -euo pipefail
cd "$(dirname "$0")/../../.."
PY=.venv/bin/python
ART=backend/models/artifacts
WORK="$ART/alfa_v3_inputs"
DATA_DIR="$ART/alfa_v3_data"
# The wide stride and left-out files alfa_v2 used, read from its record so the two sets cannot disagree
V2_MANIFEST="$ART/alfa_v2_data/MANIFEST.json"
WIDE_STRIDE=$($PY -c 'import json, sys; print(json.load(open(sys.argv[1]))["wide"]["stride"])' "$V2_MANIFEST")
WIDE_EXCLUDED=($($PY -c 'import json, sys; print(*json.load(open(sys.argv[1]))["wide"]["excluded"])' "$V2_MANIFEST"))
V2_STEP36000="$ART/alfa_v2_data/alfa_v2.step36000.npz"

inputs() {
  rm -rf "$WORK/prices_wide"
  mkdir -p "$WORK/prices_wide" "$WORK/served" "$WORK/wide"
  for f in "$PWD"/backend/database/data/prices_wide/*.csv; do
    [[ " ${WIDE_EXCLUDED[*]} " == *" $(basename "$f") "* ]] || ln -s "$f" "$WORK/prices_wide/$(basename "$f")"
  done
  ln -sf "$PWD/$ART/tokenizer.json" "$WORK/served/tokenizer.json"
  ln -sf "$PWD/$ART/tokenizer.json" "$WORK/wide/tokenizer.json"
  # advisory_casual's stride was never recorded; its AAPL validation snapshots sit 21 bars apart, the default
  $PY -u -m backend.models.training.prepare_advisory --casual --prices backend/database/data/prices \
    --artifacts "$WORK/served" > "$WORK/served/prepare.log" 2>&1
  $PY -u -m backend.models.training.prepare_advisory --casual --stride "$WIDE_STRIDE" --prices "$WORK/prices_wide" \
    --artifacts "$WORK/wide" > "$WORK/wide/prepare.log" 2>&1
}

build() {
  $PY -u -m backend.models.training.prepare_alfa_v3 --served "$WORK/served" --wide "$WORK/wide" \
    --stale-passages "$DATA_DIR/stale_passages.json" --out "$DATA_DIR" > "$DATA_DIR/prepare.log" 2>&1
}

train() {
  local run="v3_run$1" step="${2:-0}" shape
  case "$1" in
    A) shape=(--decoder-layers 4 --freeze-encoder) ;;
    B) shape=(--unfreeze-encoder-top 2) ;;
    *) echo "usage: $0 train A|B [N]" >&2; exit 2 ;;
  esac
  for f in "$DATA_DIR"/alfa_v3_*.npy "$DATA_DIR/MANIFEST.json"; do ln -sf "$PWD/$f" "$ART/$run/$(basename "$f")"; done
  local from="$ART/$run/$run.npz"
  [ "$step" = 0 ] || { from="$ART/$run/${run}_phase2.step$step.npz"; [ -f "$from" ] || { echo "no $from" >&2; exit 1; }; }
  # its own run name, so no phase-2 checkpoint overwrites the phase-1 one it starts from
  nohup caffeinate -i -m $PY -u -m backend.models.training.train_generator --artifacts "$ART/$run" --dataset alfa_v3 \
    --run-name "${run}_phase2" --init-from "$from" --start-step "$step" "${shape[@]}" --extra-split unseen --epochs 1 \
    --batch-size 8 --learning-rate 1e-4 --log-every 50 --eval-every 1000 --eval-batches 25 --checkpoint-every 1000 \
    --keep-checkpoints >> "$ART/$run/phase2.log" 2>&1 &
  echo "$run phase 2: pid $!"
}

newest() {
  # the end-of-run checkpoint once phase2.log says it was saved, as train_generator keeps no step file for it
  local run="v3_run$1"
  if grep -q "^saved " "$ART/$run/phase2.log"; then echo "$ART/$run/${run}_phase2.npz"; return; fi
  compgen -G "$ART/$run/${run}_phase2.step*.npz" > /dev/null || { echo "$run has no phase-2 checkpoint yet" >&2; exit 1; }
  # by time written: macOS's sort -V puts step9000 after step12000
  ls -t "$ART/$run/${run}_phase2".step*.npz | sed -n 1p
}

score() {
  local a b
  a=$(newest A)
  b=$(newest B)
  caffeinate -i $PY -u -m backend.models.training.score_v3 --data "$DATA_DIR" --out "${1:-$DATA_DIR/scores.md}" \
    --slots "$a" "$b" --plain "$V2_STEP36000"
}

case "${1:-}" in
  inputs|build) "$1" ;;
  train) train "${2:-}" "${3:-0}" ;;
  score) score "${2:-}" ;;
  *) echo "usage: $0 inputs|build|train A|B [N]|score [OUT]" >&2; exit 2 ;;
esac
