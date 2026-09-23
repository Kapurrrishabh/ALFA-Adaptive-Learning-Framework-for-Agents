#!/bin/bash
# Rebuild the synopsis PDF. Chrome's print engine is used because no LaTeX or pandoc is installed here.
set -euo pipefail
cd "$(dirname "$0")"

dot -Tpng -Gdpi=220 architecture.dot -o architecture.png

"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless --disable-gpu --no-pdf-header-footer --virtual-time-budget=6000 \
  --print-to-pdf="ALFA_Synopsis.pdf" "file://$PWD/synopsis.html" 2>/dev/null

echo "wrote $PWD/ALFA_Synopsis.pdf"
