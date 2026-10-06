#!/bin/bash
# Rebuild the IEEE paper. Tectonic fetches IEEEtran and runs BibTeX itself; `brew install tectonic`.
# The figures come from the served model, so rebuild them only when the model or the data changes.
set -euo pipefail
cd "$(dirname "$0")"

[ "${1:-}" = "--figures" ] && python3 make_figures.py
tectonic --keep-logs ALFA_IEEE.tex

echo "wrote $PWD/ALFA_IEEE.pdf"
