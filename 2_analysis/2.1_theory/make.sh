#!/usr/bin/env bash
# Module 2.1_theory: the theory figures and tables of the paper, written to output/.
#   - source/generate_figures.py computes every TikZ figure from the model. The build is
#     incremental: only figures that are new or whose code changed are rebuilt.
#         bash make.sh              new or changed figures only
#         bash make.sh NAME ...     just these figures (see generate_figures.py --list)
#         bash make.sh --all        every figure, from scratch (about 15 minutes)
#   - source/static/ holds the hand-written distributions table (tab_dists.tex) and two
#     figures whose generating code is not in the repository (sorting.tex, comparative.tex);
#     they are copied to output/ unchanged.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${MODULE_DIR}/output/make.log"

mkdir -p "${MODULE_DIR}/output"
echo "Making 2.1_theory: started $(date '+%Y-%m-%d %H:%M:%S')" | tee "${LOG}"

cd "${MODULE_DIR}/source"
python3 -u generate_figures.py "$@" 2>&1 | tee -a "${LOG}"
cp static/*.tex "${MODULE_DIR}/output/"
echo "  copied static/*.tex to output/" | tee -a "${LOG}"

echo "Making 2.1_theory: finished $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG}"
