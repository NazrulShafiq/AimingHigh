#!/usr/bin/env bash
# Module 4_paper: compile the paper into output/, from the LaTeX sources in source/, which
# input the figures and tables in 2_analysis/2.1_theory/output and 2_analysis/2.2_descriptives/output.
#   Writeup.tex          the paper
#   Writeup_format.tex   the same text in the Econometrica working-paper format
# The latexmk settings are in latexmkrc. latexmk keeps its auxiliary files in output/aux/
# (ignored by git) and reruns only the steps that changed sources or figures require; delete
# output/aux/ to compile from scratch. The full LaTeX log of each document goes to output/make.log.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${MODULE_DIR}/output/make.log"
DOCS=(Writeup Writeup_format)

mkdir -p "${MODULE_DIR}/output"
echo "Making 4_paper: started $(date '+%Y-%m-%d %H:%M:%S')" | tee "${LOG}"

cd "${MODULE_DIR}/source"          # the paper's relative \input paths start here
for doc in "${DOCS[@]}"; do
    echo "  compiling ${doc}.tex" | tee -a "${LOG}"
    if ! latexmk -r ../latexmkrc "${doc}.tex" >> "${LOG}" 2>&1; then
        echo "  ${doc}.tex failed; see output/aux/${doc}/${doc}.log" | tee -a "${LOG}"
        exit 1
    fi
done

echo "Making 4_paper: finished $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG}"
