#!/usr/bin/env bash
# Module 2.2_descriptives: the MSCA data figures, written to output/.
#   - source/msca_figures.py      Figures 1 and 2 of the paper
#   - source/msca_noise_pages.py  robustness of Figure 2 to the measure of reviewer noise
# Both read the clean tables in 1_data/output.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${MODULE_DIR}/output/make.log"

rm -rf "${MODULE_DIR}/output"
mkdir -p "${MODULE_DIR}/output"
echo "Making 2.2_descriptives: started $(date '+%Y-%m-%d %H:%M:%S')" | tee "${LOG}"

cd "${MODULE_DIR}/source"
python3 -u msca_figures.py 2>&1 | tee -a "${LOG}"
python3 -u msca_noise_pages.py 2>&1 | tee -a "${LOG}"

echo "Making 2.2_descriptives: finished $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG}"
