#!/usr/bin/env bash
# Module 1_data: clean the raw workbooks in 0_raw into CSV tables in output/.
#   - source/build_msca_data.py    MSCA applications by panel and reviewer scores
#   - source/build_sshrc_data.py   SSHRC Insight Grants applications and awards by committee
# Each script can also be run alone, from source/.
set -euo pipefail

MODULE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG="${MODULE_DIR}/output/make.log"

rm -rf "${MODULE_DIR}/output"
mkdir -p "${MODULE_DIR}/output"
echo "Making 1_data: started $(date '+%Y-%m-%d %H:%M:%S')" | tee "${LOG}"

cd "${MODULE_DIR}/source"
python3 -u build_msca_data.py 2>&1 | tee -a "${LOG}"
python3 -u build_sshrc_data.py 2>&1 | tee -a "${LOG}"

echo "Making 1_data: finished $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "${LOG}"
