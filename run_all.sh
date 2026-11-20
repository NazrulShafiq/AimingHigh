#!/usr/bin/env bash
# Rebuild the project from the raw data to the paper, one module after another.
# Run setup.sh once first, and activate the environment: source .venv/bin/activate
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

bash 1_data/make.sh
bash 2_analysis/2.1_theory/make.sh
bash 2_analysis/2.2_descriptives/make.sh
bash 4_paper/make.sh
