#!/usr/bin/env bash
# One-time setup: a Python virtual environment in .venv with the packages in requirements.txt,
# and a check that LaTeX (latexmk) is available for 4_paper.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python3 -m venv .venv
.venv/bin/pip install --quiet -r requirements.txt

if ! command -v latexmk > /dev/null; then
    echo "Warning: latexmk not found. Install a TeX distribution (e.g. TeX Live) to build 4_paper." >&2
fi

echo "Setup complete. Activate the environment with: source .venv/bin/activate"
