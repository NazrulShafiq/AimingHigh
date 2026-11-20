# Aiming High Under Noisy Evaluation

*Sorting into Programs Across Fields in Grantmaking.* Code, data and LaTeX sources of the
paper. The repository is organized on the
[GentzkowLabTemplate](https://github.com/gentzkowlab/GentzkowLabTemplate), © 2024 by Matthew
Gentzkow and distributed under the MIT license.

## Organization

Each module reads only the raw data or the `output/` of earlier modules, runs the code in its
`source/`, and writes everything it produces to its own `output/`, logging to
`output/make.log`. Builds are incremental where they are slow: `2.1_theory` rebuilds only the
figures that are new or whose code changed, and `4_paper` lets latexmk rerun only what changed.
The fast modules (`1_data`, `2.2_descriptives`, under a minute each) rebuild from scratch.

| Module | Contents |
| --- | --- |
| `0_raw/` | Raw MSCA workbooks, never modified (provenance in `0_raw/README.md`). |
| `1_data/` | `build_msca_data.py` cleans the raw workbooks into two CSV tables: eligible GF/EF applications by year and panel, and the reviewer scores of each proposal. |
| `2_analysis/2.1_theory/` | `generate_figures.py` computes every theory figure of the paper from the model; `static/` holds the distributions table and two figures without generating code. |
| `2_analysis/2.2_descriptives/` | `msca_figures.py` draws the data figures of the paper (Figures 1 and 2); `msca_noise_pages.py` checks the robustness of Figure 2 to the measure of reviewer noise. |
| `3_slides/` | Template placeholder, not yet used. |
| `4_paper/` | LaTeX sources of the paper (`Writeup.tex`) and of its Econometrica-format version, compiled into `4_paper/output/`. |

## Running

Set up once (Python virtual environment in `.venv`; LaTeX must be installed separately):

```sh
bash setup.sh
source .venv/bin/activate
```

Bring everything up to date, from the raw data to the paper:

```sh
bash run_all.sh
```

Or one module, e.g. `bash 4_paper/make.sh`. To add a theory figure, write its builder and add it
to `TARGETS` in `generate_figures.py`; `bash 2_analysis/2.1_theory/make.sh` then builds that
figure alone. `bash 2_analysis/2.1_theory/make.sh --all` rebuilds every theory figure from
scratch (about 15 minutes), and `bash 2_analysis/2.1_theory/make.sh <name> ...` just the named
ones (`python3 generate_figures.py --list` shows the names).

Generated files (`output/`) are never edited by hand: change the code that writes them.
