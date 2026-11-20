### Overview
Raw MSCA data: application counts for the Global (GF) and European (EF) Fellowships by
scientific panel, and the individual reviewer scores of Individual Fellowship proposals.
These files are never modified; `1_data` reads them and writes clean tables.

### Files
| File | Original name | Contents used by the code |
| --- | --- | --- |
| `MSCA_Data_and_Institutional_Changes.xlsx` | `MSCA_Data_and_Institutional_Changes (2).xlsx` | Sheet `Data`: eligible EF and GF proposals by call year and panel, 2017–2025. Sheet `Reviewer scores`: the three reviewer scores (0–100) of each of the 43,164 Individual Fellowship proposals in the 2014–2018 calls (Pina et al. 2021, eLife, Supplementary file 1). |
| `Percentiles_MSCA.xlsx` | `Percentiles MSCA.xlsx` | One sheet per call year, 2016–2024, transcribing the European Commission's published score percentiles; the row `Number of eligible proposals` gives the 2016 counts, which the main workbook lacks. |

The main workbook holds many more sheets (budgets, institutional dates, funded projects,
reports, ...) that the current code does not use.

### Source
Every source behind the main workbook is listed, with URLs and retrieval dates, in its sheet
`Sources and citations`; the sheet `Data` also gives the source URLs and provenance notes
for each row.

### When/where obtained & original form of files
TODO: record when and how each workbook was obtained.

### Notes
- The workbooks are stored with Git LFS (see `.gitattributes`): the main one is 89 MB, above
  the 50 MB file limit of Overleaf's GitHub sync, so git holds a small pointer file and GitHub
  LFS the workbook itself. Install Git LFS before cloning (`brew install git-lfs`, then
  `git lfs install`); without it the workbooks arrive as pointer files and `1_data` cannot run.
- Known quirks handled by `1_data/source/build_msca_data.py`: the 2023 percentile sheet
  labels the GF environment column `GF-ENY` instead of `GF-ENV`, and row labels vary slightly
  between years.
