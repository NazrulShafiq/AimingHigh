### Overview
Raw MSCA data: application counts for the Global (GF) and European (EF) Fellowships by
scientific panel, and the individual reviewer scores of Individual Fellowship proposals.
Raw SSHRC data: Insight Grants competition statistics, 2017-18 to 2025-26.
These files are never modified; `1_data` reads them and writes clean tables.

### Files
| File | Original name | Contents used by the code |
| --- | --- | --- |
| `MSCA_Data_and_Institutional_Changes.xlsx` | `MSCA_Data_and_Institutional_Changes (2).xlsx` | Sheet `Data`: eligible EF and GF proposals by call year and panel, 2017–2025. Sheet `Reviewer scores`: the three reviewer scores (0–100) of each of the 43,164 Individual Fellowship proposals in the 2014–2018 calls (Pina et al. 2021, eLife, Supplementary file 1). |
| `Percentiles_MSCA.xlsx` | `Percentiles MSCA.xlsx` | One sheet per call year, 2016–2024, transcribing the European Commission's published score percentiles; the row `Number of eligible proposals` gives the 2016 counts, which the main workbook lacks. |

The main workbook holds many more sheets (budgets, institutional dates, funded projects,
reports, ...) that the current code does not use.

| File | Contents used by the code |
| --- | --- |
| `ig_2017.xlsx` ... `ig_2025.xlsx` | SSHRC's competition statistics for Insight Grants, one workbook per results year (`ig_2018` = the 2018-19 results). The table BY COMMITTEE (sheet `- 6 -` in 2017-19, `- 5 -` from 2020): eligible applications, awards and dollars by merit review committee, for streams A and B separately and together. |

The other tables of these workbooks (by institution, region, discipline, team size, language,
...) are not used.

### Source
Every source behind the main workbook is listed, with URLs and retrieval dates, in its sheet
`Sources and citations`; the sheet `Data` also gives the source URLs and provenance notes
for each row.

The SSHRC workbooks are SSHRC's own competition statistics (TODO: record the download URL;
presumably https://sshrc-crsh.canada.ca/en/competition-results/statistics.aspx).

### When/where obtained & original form of files
TODO: record when and how each MSCA workbook was obtained. The SSHRC workbooks were added on
2026-11-21.

### Notes
- The workbooks are stored with Git LFS (see `.gitattributes`): the main one is 89 MB, above
  the 50 MB file limit of Overleaf's GitHub sync, so git holds a small pointer file and GitHub
  LFS the workbook itself. Install Git LFS before cloning (`brew install git-lfs`, then
  `git lfs install`); without it the workbooks arrive as pointer files and `1_data` cannot run.
- Known quirks handled by `1_data/source/build_msca_data.py`: the 2023 percentile sheet
  labels the GF environment column `GF-ENY` instead of `GF-ENV`, and row labels vary slightly
  between years.
- Known quirks handled by `1_data/source/build_sshrc_data.py`: `ig_2017` (2017-18) has no
  stream split; 2017-19 name the committees without codes (split committees numbered 1, 2, 3
  rather than A, B, C) and print success rates as percentages, 2020 on as fractions; committee
  names change slightly between years; 2022-25 spell committee 23A "Mutidisciplinary".
