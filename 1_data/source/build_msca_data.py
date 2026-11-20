#!/usr/bin/env python3
"""Build the clean MSCA tables used by the analysis from the raw workbooks in 0_raw.

Writes to 1_data/output:
  msca_applications.csv      eligible EF and GF proposals by call year and panel, 2016-2025
  msca_reviewer_scores.csv   the three reviewer scores (0-100) of each Individual Fellowship
                             proposal in the 2014-2018 calls (Pina et al. 2021), in workbook order

Sources (0_raw):
  MSCA_Data_and_Institutional_Changes.xlsx
      sheet "Data": eligible EF and GF proposals by panel, 2017-2025
      sheet "Reviewer scores": the reviewer scores of each IF proposal, 2014-2018
  Percentiles_MSCA.xlsx
      sheet "<year>", row "Number of eligible proposals": the 2016 counts, which the main
      workbook lacks. Every year both workbooks cover (2017-2024) is checked for agreement.

Requires openpyxl.
"""

import csv
from pathlib import Path

import openpyxl

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "0_raw" / "MSCA_Data_and_Institutional_Changes.xlsx"
PCT = ROOT / "0_raw" / "Percentiles_MSCA.xlsx"
OUTDIR = ROOT / "1_data" / "output"

PANELS = ["CHE", "ECO", "ENG", "ENV", "LIF", "MAT", "PHY", "SOC"]
CROSS_CHECK_YEARS = range(2017, 2025)        # years both workbooks cover
REVIEWER_COLS = ["Reviewer1 score /100", "Reviewer2 score /100", "Reviewer3 score /100"]


def main_counts(wb):
    """{(year, panel): (EF, GF)} from sheet "Data", rows with both counts only."""
    out = {}
    for r in wb["Data"].iter_rows(values_only=True):
        if r and len(r) > 4 and isinstance(r[0], int) and r[3] and r[4]:
            out[(r[0], r[1])] = (int(r[3]), int(r[4]))
    return out


def percentile_counts(wb, year):
    """{panel: (EF, GF)} from the "Number of eligible proposals" row of one percentile sheet."""
    rows = [r for r in wb[str(year)].iter_rows(values_only=True) if r]
    label = lambda r: " ".join(str(r[0]).replace("\xa0", " ").split()).lower()   # labels vary by year
    counts = next(r for r in rows if label(r).startswith("number of eligible"))
    head = next(r for r in rows if label(r).startswith("score equal"))
    fix = {"GF-ENY": "GF-ENV"}                                    # header typo in the 2023 sheet
    num = {fix.get(h, h): int(str(v).split()[0]) for h, v in zip(head[1:], counts[1:]) if h and v}
    ef = "ST" if any(str(h).startswith("ST-") for h in head if h) else "EF"   # EF is "ST-" in early years
    return {p: (num[f"{ef}-{p}"], num[f"GF-{p}"]) for p in PANELS}


def application_counts(main_wb, pct_wb):
    """{(year, panel): (EF, GF)}, 2016-2025: the main workbook plus the 2016 percentile counts."""
    counts = main_counts(main_wb)
    for year in CROSS_CHECK_YEARS:
        for panel, v in percentile_counts(pct_wb, year).items():
            if counts[(year, panel)] != v:
                raise ValueError(f"{year} {panel}: main workbook {counts[(year, panel)]} "
                                 f"!= percentile workbook {v}")
    for panel, v in percentile_counts(pct_wb, 2016).items():
        counts[(2016, panel)] = v
    return counts


def reviewer_scores(wb):
    """[(panel, call year, score 1, score 2, score 3)] in sheet order, for the proposals of the
    eight panels with all three reviewer scores."""
    rows = wb["Reviewer scores"].iter_rows(values_only=True)
    head = next(r for r in rows if r and REVIEWER_COLS[0] in r)   # skip the title rows
    i = {h: k for k, h in enumerate(head)}
    cols = [i[c] for c in REVIEWER_COLS]
    return [(r[i["Panel"]], r[i["Call year"]], *(r[c] for c in cols))
            for r in rows
            if r and r[i["Panel"]] in PANELS and all(r[c] is not None for c in cols)]


def write_csv(name, header, rows):
    with open(OUTDIR / name, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    print(f"  wrote 1_data/output/{name} ({len(rows)} rows)")


def main():
    OUTDIR.mkdir(exist_ok=True)
    main_wb = openpyxl.load_workbook(MAIN, data_only=True, read_only=True)
    pct_wb = openpyxl.load_workbook(PCT, data_only=True, read_only=True)

    counts = application_counts(main_wb, pct_wb)
    order = sorted(counts, key=lambda k: (k[0], PANELS.index(k[1])))
    write_csv("msca_applications.csv", ["year", "panel", "ef_eligible", "gf_eligible"],
              [(y, p, *counts[(y, p)]) for y, p in order])

    write_csv("msca_reviewer_scores.csv", ["panel", "call_year", "reviewer1", "reviewer2", "reviewer3"],
              reviewer_scores(main_wb))


if __name__ == "__main__":
    main()
