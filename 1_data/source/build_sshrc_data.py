#!/usr/bin/env python3
"""Clean SSHRC's Insight Grants competition statistics (0_raw/ig_2017.xlsx ... ig_2025.xlsx) into
one table of eligible applications and awards by year, stream and merit review committee.

Writes output/sshrc_ig_committees.csv, one row per year x stream x committee:
  year        first year of the SSHRC results label: 2018 = the 2018-19 results (file ig_2018)
  stream      A, B, or AB (both streams together; 2017-18, before the split, has AB only)
  code        committee code as numbered from 2020 on (01 Philosophy ... 24 Tri-Agency); the
              2017-19 committees, listed by name, are mapped to the same codes (CODES_BY_NAME)
  committee   English committee name as printed
  apps, awards               eligible applications and awards (projects)
  requested, awarded         dollars requested by all applications and awarded (multi-year)

Checks, each raising ValueError: every printed success rate equals awards / apps (catches a
misread column), the committees of each table add up to its TOTAL row, and from 2018 on the A
and B rows of each committee add up to its AB row.
"""

import csv
import re
from collections import defaultdict
from pathlib import Path

import openpyxl

MODULE = Path(__file__).resolve().parents[1]
RAW = MODULE.parent / "0_raw"
OUTDIR = MODULE / "output"
YEARS = range(2017, 2026)

# 2017-19 tables name the committees without codes; split committees are numbered 1, 2, 3 there
# and A, B, C from 2020 on (the mapping of 1 -> A, 2 -> B, 3 -> C is by order).
CODES_BY_NAME = {
    "Philosophy": "01",
    "History": "02",
    "Fine Arts, Research-Creation": "03",
    "Literature": "04",
    "Medieval, classics and religious studies": "05",
    "Medieval, classics, religious studies": "05",
    "Economics": "07",
    "Sociology, demography and related fields": "08",
    "Geography, urban planning and related fields": "09",
    "Psychology 1": "10A",
    "Psychology 2": "10B",
    "Political science and public administration": "11",
    "Political science and public administration 1": "11",
    "Education and social work 1": "12A",
    "Education and social work 2": "12B",
    "Education and social work 3": "12C",
    "Anthropology and archaeology": "13",
    "Business, management and related fields 1": "14A",
    "Business, management and related fields 2": "14B",
    "Linguistics and translation": "15",
    "Communications, media studies, library and information science, related fields": "16",
    "Communications, media studies, gender studies, library and information science, related fields": "16",
    "Law and criminology": "17",
    "Aboriginal research": "21",
    "Multidisciplinary Humanities": "22",
    "Multidisciplinary Social Sciences": "23",
}
STREAMS = {"A & B": "AB", "A": "A", "B": "B"}
FIELDS = ["year", "stream", "code", "committee", "apps", "awards", "requested", "awarded"]


def committee_sheet(wb):
    """The worksheet of the table BY COMMITTEE / SELON LE COMITÉ."""
    for ws in wb.worksheets:
        top = [c for r in ws.iter_rows(max_row=8, values_only=True) for c in r]
        if any(isinstance(c, str) and c.startswith("BY COMMITTEE") for c in top):
            return ws
    raise ValueError("no BY COMMITTEE table")


def english(cell):
    """English half of a bilingual 'English / Français' label, whitespace normalised."""
    return " ".join(re.split(r"\s/\s", cell)[0].split())


def code_of(name):
    m = re.match(r"(\d\d[A-C]?) - (.*)", name)
    if m:
        return m.group(1), m.group(2).replace("Mutidisciplinary", "Multidisciplinary")   # 23A's typo
    if name not in CODES_BY_NAME:
        raise ValueError(f"committee {name!r} has no code in CODES_BY_NAME")
    return CODES_BY_NAME[name], name


def read_year(year):
    """Rows of the committee table of ig_<year>.xlsx, checked against its TOTAL rows."""
    wb = openpyxl.load_workbook(RAW / f"ig_{year}.xlsx", read_only=True, data_only=True)
    rows, stream = [], "AB"                      # 2017-18 has one table, without a stream header
    for r in committee_sheet(wb).iter_rows(values_only=True):
        cells = [c for c in r if c is not None]
        if not cells:
            continue
        if isinstance(cells[0], str) and cells[0].startswith("Stream / Volet"):
            stream = STREAMS[cells[0].split("Volet", 1)[1].strip()]
            continue
        nums = [c for c in cells if isinstance(c, (int, float))]
        if not (isinstance(cells[0], str) and len(nums) == 8):
            continue                             # titles, column headers and notes
        apps, _, requested, awards, _, awarded, rate, _ = nums
        if year <= 2019:
            rate /= 100                          # 2017-19 print percentages, 2020 on fractions
        if apps and abs(rate - awards / apps) > 1e-6:
            raise ValueError(f"{year} {stream} {cells[0]!r}: printed success rate {rate} "
                             f"is not awards / apps = {awards}/{apps}")
        name = english(cells[0])
        if name == "TOTAL":
            got = [sum(x[k] for x in rows if x["stream"] == stream) for k in ("apps", "awards")]
            if got != [apps, awards]:
                raise ValueError(f"{year} {stream}: committees add up to {got}, "
                                 f"TOTAL row says {[apps, awards]}")
            continue
        code, name = code_of(name)
        rows.append(dict(year=year, stream=stream, code=code, committee=name,
                         apps=int(apps), awards=int(awards),
                         requested=round(requested), awarded=round(awarded)))
    return rows


def check_streams_add_up(rows):
    """From 2018 on, A + B = AB for every committee (a committee may lack an A or a B row)."""
    tot = defaultdict(lambda: [0, 0])
    for r in rows:
        if r["stream"] in ("A", "B"):
            tot[r["year"], r["code"], "A+B"][0] += r["apps"]
            tot[r["year"], r["code"], "A+B"][1] += r["awards"]
    for r in rows:
        if r["stream"] == "AB" and r["year"] >= 2018:
            if tot[r["year"], r["code"], "A+B"] != [r["apps"], r["awards"]]:
                raise ValueError(f"{r['year']} {r['code']}: A + B rows "
                                 f"{tot[r['year'], r['code'], 'A+B']} differ from the AB row")


def main():
    rows = [r for y in YEARS for r in read_year(y)]
    check_streams_add_up(rows)
    OUTDIR.mkdir(exist_ok=True)
    with open(OUTDIR / "sshrc_ig_committees.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote output/sshrc_ig_committees.csv ({len(rows)} rows)")
    for y in YEARS:
        by = {s: [r for r in rows if r["year"] == y and r["stream"] == s] for s in STREAMS.values()}
        print(f"  {y}: " + "  ".join(
            f"{s} {len(rs)} committees, {sum(r['apps'] for r in rs)} apps, "
            f"{sum(r['awards'] for r in rs)} awards" for s, rs in by.items() if rs))


if __name__ == "__main__":
    main()
