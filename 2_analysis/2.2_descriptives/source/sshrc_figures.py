#!/usr/bin/env python3
"""Figures on success-rate equalization across the merit review committees of SSHRC's Insight
Grants, within each stream (A: smaller grants, B: larger grants), 2018-19 to 2025-26.

    python3 sshrc_figures.py                       # every figure and the table
    python3 sshrc_figures.py sshrc_awards_common_A # just that one

Committees are split into two groups by split_committees(): those funded at a common success
rate within each stream (group 1), and the others, all of which are funded above it (group 2).
Indigenous research and the Tri-Agency interdisciplinary committee (SPECIAL) are in neither.

Each output file holds only the picture (tikzpicture) or the tabular: the document that inputs it
supplies the figure or table environment, the caption and the label.

Writes to output/:
  sshrc_committees.tex       tabular of the committees of each group: code and name.
  sshrc_rates_both_{A,B}.tex success rate of every committee of both groups, coloured by group,
                             by stream and year (no lines).
  sshrc_rates_common_A.tex   Stream A: success rate of each committee at the common rate, by year,
                             with the pooled success rate of the group.
  sshrc_awards_common_A.tex  Stream A: awards of each of these committees minus the awards at the
                             pooled success rate, by year, with +-1 award dashed and, above each
                             year, the pooled rate and the committees within +-1.
  sshrc_spread_common_A.tex  Stream A: spread of the success rates of these committees around their
                             pooled rate, relative to chance, by year.
  sshrc_{rates,awards,spread}_common_B.tex
                             the same for Stream B.
  sshrc_{rates,awards,spread}_above_{A,B}.tex
                             the same for the other committees.
The four figures of each kind (success rates, awards, spread) share one vertical scale.

Input: 1_data/output/sshrc_ig_committees.csv (built by 1_data/source/build_sshrc_data.py).
"""

import csv
import sys
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
DATA = MODULE.parents[1] / "1_data" / "output"
OUTDIR = MODULE / "output"

YEARS = list(range(2018, 2026))           # results years with a stream split (2018 = 2018-19)
SELECT_YEARS = range(2020, 2026)          # years over which the committees are grouped
VLINE_YEAR = 2020                         # every figure marks 2020-21 with a grey dashed line
SLOT_HALF_WIDTH = 0.1                     # committee dots spread over +-0.1 of a year around it
AWARDS_HEIGHT = "5.6cm"                   # tallest at which three fit on a page below text
T_DROP = 3.0                              # drop a committee while its |t| is at least this; about 50
                                          # tests (committees x streams), so stricter than 2-2.5
SPECIAL = {"21": "Indigenous research", "24": "Tri-Agency interdisciplinary"}
SHORT = {"01": "Philosophy", "02": "History", "03": "Fine arts", "04": "Literature",
         "05": "Medieval, classics and religious studies", "07": "Economics",
         "08": "Sociology", "09": "Geography", "10": "Psychology", "11": "Political science",
         "12": "Education", "13": "Anthropology", "14": "Business", "15": "Linguistics",
         "16": "Communications", "17": "Law and criminology",
         "22": "Multidisciplinary humanities", "23": "Multidisciplinary social sciences"}
STREAMS = ["A", "B"]
COLOR = {"A": "2A78D6", "B": "EB6834"}    # categorical slots 1 and 2, as in msca_figures.py
MARK = {"A": "*", "B": "square*"}
# figures with both groups: colour (and marker) follows the group, in categorical slots 3 and 7,
# so as not to reuse the stream colours; the pair passes the dataviz palette validator
GROUP_COLOR = {"common": "1BAF7A", "above": "6250D6"}
GROUP_MARK = {"common": "*", "above": "triangle*"}
GROUP_MARK_SIZE = {"common": "1.8pt", "above": "2.4pt"}   # a triangle looks smaller than a circle


def load():
    with open(DATA / "sshrc_ig_committees.csv", newline="") as f:
        return [dict(r, year=int(r["year"]), apps=int(r["apps"]), awards=int(r["awards"]))
                for r in csv.DictReader(f)]


def cells(rows, codes, stream, year):
    """(code, apps, awards) of the given committees in one stream and year."""
    return [(r["code"], r["apps"], r["awards"]) for r in rows
            if r["stream"] == stream and r["year"] == year and r["code"] in codes]


def pooled(cs):
    return sum(k for _, _, k in cs) / sum(n for _, n, _ in cs)


def spread(cs):
    """Pearson chi2 / df of the awards around the pooled rate: 1 = as dispersed as independent
    draws at that rate, 0 = identical rates. Also the value if awards were the rounded common-rate
    awards, the floor with whole awards."""
    n = np.array([n for _, n, _ in cs]); k = np.array([k for _, _, k in cs])
    lam = k.sum() / n.sum(); v = n * lam * (1 - lam); df = len(n) - 1
    return ((k - lam * n) ** 2 / v).sum() / df, ((np.round(lam * n) - lam * n) ** 2 / v).sum() / df


def split_committees(rows):
    """(common, above, log): codes of the committees funded at a common within-stream rate and of
    the others. Start from every committee of SELECT_YEARS except SPECIAL. A committee's deviation
    in a stream and year is its awards minus the awards at the pooled rate of the committees kept;
    drop the committee whose mean deviation in either stream has the largest |t| (standard error:
    the SD of all deviations / sqrt(years)), while that |t| is at least T_DROP."""
    keep = sorted({r["code"] for r in rows if r["year"] in SELECT_YEARS and r["stream"] in STREAMS
                   and r["code"] not in SPECIAL})
    log = []
    while True:
        dev = {}
        for s in STREAMS:
            for y in SELECT_YEARS:
                cs = cells(rows, keep, s, y); lam = pooled(cs)
                for c, n, k in cs:
                    dev.setdefault((c, s), []).append(k - lam * n)
        sd = np.std([d for ds in dev.values() for d in ds], ddof=1)
        t = {cs: np.mean(d) / (sd / np.sqrt(len(d))) for cs, d in sorted(dev.items())}
        (c, s), worst = max(t.items(), key=lambda kv: abs(kv[1]))
        if abs(worst) < T_DROP:
            break
        log.append(f"  drop {c:>4} (stream {s}: mean deviation {np.mean(dev[c, s]):+.2f} awards, "
                   f"t = {worst:+.1f})")
        keep.remove(c)
    above = sorted({r["code"] for r in rows if r["year"] in YEARS and r["stream"] in STREAMS}
                   - set(keep) - set(SPECIAL))
    return keep, above, log


def names(codes):
    """'Philosophy, ..., Psychology (10A, 10B), ...' in code order, sub-committees merged."""
    groups = {}
    for c in codes:
        groups.setdefault(c[:2], []).append(c)
    return ", ".join(SHORT[g] + (f" ({', '.join(cs)})" if len(cs) > 1 else "")
                     for g, cs in groups.items())


def label(year):
    """2018 -> '2018--19', the SSHRC results label."""
    return f"{year}--{(year + 1) % 100:02d}"


def axis_head(ylo, yhi, ylabel, height="5.5cm"):
    labels = ",".join(label(y) for y in YEARS)
    return [
        "\\begin{tikzpicture}",
        "\\begin{axis}[",
        f"  width=15cm, height={height},",
        f"  xmin={YEARS[0] - 0.4}, xmax={YEARS[-1] + 0.4}, ymin={ylo}, ymax={yhi},",
        "  xtick={" + ",".join(map(str, YEARS)) + "}, xticklabels={" + labels + "},",
        "  axis x line=bottom, axis y line=left, axis line style={gray!70},",
        "  ymajorgrids, grid style={gray!20},",
        "  tick label style={font=\\footnotesize}, label style={font=\\footnotesize},",
        f"  xlabel={{SSHRC results year}}, ylabel={{{ylabel}}}, ylabel style={{align=center}},",
        "  clip=false,",
    ]


def colors():
    return [f"\\definecolor{{sshrc{s}}}{{HTML}}{{{COLOR[s]}}}" for s in STREAMS]


def coords(pts):
    return " ".join(f"({x:.2f},{y:.2f})" for x, y in pts)


def vline(ylo, yhi):
    """Grey dashed vertical line at VLINE_YEAR, drawn first so the data sit on top of it."""
    return (f"\\draw[gray, dashed, line width=0.7pt] (axis cs:{VLINE_YEAR},{ylo}) -- "
            f"(axis cs:{VLINE_YEAR},{yhi});")


def slots(codes):
    """Horizontal offset of each committee within a year, the same in every figure of a group:
    evenly spaced within +-SLOT_HALF_WIDTH of the year."""
    return {c: SLOT_HALF_WIDTH * (2 * i / (len(codes) - 1) - 1) for i, c in enumerate(codes)}


def write(name, lines):
    """Write output/<name>.tex: the picture or tabular only. The document that inputs it supplies
    the figure or table environment, the caption and the label."""
    head = ["% GENERATED BY sshrc_figures.py - edit that file, not this one.",
            "% The document that inputs this file supplies the float, the caption and the label."]
    (OUTDIR / f"{name}.tex").write_text("\n".join(head + lines + [""]))
    print(f"  wrote output/{name}.tex")


def rate_figure(rows, codes, stream, ylim, name):
    """One stream: each committee's success rate by year (dots, each committee at its own position
    within the year) and the pooled success rate of the group (line)."""
    slot = slots(codes)
    lines = [f"% Stream {stream}. Committees, left to right within each year: {names(codes)}."]
    lines += colors()
    lines += axis_head(*ylim, "Success rate")
    lines += ["  ytick distance=10, yticklabel={\\pgfmathprintnumber{\\tick}\\%},", "]", vline(*ylim)]
    dots = [(y + slot[c], 100 * k / n) for y in YEARS for c, n, k in cells(rows, codes, stream, y)]
    pool = [(y, 100 * pooled(cells(rows, codes, stream, y))) for y in YEARS]
    lines += [f"\\addplot[only marks, mark={MARK[stream]}, mark size=1.8pt, color=sshrc{stream},",
              "  mark options={fill opacity=0.6, draw opacity=0}]",
              f"  coordinates {{{coords(dots)}}};",
              f"\\addplot[color=sshrc{stream}, line width=1.2pt, mark={MARK[stream]}, mark size=2pt] "
              f"coordinates {{{coords(pool)}}};"]
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    write(name, lines)
    print("    pooled rate: " + "  ".join(f"{y} {r:.1f}%" for y, r in pool))


def rate_limits(rows, groups):
    """Vertical axis, shared by the four success-rate figures: the committees' lowest and highest
    rates, rounded out to multiples of 10 percent."""
    rates = [100 * k / n for codes in groups for s in STREAMS for y in YEARS
             for _, n, k in cells(rows, codes, s, y)]
    return 10 * int(min(rates) // 10), 10 * int(-(-max(rates) // 10))


def figure_rates(group, stream):
    def build(rows, common, above):
        rate_figure(rows, common if group == "common" else above, stream,
                    rate_limits(rows, [common, above]), f"sshrc_rates_{group}_{stream}")
    return build


def both_rate_figure(rows, common, above, stream, ylim, name):
    """One stream: each committee's success rate by year, both groups in one plot, coloured by
    group, without lines; within each year Group 1's committees sit left of Group 2's."""
    slot = slots(common + above)
    groups = (("common", common, f"Group 1 ({len(common)} committees)"),
              ("above", above, f"Group 2 ({len(above)} committees)"))
    lines = [f"% Stream {stream}. Group 1: {names(common)}. Group 2: {names(above)}."]
    lines += [f"\\definecolor{{sshrc{g}}}{{HTML}}{{{GROUP_COLOR[g]}}}" for g, _, _ in groups]
    lines += axis_head(*ylim, "Success rate")
    lines += ["  ytick distance=10, yticklabel={\\pgfmathprintnumber{\\tick}\\%},",
              "  legend style={font=\\footnotesize, draw=none, fill=none, at={(0.5,1.02)}, anchor=south,",
              "    legend columns=2, /tikz/every even column/.append style={column sep=1em}},",
              "  legend cell align=left,", "]", vline(*ylim)]
    for g, codes, entry in groups:
        dots = [(y + slot[c], 100 * k / n) for y in YEARS for c, n, k in cells(rows, codes, stream, y)]
        lines += [f"\\addplot[only marks, mark={GROUP_MARK[g]}, mark size={GROUP_MARK_SIZE[g]}, "
                  f"color=sshrc{g},", "  mark options={fill opacity=0.5, draw opacity=0}]",
                  f"  coordinates {{{coords(dots)}}};",
                  f"\\addlegendentry{{{entry}}}"]
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    write(name, lines)


def figure_rates_both(stream):
    def build(rows, common, above):
        both_rate_figure(rows, common, above, stream, rate_limits(rows, [common, above]),
                         f"sshrc_rates_both_{stream}")
    return build


def deviations(rows, codes, stream, year):
    """Pooled success rate of the committees, and (code, awards minus awards at that rate)."""
    cs = cells(rows, codes, stream, year)
    lam = pooled(cs)
    return lam, [(c, k - lam * n) for c, n, k in cs]


def awards_figure(rows, codes, stream, lim, name):
    """One stream: each committee's awards minus the awards at the pooled success rate of the
    group, by year (dots, each committee at its own position within the year), with the pooled
    rate at 0, the +-1 award band shaded and dashed, and above each year the pooled rate and the number of committees
    within +-1 award."""
    slot = slots(codes)
    lines = [f"% Stream {stream}. Committees, left to right within each year: {names(codes)}."]
    lines += colors()
    lines += axis_head(-lim, lim, f"$s^{{{stream}}}_{{c,t}}-\\tilde{{s}}^{{{stream}}}_{{c,t}}$",
                       AWARDS_HEIGHT)
    lines += ["  ytick distance=1,", "]"]
    lines.append(f"\\fill[gray!12] (axis cs:{YEARS[0] - 0.4},-1) rectangle (axis cs:{YEARS[-1] + 0.4},1);")
    lines.append(vline(-lim, lim))
    dom = f"domain={YEARS[0] - 0.4}:{YEARS[-1] + 0.4}, samples=2"
    lines.append(f"\\addplot[color=sshrc{stream}, line width=1.2pt, forget plot, {dom}] {{0}};")
    for b in (-1, 1):
        lines.append(f"\\addplot[gray!70!black, line width=0.7pt, dashed, forget plot, {dom}] {{{b}}};")
    dots, top = [], []
    for y in YEARS:
        lam, dev = deviations(rows, codes, stream, y)
        dots += [(y + slot[c], d) for c, d in dev]
        top.append((y, f"{100 * lam:.1f}%", f"{sum(abs(d) <= 1 for _, d in dev)}/{len(dev)}"))
    lines += [f"\\addplot[only marks, mark={MARK[stream]}, mark size=1.8pt, color=sshrc{stream},",
              "  mark options={fill opacity=0.6, draw opacity=0}]",
              f"  coordinates {{{coords(dots)}}};"]
    # two rows of figures above the plot, on common baselines, labelled at the left
    for shift, text, rows_text in ((12, "pooled rate", [t[1].replace("%", "\\%") for t in top]),
                                   (3, "within $\\pm1$ award", [t[2] for t in top])):
        lines.append(f"\\node[font=\\scriptsize, anchor=base east, text=gray!80!black] at "
                     f"([yshift={shift}pt]axis cs:{YEARS[0] - 0.4},{lim}) {{{text}}};")
        lines += [f"\\node[font=\\scriptsize, anchor=base] at ([yshift={shift}pt]axis cs:{y},{lim}) "
                  f"{{{t}}};" for y, t in zip(YEARS, rows_text)]
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    write(name, lines)
    print("    pooled rate, within +-1 award: " + "  ".join(f"{y} {r} {w}" for y, r, w in top))


def awards_limit(rows, groups):
    """Half-height of the vertical axis, shared by the four awards figures: the largest deviation,
    rounded up to a whole award, and at least 2."""
    top = max(abs(d) for codes in groups for s in STREAMS for y in YEARS
              for _, d in deviations(rows, codes, s, y)[1])
    return max(2, int(np.ceil(top)))


def figure_awards(group, stream):
    def build(rows, common, above):
        awards_figure(rows, common if group == "common" else above, stream,
                      awards_limit(rows, [common, above]), f"sshrc_awards_{group}_{stream}")
    return build


def spread_figure(rows, codes, stream, yhi, name):
    """One stream: spread of the committees' success rates around their pooled rate, by year."""
    val = {y: spread(cells(rows, codes, stream, y)) for y in YEARS}
    lines = [f"% Stream {stream}. Committees: {names(codes)}.",
             "% spread (Pearson chi2/df) and its whole-award floor: " + "; ".join(
                 f"{y}: {v:.3f} ({f:.3f})" for y, (v, f) in val.items())]
    lines += colors()
    lines += axis_head(0, yhi, "Spread of committee success rates")
    lines += ["  ytick distance=0.25, yticklabel style={/pgf/number format/.cd, fixed, fixed zerofill, precision=2},",
              "]", vline(0, yhi)]
    lines.append(f"\\addplot[gray, line width=0.6pt, densely dotted, forget plot, domain={YEARS[0] - 0.4}:"
                 f"{YEARS[-1] + 0.4}, samples=2] {{1}};")
    lines.append(f"\\node[font=\\footnotesize, text=gray!80!black, anchor=south east] at "
                 f"(axis cs:{YEARS[-1] + 0.4},1) {{chance}};")
    pts = [(y, val[y][0]) for y in YEARS]
    lines.append(f"\\addplot[color=sshrc{stream}, line width=1.2pt, mark={MARK[stream]}, mark size=1.8pt] "
                 f"coordinates {{{coords(pts)}}};")
    lines += ["\\end{axis}", "\\end{tikzpicture}"]
    write(name, lines)
    print("    spread: " + "  ".join(f"{y} {val[y][0]:.2f}" for y in YEARS))


def spread_limit(rows, groups):
    """Top of the vertical axis, shared by the four spread figures: the largest spread, rounded up
    to a multiple of 0.25, and at least 1.25 so that the chance line at 1 shows."""
    top = max(spread(cells(rows, codes, s, y))[0] for codes in groups for s in STREAMS for y in YEARS)
    return max(1.25, 0.25 * int(-(-top // 0.25)))


def figure_spread(group, stream):
    def build(rows, common, above):
        spread_figure(rows, common if group == "common" else above, stream,
                      spread_limit(rows, [common, above]), f"sshrc_spread_{group}_{stream}")
    return build


def table_committees(rows, common, above):
    """The committees of each group: code and name (the latest name printed for the code)."""
    name = {r["code"]: r["committee"] for r in sorted(rows, key=lambda r: r["year"])}
    lines = ["\\begin{tabular}{ll}", "\\toprule", "Code & Committee \\\\"]
    for title, codes in ((f"Group 1: {len(common)} committees funded at a common rate within each stream",
                          common),
                         (f"Group 2: the other {len(above)} committees", above)):
        lines += ["\\midrule", f"\\multicolumn{{2}}{{l}}{{\\textit{{{title}}}}} \\\\", "\\addlinespace[2pt]"]
        lines += [f"{c} & {name[c]} \\\\" for c in codes]
    lines += ["\\bottomrule", "\\end{tabular}"]
    write("sshrc_committees", lines)


TARGETS = {"sshrc_committees": table_committees}
for _s in STREAMS:
    TARGETS[f"sshrc_rates_both_{_s}"] = figure_rates_both(_s)
for _g in ("common", "above"):
    for _s in STREAMS:
        TARGETS[f"sshrc_rates_{_g}_{_s}"] = figure_rates(_g, _s)
        TARGETS[f"sshrc_awards_{_g}_{_s}"] = figure_awards(_g, _s)
        TARGETS[f"sshrc_spread_{_g}_{_s}"] = figure_spread(_g, _s)

if __name__ == "__main__":
    targets = sys.argv[1:] or list(TARGETS)        # no argument: build every figure
    unknown = [t for t in targets if t not in TARGETS]
    if unknown:
        sys.exit(f"unknown target(s) {unknown}; choose from {list(TARGETS)}")
    OUTDIR.mkdir(exist_ok=True)
    rows = load()
    common, above, log = split_committees(rows)
    print("\n".join(log))
    print(f"  common rate ({len(common)}): {' '.join(common)}")
    print(f"  above it ({len(above)}): {' '.join(above)}")
    for t in targets:
        TARGETS[t](rows, common, above)
