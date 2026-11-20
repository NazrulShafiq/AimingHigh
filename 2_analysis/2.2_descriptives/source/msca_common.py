"""Data, constants and helpers shared by the MSCA figure scripts of 2.2_descriptives.

Inputs (1_data/output, built by 1_data/source/build_msca_data.py):
  msca_applications.csv      eligible EF and GF proposals by call year and panel, 2016-2025
  msca_reviewer_scores.csv   the three reviewer scores (0-100) of each Individual Fellowship
                             proposal in the 2014-2018 calls (Pina et al. 2021)
"""

import csv
from pathlib import Path

import numpy as np

MODULE = Path(__file__).resolve().parents[1]
DATA = MODULE.parents[1] / "1_data" / "output"
OUTDIR = MODULE / "output"

YEARS = list(range(2016, 2026))
PANELS = ["CHE", "ECO", "ENG", "ENV", "LIF", "MAT", "PHY", "SOC"]
GRADE_CUTS = [10, 30, 50, 70, 90]    # 0-100 total -> the six grades of the 0-5 scale
N_GRADES = len(GRADE_CUTS) + 1


# ---------------------------------------------------------------- data
def counts():
    """{(year, panel): (EF, GF)} eligible proposals."""
    with open(DATA / "msca_applications.csv", newline="") as f:
        return {(int(r["year"]), r["panel"]): (int(r["ef_eligible"]), int(r["gf_eligible"]))
                for r in csv.DictReader(f)}


def reviewer_scores():
    """{panel: (n x 3 array of reviewer totals, n call years)}, proposals in workbook order."""
    by_panel = {}
    with open(DATA / "msca_reviewer_scores.csv", newline="") as f:
        for r in csv.DictReader(f):
            by_panel.setdefault(r["panel"], []).append(r)
    return {p: (np.array([[float(r[f"reviewer{k}"]) for k in (1, 2, 3)] for r in rows]),
                np.array([int(r["call_year"]) for r in rows]))
            for p, rows in by_panel.items()}


def grades(scores):
    """The 0-5 grade of each 0-100 score."""
    return np.searchsorted(GRADE_CUTS, scores, side="right")


# ---------------------------------------------------------------- agreement
def agreement(codes, q=N_GRADES, power=0.25):
    """Weighted percent agreement and Gwet's AC2 of an n x m array of category codes 0..q-1,
    with power weights 1 - (d/(q-1))^power, and the analytic standard error of AC2
    (Gwet 2008, as in irrCAC). Returns {"pa", "gwet", "gwet_se"}."""
    n, m = codes.shape
    k = np.arange(q)
    W = 1 - (np.abs(k[:, None] - k[None, :]) / (q - 1)) ** power
    cnt = np.stack([(codes == j).sum(1) for j in k], 1).astype(float)
    pa_i = np.sum(cnt * (cnt @ W.T - 1), 1) / (m * (m - 1))
    pi = cnt.mean(0) / m
    tw = W.sum() / (q * (q - 1))
    pe = tw * np.sum(pi * (1 - pi))
    pa = pa_i.mean()
    ac = (pa - pe) / (1 - pe)
    pe_i = tw * (cnt @ (1 - pi)) / m
    ac_i = (pa_i - pe) / (1 - pe) - 2 * (1 - ac) * (pe_i - pe) / (1 - pe)
    return {"pa": pa, "gwet": ac, "gwet_se": np.sqrt(np.sum((ac_i - ac) ** 2) / (n * (n - 1)))}


# ---------------------------------------------------------------- point labels
def place_labels_clear(pts, xbars, ybars, line, legend, w_cm, h_cm, names, char_w=0.23, lab_h=0.32):
    """Label centre for each point, chosen to keep the label clear of every confidence interval
    (bars and caps), every marker, the fitted line, the legend and the other labels.

    All positions are in normalised axis units: pts {name: (u, v)}; xbars, ybars {name: half-width
    of the 95% interval}; line ((u0, v0), (u1, v1)); legend (u0, v0, u1, v1) or None. w_cm, h_cm:
    plot-area size; char_w, lab_h: label size in cm. Returns {name: (u, v)} label centres."""
    P = {f: (u * w_cm, v * h_cm) for f, (u, v) in pts.items()}
    segs = []                                                # (x0, y0, x1, y1, weight, owner)
    for f, (x, y) in P.items():
        hx, hy, cap = xbars[f] * w_cm, ybars[f] * h_cm, 0.09
        segs += [(x - hx, y, x + hx, y, 1.0, f), (x, y - hy, x, y + hy, 1.0, f),
                 (x - hx, y - cap, x - hx, y + cap, 1.0, f), (x + hx, y - cap, x + hx, y + cap, 1.0, f),
                 (x - cap, y - hy, x + cap, y - hy, 1.0, f), (x - cap, y + hy, x + cap, y + hy, 1.0, f)]
    (a0, b0), (a1, b1) = line
    segs.append((a0 * w_cm, b0 * h_cm, a1 * w_cm, b1 * h_cm, 0.6, None))
    boxes = [(x - 0.1, y - 0.1, x + 0.1, y + 0.1, 3.0, f) for f, (x, y) in P.items()]       # markers
    if legend:
        boxes.append((legend[0] * w_cm, legend[1] * h_cm, legend[2] * w_cm, legend[3] * h_cm, 4.0, None))

    def seg_hits(r, s):                                      # Liang-Barsky clip of segment s against box r
        x0, y0, x1, y1 = s[:4]; dx, dy = x1 - x0, y1 - y0; t0, t1 = 0.0, 1.0
        for p_, q_ in ((-dx, x0 - r[0]), (dx, r[2] - x0), (-dy, y0 - r[1]), (dy, r[3] - y0)):
            if p_ == 0:
                if q_ < 0:
                    return False
            else:
                t = q_ / p_
                t0, t1 = (max(t0, t), t1) if p_ < 0 else (t0, min(t1, t))
                if t0 > t1:
                    return False
        return True

    overlaps = lambda r, o: not (r[2] <= o[0] or o[2] <= r[0] or r[3] <= o[1] or o[3] <= r[1])

    def candidates(f):
        x, y = P[f]; w, h = char_w * len(names[f]), lab_h
        return [(x + ex * (g + w / 2), y + ey * (g + h / 2), w, h, g)
                for g in (0.12, 0.24, 0.38, 0.55)
                for ex, ey in ((1, 1), (-1, 1), (1, -1), (-1, -1), (1, 0), (-1, 0), (0, 1), (0, -1))]

    def penalty(f, cd, placed):
        cx, cy, w, h, g = cd; m = 0.05
        r = (cx - w / 2 - m, cy - h / 2 - m, cx + w / 2 + m, cy + h / 2 + m)
        pen = sum(s[4] for s in segs if seg_hits(r, s))
        pen += sum(b[4] for b in boxes if b[5] != f and overlaps(r, b))
        pen += sum(5.0 for o in placed.values() if overlaps(r, o))
        if r[0] < 0 or r[2] > w_cm or r[1] < 0 or r[3] > h_cm:
            pen += 6.0
        return pen + 0.2 * g                                 # among clear positions, the closest

    placed, out, todo = {}, {}, set(P)
    box = lambda cd: (cd[0] - cd[2] / 2, cd[1] - cd[3] / 2, cd[0] + cd[2] / 2, cd[1] + cd[3] / 2)
    while todo:                                              # most constrained label first
        best = {f: min(candidates(f), key=lambda cd: penalty(f, cd, placed)) for f in todo}
        f = max(sorted(todo), key=lambda f: penalty(f, best[f], placed))
        placed[f] = box(best[f]); out[f] = best[f]; todo.remove(f)
    for f in sorted(P):                                      # one pass with every other label fixed
        others = {k: v for k, v in placed.items() if k != f}
        cd = min(candidates(f), key=lambda cd: penalty(f, cd, others))
        placed[f] = box(cd); out[f] = cd
    return {f: (cd[0] / w_cm, cd[1] / h_cm) for f, cd in out.items()}
