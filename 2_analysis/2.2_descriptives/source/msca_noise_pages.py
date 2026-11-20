#!/usr/bin/env python3
"""Robustness of the GF/EF-against-noise pattern to the measure of reviewer noise.

    python3 msca_noise_pages.py              # every figure and page, every measure
    python3 msca_noise_pages.py a            # just the pooled scatter plots (figures a)
    python3 msca_noise_pages.py a c page ns  # targets and measures can be combined

Measures, all from the IF 2014-2018 reviewer scores:
  pa       weighted percent agreement, 0-5 grade (power weights with exponent 1/4)
  gwet     Gwet's AC, 0-5 grade (same weights)
  gwet1618 Gwet's AC, 0-5 grade, on the reviewer scores of the 2016-2018 calls only (the calls
           that overlap the 2016-2020 GF/EF window)
  gwetrs   Gwet's AC on rescaled grades: within each panel and call, the totals are rescaled to
           the mean and SD of all IF totals and then graded 0-5 as for gwet; per-call AC,
           weighted by proposals
  ns       noise-to-signal ratio of the 0-100 totals: within-proposal SD of the three reviewers'
           totals divided by the between-proposal SD, both from a one-way ANOVA by panel
  icc      one-way intraclass correlation ICC(1) of the same totals, 1 / (1 + ns^2)
  iccr     the same ICC after replacing each total by the normal score of its rank among all
           reviewer totals of its panel and call (within-call ranks; robust to the tail of low scores)

For each measure, writes to output/:
  MSCA_noise_a_<measure>_<scale>.pdf  GF/EF applications pooled over 2016-2020, with 95% CIs,
                                      against the measure; fitted lines for all panels and
                                      without physics (PHY).
  MSCA_noise_c_<measure>_<scale>.pdf  annual GF/EF applications, 2016-2025, coloured by the measure.
and, for gwet1618, gwetrs, ns, icc and iccr, a LaTeX page with both figures and a table
(target "page"): Pages_MSCA_Gwet1618.tex, Pages_MSCA_GwetRescaled.tex, Pages_MSCA_SNR.tex,
Pages_MSCA_ICC.tex and Pages_MSCA_ICCrank.tex. For pa and gwet no page is written; their
estimates and bootstrap standard errors are printed to the log.

Inputs: the clean tables in 1_data/output (see msca_common.py). Requires scipy and matplotlib.
"""

import sys

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm, colors
from scipy import stats

from msca_common import OUTDIR, PANELS as F, YEARS
from msca_common import agreement, counts, grades, place_labels_clear, reviewer_scores

# code, name on the axis, kind ("agree": plotted as 1 - value; "ratio": plotted as is), file tag, scale
MEAS = [("pa", "percent agreement", "agree", "g5", "0–5 grade"),
        ("gwet", "Gwet's AC", "agree", "g5", "0–5 grade"),
        ("gwet1618", "Gwet's AC", "agree", "g5", "0–5 grade, 2016–18"),
        ("gwetrs", "Gwet's AC", "agree", "g5rs", "rescaled grade"),
        ("ns", "noise-to-signal ratio", "ratio", "raw", "0–100 score"),
        ("icc", "ICC", "agree", "raw", "0–100 score"),
        ("iccr", "ICC", "agree", "rank", "within-call ranks")]
WINDOW = ("pooled 2016–20", list(range(2016, 2021)))
EXCLUDE = "PHY"                      # panel left out of the dashed fit
NO_DASHED = {"gwet"}                 # measures whose pooled graph shows only the fit for all panels
PAPER_STYLE = {"gwet"}               # pooled graph drawn like Figure 2 of the paper: GF share, 13 x 8 cm
B, SEED = 1000, 11                   # bootstrap replications and seed
YT = [0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25]
PDF_METADATA = {"CreationDate": None}   # no timestamp: an unchanged figure gives an unchanged file
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "cm", "font.size": 9.5,
                     "axes.spines.top": False, "axes.spines.right": False})


def anova(X):
    """Within- and between-proposal SDs of an n x k array of reviewer totals (one-way ANOVA)."""
    n, k = X.shape
    rm = X.mean(1)
    msb = k * np.sum((rm - X.mean()) ** 2) / (n - 1)
    msw = np.sum((X - rm[:, None]) ** 2) / (n * (k - 1))
    return np.sqrt(msw), np.sqrt((msb - msw) / k)


def icc_rank(X, years):
    """ICC of the normal scores of the reviewer totals' ranks within each call."""
    Z = np.empty_like(X)
    for y in np.unique(years):
        k = years == y
        s = X[k].ravel()
        Z[k] = stats.norm.ppf((stats.rankdata(s) - 0.5) / len(s)).reshape(-1, X.shape[1])
    sw, sb = anova(Z)
    return sb ** 2 / (sb ** 2 + sw ** 2)


def gwet_rescaled(X, years, mu, sd):
    """Gwet's AC on the 0-5 grade after rescaling each call's totals to mean mu and SD sd;
    the per-call values are averaged with weights equal to the number of proposals."""
    vals, wts = [], []
    for y in np.unique(years):
        k = years == y
        Z = mu + sd * (X[k] - X[k].mean()) / X[k].std()
        vals.append(agreement(grades(Z))["gwet"])
        wts.append(k.sum())
    return np.average(vals, weights=wts)


def measures():
    """{(measure, panel): estimate} and bootstrap standard errors; also the SDs behind ns and icc.
    Each block draws from its own stream seeded with SEED, so its SEs do not depend on the others."""
    R = reviewer_scores()
    est, se = {}, {}
    rng = np.random.default_rng(SEED)
    for f in F:
        C = grades(R[f][0])
        a = agreement(C)
        bs = [agreement(C[rng.integers(0, len(C), len(C))]) for _ in range(B)]
        for m in ("pa", "gwet"):
            est[(m, f)] = a[m]
            se[(m, f)] = np.std([b[m] for b in bs])
    rng = np.random.default_rng(SEED)
    for f in F:
        X = R[f][0]
        sw, sb = anova(X)
        est[("sw", f)], est[("sb", f)], est[("ns", f)] = sw, sb, sw / sb
        est[("icc", f)] = 1 / (1 + (sw / sb) ** 2)
        bs = []
        for _ in range(B):
            w, b = anova(X[rng.integers(0, len(X), len(X))])
            bs.append(w / b)
        se[("ns", f)] = np.std(bs)
        se[("icc", f)] = np.std([1 / (1 + r ** 2) for r in bs])
    rng = np.random.default_rng(SEED)
    for f in F:
        X, years = R[f]
        est[("iccr", f)] = icc_rank(X, years)
        draws = [rng.integers(0, len(X), len(X)) for _ in range(B)]
        se[("iccr", f)] = np.std([icc_rank(X[i], years[i]) for i in draws])
    rng = np.random.default_rng(SEED)
    for f in F:                              # 0-5 grade, calls 2016-2018 only
        X, years = R[f]
        C = grades(X[(years >= 2016) & (years <= 2018)])
        est[("gwet1618", f)] = agreement(C)["gwet"]
        se[("gwet1618", f)] = np.std([agreement(C[rng.integers(0, len(C), len(C))])["gwet"] for _ in range(B)])
    allx = np.concatenate([R[f][0].ravel() for f in F])
    mu, sd = allx.mean(), allx.std()          # mean and SD of all IF totals, all panels and calls
    rng = np.random.default_rng(SEED)
    for f in F:
        X, years = R[f]
        est[("gwetrs", f)] = gwet_rescaled(X, years, mu, sd)
        draws = [rng.integers(0, len(X), len(X)) for _ in range(B)]
        se[("gwetrs", f)] = np.std([gwet_rescaled(X[i], years[i], mu, sd) for i in draws])
    return est, se


def logy(ax, lo, hi):
    ax.set_yscale("log"); ax.set_ylim(lo, hi); ax.minorticks_off()
    t = [v for v in YT if lo <= v <= hi]
    ax.set_yticks(t); ax.set_yticklabels([f"{v:.2f}" for v in t])


def setup(meas, est, se):
    m, short, kind, _, scale = meas
    D = {f: (1 - est[(m, f)]) if kind == "agree" else est[(m, f)] for f in F}
    S = {f: se[(m, f)] for f in F}
    norm = colors.Normalize(min(D.values()), max(D.values()))
    cmap = matplotlib.colormaps["viridis"]
    col = {f: cmap(0.08 + 0.84 * norm(D[f])) for f in F}
    xlab = f"disagreement: $1-$ {short} ({scale})" if kind == "agree" else f"{short} ({scale})"
    return D, S, col, cmap, xlab


def save(fig, name, note=""):
    fig.savefig(OUTDIR / name, metadata=PDF_METADATA)
    plt.close(fig)
    print(f"  wrote output/{name}{note}")


def figure_a(c, est, se, chosen):
    title, years = WINDOW
    g = {f: sum(c[(y, f)][1] for y in years) for f in F}
    e = {f: sum(c[(y, f)][0] for y in years) for f in F}
    pooled = {f: g[f] / e[f] for f in F}
    lse = {f: np.sqrt(1 / g[f] + 1 / e[f]) for f in F}         # SE of the log ratio
    rest = [f for f in F if f != EXCLUDE]
    for meas in chosen:
        m, tag = meas[0], meas[3]
        if m in PAPER_STYLE:
            figure_a_paper(c, est, se, meas)
            continue
        D, S, col, _, xlab = setup(meas, est, se)
        lo_ci = min(D[f] - 1.96 * S[f] for f in F); hi_ci = max(D[f] + 1.96 * S[f] for f in F)
        xlo, xhi = lo_ci - 0.05 * (hi_ci - lo_ci), hi_ci + 0.05 * (hi_ci - lo_ci)
        fits = {key: stats.linregress([D[f] for f in fs], np.log([pooled[f] for f in fs]))
                for key, fs in (("all", F), ("rest", rest))}
        fig, ax = plt.subplots(figsize=(6.69, 3.0))
        xs = np.linspace(xlo, xhi, 50)
        ax.plot(xs, np.exp(fits["all"].intercept + fits["all"].slope * xs), color="0.4", lw=1.1,
                label=f"all: $R^2$ = {fits['all'].rvalue ** 2:.2f}")
        if m not in NO_DASHED:
            ax.plot(xs, np.exp(fits["rest"].intercept + fits["rest"].slope * xs), color="0.4", lw=1.0, ls="--",
                    label=f"without {EXCLUDE}: $R^2$ = {fits['rest'].rvalue ** 2:.2f}")
        for f in F:
            lo, hi = pooled[f] * np.exp(-1.96 * lse[f]), pooled[f] * np.exp(1.96 * lse[f])
            ax.errorbar(D[f], pooled[f], xerr=1.96 * S[f], yerr=[[pooled[f] - lo], [hi - pooled[f]]], fmt="o", ms=6.5,
                        color=col[f], mec="k", mew=0.6, elinewidth=1.1, capsize=2.5, zorder=3)
        logy(ax, 0.036, 0.24); ax.set_xlim(xlo, xhi)
        ax.set_ylabel("GF/EF applications (log scale)"); ax.set_xlabel(xlab)
        ax.set_title(title + ", 95% CIs", fontsize=9.5)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.canvas.draw()
        # label each point on the side that keeps it furthest from the other points
        pts = {f: ax.transData.transform((D[f], pooled[f])) for f in F}
        cand = [((7, 5), "left", "bottom"), ((7, -7), "left", "top"), ((-7, 5), "right", "bottom"),
                ((-7, -7), "right", "top"), ((0, 9), "center", "bottom"), ((0, -9), "center", "top")]
        for f in F:
            def score(cd):
                (ox, oy), ha, va = cd; w = 22 if ha != "center" else 0; h = 8
                cx = pts[f][0] + ox + (w / 2 if ha == "left" else -w / 2 if ha == "right" else 0)
                cy = pts[f][1] + oy + (h / 2 if va == "bottom" else -h / 2)
                return min(np.hypot(cx - pts[k][0], cy - pts[k][1]) for k in F if k != f)
            (ox, oy), ha, va = max(cand, key=score)
            ax.annotate(f, (D[f], pooled[f]), xytext=(ox, oy), textcoords="offset points", fontsize=8.5, ha=ha, va=va)
        save(fig, f"MSCA_noise_a_{m}_{tag}.pdf")
        for key, lab in (("all", "all panels"), ("rest", f"without {EXCLUDE}")):
            ft = fits[key]
            print(f"    {lab}: slope {ft.slope:.2f}, R2 {ft.rvalue ** 2:.2f}, p {ft.pvalue:.3f}")


def figure_a_paper(c, est, se, meas):
    """The pooled graph in the style of Figure 2 of the paper (msca_figures.py): share of applicants
    choosing the GF, 2016-2020, against disagreement, all panels, one dashed fit; 14.5 x 8.5 cm, Times."""
    from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MultipleLocator
    m, tag = meas[0], meas[3]
    _, years = WINDOW
    G = {f: sum(c[(y, f)][1] for y in years) for f in F}
    N = {f: sum(sum(c[(y, f)]) for y in years) for f in F}
    s = {f: 100 * G[f] / N[f] for f in F}
    ds = {f: 1.96 * 100 * np.sqrt(G[f] / N[f] * (1 - G[f] / N[f]) / N[f]) for f in F}
    D, S, *_ = setup(meas, est, se)
    lo = min(D[f] - 1.96 * S[f] for f in F); hi = max(D[f] + 1.96 * S[f] for f in F)
    xlo, xhi = lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo)
    ylo, yhi = 0, 5 * int(np.ceil((max(s.values()) + 2) / 5))
    fit = stats.linregress([D[f] for f in F], [s[f] for f in F])
    with plt.rc_context({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 10}):
        fig, ax = plt.subplots(figsize=(14.5 / 2.54, 8.5 / 2.54))
        ax.yaxis.grid(True, color="0.9", lw=0.5); ax.set_axisbelow(True)
        ax.plot([xlo, xhi], [fit.intercept + fit.slope * v for v in (xlo, xhi)], color="0.35", lw=1.0, ls=(0, (3, 3)))
        for f in F:
            ax.errorbar(D[f], s[f], xerr=1.96 * S[f], yerr=ds[f], fmt="o", ms=5.2, color="#2A78D6", mec="white", mew=0.6,
                        ecolor="#7FAEE6", elinewidth=0.6, capsize=2.5, capthick=0.6, zorder=3)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        ax.xaxis.set_major_locator(MultipleLocator(0.02)); ax.xaxis.set_minor_locator(MultipleLocator(0.01))
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
        for side in ("top", "right"): ax.spines[side].set_visible(False)
        for side in ("left", "bottom"): ax.spines[side].set_color("0.65"); ax.spines[side].set_linewidth(0.6)
        ax.tick_params(direction="inout", length=4, width=0.5, color="0.65")
        ax.tick_params(axis="x", which="minor", direction="inout", length=2.5, width=0.5, color="0.65")
        ax.plot(1, 0, ">", transform=ax.transAxes, clip_on=False, color="0.65", ms=3)
        ax.plot(0, 1, "^", transform=ax.transAxes, clip_on=False, color="0.65", ms=3)
        ax.set_xlabel("Reviewer disagreement: 1 − Gwet's AC")
        ax.set_ylabel("Share of total applicants who choose GF")
        fig.tight_layout()
        # point labels placed as in Figure 2: clear of the intervals, marks and fit
        w_cm, h_cm = 12.9, 6.9                  # Figure 2's plot area, so both place the labels identically
        norm = {f: ((D[f] - xlo) / (xhi - xlo), (s[f] - ylo) / (yhi - ylo)) for f in F}
        seg = [((v - xlo) / (xhi - xlo), (fit.intercept + fit.slope * v - ylo) / (yhi - ylo)) for v in (xlo, xhi)]
        lab = place_labels_clear(norm, {f: 1.96 * S[f] / (xhi - xlo) for f in F}, {f: ds[f] / (yhi - ylo) for f in F},
                                 seg, None, w_cm, h_cm, {f: f for f in F})
        for f, (u, v) in lab.items():
            ax.text(xlo + u * (xhi - xlo), ylo + v * (yhi - ylo), f, ha="center", va="center")
        save(fig, f"MSCA_noise_a_{m}_{tag}.pdf", " (paper style)")
    print(f"    GF share on disagreement, all panels: slope {fit.slope:.1f} pp per unit, R2 {fit.rvalue ** 2:.2f}, p {fit.pvalue:.3f}")


def figure_c(c, est, se, chosen):
    ratio = {(f, y): c[(y, f)][1] / c[(y, f)][0] for f in F for y in YEARS}
    for meas in chosen:
        m, tag = meas[0], meas[3]
        D, _, col, cmap, xlab = setup(meas, est, se)
        fig, ax = plt.subplots(figsize=(6.69, 3.0))
        for f in F:
            y = [ratio[(f, yy)] for yy in YEARS]
            ax.plot(YEARS[:6], y[:6], color=col[f], lw=1.1, alpha=0.3)
            ax.plot(YEARS[5:], y[5:], color=col[f], lw=1.9, marker="o", ms=3.5)
        ends = sorted(F, key=lambda f: ratio[(f, 2025)])
        ly = np.log([ratio[(f, 2025)] for f in ends])
        for i in range(1, len(ly)): ly[i] = max(ly[i], ly[i - 1] + 0.105)
        for f, v in zip(ends, ly):
            ax.annotate(f, (2025, ratio[(f, 2025)]), xytext=(2025.3, np.exp(v)), textcoords="data", fontsize=8.5,
                        va="center", arrowprops=dict(arrowstyle="-", color="0.7", lw=0.5))
        ax.axvline(2020.5, color="0.6", ls=":", lw=0.9); ax.text(2020.58, 0.0275, "PF reform", fontsize=8, color="0.4")
        logy(ax, 0.026, 0.29); ax.set_xticks(YEARS); ax.set_xlim(2015.7, 2026.0)
        ax.set_xlabel("call year"); ax.set_ylabel("GF/EF applications (log scale)")
        sm = cm.ScalarMappable(norm=colors.Normalize(min(D.values()), max(D.values())),
                               cmap=colors.ListedColormap(cmap(np.linspace(0.08, 0.92, 256))))
        cb = fig.colorbar(sm, ax=ax, fraction=0.04, pad=0.02)
        lab = xlab if len(xlab) <= 42 else xlab.replace(" (", "\n(", 1)   # long labels on two lines
        cb.set_label(lab, fontsize=8.5); cb.ax.tick_params(labelsize=8)
        fig.tight_layout()
        save(fig, f"MSCA_noise_c_{m}_{tag}.pdf")


def page(c, est, se, chosen):
    """Write the LaTeX page of each chosen measure that has one: graphs (a), (c) and a table."""
    pf = lambda p: "$<$0.001" if p < 0.001 else f"{p:.3f}"
    head = lambda what, detail="IF 2014-2018 reviewer totals (0-100); one-way ANOVA by panel; SE: bootstrap (1,000).": [
        f"% Page: GF/EF applications against {what} of panel reviewer scores.",
        "% GENERATED BY msca_noise_pages.py - edit that file, not this one.",
        f"% {detail}", ""]
    def body(title, m, tag, order, rows):
        return [r"\clearpage", r"\subsection*{" + title + "}", r"\begin{center}",
                r"\includegraphics[width=\textwidth]{" + f"MSCA_noise_a_{m}_{tag}.pdf" + "}", r"\par\medskip",
                r"\includegraphics[width=\textwidth]{" + f"MSCA_noise_c_{m}_{tag}.pdf" + "}", r"\par\medskip",
                r"\small\renewcommand{\arraystretch}{1.15}\setlength{\tabcolsep}{4pt}",
                r"\begin{tabular}{@{}l" + "c" * len(F) + r"@{}}", r"\toprule",
                " & " + " & ".join(order) + r" \\", r"\midrule"] + rows + [r"\bottomrule", r"\end{tabular}", r"\end{center}", ""]
    row = lambda label, vals: label + " & " + " & ".join(vals) + r" \\"
    for m, _, _, tag, _ in chosen:
        if m == "ns":
            order = sorted(F, key=lambda f: -est[("ns", f)])            # noisiest first, as in the other tables
            L = head("the noise-to-signal ratio") + body("Noise-to-signal ratio --- 0–100 score", m, tag, order, [
                row("Noise: within-proposal SD", [f"{est[('sw', f)]:.1f}" for f in order]),
                row("Signal: between-proposal SD", [f"{est[('sb', f)]:.1f}" for f in order]),
                row("Noise-to-signal ratio", [f"{est[('ns', f)]:.3f}" for f in order]),
                row("SE", [f"({se[('ns', f)]:.3f})" for f in order])])
            name = "Pages_MSCA_SNR.tex"
        elif m == "icc":
            order = sorted(F, key=lambda f: est[("icc", f)])            # least agreement first
            L = head("the intraclass correlation") + body("Intraclass correlation (ICC) --- 0–100 score", m, tag, order, [
                row("ICC (0–100 score)", [f"{est[('icc', f)]:.3f}" for f in order]),
                row("SE", [f"({se[('icc', f)]:.3f})" for f in order]),
                row("$p$", [pf(2 * stats.norm.sf(est[("icc", f)] / se[("icc", f)])) for f in order])])
            name = "Pages_MSCA_ICC.tex"
        elif m == "iccr":
            order = sorted(F, key=lambda f: est[("iccr", f)])           # least agreement first
            L = head("the intraclass correlation of within-call ranks",
                     "IF 2014-2018 reviewer totals, each replaced by the normal score of its rank within its panel and call; "
                     "one-way ANOVA by panel; SE: bootstrap (1,000).") + \
                body("Intraclass correlation (ICC) --- within-call ranks", m, tag, order, [
                    row("ICC (within-call ranks)", [f"{est[('iccr', f)]:.3f}" for f in order]),
                    row("SE", [f"({se[('iccr', f)]:.3f})" for f in order]),
                    row("$p$", [pf(2 * stats.norm.sf(est[("iccr", f)] / se[("iccr", f)])) for f in order])])
            name = "Pages_MSCA_ICCrank.tex"
        elif m == "gwetrs":
            order = sorted(F, key=lambda f: est[("gwetrs", f)])         # least agreement first
            L = head("Gwet's AC on rescaled grades",
                     "IF 2014-2018 reviewer totals, rescaled within each panel and call to the mean and SD of all IF totals, "
                     "then graded 0-5 (cuts at 10, 30, 50, 70, 90); power weights (exponent 1/4); AC per call, weighted by "
                     "proposals; SE: bootstrap (1,000).") + \
                body("Gwet's AC --- rescaled 0–5 grade", m, tag, order, [
                    row("Gwet's AC (rescaled 0–5 grade)", [f"{est[('gwetrs', f)]:.3f}" for f in order]),
                    row("SE", [f"({se[('gwetrs', f)]:.3f})" for f in order]),
                    row("$p$", [pf(2 * stats.norm.sf(est[("gwetrs", f)] / se[("gwetrs", f)])) for f in order])])
            name = "Pages_MSCA_GwetRescaled.tex"
        elif m == "gwet1618":
            order = sorted(F, key=lambda f: est[("gwet1618", f)])       # least agreement first
            L = head("Gwet's AC (0-5 grade) from the 2016-2018 calls",
                     "IF 2016-2018 reviewer totals only (the calls that overlap the 2016-2020 GF/EF window); 0-5 grade "
                     "(score/20 rounded); power weights (exponent 1/4); SE: bootstrap (1,000).") + \
                body("Gwet's AC --- 0–5 grade, reviewer scores 2016–18", m, tag, order, [
                    row("Gwet's AC (2016–18)", [f"{est[('gwet1618', f)]:.3f}" for f in order]),
                    row("SE", [f"({se[('gwet1618', f)]:.3f})" for f in order]),
                    row("$p$", [pf(2 * stats.norm.sf(est[("gwet1618", f)] / se[("gwet1618", f)])) for f in order])])
            name = "Pages_MSCA_Gwet1618.tex"
        else:
            continue
        (OUTDIR / name).write_text("\n".join(L))
        print(f"  wrote output/{name}")


TARGETS = {"a": figure_a, "c": figure_c, "page": page}

if __name__ == "__main__":
    args = sys.argv[1:]
    codes = [meas[0] for meas in MEAS]
    unknown = [a for a in args if a not in TARGETS and a not in codes]
    if unknown:
        sys.exit(f"unknown argument(s) {unknown}; targets {list(TARGETS)}, measures {codes}")
    names = [a for a in args if a in TARGETS] or list(TARGETS)
    chosen = [meas for meas in MEAS if meas[0] in args] or MEAS
    OUTDIR.mkdir(exist_ok=True)
    c = counts()
    est, se = measures()
    for m, short, *_ in MEAS:
        order = sorted(F, key=lambda f: est[(m, f)])
        print(f"  {short}: " + "  ".join(f"{f} {est[(m, f)]:.3f} ({se[(m, f)]:.3f})" for f in order))
    for n in names:
        TARGETS[n](c, est, se, chosen)
