#!/usr/bin/env python3
"""Generate the TikZ figures of the theory sections of the paper (4_paper).

Each figure is a named target, named after the file it writes into the
module's output/ folder (2_analysis/2.1_theory/output):

    python3 generate_figures.py                  # only what is new or changed
    python3 generate_figures.py oc_acceptance    # just that one
    python3 generate_figures.py 'misc*'          # shell-style patterns
    python3 generate_figures.py --all            # everything, from scratch
    python3 generate_figures.py --list           # the targets and what they are

Builds are incremental: a plain run rebuilds a target only if it is new, if
the code it depends on (its builder and every function, class and constant of
this file that the builder uses) has changed, or if one of its files is
missing.  The record is kept in output/build_state.json.  A full build takes
about 15 minutes, nearly all of it on the flow trajectories of general_eq,
payline_rise, br_shift and br_shift_arrows.

To add a figure: write its builder, then add one line to TARGETS at the bottom
mapping the file stem to a description and a no-argument thunk; the next plain
run builds that figure alone.  Every number in the output is computed here;
nothing is hand-placed, and the output files are never edited by hand.

Requires numpy and scipy.
"""

from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import sys
import time
import zlib
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy import optimize, stats

OUTDIR = Path(__file__).resolve().parent.parent / "output"

# Gauss-Legendre nodes/weights on [0, 1]; weights sum to 1.
_NODES, _WTS = np.polynomial.legendre.leggauss(200)
_NODES = 0.5 * (_NODES + 1.0)
_WTS = 0.5 * _WTS


# --------------------------------------------------------------------------
# model
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class Model:
    """Primitives of the vertical model.

    The noise distribution F is standard logistic and the merit distribution G
    is Beta(a, b) rescaled onto [theta_lo, theta_hi].  Both satisfy the
    log-concavity and MLRP assumptions the paper requires.
    """

    sigma: float = 0.70          # evaluation noise
    lambda_T: float = 0.35       # top program payline
    lambda_R: float = 0.55       # regular program payline
    v_ratio: float = 0.35        # v^R / v^T
    cost_ratio: float = 0.40     # c / v^R
    beta_a: float = 1.5
    beta_b: float = 1.5
    theta_lo: float = -1.6
    theta_hi: float = 1.6
    noise: str = "logistic"      # "logistic" or "normal"; both log-concave, both MLRP
    # "beta" (above), or "texp": density proportional to exp(-kappa theta) on
    # [theta_lo, theta_hi], log-concave with a concave mills ratio for kappa > 0,
    # or "normal": standard normal, with a convex mills ratio; theta_lo and
    # theta_hi then serve only as brackets for the root finders
    merit: str = "beta"
    kappa: float = 0.0

    # ---- noise ----------------------------------------------------------
    @property
    def _F(self):
        return {"logistic": stats.logistic, "normal": stats.norm}[self.noise]

    def Fbar(self, z):
        return self._F.sf(z)

    def Finv(self, p):
        return self._F.ppf(p)

    @property
    def noise_name(self):
        return f"standard {self.noise}"

    # ---- merit ----------------------------------------------------------
    @property
    def _beta(self):
        return stats.beta(self.beta_a, self.beta_b)

    @property
    def _width(self):
        return self.theta_hi - self.theta_lo

    def Gcdf(self, t):
        if self.merit == "normal":
            return stats.norm.cdf(t)
        if self.merit == "texp":
            u = np.clip(np.asarray(t) - self.theta_lo, 0.0, self._width)
            return np.expm1(-self.kappa * u) / np.expm1(-self.kappa * self._width)
        return self._beta.cdf((t - self.theta_lo) / self._width)

    def Ginv(self, q):
        if self.merit == "normal":
            return stats.norm.ppf(q)
        if self.merit == "texp":
            return self.theta_lo - np.log1p(
                np.asarray(q) * np.expm1(-self.kappa * self._width)) / self.kappa
        return self.theta_lo + self._width * self._beta.ppf(q)

    def mills(self, t):
        """r = (1 - G)/g, the mills ratio of the merit distribution."""
        if self.merit == "normal":
            return stats.norm.sf(t) / stats.norm.pdf(t)
        if self.merit == "texp":
            return -np.expm1(-self.kappa * (self.theta_hi - np.asarray(t))) / self.kappa
        u = (np.asarray(t) - self.theta_lo) / self._width
        return self._width * self._beta.sf(u) / self._beta.pdf(u)

    @property
    def merit_name(self):
        if self.merit == "normal":
            return "$G$ standard normal"
        if self.merit == "texp":
            return (f"$G$ exponential with rate ${self.kappa:g}$ truncated to "
                    f"$[{self.theta_lo:g},{self.theta_hi:g}]$")
        return (f"$G=\\mathrm{{Beta}}({self.beta_a:g},{self.beta_b:g})$ on "
                f"$[{self.theta_lo:g},{self.theta_hi:g}]$")

    # ---- top program ----------------------------------------------------
    def x_supply_top(self, theta_T):
        """Cutoff solving the top payline, eq (supply_top).

        The 1/Gbar(theta_T) prefactor is cancelled exactly by substituting
        u = G(theta), which also removes the vanishing denominator as
        theta_T -> theta_hi.
        """
        q = self.Gcdf(theta_T)
        if q >= 1 - 1e-12:
            return theta_T + self.sigma * self.Finv(1 - self.lambda_T)
        grid = self.Ginv(q + (1 - q) * _NODES)
        span = 40 * self.sigma
        return optimize.brentq(
            lambda x: float(np.dot(_WTS, self.Fbar((x - grid) / self.sigma))) - self.lambda_T,
            self.theta_lo - span, self.theta_hi + span, xtol=1e-13)

    def x_demand_top(self, theta_T, x_R):
        """Top cutoff needed to induce marginal top applicant theta_T."""
        return theta_T + self.sigma * self.Finv(
            1 - self.v_ratio * self.Fbar((x_R - theta_T) / self.sigma))

    def top_equilibria(self, x_R, grid):
        """All partial equilibria of the top program on `grid`, ascending."""
        gap = np.array([self.x_supply_top(t) - self.x_demand_top(t, x_R) for t in grid])
        flips = np.where(np.sign(gap)[1:] != np.sign(gap)[:-1])[0]
        return [optimize.brentq(
                    lambda t: self.x_supply_top(t) - self.x_demand_top(t, x_R),
                    grid[i], grid[i + 1], xtol=1e-12)
                for i in flips]

    def theta_top(self, x_R, grid):
        """Stable partial equilibrium of the top program.

        Stability means the supply curve cuts demand from above, which is the
        largest root; it is the branch that is decreasing in x_R (prop 3.1.6).
        Returns None when the saddle-node has destroyed all equilibria.
        """
        roots = self.top_equilibria(x_R, grid)
        return roots[-1] if roots else None

    # ---- regular program ------------------------------------------------
    def x_supply_reg(self, theta_R, theta_T):
        """Cutoff solving the regular payline, eq (supply_regular)."""
        u_lo, u_hi = self.Gcdf(theta_R), self.Gcdf(theta_T)
        if u_hi - u_lo < 1e-10:
            return theta_T + self.sigma * self.Finv(1 - self.lambda_R)
        grid = self.Ginv(u_lo + (u_hi - u_lo) * _NODES)
        span = 40 * self.sigma
        return optimize.brentq(
            lambda x: float(np.dot(_WTS, self.Fbar((x - grid) / self.sigma))) - self.lambda_R,
            self.theta_lo - span, self.theta_hi + span, xtol=1e-13)

    @property
    def reg_intercept(self):
        """k in xhat^R_D = theta_R + k.  Demand has slope exactly one."""
        return self.sigma * self.Finv(1 - self.cost_ratio)

    def x_demand_reg(self, theta_R):
        return theta_R + self.reg_intercept

    def reg_equilibrium(self, theta_T):
        """(marginal regular applicant, prevailing cutoff), or None.

        Unique whenever it exists: demand has slope one while supply has slope
        h(theta_R)/h(xhat) < 1 for the theta_T-truncated merit law, because a
        log-concave density has an increasing hazard rate.
        """
        lo = self.theta_lo + 1e-4
        hi = theta_T - 1e-6
        if hi <= lo:
            return None
        f = lambda t: self.x_supply_reg(t, theta_T) - self.x_demand_reg(t)
        if f(lo) * f(hi) > 0:
            return None
        root = optimize.brentq(f, lo, hi, xtol=1e-12)
        return root, self.x_demand_reg(root)

    def x_reg(self, theta_T):
        """Regular cutoff at the regular program's partial equilibrium."""
        eq = self.reg_equilibrium(theta_T)
        return None if eq is None else eq[1]

    # ---- general equilibrium --------------------------------------------
    def general_equilibrium(self, x_grid, theta_grid):
        """Fixed point of the two best responses, plus the composite slope.

        theta_top falls and x_reg rises, so the composite map falls and the
        cobweb spirals; |slope| < 1 is exactly stability.
        """
        def excess(x):
            t = self.theta_top(x, theta_grid)
            if t is None:
                return None
            xr = self.x_reg(t)
            return None if xr is None else (xr - x, t)

        pts = [(x, excess(x)) for x in x_grid]
        pts = [(x, e) for x, e in pts if e is not None]
        star = None
        for (xa, ea), (xb, eb) in zip(pts, pts[1:]):
            if ea[0] * eb[0] < 0:
                star = optimize.brentq(lambda x: excess(x)[0], xa, xb, xtol=1e-11)
                break
        if star is None:
            return None
        theta_star = self.theta_top(star, theta_grid)
        theta_R_star = self.reg_equilibrium(theta_star)[0]
        h = 2e-3
        d_top = (self.theta_top(star + h, theta_grid)
                 - self.theta_top(star - h, theta_grid)) / (2 * h)
        d_reg = (self.x_reg(theta_star + h) - self.x_reg(theta_star - h)) / (2 * h)
        return star, theta_star, theta_R_star, d_top * d_reg

    def best_response_flow(self, start, theta_grid, x_lo, x_hi, t_lo, t_hi,
                           kappa_R=1.0, kappa_T=1.0, t_end=30.0, n=240,
                           stop_within=None):
        """Solution path of the Proposition 4 partial-adjustment system

            d(theta^T)/dt = kappa_T [theta^T_top(x^R) - theta^T]
            d(x^R)/dt     = kappa_R [x^R_reg(theta^T)  - x^R]

        The best responses are clamped to their domains so the integrator
        cannot wander outside where they are defined.  The Jacobian at a fixed
        point has trace -(kappa_R+kappa_T) < 0 and determinant
        kappa_R kappa_T (1 - B_T' B_R'), so local asymptotic stability is
        exactly B_T' B_R' < 1.
        """
        from scipy.integrate import solve_ivp
        cx = lambda v: min(max(v, x_lo), x_hi)
        ct = lambda v: min(max(v, t_lo), t_hi)

        def rhs(_, y):
            x, th = y
            return [kappa_R * (self.x_reg(ct(th)) - x),
                    kappa_T * (self.theta_top(cx(x), theta_grid) - th)]

        sol = solve_ivp(rhs, [0.0, t_end], list(start),
                        rtol=1e-8, atol=1e-10, dense_output=True)
        grid = np.linspace(0.0, t_end, n)
        xs, ths = sol.sol(grid)
        if stop_within is not None:
            d = np.hypot(xs - xs[-1], ths - ths[-1])
            keep = np.where(d > stop_within)[0]
            cut = (keep[-1] + 2) if len(keep) else len(xs)
            xs, ths = xs[:cut], ths[:cut]
        # Convergence is exponential, so time-spaced samples bunch up at the
        # end; that makes pgfplots' "smooth" throw Dimension too large and puts
        # every decoration arrow in the same spot.  Resample evenly by arc
        # length instead.
        seg = np.hypot(np.diff(xs), np.diff(ths))
        arc = np.concatenate([[0.0], np.cumsum(seg)])
        if arc[-1] <= 0:
            return xs, ths
        even = np.linspace(0.0, arc[-1], 70)
        return np.interp(even, arc, xs), np.interp(even, arc, ths)

    def cobweb(self, x_start, theta_grid, y_start, min_dx, min_dy, max_steps=9):
        """Tatonnement path: vertical onto the top best response, horizontal
        onto the regular one.  Stops once a step is too short to draw."""
        segs, x, y = [], x_start, y_start
        for _ in range(max_steps):
            t = self.theta_top(x, theta_grid)
            if t is None or abs(t - y) < min_dy:
                break
            segs.append(("v", x, y, t))
            y = t
            nxt = self.x_reg(y)
            if nxt is None or abs(nxt - x) < min_dx:
                break
            segs.append(("h", x, y, nxt))
            x = nxt
        return segs

    # ---- the delta parameterisation of Vertical_Model_4.tex ---------------
    # delta = (theta^T - theta^R)/sigma is the normalised opportunity cost.
    @property
    def z_R(self):
        """z^R = F^{-1}(1 - c/v^R);  note reg_intercept = sigma * z^R."""
        return self.Finv(1 - self.cost_ratio)

    def z_T(self, d):
        return self.Finv(1 - self.v_ratio * self.Fbar(self.z_R - d))

    def top_rate(self, theta_T, d):
        """T(theta^T, sigma, delta): average acceptance rate in the top program."""
        q = self.Gcdf(theta_T)
        if q >= 1 - 1e-12:
            return self.Fbar(self.z_T(d))
        grid = self.Ginv(q + (1 - q) * _NODES)
        return float(np.dot(_WTS, self.Fbar(self.z_T(d) - (grid - theta_T) / self.sigma)))

    def theta_T_of_delta(self, d):
        """Unique root of T = lambda^T; None if delta is outside D^T."""
        lo, hi = self.theta_lo + 1e-6, self.theta_hi - 1e-9
        if (self.top_rate(lo, d) - self.lambda_T) * (self.top_rate(hi, d) - self.lambda_T) > 0:
            return None
        root = optimize.brentq(lambda t: self.top_rate(t, d) - self.lambda_T,
                               lo, hi, xtol=1e-13)
        # top_rate switches to a closed form once G(theta^T) is 1 to machine
        # precision, and that jump can itself carry the sign change brentq
        # brackets.  Anything that is not actually a root is outside D^T.
        if abs(self.top_rate(root, d) - self.lambda_T) > 1e-9:
            return None
        return root

    def R_of_delta(self, d):
        """R(sigma, delta): the regular acceptance rate.  None outside D^R."""
        tT = self.theta_T_of_delta(d)
        if tT is None:
            return None
        tR = tT - self.sigma * d
        if tR <= self.theta_lo:
            return None
        uR, uT = self.Gcdf(tR), self.Gcdf(tT)
        if uT - uR < 1e-12:
            return None
        grid = self.Ginv(uR + (uT - uR) * _NODES)
        return float(np.dot(_WTS, self.Fbar(self.z_R - (grid - tR) / self.sigma)))

    @lru_cache(maxsize=None)
    def R_curve(self, dmax=8.0, n=260):
        """Cached: R_of_delta is a quadrature inside a root-solve, and the
        acceptance, OC and triptych figures all sweep the same grid."""
        ds = np.linspace(0.05, dmax, n)
        pts = [(d, self.R_of_delta(d)) for d in ds]
        pts = [(d, v) for d, v in pts if v is not None]
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])

    def oc_of_delta(self, d):
        """OC = v^R Fbar(z^R - delta), the opportunity cost behind delta (v^T = 1).

        delta = kappa(OC) = z^R - F^{-1}(1 - OC/v^R) inverted; it is strictly
        increasing, so the hump in R(sigma, .) survives the reparameterisation.
        At delta = 0 it returns c, the lower end of the domain OC > c.
        """
        return self.v_ratio * self.Fbar(self.z_R - d)

    @lru_cache(maxsize=None)
    def delta_equilibria(self, dmax=8.0):
        """All delta with R = lambda^R, ascending.  Cached alongside R_curve."""
        xs, ys = self.R_curve(dmax)
        g = ys - self.lambda_R
        flips = np.where(np.sign(g)[1:] != np.sign(g)[:-1])[0]
        return [optimize.brentq(lambda d: self.R_of_delta(d) - self.lambda_R,
                                xs[i], xs[i + 1], xtol=1e-12) for i in flips]

    # ---- caption helper --------------------------------------------------
    def caption_prefix(self, *, top=True, regular=True):
        bits = [f"$F$ {self.noise_name}, $\\sigma={self.sigma:g}$",
                f"$G=\\mathrm{{Beta}}({self.beta_a:g},{self.beta_b:g})$ on "
                f"$[{self.theta_lo:g},{self.theta_hi:g}]$"]
        tail = []
        if top:
            tail += [f"$\\lambda^T={self.lambda_T:g}$", f"$v^R/v^T={self.v_ratio:g}$"]
        if regular:
            tail += [f"$\\lambda^R={self.lambda_R:g}$", f"$c/v^R={self.cost_ratio:g}$"]
        return "; ".join(bits) + "; " + ", ".join(tail) + "."


# --------------------------------------------------------------------------
# tikz helpers
# --------------------------------------------------------------------------
def coords(xs, ys, per_line=5):
    items = [f"({x:.4f},{y:.4f})" for x, y in zip(xs, ys)]
    body = "\n".join(" ".join(items[i:i + per_line]) for i in range(0, len(items), per_line))
    return "\n" + body


def axis_style(name, *, width, height, xmin, xmax, ymin, ymax,
               xlabel, ylabel, extra=""):
    """Axis style with both axis labels sitting *after* the arrowhead.

    Same convention as Figures 1 and 2: the x label hangs off the right of the
    x axis and the y label sits above the top of the y axis, in both cases
    clear of the Stealth tip.
    """
    return f"""\\pgfplotsset{{{name}/.style={{
  width={width}cm, height={height}cm,
  xmin={xmin:.2f}, xmax={xmax:.2f}, ymin={ymin:.3f}, ymax={ymax:.3f},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=6pt]}},black}}, clip=false,
  xlabel={{{xlabel}}}, ylabel={{{ylabel}}},
  every axis x label/.style={{at={{(current axis.south east)}},anchor=west,xshift=9pt}},
  every axis y label/.style={{at={{(current axis.north west)}},anchor=south,yshift=7pt}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\footnotesize}},
  title style={{font=\\footnotesize, yshift=2pt}},
  tick style={{black}},{extra}
}}}}"""


def shift_arrow(colour, x, y_from, y_to):
    """Arrow drawn from a point on the original curve to the shifted one."""
    return (f"  \\draw[{colour},thick,-{{Stealth[length=5pt]}}] "
            f"(axis cs:{x:.4f},{y_from:.4f}) -- (axis cs:{x:.4f},{y_to:.4f});")


def shift_arrow_h(colour, y, x_from, x_to):
    return (f"  \\draw[{colour},thick,-{{Stealth[length=5pt]}}] "
            f"(axis cs:{x_from:.4f},{y:.4f}) -- (axis cs:{x_to:.4f},{y:.4f});")


def widest_gap(xs, lower, upper, lo_frac=0.0, hi_frac=1.0):
    """x where two curves are furthest apart, searched over a slice of the range.

    Used to hang the shift arrow where it is most legible.
    """
    n = len(xs)
    a, b = int(lo_frac * n), max(int(hi_frac * n), int(lo_frac * n) + 1)
    j = a + int(np.argmax(np.abs(np.asarray(upper)[a:b] - np.asarray(lower)[a:b])))
    return j


def subfigure(body, label, frac=0.49):
    return (f"\\begin{{subfigure}}[t]{{{frac}\\textwidth}}\\centering\n"
            f"{body}\n\\phantomsubcaption\n\\label{{{label}}}\n\\end{{subfigure}}")


def figure_env(content, caption, label):
    return (f"\\begin{{figure}}[htbp]\n\\centering\n{content}\n"
            f"\\caption{{{caption}}}\n\\label{{{label}}}\n\\end{{figure}}\n")


def write(name, text):
    OUTDIR.mkdir(exist_ok=True)
    (OUTDIR / name).write_text(text)
    WRITTEN.append(name)
    print(f"  wrote output/{name}")


WRITTEN: list[str] = []      # files written by the target being built (see main)


# --------------------------------------------------------------------------
# Figure 3 - partial equilibrium in the top program (section 3.1.6)
# --------------------------------------------------------------------------
def figure_partial_eq(m: Model, x_R_before: float, x_R_after: float, fname="partial_eq.tex"):
    pad = 0.02 * (m.theta_hi - m.theta_lo)
    lo, hi = m.theta_lo + pad, m.theta_hi - pad
    th = np.linspace(lo, hi, 180)
    S = np.array([m.x_supply_top(t) for t in th])
    D0 = np.array([m.x_demand_top(t, x_R_before) for t in th])
    D1 = np.array([m.x_demand_top(t, x_R_after) for t in th])
    r0 = m.top_equilibria(x_R_before, th)
    r1 = m.top_equilibria(x_R_after, th)
    assert len(r0) == 2 and len(r1) == 2, (r0, r1)

    ylo = min(S.min(), D0.min(), D1.min()) - 0.12
    yhi = max(S.max(), D0.max(), D1.max()) + 0.10
    xmin, xmax = lo - 0.28, hi + 0.46
    depth = ylo - 0.05
    style = axis_style("peaxis", width=8.4, height=7.2,
                       xmin=xmin, xmax=xmax, ymin=ylo, ymax=yhi,
                       xlabel="$\\hat{\\theta}^T$", ylabel="$\\hat{x}^T$",
                       extra="\n  ytick=\\empty, xtick=\\empty,")

    # the curves are named in one legend beneath both panels
    common = []
    # hang the shift arrow on the left half, where the two demand curves are
    # furthest apart and the supply curve is out of the way
    j_shift = widest_gap(th, D0, D1, 0.05, 0.40)

    panel_a = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        "\\begin{axis}[peaxis, title={(a) Two partial equilibria}]",
        *[f"  \\node[anchor=south east,font=\\footnotesize] "
          f"at (axis cs:{r - 0.05:.4f},{m.x_supply_top(r) + 0.03:.4f}) {{({tag})}};"
          for r, tag in zip(r0, ("i", "ii"))],
        *common,
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(th, S)}}};",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(th, D0)}}};",
        "\\end{axis}", "\\end{tikzpicture}"])

    panel_b = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        "\\begin{axis}[peaxis, title={(b) A rise in $\\hat{x}^R$ to $\\hat{x}^{R\\prime}$ shifts $\\hat{x}^T_D$ up}]",
        f"  \\draw[gray,dashed] (axis cs:{r1[1]:.5f},{depth:.3f}) -- "
        f"(axis cs:{r1[1]:.5f},{m.x_supply_top(r1[1]):.5f});",
        f"  \\draw[gray,dashed] (axis cs:{r0[1]:.5f},{depth:.3f}) -- "
        f"(axis cs:{r0[1]:.5f},{m.x_supply_top(r0[1]):.5f});",
        f"  \\node[anchor=north east,font=\\footnotesize] at (axis cs:{r1[1] + 0.03:.5f},"
        f"{depth:.3f}) {{$\\hat{{\\theta}}^T_{{top}}(\\hat{{x}}^{{R\\prime}})$}};",
        f"  \\node[anchor=north west,font=\\footnotesize] at (axis cs:{r0[1] - 0.03:.5f},"
        f"{depth:.3f}) {{$\\hat{{\\theta}}^T_{{top}}(\\hat{{x}}^R)$}};",
        f"  \\draw[-{{Stealth[length=5pt]}},thick] (axis cs:{r0[1]:.4f},{ylo + 0.06:.3f}) -- "
        f"(axis cs:{r1[1]:.4f},{ylo + 0.06:.3f});",
        *common,
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(th, S)}}};",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(th, D0)}}};",
        f"  \\addplot[red,very thick,densely dotted,smooth] coordinates {{{coords(th, D1)}}};",
        shift_arrow("red", th[j_shift], D0[j_shift], D1[j_shift]),
        "\\end{axis}", "\\end{tikzpicture}"])

    caption = (m.caption_prefix(regular=False) +
               f" Panel (a) $\\hat{{x}}^R={x_R_before:g}$; "
               f"panel (b) $\\hat{{x}}^R$ rises to ${x_R_after:g}$.")
    header = (
        "% Figure 3 - partial equilibrium in the top program (section 3.1.6).\n"
        "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
        f"% Panel (a): equilibria {', '.join(f'{v:.4f}' for v in r0)}\n"
        f"% Panel (b): equilibria {', '.join(f'{v:.4f}' for v in r1)}\n")
    sample = "\\tikz[baseline=-0.6ex]\\draw[{}] (0,0) -- (0.7,0);"
    legend = ("{\\footnotesize "
              + sample.format("blue,very thick")
              + "\\ $\\hat{\\theta}^T\\mapsto\\hat{x}^T_S(\\hat{\\theta}^T|\\lambda^T,\\sigma)$\\qquad"
              + sample.format("red,very thick")
              + "\\ $\\hat{\\theta}^T\\mapsto\\hat{x}^T_D(\\hat{\\theta}^T|\\hat{x}^R,\\sigma)$\\qquad"
              + sample.format("red,very thick,densely dotted")
              + "\\ $\\hat{\\theta}^T\\mapsto\\hat{x}^T_D(\\hat{\\theta}^T|\\hat{x}^{R\\prime},\\sigma)$\\par}")
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(panel_a, "fig:pe_a"), "\\hfill", subfigure(panel_b, "fig:pe_b"),
        "\\par\\medskip", legend,
        "\\caption{}", f"% \\caption{{\\footnotesize {caption}}}",
        "\\label{fig:pe}", "\\end{figure}", ""])
    write(fname, header + style + "\n\n" + env)
    print(f"    (a) equilibria {r0[0]:.4f}, {r0[1]:.4f}   "
          f"(b) equilibria {r1[0]:.4f}, {r1[1]:.4f}")


# --------------------------------------------------------------------------
# Figure 4 - partial equilibrium in the regular program (section 3.1.7)
# --------------------------------------------------------------------------
def figure_regular_eq(m: Model, theta_T_before: float, theta_T_after: float,
                      fname="regular_eq.tex"):
    k = m.reg_intercept
    r0, c0 = m.reg_equilibrium(theta_T_before)
    r1, c1 = m.reg_equilibrium(theta_T_after)
    pad = 0.02 * (m.theta_hi - m.theta_lo)
    lo = m.theta_lo + pad
    t0 = np.linspace(lo, theta_T_before - 0.004, 150)
    t1 = np.linspace(lo, theta_T_after - 0.004, 150)
    S0 = np.array([m.x_supply_reg(t, theta_T_before) for t in t0])
    S1 = np.array([m.x_supply_reg(t, theta_T_after) for t in t1])
    win_hi = theta_T_after + 0.05
    edge = np.array([lo, win_hi])
    RD = edge + k

    ylo = min(S0.min(), S1.min(), RD.min()) - 0.12
    yhi = max(S0.max(), S1.max(), RD.max()) + 0.10
    xmin, xmax = lo - 0.10, win_hi + 0.34
    depth = ylo - 0.05
    ytk_a = f"\n  ytick={{{c0:.4f}}}, yticklabels={{$\\hat{{x}}^R_{{reg}}(\\hat{{\\theta}}^{{T}})$}},"
    ytk_b = (f"\n  ytick={{{c0:.4f},{c1:.4f}}}, yticklabels={{"
             f"$\\hat{{x}}^R_{{reg}}(\\hat{{\\theta}}^{{T}})$,"
             f"$\\hat{{x}}^R_{{reg}}(\\hat{{\\theta}}^{{T\\prime}})$}},")
    # width 7.4cm: the y tick labels are wide, the x label hangs off the right,
    # and each subfigure only has 0.49\textwidth = 237pt to play with.
    common_axis = dict(width=7.4, height=7.2, xmin=xmin, xmax=xmax, ymin=ylo, ymax=yhi,
                       xlabel="$\\hat{\\theta}^R$", ylabel="$\\hat{x}^R$")
    style = (axis_style("regaxisA", **common_axis, extra="\n  xtick=\\empty," + ytk_a)
             + "\n" +
             axis_style("regaxisB", **common_axis, extra="\n  xtick=\\empty," + ytk_b))

    def theta_T_mark(t, series_x, series_y):
        top = float(np.interp(t - 0.004, series_x, series_y))
        return (f"  \\draw[gray,dashed] (axis cs:{t:.5f},{depth:.3f}) -- "
                f"(axis cs:{t:.5f},{top:.5f});")

    # the curves are named in one legend beneath both panels; the shift arrow
    # hangs over the open space above the dotted curve on the left
    lab_dotted = lo + 0.18 * (theta_T_after - lo)

    panel_a = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        "\\begin{axis}[regaxisA, title={(a) the stable partial equilibrium}]",
        f"  \\draw[gray,dashed] (axis cs:{xmin:.2f},{c0:.5f}) -- (axis cs:{r0:.5f},{c0:.5f});",
        f"  \\draw[gray,dashed] (axis cs:{r0:.5f},{depth:.3f}) -- (axis cs:{r0:.5f},{c0:.5f});",
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{r0:.5f},{depth:.3f}) "
        f"{{$\\hat{{\\theta}}^{{R,*}}$}};",
        theta_T_mark(theta_T_before, t0, S0),
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{theta_T_before:.5f},"
        f"{depth:.3f}) {{$\\hat{{\\theta}}^{{T}}$}};",
        f"  \\addplot[red,very thick] coordinates {{{coords(edge, RD)}}};",
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(t0, S0)}}};",
        "\\end{axis}", "\\end{tikzpicture}"])

    panel_b = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        "\\begin{axis}[regaxisB, title={(b) a rise in $\\hat{\\theta}^T$ shifts "
        "$\\hat{x}^R_S$ up}]",
        f"  \\draw[gray,dashed] (axis cs:{xmin:.2f},{c0:.5f}) -- (axis cs:{r0:.5f},{c0:.5f});",
        f"  \\draw[gray,dashed] (axis cs:{xmin:.2f},{c1:.5f}) -- (axis cs:{r1:.5f},{c1:.5f});",
        f"  \\draw[-{{Stealth[length=5pt]}},thick] (axis cs:{xmin + 0.13:.2f},{c0:.4f}) -- "
        f"(axis cs:{xmin + 0.13:.2f},{c1:.4f});",
        theta_T_mark(theta_T_before, t0, S0),
        theta_T_mark(theta_T_after, t1, S1),
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{theta_T_before:.5f},"
        f"{depth:.3f}) {{$\\hat{{\\theta}}^{{T}}$}};",
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{theta_T_after:.5f},"
        f"{depth:.3f}) {{$\\hat{{\\theta}}^{{T\\prime}}$}};",
        f"  \\addplot[red,very thick] coordinates {{{coords(edge, RD)}}};",
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(t0, S0)}}};",
        f"  \\addplot[blue,very thick,densely dotted,smooth] coordinates {{{coords(t1, S1)}}};",
        shift_arrow("blue", lab_dotted,
                    m.x_supply_reg(lab_dotted, theta_T_before),
                    m.x_supply_reg(lab_dotted, theta_T_after)),
        "\\end{axis}", "\\end{tikzpicture}"])

    caption = (m.caption_prefix(top=False) +
               f" Panel (a) $\\hat{{\\theta}}^T={theta_T_before:g}$; "
               f"panel (b) $\\hat{{\\theta}}^T$ rises to ${theta_T_after:g}$.")
    header = (
        "% Figure 4 - partial equilibrium in the regular program (section 3.1.7).\n"
        "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
        "% Demand xhat^R_D = theta_R + sigma F^{-1}(1 - c/v^R) has slope EXACTLY one,\n"
        "% while supply has slope h(theta_R)/h(xhat) < 1 for the theta_T-truncated merit\n"
        "% law (log-concave g => increasing hazard).  Hence the crossing is unique and\n"
        "% necessarily stable.  Multiplicity would require a non-log-concave g.\n"
        f"% Panel (a): theta^T={theta_T_before:g}, theta^R*={r0:.4f}, cutoff={c0:.4f}\n"
        f"% Panel (b): theta^T={theta_T_after:g}, theta^R*={r1:.4f}, cutoff={c1:.4f}\n")
    sample = "\\tikz[baseline=-0.6ex]\\draw[{}] (0,0) -- (0.7,0);"
    legend = ("{\\footnotesize "
              + sample.format("blue,very thick")
              + "\\ $\\hat{\\theta}^R\\mapsto\\hat{x}^R_S(\\hat{\\theta}^R|\\lambda^R,\\hat{\\theta}^T,\\sigma)$\\qquad"
              + sample.format("blue,very thick,densely dotted")
              + "\\ $\\hat{\\theta}^R\\mapsto\\hat{x}^R_S(\\hat{\\theta}^R|\\lambda^R,\\hat{\\theta}^{T\\prime},\\sigma)$\\qquad"
              + sample.format("red,very thick")
              + "\\ $\\hat{\\theta}^R\\mapsto\\hat{x}^R_D(\\hat{\\theta}^R|\\sigma)$\\par}")
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(panel_a, "fig:reg_a"), "\\hfill", subfigure(panel_b, "fig:reg_b"),
        "\\par\\medskip", legend,
        "\\caption{}", f"% \\caption{{\\footnotesize {caption}}}",
        "\\label{fig:reg}", "\\end{figure}", ""])
    write(fname, header + style + "\n\n" + env)
    print(f"    (a) theta^R*={r0:.4f} cutoff={c0:.4f}   (b) theta^R*={r1:.4f} cutoff={c1:.4f}")


# --------------------------------------------------------------------------
# Figure 5 - general fixed payline equilibrium (section 3.1.8)
# --------------------------------------------------------------------------
def figure_general_eq(m: Model, x_start: float, fname="general_eq.tex", *,
                      x_range=(-0.85, 0.17), t_range=(0.60, 1.55), n_blue=90,
                      pads=(0.10, 0.22, 0.13, 0.09), stars=("$\\hat{x}^{R,*}$", "$\\hat{\\theta}^{T,*}$"),
                      label="fig:geneq", title="Figure 5 - general fixed payline "
                      "equilibrium (section 3.1.8).", cobweb=True, note=None, flow=True,
                      eq_dot=None, iso_oc=(), window=None, size=(11.2, 7.6),
                      sig="\\sigma", second=None, colors=("blue", "red"), legend=False,
                      split=(0.60, 0.38), arrows=(), fade=None):
    """The two best responses, their intersection and the Proposition 4 flow.

    The keywords only move the window and the labels, so the same figure can
    be drawn at another equilibrium.  cobweb=True also demands |B_T' B_R'| < 1;
    Proposition 4 itself needs only B_T' B_R' < 1, which cobweb=False checks.
    note, when given, replaces the typeset caption: the figure then carries a
    number only and the caption is kept as a comment.  flow=False draws the two
    best responses and their intersection only, without U and its paths.
    eq_dot, when given, is the colour of a dot marking the intersection.
    window, when given, fixes (xmin, xmax, ymin, ymax) instead of padding the
    curves; size is the axis (width, height) in cm.  iso_oc lists
    (sigma, label, frac, anchor, x_cap): for each, the line of regular cutoffs and
    marginal top applicants at which the marginal top applicant faces the
    equilibrium opportunity cost when noise is sigma.  OC = v^R Fbar((x^R -
    theta^T)/sigma) is constant along x^R - theta^T = sigma F^{-1}(1 - OC/v^R),
    a line of slope one; a change in sigma shifts it and does not rotate it.
    The label sits at fraction frac along the visible segment, at anchor; the
    segment stops at x_cap and at the top of the merit support.  sig is the
    noise symbol in the curve labels.  second, when given, adds a dashed pair of
    best responses for another model: a dict with keys model, x_range, t_range,
    dot, sig, lam (the payline symbol), top_lab / reg_lab = (anchor, end) and
    optionally colors and third, a further regular best response (keys model,
    color, dot, lam) meeting the same top best response.  colors are the first
    pair's colours.  legend=True names the curves in a legend under the figure
    instead of beside them; split gives the widths (fractions of the text
    width) of the plot and of a legend on the right.  arrows lists pairs of
    equilibria ("first", "second", "third") joined by an arrow, in order.
    fade, when given, is the opacity of the curves and OC lines, so that the
    arrows and dots stand out.
    """
    op = f",opacity={fade}" if fade else ""
    pad = 0.02 * (m.theta_hi - m.theta_lo)
    theta_grid = np.linspace(m.theta_lo + pad, m.theta_hi - pad, 180)

    xs = np.linspace(*x_range, n_blue)
    blue = [(x, m.theta_top(x, theta_grid)) for x in xs]
    blue = [(x, t) for x, t in blue if t is not None]
    bx = np.array([p[0] for p in blue])
    by = np.array([p[1] for p in blue])

    red = [(m.x_reg(t), t) for t in np.linspace(*t_range, 90)]
    red = [(x, t) for x, t in red if x is not None]
    rx = np.array([p[0] for p in red])
    ry = np.array([p[1] for p in red])

    assert np.all(np.diff(by) < 0), "top best response must be decreasing (3.1.6)"
    assert np.all(np.diff(rx) > 0), "regular best response must be increasing (3.1.7)"

    star = m.general_equilibrium(bx, theta_grid)
    assert star is not None, "no general equilibrium in range"
    x_star, t_star, r_star, composite = star
    assert (abs(composite) < 1 if cobweb else composite < 1), \
        f"equilibrium is unstable: composite slope {composite:.3f}"

    extra_curves, extra_marks = [], []
    if second is not None:
        m2 = second["model"]

        def stable_branch(xr):
            """The stable top branch through the second equilibrium: the largest
            root, kept only over the unbroken decreasing run."""
            pts = [(x, m2.theta_top(x, theta_grid)) for x in np.linspace(*xr, n_blue)]
            pts = [q for q in pts if q[1] is not None]
            runs, cur = [], [pts[0]]
            for q0, q1 in zip(pts, pts[1:]):
                if q1[1] < q0[1] and q0[1] - q1[1] < 0.15:
                    cur.append(q1)
                else:
                    runs.append(cur); cur = [q1]
            runs.append(cur)
            run = max(runs, key=len)
            return np.array([q[0] for q in run]), np.array([q[1] for q in run])
        bx2, by2 = stable_branch(second["x_range"])
        red2 = [(m2.x_reg(t), t) for t in np.linspace(*second["t_range"], 90)]
        red2 = [q for q in red2 if q[0] is not None]
        rx2, ry2 = np.array([q[0] for q in red2]), np.array([q[1] for q in red2])
        assert np.all(np.diff(rx2) > 0), "second regular best response must be increasing"
        star2 = m2.general_equilibrium(bx2, theta_grid)
        assert star2 is not None, "the second pair of best responses does not meet"
        x_star2, t_star2 = star2[0], star2[1]
        third = second.get("third")
        if third is not None:
            m3 = third["model"]
            red3 = [(m3.x_reg(t), t) for t in np.linspace(*second["t_range"], 90)]
            red3 = [q for q in red3 if q[0] is not None]
            rx3, ry3 = np.array([q[0] for q in red3]), np.array([q[1] for q in red3])
            assert np.all(np.diff(rx3) > 0), "third regular best response must be increasing"
            # the top best response does not involve the regular payline
            star3 = m3.general_equilibrium(bx2, theta_grid)
            assert star3 is not None, "the third regular best response misses the top one"
            x_star3, t_star3 = star3[0], star3[1]
    more = ((bx2.min(), rx2.min(), *((rx3.min(),) if second.get("third") else ()))
            if second else ())
    xmin = min(bx.min(), rx.min(), *more) - pads[0]
    xmax = max(bx.max(), rx.max(), *((bx2.max(), rx2.max(),
                                       *((rx3.max(),) if second.get("third") else ()))
                                      if second else ())) + pads[1]
    ymin = min(by.min(), ry.min(), *((by2.min(), ry2.min()) if second else ())) - pads[2]
    ymax = max(by.max(), ry.max(), *((by2.max(), ry2.max()) if second else ())) + pads[3]
    if window is not None:
        xmin, xmax, ymin, ymax = window
    # Proposition 4 is a LOCAL statement: every initial point is taken inside an
    # open neighbourhood U of the equilibrium, and the solution both stays close
    # and converges.  Size U so it renders as a circle on the page, then shrink
    # it if needed to stay inside both best-response domains.
    R_CM = 1.95
    ux = R_CM * (xmax - xmin) / 11.2
    uy = R_CM * (ymax - ymin) / 7.6
    shrink = min(1.0,
                 0.92 * min(x_star - bx.min(), bx.max() - x_star) / ux,
                 0.92 * min(t_star - ry.min(), ry.max() - t_star) / uy)
    ux, uy = ux * shrink, uy * shrink
    ang = np.linspace(0, 2 * np.pi, 181)
    u_boundary = (x_star + ux * np.cos(ang), t_star + uy * np.sin(ang))

    # Initial points inside U.  Fully random angles clump and read as a tangle,
    # while a fixed radius reads as a literal circle; so jitter the angles about
    # an even spread and vary the radii.  Seeded, so the figure is reproducible.
    # Radii stay off the boundary because the spiral overshoots before decaying.
    rng = np.random.default_rng(5)
    n_paths = 5
    base = np.linspace(0, 2 * np.pi, n_paths, endpoint=False)
    jitter = rng.uniform(-0.30, 0.30, n_paths)
    radii = rng.uniform(0.62, 0.86, n_paths)
    starts = [(x_star + r * ux * np.cos(a + j), t_star + r * uy * np.sin(a + j))
              for a, j, r in zip(base, jitter, radii)]
    paths = []
    escaped = 0.0
    for st in (starts if flow else []):
        px, pt = m.best_response_flow(
            st, theta_grid, bx.min(), bx.max(), ry.min(), ry.max(),
            stop_within=0.010 * max(ux, uy))
        escaped = max(escaped,
                      np.hypot((px - x_star) / ux, (pt - t_star) / uy).max())
        paths.append((px, pt))
    assert escaped <= 1.0, f"a trajectory left U (max normalised radius {escaped:.3f})"
    # NB: "smooth" together with a "mark=between positions" decoration makes
    # pgfplots raise Dimension too large.  The path is resampled evenly by arc
    # length, so a plain polyline already reads as a smooth curve.
    web = "\n".join(
        "  \\addplot[darkgreen,thick,postaction={decorate,decoration={markings,"
        "mark=between positions 0.12 and 0.92 step 0.3 with "
        "{\\arrow{Stealth[length=5pt]}}}}] coordinates {%s};" % coords(px, pt)
        for px, pt in paths)

    style = axis_style(
        "geaxis", width=size[0], height=size[1], xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax,
        xlabel="$\\hat{x}^R$", ylabel="$\\hat{\\theta}^T$",
        extra=(f"\n  xtick={{{x_star:.4f}}}, "
               f"xticklabels={{{stars[0]}}},"
               f"\n  ytick={{{t_star:.4f}}}, "
               f"yticklabels={{{stars[1]}}},"))

    def inside(cx, cy):
        """With a window, the part of a curve inside the frame: the figure is
        a close-up, and the curves run off its edges."""
        if window is None:
            return coords(cx, cy)
        k = (cx >= xmin) & (cx <= xmax) & (cy >= ymin) & (cy <= ymax)
        return coords(np.asarray(cx)[k], np.asarray(cy)[k])

    top1_lab = f"$\\hat{{x}}^R\\mapsto\\hat{{\\theta}}^T_{{top}}(\\hat{{x}}^R|{sig})$"
    reg1_lab = f"$\\hat{{\\theta}}^T\\mapsto\\hat{{x}}^R_{{reg}}(\\hat{{\\theta}}^T|\\lambda^R,{sig})$"
    if second is not None:
        dash = "dash pattern=on 6pt off 3pt"
        s2sym, lam2 = second["sig"], second["lam"]
        ta, te = second["top_lab"]
        ra, re_ = second["reg_lab"]
        tpt = (bx2[-1], by2[-1]) if te == "last" else (bx2[0], by2[0])
        rpt = (rx2[-1], ry2[-1]) if re_ == "last" else (rx2[0], ry2[0])
        c_top, c_reg = second.get("colors", ("blue", "red"))
        sty = "" if "colors" in second else f",{dash}"
        top2_lab = f"$\\hat{{x}}^R\\mapsto\\hat{{\\theta}}^T_{{top}}(\\hat{{x}}^R|{s2sym})$"
        reg2_lab = (f"$\\hat{{\\theta}}^T\\mapsto\\hat{{x}}^R_{{reg}}"
                    f"(\\hat{{\\theta}}^T|{lam2},{s2sym})$")
        extra_curves = [
            "  % the best responses at the higher noise, whose payline is the",
            "  % regular acceptance rate at the original opportunity cost",
            f"  \\addplot[{c_top},very thick{sty}{op},smooth] coordinates {{{inside(bx2, by2)}}};",
            f"  \\addplot[{c_reg},very thick{sty}{op},smooth] coordinates {{{inside(rx2, ry2)}}};"]
        if not legend:
            extra_curves += [
                f"  \\node[{c_top},anchor={ta},font=\\footnotesize,inner sep=3pt] "
                f"at (axis cs:{tpt[0]:.4f},{tpt[1]:.4f}) {{{top2_lab}}};",
                f"  \\node[{c_reg},anchor={ra},font=\\footnotesize,inner sep=3pt] "
                f"at (axis cs:{rpt[0]:.4f},{rpt[1]:.4f}) {{{reg2_lab}}};"]
        extra_marks = [f"  \\filldraw[fill={second['dot']},draw=white,line width=0.8pt] "
                       f"(axis cs:{x_star2:.5f},{t_star2:.5f}) circle (3.2pt);"]

        def guide(spec, px, py):
            """Dashed guides from a dot to both axes, with its coordinates as
            axis labels; xa anchors the horizontal label (north, north east or
            north west) so neighbouring labels can hang away from each other."""
            if spec is None:
                return []
            xl, yl, xa = spec[:3]
            # drop > 0 carries the dashed guide that many pt below the axis and
            # hangs the label from its end, a row below its neighbour's
            drop = spec[3] if len(spec) > 3 else 0
            return [f"  \\draw[gray,dashed] (axis cs:{px:.5f},{ymin:.4f}) -- "
                    f"(axis cs:{px:.5f},{py:.5f}) -- (axis cs:{xmin:.4f},{py:.5f});",
                    f"  \\draw[black] (axis cs:{px:.5f},{ymin:.4f}) -- ++(0pt,-3pt);",
                    *([f"  \\draw[gray,dashed] (axis cs:{px:.5f},{ymin:.4f}) -- ++(0pt,-{drop}pt);"]
                      if drop else []),
                    f"  \\draw[black] (axis cs:{xmin:.4f},{py:.5f}) -- ++(-3pt,0pt);",
                    f"  \\node[anchor={xa},font=\\footnotesize,inner sep=2pt,"
                    f"yshift=-{max(3, drop)}pt] at (axis cs:{px:.5f},{ymin:.4f}) {{{xl}}};",
                    f"  \\node[anchor=east,font=\\footnotesize,inner sep=2pt,xshift=-3pt] "
                    f"at (axis cs:{xmin:.4f},{py:.5f}) {{{yl}}};"]
        extra_marks = guide(second.get("guide"), x_star2, t_star2) + extra_marks
        legend_rows = []
        if second.get("third") is not None:
            third = second["third"]
            reg3_lab = (f"$\\hat{{\\theta}}^T\\mapsto\\hat{{x}}^R_{{reg}}"
                        f"(\\hat{{\\theta}}^T|{third['lam']},{s2sym})$")
            dash3 = third.get("dash")
            extra_curves += [
                "  % back at the original regular payline, still at the higher noise",
                f"  \\addplot[{third['color']},very thick{',' + dash3 if dash3 else ''}{op},smooth] "
                f"coordinates {{{inside(rx3, ry3)}}};"]
            if not legend and "lab_at" in third:
                lead = (float(np.interp(third["lead_t"], ry3, rx3)), third["lead_t"])
                extra_curves += [
                    f"  \\node[{third['color']},anchor={third['lab_anchor']},font=\\footnotesize,"
                    f"inner sep=2pt] (reg3) at (axis cs:{third['lab_at'][0]:.4f},"
                    f"{third['lab_at'][1]:.4f}) {{{reg3_lab}}};",
                    f"  \\draw[{third['color']},thin] (reg3) -- "
                    f"(axis cs:{lead[0]:.4f},{lead[1]:.4f});"]
            extra_marks.append(f"  \\filldraw[fill={third['dot']},draw=white,line width=0.8pt] "
                               f"(axis cs:{x_star3:.5f},{t_star3:.5f}) circle (3.2pt);")
            extra_marks = guide(third.get("guide"), x_star3, t_star3) + extra_marks
    # arrows between the equilibria, drawn under the dots and stopping short
    # of them so the dots stay whole
    pts = {"first": (x_star, t_star)}
    if second is not None:
        pts["second"] = (x_star2, t_star2)
        if second.get("third") is not None:
            pts["third"] = (x_star3, t_star3)
    arrow_lines = [
        f"  \\draw[black,thick,-{{Stealth[length=6pt]}},shorten <=5pt,shorten >=5pt] "
        f"(axis cs:{pts[a][0]:.5f},{pts[a][1]:.5f}) -- (axis cs:{pts[b][0]:.5f},{pts[b][1]:.5f});"
        for a, b in arrows]
    # opportunity cost indifference lines through the equilibrium's OC
    iso_lines = []
    z_eq = (x_star - t_star) / m.sigma        # F^{-1}(1 - OC*/v^R)
    oc_eq = m.v_ratio * m.Fbar(z_eq)
    for s_iso, lab, frac, anchor, x_cap in iso_oc:
        # theta^T = x^R - s z_eq, clipped to the window and the merit support
        xa = max(xmin, ymin + s_iso * z_eq)
        xb = min(xmax, x_cap, min(ymax, m.theta_hi) + s_iso * z_eq)
        assert xa < xb, f"the OC line at sigma={s_iso} misses the window"
        assert abs(m.v_ratio * m.Fbar((xa - (xa - s_iso * z_eq)) / s_iso) - oc_eq) < 1e-12
        iso_lines += [
            f"  \\addplot[darkgreen!45,very thick{op},samples=2,domain={xa:.4f}:{xb:.4f}] "
            f"{{x {'-' if s_iso * z_eq >= 0 else '+'} {abs(s_iso * z_eq):.5f}}};",
            f"  \\node[darkgreen,anchor={anchor},font=\\footnotesize,inner sep=3pt] at (axis cs:"
            f"{xa + frac * (xb - xa):.4f},{xa + frac * (xb - xa) - s_iso * z_eq:.4f}) {{{lab}}};"]
    body = "\n".join([
        "\\begin{tikzpicture}", "\\begin{axis}[geaxis]",
        f"  \\draw[gray,dashed] (axis cs:{x_star:.5f},{ymin:.3f}) -- "
        f"(axis cs:{x_star:.5f},{t_star:.5f});",
        f"  \\draw[gray,dashed] (axis cs:{xmin:.2f},{t_star:.5f}) -- "
        f"(axis cs:{x_star:.5f},{t_star:.5f});",
        *([f"  \\addplot[gray,dashed,thick] coordinates "
           f"{{{coords(u_boundary[0], u_boundary[1])}}};",
           # U sits on top of its circle: the two best responses cross it on
           # all four diagonals, and the red label hangs to the upper right
           f"  \\node[gray,anchor=south,font=\\footnotesize] at (axis cs:"
           f"{x_star:.4f},{t_star + uy:.4f}) {{$U$}};",
           web] if flow else []),
        *iso_lines,
        f"  \\addplot[{colors[0]},very thick{op},smooth] coordinates {{{inside(bx, by)}}};",
        f"  \\addplot[{colors[1]},very thick{op},smooth] coordinates {{{inside(rx, ry)}}};",
        *extra_curves,
        *arrow_lines,
        *([f"  \\filldraw[fill={eq_dot},draw=white,line width=0.8pt] "
           f"(axis cs:{x_star:.5f},{t_star:.5f}) circle (3.2pt);"] if eq_dot else []),
        *extra_marks,
        # each label hangs off the end of its own curve: the top best response
        # below its lower end, the regular best response right of its upper end
        *([f"  \\node[{colors[0]},anchor=north west,font=\\footnotesize,xshift=2pt,yshift=-2pt] "
           f"at (axis cs:{bx[-1]:.4f},{by[-1]:.4f}) {{{top1_lab}}};",
           f"  \\node[{colors[1]},anchor=west,font=\\footnotesize,xshift=4pt] "
           f"at (axis cs:{rx[-1]:.4f},{ry[-1]:.4f}) {{{reg1_lab}}};"] if not legend else []),
        "\\end{axis}", "\\end{tikzpicture}"])
    if legend:
        # the curves are named in a legend - under the figure, or to its right
        # (legend="right") - instead of beside them; each entry shows a short
        # sample of its line in its own colour and dash
        def swatch(c, st=""):
            return (f"\\tikz[baseline=-0.6ex]\\draw[{c},very thick{',' + st if st else ''}] "
                    f"(0,0) -- (0.8,0);")
        dash2 = "" if (second is None or "colors" in second) else "dash pattern=on 6pt off 3pt"
        entries = [(colors[0], "", top1_lab)]
        if second is not None:
            entries.append((c_top, dash2, top2_lab))
        entries.append((colors[1], "", reg1_lab))
        if second is not None:
            entries.append((c_reg, dash2, reg2_lab))
            if second.get("third") is not None:
                entries.append((second["third"]["color"], second["third"].get("dash", ""),
                                reg3_lab))
        if legend == "inside":
            # pgfplots' own legend, placed in an empty corner of the frame: the
            # legend images come first, so the entries attach to them only
            images = [f"  \\addlegendimage{{{col},very thick{',' + st if st else ''}{op}}}"
                      for col, st, _ in entries]
            names = ",".join(f"{{{lab}}}" for _, _, lab in entries)
            body = body.replace(
                "\\begin{axis}[geaxis]",
                "\\begin{axis}[geaxis, legend cell align=left,\n"
                f"  legend entries={{{names}}},\n"
                "  legend style={at={(0.99,0.02)}, anchor=south east, font=\\footnotesize,\n"
                "    draw=gray!40, fill=white, row sep=1pt}]\n" + "\n".join(images), 1)
        elif legend == "right":
            rows = [f"{swatch(c, st)}\\ {lab} \\\\[2pt]" for c, st, lab in entries]
            # plot and legend sit side by side at their natural widths, centred
            # vertically on each other, with a small fixed gap between them
            body = body.replace("\\begin{tikzpicture}",
                                "\\begin{tikzpicture}[baseline=(current bounding box.center)]", 1)
            body = "\n".join([
                body + "%", "\\hspace{0.6em}%",
                "{\\footnotesize\\begin{tabular}[c]{@{}l@{}}", *rows, "\\end{tabular}}"])
        else:
            left = [e for e in entries if "top" in e[2]]
            right = [e for e in entries if "reg" in e[2]]
            rows = []
            for k in range(max(len(left), len(right))):
                lc = f"{swatch(left[k][0], left[k][1])}\\ {left[k][2]}" if k < len(left) else ""
                rc = f"{swatch(right[k][0], right[k][1])}\\ {right[k][2]}" if k < len(right) else ""
                rows.append(f"{lc} & {rc} \\\\")
            body += "\n".join(["", "\\par\\medskip",
                                "{\\footnotesize\\begin{tabular}{ll}", *rows,
                                "\\end{tabular}\\par}"])

    caption = m.caption_prefix() + (
        " Green: solution paths of the partial-adjustment system of "
        "Proposition~4. Each starts inside the neighbourhood $U$ of the "
        "equilibrium, remains in $U$, and converges to the intersection."
        if flow else "")
    header = (
        f"% {title}\n"
        "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
        "% Blue : xhat^R     |-> thetahat^T_top(xhat^R|sigma)   from 3.1.6, DECREASING.\n"
        "% Red  : thetahat^T |-> xhat^R_reg(thetahat^T|sigma)   from 3.1.7, INCREASING.\n"
        "% One falls and one rises, so they meet exactly once.\n"
        "% Green: trajectories of the Proposition 4 partial-adjustment ODE.\n"
        "% Its Jacobian has trace -(kappa_R+kappa_T) < 0 and determinant\n"
        "% kappa_R kappa_T (1 - B_T' B_R'), so local asymptotic stability is\n"
        f"% exactly B_T' B_R' < 1; here B_T' B_R' = {composite:.4f}, so the det\n"
        f"% factor is {1 - composite:.4f} > 0.\n"
        f"% Equilibrium: xhat^R*={x_star:.4f}, thetahat^T*={t_star:.4f} "
        f"(implied thetahat^R*={r_star:.4f}).\n")
    if note is None:            # the caption is kept, but not typeset
        env = "\n".join([
            "\\begin{figure}[htbp]", "\\centering", body,
            "\\caption{}", f"% \\caption{{\\footnotesize {caption}}}",
            f"\\label{{{label}}}", "\\end{figure}", ""])
    else:
        env = "\n".join([
            "\\begin{figure}[htbp]", "\\centering", body, "\\caption{}",
            "% Caption kept here, deliberately not typeset - the figure carries a",
            "% number only.  To restore it, move the line below into \\caption{}.",
            f"% {note} {caption}", f"\\label{{{label}}}", "\\end{figure}", ""])
    write(fname, header + style + "\n\n" + env)
    print(f"    equilibrium x^R*={x_star:.4f}, theta^T*={t_star:.4f}, "
          f"theta^R*={r_star:.4f}; B_T'B_R'={composite:.4f}, "
          f"det factor {1 - composite:.4f} > 0 ({len(paths)} trajectories)")


# --------------------------------------------------------------------------
# Figure 6 - a rise in the regular payline (section 3.1.9)
# --------------------------------------------------------------------------
def figure_payline_rise(m_lo: Model, lambda_R_hi: float, fname="payline_rise.tex"):
    """Panel (a): a higher lambda^R pushes xhat^R_S down, so the regular
    program's partial equilibrium cutoff falls.  Panel (b): that drags the
    regular best response left, and the cobweb converges to a new equilibrium
    with a lower xhat^R and a higher thetahat^T -- a larger opportunity cost
    for the marginal top applicant."""
    m_hi = replace(m_lo, lambda_R=lambda_R_hi)
    pad = 0.02 * (m_lo.theta_hi - m_lo.theta_lo)
    theta_grid = np.linspace(m_lo.theta_lo + pad, m_lo.theta_hi - pad, 180)

    x_old, t_old, _, _ = m_lo.general_equilibrium(np.linspace(-1.6, 0.3, 90), theta_grid)
    x_new, t_new, _, comp_new = m_hi.general_equilibrium(np.linspace(-1.6, 0.3, 90), theta_grid)
    assert x_new < x_old and t_new > t_old, "3.1.9 predicts x^R falls and theta^T rises"
    assert abs(comp_new) < 1, f"new equilibrium unstable: {comp_new:.3f}"

    # ---------------- panel (a): theta^T held at the OLD equilibrium ----------
    k = m_lo.reg_intercept
    lo = m_lo.theta_lo + pad
    th = np.linspace(lo, t_old - 0.004, 150)
    S_lo = np.array([m_lo.x_supply_reg(t, t_old) for t in th])
    S_hi = np.array([m_hi.x_supply_reg(t, t_old) for t in th])
    assert np.all(S_hi < S_lo), "a higher payline must push the supply curve down"
    r_lo, c_lo = m_lo.reg_equilibrium(t_old)
    r_hi, c_hi = m_hi.reg_equilibrium(t_old)
    win_hi = t_old + 0.05
    edge = np.array([lo, win_hi])
    RD = edge + k

    a_ylo = min(S_lo.min(), S_hi.min(), RD.min()) - 0.12
    a_yhi = max(S_lo.max(), S_hi.max(), RD.max()) + 0.10
    a_xmin, a_xmax = lo - 0.10, win_hi + 0.34
    depth = a_ylo - 0.05
    style_a = axis_style(
        "payaxisA", width=7.6, height=7.2, xmin=a_xmin, xmax=a_xmax,
        ymin=a_ylo, ymax=a_yhi, xlabel="$\\hat{\\theta}^R$", ylabel="$\\hat{x}^R$",
        extra=(f"\n  xtick=\\empty,\n  ytick={{{c_hi:.4f},{c_lo:.4f}}}, "
               f"yticklabels={{$\\hat{{x}}^{{R\\prime}}$,$\\hat{{x}}^{{R,*}}$}},"))

    j_a = widest_gap(th, S_hi, S_lo, 0.45, 0.85)
    panel_a = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        "\\begin{axis}[payaxisA, title={(a) higher $\\lambda^R$ lowers "
        "$\\hat{x}^R_S$}]",
        f"  \\draw[gray,dashed] (axis cs:{a_xmin:.2f},{c_lo:.5f}) -- "
        f"(axis cs:{r_lo:.5f},{c_lo:.5f});",
        f"  \\draw[gray,dashed] (axis cs:{a_xmin:.2f},{c_hi:.5f}) -- "
        f"(axis cs:{r_hi:.5f},{c_hi:.5f});",
        f"  \\draw[-{{Stealth[length=5pt]}},thick] (axis cs:{a_xmin + 0.13:.2f},"
        f"{c_lo:.4f}) -- (axis cs:{a_xmin + 0.13:.2f},{c_hi:.4f});",
        f"  \\draw[gray,dashed] (axis cs:{t_old:.5f},{depth:.3f}) -- "
        f"(axis cs:{t_old:.5f},{float(np.interp(t_old - 0.004, th, S_lo)):.5f});",
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{t_old:.5f},"
        f"{depth:.3f}) {{$\\hat{{\\theta}}^{{T,*}}$}};",
        f"  \\addplot[red,very thick] coordinates {{{coords(edge, RD)}}};",
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(th, S_lo)}}};",
        f"  \\addplot[blue,very thick,densely dotted,smooth] coordinates "
        f"{{{coords(th, S_hi)}}};",
        shift_arrow("blue", th[j_a], S_lo[j_a], S_hi[j_a]),
        "\\end{axis}", "\\end{tikzpicture}"])

    # ---------------- panel (b): the two best responses -----------------------
    xs = np.linspace(-0.85, 0.17, 90)
    blue = [(x, m_lo.theta_top(x, theta_grid)) for x in xs]
    blue = [(x, t) for x, t in blue if t is not None]
    bx = np.array([p[0] for p in blue])
    by = np.array([p[1] for p in blue])
    ts = np.linspace(0.60, 1.55, 90)
    red_lo = [(m_lo.x_reg(t), t) for t in ts]
    red_lo = [(x, t) for x, t in red_lo if x is not None]
    red_hi = [(m_hi.x_reg(t), t) for t in ts]
    red_hi = [(x, t) for x, t in red_hi if x is not None]
    rlx = np.array([p[0] for p in red_lo]); rly = np.array([p[1] for p in red_lo])
    rhx = np.array([p[0] for p in red_hi]); rhy = np.array([p[1] for p in red_hi])

    b_xmin = min(bx.min(), rlx.min(), rhx.min()) - 0.10
    b_xmax = max(bx.max(), rlx.max(), rhx.max()) + 0.22
    b_ymin = min(by.min(), rly.min(), rhy.min()) - 0.13
    b_ymax = max(by.max(), rly.max(), rhy.max()) + 0.09
    # The old equilibrium is no longer a rest point once lambda^R rises, so the
    # adjustment from old to new is the solution of the Proposition 4 system
    # under the RAISED payline, started at the old equilibrium.
    px, pt = m_hi.best_response_flow(
        (x_old, t_old), theta_grid, bx.min(), bx.max(), rhy.min(), rhy.max(),
        stop_within=0.006 * max(b_xmax - b_xmin, b_ymax - b_ymin))
    drift = np.hypot(px[-1] - x_new, pt[-1] - t_new)
    assert drift < 0.02, f"flow did not reach the new equilibrium (off by {drift:.3f})"
    # "smooth" plus a between-positions decoration overflows pgfplots; the path
    # is already resampled by arc length, so a polyline reads as a curve.
    web = ("  \\addplot[darkgreen,thick,postaction={decorate,decoration={markings,"
           "mark=between positions 0.10 and 0.94 step 0.28 with "
           "{\\arrow{Stealth[length=5pt]}}}}] coordinates {%s};" % coords(px, pt))
    # shift arrow low down, clear of the cobweb and of the blue curve
    j_b = int(0.12 * len(rly))
    style_b = axis_style(
        "payaxisB", width=7.6, height=7.2, xmin=b_xmin, xmax=b_xmax,
        ymin=b_ymin, ymax=b_ymax, xlabel="$\\hat{x}^R$", ylabel="$\\hat{\\theta}^T$",
        extra=(f"\n  xtick={{{x_new:.4f},{x_old:.4f}}}, "
               f"xticklabels={{$\\hat{{x}}^{{R,*\\prime}}$,$\\hat{{x}}^{{R,*}}$}},"
               f"\n  ytick={{{t_old:.4f},{t_new:.4f}}}, "
               f"yticklabels={{$\\hat{{\\theta}}^{{T,*}}$,$\\hat{{\\theta}}^{{T,*\\prime}}$}},"))

    panel_b = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        "\\begin{axis}[payaxisB, title={(b) $\\hat{x}^R_{reg}$ shifts left}]",
        *[f"  \\draw[gray,dashed] (axis cs:{x:.5f},{b_ymin:.3f}) -- "
          f"(axis cs:{x:.5f},{t:.5f});" for x, t in ((x_old, t_old), (x_new, t_new))],
        *[f"  \\draw[gray,dashed] (axis cs:{b_xmin:.2f},{t:.5f}) -- "
          f"(axis cs:{x:.5f},{t:.5f});" for x, t in ((x_old, t_old), (x_new, t_new))],
        web,
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(bx, by)}}};",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(rlx, rly)}}};",
        f"  \\addplot[red,very thick,densely dotted,smooth] coordinates "
        f"{{{coords(rhx, rhy)}}};",
        shift_arrow_h("red", rly[j_b], rlx[j_b],
                      float(np.interp(rly[j_b], rhy, rhx))),
        "\\end{axis}", "\\end{tikzpicture}"])

    caption = (m_lo.caption_prefix() +
               f" The regular payline rises to $\\lambda^{{R\\prime}}={lambda_R_hi:g}$. "
               "Panel (a) holds $\\hat{\\theta}^T$ at its old equilibrium value; "
               "green in panel (b) is the path of the partial-adjustment rule of "
               "Proposition~4 under the raised payline, started at the old "
               "equilibrium.")
    header = (
        "% Figure 6 - a rise in the regular payline (section 3.1.9).\n"
        "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
        "% (a) a higher lambda^R forces a lower regular cutoff, so xhat^R_S shifts DOWN\n"
        "%     and the partial equilibrium cutoff falls.\n"
        "% (b) the regular best response therefore shifts LEFT; the cobweb converges to\n"
        "%     a new equilibrium with lower xhat^R and higher thetahat^T, i.e. a larger\n"
        "%     opportunity cost for the marginal top applicant.\n"
        f"% old equilibrium: x^R*={x_old:.4f}, theta^T*={t_old:.4f}\n"
        f"% new equilibrium: x^R*={x_new:.4f}, theta^T*={t_new:.4f} "
        f"(composite slope {comp_new:.4f})\n"
        f"% panel (a) cutoff at theta^T*: {c_lo:.4f} -> {c_hi:.4f}\n")
    # each panel names its own curves in a legend beneath it
    sample = "\\tikz[baseline=-0.6ex]\\draw[{}] (0,0) -- (0.7,0);"

    def legend(entries):
        rows = " \\\\\n".join(f"{sample.format(st)}\\ {txt}" for st, txt in entries)
        return ("\\par\\smallskip\n{\\footnotesize\\begin{tabular}{@{}l@{}}\n"
                + rows + "\n\\end{tabular}\\par}")
    leg_a = legend([
        ("blue,very thick",
         "$\\hat{\\theta}^R\\mapsto\\hat{x}^R_S(\\hat{\\theta}^R|\\lambda^R,\\hat{\\theta}^{T,*},\\sigma)$"),
        ("blue,very thick,densely dotted",
         "$\\hat{\\theta}^R\\mapsto\\hat{x}^R_S(\\hat{\\theta}^R|\\lambda^{R\\prime},\\hat{\\theta}^{T,*},\\sigma)$"),
        ("red,very thick", "$\\hat{\\theta}^R\\mapsto\\hat{x}^R_D(\\hat{\\theta}^R|\\sigma)$")])
    leg_b = legend([
        ("blue,very thick",
         "$\\hat{x}^R\\mapsto\\hat{\\theta}^T_{top}(\\hat{x}^R|\\lambda^T,\\sigma)$"),
        ("red,very thick",
         "$\\hat{\\theta}^T\\mapsto\\hat{x}^R_{reg}(\\hat{\\theta}^T|\\lambda^R,\\sigma)$"),
        ("red,very thick,densely dotted",
         "$\\hat{\\theta}^T\\mapsto\\hat{x}^R_{reg}(\\hat{\\theta}^T|\\lambda^{R\\prime},\\sigma)$")])
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(panel_a + "\n" + leg_a, "fig:pay_a"), "\\hfill",
        subfigure(panel_b + "\n" + leg_b, "fig:pay_b"),
        "\\caption{}", f"% \\caption{{\\footnotesize {caption}}}",
        "\\label{fig:pay}", "\\end{figure}", ""])
    write(fname, header + style_a + "\n" + style_b + "\n\n" + env)
    print(f"    old eq ({x_old:.4f}, {t_old:.4f}) -> new eq ({x_new:.4f}, {t_new:.4f}); "
          f"panel (a) cutoff {c_lo:.4f} -> {c_hi:.4f}; flow ends {drift:.2e} from the new eq")


# --------------------------------------------------------------------------
# Top program acceptance rate function (section 4.0.1)
# --------------------------------------------------------------------------
def figure_top_acceptance(fname="top_acceptance.tex", sigma=2.0, v_T=1.0,
                          theta_T=-1.0, OC=0.30, lo=-1.6, hi=1.6):
    """T(theta^T|sigma,OC) as the average of top-program winning probabilities.

    Setup of Figure 2: F standard normal, sigma = 2, v^T = 1, support
    [-1.6, 1.6].  G is uniform on the support, so dG is proportional to
    d(theta) and T is the PLAIN average of the winning probability over
    [theta^T, thetabar].  Equivalently, the horizontal line at T splits the
    region between the curve and the line into two pieces of equal area:
    the shortfall to its lower left and the excess to its upper right.
    (For non-uniform G the balance holds with areas weighted by g.)
    With v^T = 1 the curve meets OC/v^T = OC exactly at theta^T.
    """
    Fbar, Finv = stats.norm.sf, stats.norm.ppf
    x_T = theta_T + sigma * Finv(1 - OC / v_T)          # eq (fixedoc_topdemand)
    win = lambda th: Fbar((x_T - th) / sigma)            # theta -> Fbar_sigma(x^T|theta)
    assert abs(v_T * win(theta_T) - OC) < 1e-10

    # T under uniform G: plain average over [theta^T, thetabar]
    T = float(np.dot(_WTS, win(theta_T + (hi - theta_T) * _NODES)))
    t_star = optimize.brentq(lambda th: win(th) - T, theta_T, hi, xtol=1e-13)

    # equal-area check: shortfall to the left of t_star, excess to the right
    short = float((t_star - theta_T) * np.dot(_WTS, T - win(theta_T + (t_star - theta_T) * _NODES)))
    excess = float((hi - t_star) * np.dot(_WTS, win(t_star + (hi - t_star) * _NODES) - T))
    assert abs(short - excess) < 1e-10, (short, excess)

    # theta_T = -1 centres the shaded interval [theta_T, thetabar] in the frame;
    # the y-range follows the curve, which now climbs higher at thetabar.
    xmin, xmax, ymin = -1.75, 2.30, 0.0
    ymax = round(max(0.62, win(hi) + 0.12), 2)
    full = np.linspace(lo, hi, 90)
    left = full[full <= theta_T]
    right = np.linspace(theta_T, hi, 60)
    red_x = np.linspace(theta_T, t_star, 40)
    grn_x = np.linspace(t_star, hi, 40)
    red_poly = coords(np.r_[red_x, theta_T], np.r_[win(red_x), T])
    grn_poly = coords(np.r_[grn_x, hi], np.r_[win(grn_x), T])

    style = f"""\\pgfplotsset{{topaxis/.style={{
  width=14.8cm, height=7.6cm,
  xmin={xmin}, xmax={xmax}, ymin={ymin}, ymax={ymax},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=7pt]}},black}},
  clip=false,
  xlabel={{$\\theta$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=10pt}},
  tick label style={{font=\\small}}, label style={{font=\\small}},
  xtick=\\empty, tick style={{black}},
  ytick={{{OC:.4f},{T:.4f}}},
  yticklabels={{$OC$,{{$\\mathcal{{T}}(\\hat{{\\theta}}^T|\\sigma,OC)$}}}},
}}}}"""
    body = "\n".join([
        "\\begin{tikzpicture}", "\\begin{axis}[topaxis]",
        "  % shortfall below the average (red) and excess above it (green): equal areas",
        f"  \\addplot[draw=none,fill=red!30,forget plot] coordinates {{{red_poly}}} --cycle;",
        f"  \\addplot[draw=none,fill=green!30,forget plot] coordinates {{{grn_poly}}} --cycle;",
        f"  \\draw[gray,dashed] (axis cs:{xmin},{OC:.5f}) -- (axis cs:{theta_T:.5f},{OC:.5f});",
        f"  \\draw[black,dashed,thick] (axis cs:{xmin},{T:.5f}) -- (axis cs:{hi},{T:.5f});",
        f"  \\draw[gray,dashed] (axis cs:{theta_T:.5f},0) -- (axis cs:{theta_T:.5f},{OC:.5f});",
        f"  \\draw[gray,dashed] (axis cs:{hi},0) -- (axis cs:{hi},{win(hi):.5f});",
        f"  \\node[anchor=north,font=\\small] at (axis cs:{theta_T:.5f},-0.012) {{$\\hat{{\\theta}}^T$}};",
        f"  \\node[anchor=north,font=\\small] at (axis cs:{hi},-0.012) {{$\\bar{{\\theta}}$}};",
        "  % winning probabilities: faint where types do not apply to the top program",
        f"  \\addplot[red!35,thick,smooth] coordinates {{{coords(left, win(left))}}};",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(right, win(right))}}};",
        f"  \\node[red,anchor=west,font=\\small] at (axis cs:{hi + 0.10},{win(hi):.4f}) "
        f"{{$\\theta\\mapsto \\bar{{F}}_{{\\sigma}}(\\hat{{x}}^T|\\theta)$}};",
        "\\end{axis}", "\\end{tikzpicture}"])
    caption = ("The top program acceptance rate function. The red curve is the winning "
               "probability $\\bar{F}_{\\sigma}(\\hat{x}^T|\\theta)$ at the score cutoff "
               "\\eqref{eq:fixedoc_topdemand}, which meets $OC$ at the marginal top "
               "applicant $\\hat{\\theta}^T$. $\\mathcal{T}(\\hat{\\theta}^T|\\sigma,OC)$ is its "
               "average over $[\\hat{\\theta}^T,\\bar{\\theta}]$: the red and green regions have "
               "equal area. $F$ standard normal, $\\sigma=2$, $v^T=1$, $G$ uniform on "
               "$[-1.6,1.6]$, $OC=0.3$, $\\hat{\\theta}^T=-1$.")
    header = ("% Top program acceptance rate function T(theta^T|sigma,OC) (section 4.0.1).\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              "% Uniform G, so T is the plain average and the red and green areas are equal.\n"
              f"% xhat^T={x_T:.4f}, T={T:.5f}, curve crosses T at theta={t_star:.4f}; "
              f"areas {short:.6f} vs {excess:.6f}\n")
    write(fname, header + style + "\n\n" +
          figure_env(body, caption, "fig:topacc").replace("[htbp]", "[!ht]"))
    print(f"    xhat^T={x_T:.4f}  T={T:.4f}  crossing at theta={t_star:.4f}  "
          f"red area={short:.6f}  green area={excess:.6f}")


def figure_top_tradeoff(fname="top_tradeoff.tex", sigma=2.0, v_T=1.0, OC=0.30,
                        u_H=0.375, u_L=0.1875):
    """Intensive gain versus extensive loss, in two panels: (a) gain dominates,
    (b) loss dominates.

    Hold OC fixed and lower the marginal top applicant from theta_H to theta_L.
    With u = G(theta), for ANY merit law G,

        T_L - T_H = [ A_int - A_ext ] / (1 - u_L)
        A_int = int_{u_H}^{1}   (w_L - w_H) du        intensive margin, gain
        A_ext = int_{u_L}^{u_H} (T_H - w_L) du        extensive margin, loss

    so plotting against the merit QUANTILE makes the shaded plain areas equal
    the g-weighted gain and loss exactly.  That matters: by Proposition 7 the
    loss can dominate only if the Mills ratio is not strictly decreasing, so
    panel (b) needs a non-uniform G, where areas in theta would mislead.  For
    the uniform G of panel (a), u is an affine rescaling of theta.

    Both panels lower the marginal applicant by the same quantile amount.
    """
    Fbar, Finv = stats.norm.sf, stats.norm.ppf
    z = Finv(1 - OC / v_T)
    lo = -1.6
    panels = [
        dict(tag="a", title="(a) intensive margin dominates",
             Ginv=lambda u: lo + 3.2 * u, top=1.6, label="fig:toptrade_a"),
        dict(tag="b", title="(b) extensive margin dominates",
             Ginv=lambda u: lo + 2.0 * ((1 - u) ** (-1 / 1.2) - 1), top=np.inf,
             label="fig:toptrade_b"),
    ]
    U_END = 1 - 1e-12
    avg = lambda f, a, b: float(np.dot(_WTS, f(a + (b - a) * _NODES)))
    styles, bodies, info = [], [], []
    for P in panels:
        Gi = P["Ginv"]
        t_H, t_L = Gi(u_H), Gi(u_L)
        w = lambda u, tm: Fbar(z - (Gi(u) - tm) / sigma)
        T_H = avg(lambda u: w(u, t_H), u_H, U_END)
        T_L = avg(lambda u: w(u, t_L), u_L, U_END)
        A_int = (U_END - u_H) * avg(lambda u: w(u, t_L) - w(u, t_H), u_H, U_END)
        A_ext = (u_H - u_L) * avg(lambda u: T_H - w(u, t_L), u_L, u_H)
        assert abs((T_L - T_H) - (A_int - A_ext) / (1 - u_L)) < 1e-9, "decomposition"
        assert w(u_H, t_L) < T_H, "extensive region must be a pure shortfall"
        if P["tag"] == "a":
            assert T_L > T_H and A_int > A_ext
        else:
            assert T_L < T_H and A_ext > A_int

        # curves on [u, 1]; the heavy-tailed panel pinches to 1 at u = 1
        def curve(u0, tm, n=90):
            uu = np.r_[np.linspace(u0, 0.9985, n), 1.0]
            ww = np.r_[w(uu[:-1], tm), Fbar(z - (P["top"] - tm) / sigma)
                       if np.isfinite(P["top"]) else 1.0]
            return uu, ww
        uo, wo = curve(u_H, t_H)
        un, wn = curve(u_L, t_L)
        g_u = uo
        green = coords(np.r_[g_u, g_u[::-1]],
                       np.r_[np.interp(g_u, un, wn), wo[::-1]])
        r_u = np.linspace(u_L, u_H, 40)
        red = coords(np.r_[r_u, u_H, u_L], np.r_[w(r_u, t_L), T_H, T_H])

        ytop = round(max(wn.max(), wo.max()) + 0.10, 2)
        xmin, xmax = -0.05, 1.12
        name = f"tt{P['tag']}"
        styles.append(axis_style(
            name, width=8.1, height=7.0, xmin=xmin, xmax=xmax, ymin=0.0, ymax=ytop,
            xlabel="$G(\\theta)$", ylabel="",
            extra=(f"\n  xtick={{{u_L:.4f},{u_H:.4f},1}}, "
                   f"xticklabels={{$\\hat{{\\theta}}^T_L$,$\\hat{{\\theta}}^T_H$,$1$}},"
                   f"\n  ytick={{{OC:.4f},{T_H:.4f}}}, "
                   f"yticklabels={{$OC$,{{$\\mathcal{{T}}(\\hat{{\\theta}}^T_H)$}}}},")))
        band = np.interp(uo, un, wn) - wo
        u_peak = float(uo[int(np.argmax(band[:-1]))])
        arrow_u = max(u_H + 0.04, u_peak - 0.10)
        lab_u = min(0.90, u_peak + 0.12)
        mid = lambda u: 0.5 * (np.interp(u, un, wn) + np.interp(u, uo, wo))
        direction = "rises" if T_L > T_H else "falls"
        bodies.append("\n".join([
            "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
            f"\\begin{{axis}}[{name}, title={{{P['title']}}}]",
            f"  \\addplot[draw=none,fill=green!30,forget plot] coordinates {{{green}}} --cycle;",
            f"  \\addplot[draw=none,fill=red!30,forget plot] coordinates {{{red}}} --cycle;",
            f"  \\draw[gray,dashed] (axis cs:{xmin},{OC:.5f}) -- (axis cs:{u_H:.5f},{OC:.5f});",
            f"  \\draw[black,dashed,thick] (axis cs:{xmin},{T_H:.5f}) -- (axis cs:{u_H:.5f},{T_H:.5f});",
            "  % new average, dotted: above the old one in (a), below it in (b)",
            f"  \\draw[black,densely dotted,thick] (axis cs:{xmin},{T_L:.5f}) -- (axis cs:{u_L:.5f},{T_L:.5f});",
            f"  \\draw[gray,dashed] (axis cs:{u_L:.5f},0) -- (axis cs:{u_L:.5f},{OC:.5f});",
            f"  \\draw[gray,dashed] (axis cs:{u_H:.5f},0) -- (axis cs:{u_H:.5f},{OC:.5f});",
            f"  \\draw[gray,dashed] (axis cs:1,0) -- (axis cs:1,{wn[-1]:.5f});",
            f"  \\addplot[red,very thick] coordinates {{{coords(uo, wo)}}};",
            f"  \\addplot[red,very thick,densely dotted] coordinates {{{coords(un, wn)}}};",
            f"  \\draw[red,thick,-{{Stealth[length=4pt]}}] (axis cs:{arrow_u:.4f},"
            f"{np.interp(arrow_u, uo, wo):.4f}) -- (axis cs:{arrow_u:.4f},{np.interp(arrow_u, un, wn):.4f});",
            f"  \\path (axis cs:{lab_u - 0.14:.4f},{mid(lab_u - 0.14):.4f}) -- "
            f"(axis cs:{lab_u + 0.14:.4f},{mid(lab_u + 0.14):.4f}) "
            f"node[midway,sloped,font=\\scriptsize,text=green!40!black] {{intensive}};",
            f"  \\node[anchor=south,font=\\scriptsize,text=red!65!black] at (axis cs:"
            f"{0.5 * (u_L + u_H):.4f},{max(T_H, T_L) + 0.015:.4f}) {{extensive}};",
            "\\end{axis}", "\\end{tikzpicture}"]))
        info.append((P["tag"], t_H, t_L, T_H, T_L, A_int, A_ext, direction))

    caption = ("The trade-off from a lower marginal top applicant, holding $OC$ fixed. In "
               "each panel the marginal top applicant falls from $\\hat{\\theta}^T_H$ to "
               "$\\hat{\\theta}^T_L$, so winning probabilities rise from the solid to the dotted "
               "curve. Green is the intensive margin (applicants above $\\hat{\\theta}^T_H$ "
               "win more often); red is the extensive margin (the entrants win less often than "
               "the previous average $\\mathcal{T}(\\hat{\\theta}^T_H)$, dashed). The average "
               "acceptance rate moves to the dotted level, rising in (a) and falling in (b). "
               "The horizontal axis is the merit quantile $G(\\theta)$, so the shaded areas "
               "are exactly the gain and loss for any $G$. (a) $G$ uniform on $[-1.6,1.6]$, "
               "with strictly decreasing Mills ratio; (b) $G$ Lomax with shape $1.2$ and "
               "scale $2$ on $[-1.6,\\infty)$, whose Mills ratio is increasing. $F$ standard "
               "normal, $\\sigma=2$, $v^T=1$, $OC=0.3$; in both panels $\\hat{\\theta}^T_H$ and "
               "$\\hat{\\theta}^T_L$ are the $37.5$th and $18.75$th percentiles of merit. "
               "$\\mathcal{T}(\\cdot)$ suppresses the fixed arguments $\\sigma$ and $OC$.")
    header = ("% Intensive versus extensive margin, two panels (section 4.0.1).\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              "% Horizontal axis is the merit quantile u = G(theta), so plain areas are the\n"
              "% exact g-weighted gain (green) and loss (red):\n"
              "%   T_L - T_H = (A_int - A_ext)/(1 - u_L)\n"
              + "".join(f"% ({t}) theta_H={a:.4f} theta_L={b:.4f}: T {c:.4f} -> {d:.4f} ({e}); "
                        f"gain {g:.4f}, loss {h:.4f}\n"
                        for t, a, b, c, d, g, h, e in info))
    body = (subfigure(bodies[0], panels[0]["label"]) + "\n\\hfill\n" +
            subfigure(bodies[1], panels[1]["label"]))
    write(fname, header + "\n".join(styles) + "\n\n" +
          figure_env(body, caption, "fig:toptrade").replace("[htbp]", "[!ht]"))
    for t, a, b, c, d, g, h, e in info:
        print(f"    ({t}) T {c:.4f} -> {d:.4f} ({e});  gain {g:.4f}  loss {h:.4f}")


# --------------------------------------------------------------------------
# Miscellaneous: a parameterisation with two general equilibria
# --------------------------------------------------------------------------
def figure_misc_acceptance(m: Model, fname="misc_acceptance.tex"):
    """R(sigma, .) is hump shaped here, so lambda^R is crossed twice."""
    xs, ys = m.R_curve()
    roots = m.delta_equilibria()
    assert len(roots) == 2, f"expected two crossings, got {roots}"
    jp = int(np.argmax(ys))
    assert 0 < jp < len(ys) - 1, "R is monotone: no interior peak"
    slopes = [(m.R_of_delta(r + 1e-4) - m.R_of_delta(r - 1e-4)) / 2e-4 for r in roots]

    ylo, yhi = ys.min() - 0.03, ys.max() + 0.035
    xmin, xmax = xs.min() - 0.15, xs.max() + 0.40
    style = axis_style("miscacc", width=12.0, height=7.4, xmin=xmin, xmax=xmax,
                       ymin=ylo, ymax=yhi, xlabel="$\\delta$",
                       ylabel="$\\mathcal{R}(\\sigma,\\delta)$",
                       extra=(f"\n  xtick={{{roots[0]:.4f},{roots[1]:.4f}}}, "
                              f"xticklabels={{$\\delta^{{*}}_{{1}}$,$\\delta^{{*}}_{{2}}$}},"
                              f"\n  ytick={{{m.lambda_R:.4f}}}, "
                              f"yticklabels={{$\\lambda^{{R}}$}},"))
    body = "\n".join([
        "\\begin{tikzpicture}", "\\begin{axis}[miscacc]",
        f"  \\addplot[black,dashed,thick,samples=2,domain={xmin:.2f}:{xmax:.2f}] "
        f"{{{m.lambda_R:.4f}}};",
        *[f"  \\draw[gray,dashed] (axis cs:{r:.5f},{ylo:.3f}) -- "
          f"(axis cs:{r:.5f},{m.lambda_R:.5f});" for r in roots],
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(xs, ys)}}};",
        "\\end{axis}", "\\end{tikzpicture}"])
    caption = ("The regular acceptance rate $\\mathcal{R}(\\sigma,\\delta)$ of "
               "Vertical\\_Model\\_4 against the normalised opportunity cost "
               "$\\delta=(\\hat{\\theta}^T-\\hat{\\theta}^R)/\\sigma$. It is hump shaped, "
               "so the payline $\\lambda^R$ is crossed twice: $\\delta^*_1$ is stable "
               "($\\partial\\mathcal{R}/\\partial\\delta>0$) and $\\delta^*_2$ is not."
               + footnote_params(m))
    header = ("% Miscellaneous - R(sigma,delta) with two crossings of lambda^R.\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              f"% delta*_1={roots[0]:.4f} (dR/ddelta={slopes[0]:+.4f}), "
              f"delta*_2={roots[1]:.4f} (dR/ddelta={slopes[1]:+.4f})\n"
              f"% R runs over [{ys.min():.4f}, {ys.max():.4f}]; two crossings for any "
              f"lambda^R in ({max(ys[0], ys[-1]):.4f}, {ys.max():.4f})\n")
    write(fname, header + style + "\n\n" + figure_env(body, caption, "fig:misc_acc"))
    print(f"    delta*_1={roots[0]:.4f} (slope {slopes[0]:+.4f}), "
          f"delta*_2={roots[1]:.4f} (slope {slopes[1]:+.4f})")
    return roots


# --------------------------------------------------------------------------
# R(OC, sigma) against the opportunity cost - the multiplicity of Proposition 9
# --------------------------------------------------------------------------
def figure_oc_acceptance(m: Model, fname="oc_acceptance.tex"):
    """The same hump, drawn against OC: R(OC,sigma) = lambda^R has two roots."""
    ds, ys = m.R_curve()
    xs = np.array([m.oc_of_delta(d) for d in ds])
    assert np.all(np.diff(xs) > 0), "OC(delta) must be increasing"
    roots = [m.oc_of_delta(d) for d in m.delta_equilibria()]
    assert len(roots) == 2, f"expected two crossings, got {roots}"
    jp = int(np.argmax(ys))
    assert 0 < jp < len(ys) - 1, "R is monotone in OC: no interior peak"
    cost = m.cost_ratio * m.v_ratio           # c, in units of v^T
    slopes = [(m.R_of_delta(d + 1e-4) - m.R_of_delta(d - 1e-4))
              / (m.oc_of_delta(d + 1e-4) - m.oc_of_delta(d - 1e-4))
              for d in m.delta_equilibria()]

    ylo, yhi = ys.min() - 0.03, ys.max() + 0.035
    xmin, xmax = cost - 0.02, xs.max() + 0.06
    style = axis_style("ocacc", width=12.0, height=7.4, xmin=xmin, xmax=xmax,
                       ymin=ylo, ymax=yhi, xlabel="$OC$",
                       ylabel="$\\mathcal{R}(OC,\\sigma)$",
                       extra=(f"\n  xtick={{{cost:.4f},{roots[0]:.4f},{roots[1]:.4f}}}, "
                              f"xticklabels={{$c$,$OC^{{*}}_{{1}}$,$OC^{{*}}_{{2}}$}},"
                              f"\n  ytick={{{m.lambda_R:.4f}}}, "
                              f"yticklabels={{$\\lambda^{{R}}$}},"))
    body = "\n".join([
        "\\begin{tikzpicture}", "\\begin{axis}[ocacc]",
        f"  \\addplot[black,dashed,thick,samples=2,domain={xmin:.2f}:{xmax:.2f}] "
        f"{{{m.lambda_R:.4f}}};",
        *[f"  \\draw[gray,dashed] (axis cs:{r:.5f},{ylo:.3f}) -- "
          f"(axis cs:{r:.5f},{m.lambda_R:.5f});" for r in roots],
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(xs, ys)}}};",
        "\\end{axis}", "\\end{tikzpicture}"])
    caption = ("The regular program acceptance rate $\\mathcal{R}(OC,\\sigma)$ against "
               "the opportunity cost $OC>c$ faced by the marginal top applicant, in units "
               "of $v^T$. It is not monotone, so the regular payline $\\lambda^R$ is met "
               "at two opportunity costs $OC^*_1$ and $OC^*_2$, each of which induces an "
               "equilibrium in the sense of Definition \\ref{def:equilibrium}. "
               + m.caption_prefix())
    header = ("% R(OC,sigma) crossing lambda^R twice - the multiplicity after "
              "Proposition 9.\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              f"% OC*_1={roots[0]:.4f} (dR/dOC={slopes[0]:+.4f}), "
              f"OC*_2={roots[1]:.4f} (dR/dOC={slopes[1]:+.4f})\n"
              f"% c={cost:.4f}; OC runs over [{xs.min():.4f}, {xs.max():.4f}], "
              f"R over [{ys.min():.4f}, {ys.max():.4f}]\n")
    env = "\n".join([            # the caption is kept, but not typeset
        "\\begin{figure}[htbp]", "\\centering", body,
        "\\caption{}", f"% \\caption{{{caption}}}",
        "\\label{fig:oc_acc}", "\\end{figure}", ""])
    write(fname, header + style + "\n\n" + env)
    print(f"    OC*_1={roots[0]:.4f} (slope {slopes[0]:+.4f}), "
          f"OC*_2={roots[1]:.4f} (slope {slopes[1]:+.4f}); peak at OC={xs[jp]:.4f}")
    return roots


# --------------------------------------------------------------------------
# Section 4.3.1 - the curve of Figure oc_acc, shifted up by higher noise
# --------------------------------------------------------------------------
def figure_oc_noise_shift(m: Model, sigma_2=0.665, fname="oc_noise_shift.tex"):
    """R(OC, sigma_1) and R(OC, sigma_2) after a rise in noise.

    Beta(2, 2) merit has a convex mills ratio, so higher noise lifts the curve
    (Proposition reg_noise).  Where the curve crosses lambda^R from below it is
    increasing, so the lift moves that crossing to a lower opportunity cost.
    The equilibria at OC_1 and at OC_2 are both stable in the sense of
    Definition def:equilibrium, which is checked below.  Among the parameterisations
    searched, a stable crossing from below never came with a later crossing
    from above, so OC_unstable is drawn only when one exists.
    """
    m2 = replace(m, sigma=sigma_2)
    B = stats.beta(m.beta_a, m.beta_b)
    u = np.linspace(0.01, 0.99, 600)
    r = B.sf(u) / B.pdf(u)
    assert np.all(np.diff(r) < 0) and np.all(np.diff(r, 2) > -1e-12), \
        "the mills ratio must be decreasing and convex"

    d1, y1 = m.R_curve()
    d2, y2 = m2.R_curve()
    x1 = np.array([m.oc_of_delta(d) for d in d1])
    x2 = np.array([m2.oc_of_delta(d) for d in d2])
    # higher noise lifts the curve wherever both are defined
    common = np.intersect1d(np.round(d1, 10), np.round(d2, 10))
    f1, f2 = dict(zip(np.round(d1, 10), y1)), dict(zip(np.round(d2, 10), y2))
    assert all(f2[c] >= f1[c] - 1e-12 for c in common), "higher noise must lift R"
    e1, e2 = m.delta_equilibria(), m2.delta_equilibria()
    r1 = [m.oc_of_delta(d) for d in e1]
    r2 = [m2.oc_of_delta(d) for d in e2]
    assert r1 and r2, "both curves must cross lambda^R"
    assert abs(x1.min() - x2.min()) < 1e-3, "both curves must start at the same opportunity cost"

    def slope(mm, d):
        return ((mm.R_of_delta(d + 1e-4) - mm.R_of_delta(d - 1e-4))
                / (mm.oc_of_delta(d + 1e-4) - mm.oc_of_delta(d - 1e-4)))
    s1 = [slope(m, d) for d in e1]
    s2 = [slope(m2, d) for d in e2]
    assert s1[0] > 0 and s2[0] > 0, "the first crossing is from below"
    assert len(s1) < 2 or s1[1] < 0, "the second crossing is from above"
    oc_1, oc_2 = r1[0], r2[0]
    oc_unstable = r1[1] if len(r1) > 1 else None
    assert oc_2 < oc_1, "higher noise must lower the stable opportunity cost"

    def stable(mm, d):
        """Definition def:equilibrium: in each program the implemented cutoff
        crosses the cutoff needed to induce the marginal type from above."""
        tT = mm.theta_T_of_delta(d)
        tR = tT - mm.sigma * d
        xR = mm.x_demand_reg(tR)
        h = 1e-4
        gR = lambda t: mm.x_supply_reg(t, tT) - mm.x_demand_reg(t)
        gT = lambda t: mm.x_supply_top(t) - mm.x_demand_top(t, xR)
        assert abs(gR(tR)) < 1e-8 and abs(gT(tT)) < 1e-8, "not an equilibrium"
        return gR(tR + h) - gR(tR - h) < 0, gT(tT + h) - gT(tT - h) < 0
    st_1, st_2 = stable(m, e1[0]), stable(m2, e2[0])
    assert all(st_1), f"the initial equilibrium at OC_1 is not stable: {st_1}"
    assert all(st_2), f"the new equilibrium at OC_2 is not stable: {st_2}"

    cost = m.cost_ratio * m.v_ratio           # c, in units of v^T
    # the axes hug the curves: they span c < OC < v^R only, and the space to
    # their right is just enough for the curve labels
    yspan = max(y1.max(), y2.max()) - min(y1.min(), y2.min())
    ylo, yhi = min(y1.min(), y2.min()) - 0.10 * yspan, max(y1.max(), y2.max()) + 0.08 * yspan
    xspan = max(x1.max(), x2.max()) - cost
    xmin, xmax = cost - 0.04 * xspan, max(x1.max(), x2.max()) + 0.20 * xspan
    lam = m.lambda_R
    depth = ylo - 0.01 * yspan                # tick labels hang below the axis
    marks = [cost, oc_2, oc_1] + ([oc_unstable] if oc_unstable is not None else [])
    style = axis_style("ocshift", width=12.0, height=8.4, xmin=xmin, xmax=xmax,
                       ymin=ylo, ymax=yhi, xlabel="$OC$",
                       ylabel="$\\mathcal{R}(OC,\\sigma)$",
                       extra=(f"\n  xtick={{{','.join(f'{v:.4f}' for v in marks)}}}, "
                              f"xticklabels={{{','.join('' for _ in marks)}}},"
                              f"\n  ytick={{{lam:.4f}}}, yticklabels={{$\\lambda^{{R}}$}},"
                              "\n  tick label style={font=\\small}, label style={font=\\small},"))
    # centred labels, unless the two stable crossings sit too close together:
    # then the new one hangs to the left and the old one to the right
    close = (oc_1 - oc_2) / (xmax - xmin) < 0.08
    ticks = [(cost, "north", "$c$"), (oc_2, "north east" if close else "north", "$OC_2$"),
             (oc_1, "north west" if close else "north", "$OC_1$")]
    guides = [(oc_2, lam)]
    if oc_unstable is not None:
        ticks.append((oc_unstable, "north", "$OC_{\\mathrm{unstable}}$"))
        guides.append((oc_unstable, lam))
    # the guide at OC_1 runs up to the new curve, showing the shift there
    assert oc_1 <= x2.max(), "the new curve must reach OC_1"
    top_1 = float(np.interp(oc_1, x2, y2))
    assert top_1 > lam, "the new curve must lie above lambda^R at OC_1"
    guides.append((oc_1, top_1))
    # label each curve at its right-hand end, pushed apart if the ends are close
    ly = stack_labels([y2[-1], y1[-1]], 0.08 * yspan) if y2[-1] >= y1[-1] else [y2[-1], y1[-1]]
    lx = max(x1[-1], x2[-1]) + 0.02 * xspan
    blue_dots = [oc_1] + ([oc_unstable] if oc_unstable is not None else [])
    body = "\n".join([
        "\\begin{tikzpicture}", "\\begin{axis}[ocshift]",
        f"  \\addplot[black,dashed,thick,samples=2,domain={xmin:.2f}:{xmax:.2f}] {{{lam:.4f}}};",
        *[f"  \\draw[gray,dashed] (axis cs:{r_:.5f},{ylo:.3f}) -- (axis cs:{r_:.5f},{top:.5f});"
          for r_, top in guides],
        *[f"  \\node[anchor={a},font=\\small] at (axis cs:{x:.5f},{depth:.4f}) {{{t}}};"
          for x, a, t in ticks],
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(x1, y1)}}};",
        f"  \\addplot[blue,very thick,dash pattern=on 6pt off 3pt,smooth] coordinates "
        f"{{{coords(x2, y2)}}};",
        "  % the dots carry a white rim so they stand out against the curves",
        "  \\filldraw[fill=darkgreen,draw=white,line width=0.8pt] "
        + " ".join(f"(axis cs:{v:.5f},{lam:.5f}) circle (3.2pt)" for v in blue_dots) + ";",
        "  % the new stable crossing",
        f"  \\filldraw[fill=red,draw=white,line width=0.8pt] "
        f"(axis cs:{oc_2:.5f},{lam:.5f}) circle (3.2pt);",
        "  % where the guide at OC_1 meets the new curve",
        f"  \\filldraw[fill=orange,draw=white,line width=0.8pt] "
        f"(axis cs:{oc_1:.5f},{top_1:.5f}) circle (3.2pt);",
        f"  \\node[blue,anchor=west,font=\\small] at (axis cs:{lx:.4f},{ly[0]:.4f}) "
        f"{{$\\mathcal{{R}}(OC,\\sigma_2)$}};",
        f"  \\node[blue,anchor=west,font=\\small] at (axis cs:{lx:.4f},{ly[1]:.4f}) "
        f"{{$\\mathcal{{R}}(OC,\\sigma_1)$}};",
        "\\end{axis}", "\\end{tikzpicture}"])
    caption = ("The regular acceptance rate curve before (solid) and after (dashed) a "
               "rise in noise from $\\sigma_1$ to $\\sigma_2$. "
               "The mills ratio is convex, so the curve shifts upwards. It crosses "
               "$\\lambda^R$ from below at $OC_1$, a stable equilibrium in the sense of "
               "Definition \\ref{def:equilibrium}; after the shift, the crossing from "
               "below moves to the lower opportunity cost $OC_2$, which is also stable. "
               f"$\\sigma_1={m.sigma:g}$, $\\sigma_2={sigma_2:g}$; " + m.caption_prefix())
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering", style, "", body,
        "\\caption{}",
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {caption}",
        "\\label{fig:oc_noise_shift}", "\\end{figure}", ""])
    header = ("% Section 4.3.1 - R(OC,sigma) shifted up by higher noise, convex mills ratio.\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              f"% sigma_1={m.sigma:g}: OC_1={oc_1:.4f} (dR/dOC={s1[0]:+.4f}), stable "
              f"(regular {st_1[0]}, top {st_1[1]})"
              + (f"; OC_unstable={oc_unstable:.4f}" if oc_unstable is not None else
                 "; no crossing from above") + "\n"
              f"% sigma_2={sigma_2:g}: OC_2={oc_2:.4f} (dR/dOC={s2[0]:+.4f}), "
              f"Definition 3 conditions (regular {st_2[0]}, top {st_2[1]})\n")
    write(fname, header + env)
    print(f"    OC_1={oc_1:.4f} (stable) -> OC_2={oc_2:.4f}")
    return oc_1, oc_2, oc_unstable


# --------------------------------------------------------------------------
# Section 6.0.1 - the regular acceptance curves of a low and a high noise field
# --------------------------------------------------------------------------
def def3_stable(m: Model, d):
    """Definition def:equilibrium at the equilibrium with normalised OC d: in
    each program the implemented cutoff crosses the needed cutoff from above.
    Returns (regular condition, top condition)."""
    tT = m.theta_T_of_delta(d)
    tR = tT - m.sigma * d
    xR = m.x_demand_reg(tR)
    h = 1e-4
    gR = lambda t: m.x_supply_reg(t, tT) - m.x_demand_reg(t)
    gT = lambda t: m.x_supply_top(t) - m.x_demand_top(t, xR)
    assert abs(gR(tR)) < 1e-8 and abs(gT(tT)) < 1e-8, "not an equilibrium"
    return bool(gR(tR + h) - gR(tR - h) < 0), bool(gT(tT + h) - gT(tT - h) < 0)


def figure_field_curves(m: Model, sigma_2, curvature, fname=None, label=None, *,
                        panel=None):
    """R(OC, sigma_1) of the low noise field (blue) and R(OC, sigma_2) of the
    high noise field (red), with the common regular payline lambda^{R,*}.

    Each field's equilibrium opportunity cost OC_i^* is where its curve crosses
    the payline from below, and both equilibria are stable in the sense of
    Definition def:equilibrium.  A convex (concave) mills ratio puts the high
    noise curve above (below) the low noise one (Proposition reg_noise), so
    OC_2^* lies below (above) OC_1^*.  With panel=(name, title, width, height)
    it returns (axis style, picture, summary) for a panel of a larger figure
    instead of writing a figure of its own; the curves are then named in that
    figure's legend.
    """
    m2 = replace(m, sigma=sigma_2)
    tm = np.linspace(m.theta_lo + 0.05 * m._width, m.theta_hi - 0.05 * m._width, 400)
    r = m.mills(tm)
    assert np.all(np.diff(r) < 0), "the mills ratio must be decreasing"
    assert (np.all(np.diff(r, 2) > -1e-12) if curvature == "convex"
            else np.all(np.diff(r, 2) < 1e-12)), f"the mills ratio is not {curvature}"

    d1, y1 = m.R_curve()
    d2, y2 = m2.R_curve()
    x1 = np.array([m.oc_of_delta(d) for d in d1])
    x2 = np.array([m2.oc_of_delta(d) for d in d2])
    assert abs(x1.min() - x2.min()) < 1e-3, "both curves must start at the same opportunity cost"
    y2_at = np.interp(x1, x2, y2)
    lap = (x1 >= x2.min()) & (x1 <= x2.max())
    gap = (y2_at - y1)[lap]
    assert (np.all(gap >= -1e-9) if curvature == "convex" else np.all(gap <= 1e-9)), \
        "the high noise curve must lie above (convex) or below (concave) the low noise one"

    def slope(mm, d):
        return ((mm.R_of_delta(d + 1e-4) - mm.R_of_delta(d - 1e-4))
                / (mm.oc_of_delta(d + 1e-4) - mm.oc_of_delta(d - 1e-4)))
    # each field's equilibrium: its first crossing of the payline from below
    up1 = [d for d in m.delta_equilibria() if slope(m, d) > 0]
    up2 = [d for d in m2.delta_equilibria() if slope(m2, d) > 0]
    assert up1 and up2, "each curve must cross the payline from below"
    e1, e2 = up1[0], up2[0]
    oc_1, oc_2 = m.oc_of_delta(e1), m2.oc_of_delta(e2)
    st_1, st_2 = def3_stable(m, e1), def3_stable(m2, e2)
    assert all(st_1) and all(st_2), f"both equilibria must be stable: {st_1}, {st_2}"
    assert (oc_2 < oc_1) if curvature == "convex" else (oc_2 > oc_1), \
        "the equilibrium opportunity costs are in the wrong order"

    cost = m.cost_ratio * m.v_ratio           # c, in units of v^T
    lam = m.lambda_R
    # the high noise field's acceptance rate at the low noise field's
    # equilibrium cost OC_1^*: the guide at OC_1^* runs up to it
    assert x2.min() <= oc_1 <= x2.max(), "the high noise curve must reach OC_1^*"
    r2_at_1 = float(np.interp(oc_1, x2, y2))
    yspan = max(y1.max(), y2.max()) - min(y1.min(), y2.min())
    ylo, yhi = min(y1.min(), y2.min()) - 0.10 * yspan, max(y1.max(), y2.max()) + 0.08 * yspan
    xspan = max(x1.max(), x2.max()) - cost
    xmin, xmax = cost - 0.04 * xspan, max(x1.max(), x2.max()) + 0.20 * xspan
    depth = ylo - 0.01 * yspan
    if panel is not None:
        # a narrower frame needs less room on the right: no curve labels there
        xmax = max(x1.max(), x2.max()) + 0.05 * xspan
    close = abs(oc_1 - oc_2) / (xmax - xmin) < (0.12 if panel else 0.08)
    lo_oc, hi_oc = sorted((oc_1, oc_2))
    labs = {oc_1: "$OC_1^{*}$", oc_2: "$OC_2^{*}$"}
    ticks = [(cost, "north", "$c$"),
             (lo_oc, "north east" if close else "north", labs[lo_oc]),
             (hi_oc, "north west" if close else "north", labs[hi_oc])]
    name, title, wd, ht = panel if panel is not None else ("fieldR", None, 12.0, 8.4)
    fs = "\\footnotesize" if panel is not None else "\\small"
    style = axis_style(name, width=wd, height=ht, xmin=xmin, xmax=xmax,
                       ymin=ylo, ymax=yhi, xlabel="$OC$",
                       ylabel="$\\mathcal{R}(OC,\\sigma)$",
                       extra=(f"\n  xtick={{{cost:.4f},{oc_1:.4f},{oc_2:.4f}}}, xticklabels={{,,}},"
                              f"\n  ytick={{{lam:.4f}}}, yticklabels={{$\\lambda^{{R,*}}$}},"
                              f"\n  tick label style={{font={fs}}}, label style={{font={fs}}},"
                              + (f"\n  title={{{title}}}, title style={{font=\\small, yshift=20pt}},"
                                 if title else "")))
    ends = sorted([(y1[-1], "blue", "$\\mathcal{R}(OC,\\sigma_1)$"),
                   (y2[-1], "red", "$\\mathcal{R}(OC,\\sigma_2)$")], reverse=True)
    ly = stack_labels([e[0] for e in ends], 0.08 * yspan)
    lx = max(x1[-1], x2[-1]) + 0.02 * xspan
    body = "\n".join([
        f"\\begin{{tikzpicture}}[baseline=(current axis.south west)]" if panel else
        "\\begin{tikzpicture}", f"\\begin{{axis}}[{name}]",
        f"  \\addplot[black,dashed,thick,samples=2,domain={xmin:.2f}:{xmax:.2f}] {{{lam:.4f}}};",
        *[f"  \\draw[gray,dashed] (axis cs:{v:.5f},{ylo:.3f}) -- (axis cs:{v:.5f},{top:.5f});"
          for v, top in ((oc_1, max(lam, r2_at_1)), (oc_2, lam))],
        *[f"  \\node[anchor={a},font={fs}] at (axis cs:{x:.5f},{depth:.4f}) {{{t}}};"
          for x, a, t in ticks],
        "  % the low noise field in blue, the high noise field in red",
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(x1, y1)}}};",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(x2, y2)}}};",
        "  % each field's equilibrium, with a white rim so it stands out",
        f"  \\filldraw[fill=blue,draw=white,line width=0.8pt] (axis cs:{oc_1:.5f},{lam:.5f}) circle (3.2pt);",
        f"  \\filldraw[fill=red,draw=white,line width=0.8pt] (axis cs:{oc_2:.5f},{lam:.5f}) circle (3.2pt);",
        "  % the high noise field's acceptance rate at OC_1^*",
        f"  \\filldraw[fill=orange,draw=white,line width=0.8pt] (axis cs:{oc_1:.5f},{r2_at_1:.5f}) circle (3.2pt);",
        *([f"  \\node[{col},anchor=west,font=\\small] at (axis cs:{lx:.4f},{yy:.4f}) {{{t}}};"
           for (_, col, t), yy in zip(ends, ly)] if panel is None else []),
        "\\end{axis}", "\\end{tikzpicture}"])
    word = "above" if curvature == "convex" else "below"
    caption = (f"Regular acceptance rate curves of the low noise field (blue, $\\sigma_1$) and "
               f"the high noise field (red, $\\sigma_2$) under a {curvature} mills ratio: the "
               f"high noise curve lies {word} the low noise one. Each field's equilibrium "
               "opportunity cost $OC_i^*$ is where its curve crosses the common regular payline "
               "$\\lambda^{R,*}$ from below; both equilibria are stable in the sense of "
               "Definition \\ref{def:equilibrium}. "
               f"$\\sigma_1={m.sigma:g}$, $\\sigma_2={sigma_2:g}$; F {m.noise_name}; "
               f"{m.merit_name}; $\\lambda^T={m.lambda_T:g}$, $v^R/v^T={m.v_ratio:g}$, "
               f"$\\lambda^{{R,*}}={lam:g}$, $c/v^R={m.cost_ratio:g}$.")
    summary = (f"% {curvature}: sigma_1={m.sigma:g}: OC_1*={oc_1:.4f}; sigma_2={sigma_2:g}: "
               f"OC_2*={oc_2:.4f}; both stable (Definition 3).\n")
    if panel is not None:
        print(f"    {curvature}: OC_1*={oc_1:.4f}, OC_2*={oc_2:.4f}, both stable")
        return style, body, summary, caption
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering", style, "", body, "\\caption{}",
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {caption}", f"\\label{{{label}}}", "\\end{figure}", ""])
    header = (f"% Section 6.0.1 - regular acceptance curves of two fields, {curvature} mills ratio.\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              f"% sigma_1={m.sigma:g}: OC_1*={oc_1:.4f}; sigma_2={sigma_2:g}: OC_2*={oc_2:.4f}; "
              f"both stable (Definition 3).\n")
    write(fname, header + env)
    print(f"    {curvature}: OC_1*={oc_1:.4f}, OC_2*={oc_2:.4f}, both stable")


def figure_field_pair(fname="field_pair.tex"):
    """Section 6.0.1 in two panels: (a) convex and (b) concave mills ratio,
    with one legend for the low (blue) and high (red) noise field."""
    parts = [figure_field_curves(FIELD_CONVEX, 2.25, "convex",
                                 panel=("fieldRa", "(a) Convex mills ratio", 7.6, 6.6)),
             figure_field_curves(FIELD_CONCAVE, 0.60, "concave",
                                 panel=("fieldRb", "(b) Concave mills ratio", 7.6, 6.6))]
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(parts[0][0] + "\n\n" + parts[0][1], "fig:field_convex", frac=0.49),
        "\\hfill",
        subfigure(parts[1][0] + "\n\n" + parts[1][1], "fig:field_concave", frac=0.49),
        "\\par\\medskip",
        "{\\footnotesize\\tikz[baseline=-0.6ex]\\draw[blue,very thick] (0,0) -- (0.7,0);"
        "\\ $\\mathcal{R}(OC,\\sigma_1)$, low noise field\\qquad"
        "\\tikz[baseline=-0.6ex]\\draw[red,very thick] (0,0) -- (0.7,0);"
        "\\ $\\mathcal{R}(OC,\\sigma_2)$, high noise field\\par}",
        "\\caption{}",
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move a line below into \\caption{}.",
        f"% (a) {parts[0][3]}", f"% (b) {parts[1][3]}",
        "\\label{fig:field}", "\\end{figure}", ""])
    write(fname, "% Section 6.0.1 - regular acceptance curves of two fields, convex and concave.\n"
                 "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
                 + parts[0][2] + parts[1][2] + env)


# --------------------------------------------------------------------------
# Higher noise in the regular program evaluation only (section 4.0.4)
# --------------------------------------------------------------------------
def stack_labels(ends, gap):
    """Label heights for curves whose right-hand ends may be too close together.

    Keeps each label next to its own curve and in the same order, pushing one
    down only when its neighbour above is within gap.
    """
    ys = list(ends)
    for i in range(1, len(ys)):
        ys[i] = min(ys[i], ys[i - 1] - gap)
    return ys


def figure_noise_rotation(stem="noise_rotation", sigma=1.50, sigma_hi=2.70,
                          v_T=1.0, v_R=0.35, cost=0.20, theta_R=-0.30,
                          theta_T=1.10, lo=-2.2, hi=2.2, height=6.3, merit_sd=0.40):
    """Section 4.0.4 as three figures, on one merit scale, in the style of Figure 1.

    Panel (a): noise rises from sigma to sigma' in the REGULAR program alone.
    The regular curve rotates clockwise about (x^R, v^R/2): the marginal top
    applicant and its opportunity cost both fall.  Cutting the regular cutoff
    to x^R' restores OC at the original theta^T, which lifts every
    inframarginal applicant (blue band) and draws in new ones (grey band).

    Panel (b) starts where (a) ended - the green curve and the top curve - and
    raises noise in the TOP program.  Holding OC_i fixed, the top program keeps
    its acceptance rate at the payline lambda^T of the initial state, computed
    for normal merit with standard deviation merit_sd.  By Proposition 7 this
    takes a lower marginal top applicant theta^T_f and a new top cutoff x^T_f.
    The dashed top curve is the one at x^T_f, through (theta^T_f, OC_i); no
    rotation is drawn.  It meets the green curve at theta^T'' and OC'' < OC_i.
    Restoring OC_i takes a second cut in the REGULAR cutoff, to x^R_f, whose
    curve meets the dashed top curve at (theta^T_f, OC_i).

    Panel (c) sets the initial state against the final one.  Each panel is
    its own figure, in its own file <stem>_a.tex, _b.tex, _c.tex, labelled
    fig:noise_rot_a, _b and _c, so the three can sit in separate places.
    """
    Fbar, Finv = stats.norm.sf, stats.norm.ppf
    z_c = Finv(1 - cost / v_R)                        # (x^R - theta^R)/sigma at c
    x_R = theta_R + sigma * z_c
    OC = v_R * Fbar((x_R - theta_T) / sigma)
    z_T = Finv(1 - OC / v_T)
    x_T = theta_T + sigma * z_T
    x_R2 = theta_T + sigma_hi * Finv(1 - OC / v_R)    # panel (a): restores OC
    theta_R2 = x_R2 - sigma_hi * z_c
    G = stats.norm(0, merit_sd)                       # merit, needed for the top payline

    def top_rate(t, s):
        """T(theta^T | s, OC): average top acceptance rate at the fixed OC."""
        u = G.cdf(t)
        grid = G.ppf(u + (1 - u) * _NODES)
        return float(np.dot(_WTS, Fbar(z_T - (grid - t) / s)))

    lambda_T = top_rate(theta_T, sigma)               # the top payline, held fixed
    t_c = optimize.brentq(lambda t: top_rate(t, sigma_hi) - lambda_T,
                          theta_T - 4.0, theta_T, xtol=1e-13)   # panel (b): theta^T_f
    x_T_f = t_c + sigma_hi * z_T                      # panel (b): new top cutoff
    x_R3 = t_c + sigma_hi * Finv(1 - OC / v_R)        # panel (b): restores OC
    theta_R3 = x_R3 - sigma_hi * z_c

    top = lambda t: v_T * Fbar((x_T - t) / sigma)
    top_hi = lambda t: v_T * Fbar((x_T_f - t) / sigma_hi)
    reg0 = lambda t: v_R * Fbar((x_R - t) / sigma)
    reg1 = lambda t: v_R * Fbar((x_R - t) / sigma_hi)
    reg2 = lambda t: v_R * Fbar((x_R2 - t) / sigma_hi)
    reg3 = lambda t: v_R * Fbar((x_R3 - t) / sigma_hi)
    theta_T1 = optimize.brentq(lambda t: top(t) - reg1(t), lo, hi, xtol=1e-13)
    OC1 = top(theta_T1)
    t_b = optimize.brentq(lambda t: top_hi(t) - reg2(t), lo, hi, xtol=1e-13)
    OC2 = top_hi(t_b)

    grid = np.linspace(lo, hi, 400)
    # the construction, panel (a)
    assert abs(reg0(theta_R) - cost) < 1e-12 and abs(reg2(theta_R2) - cost) < 1e-12
    assert abs(top(theta_T) - OC) < 1e-12 and abs(reg0(theta_T) - OC) < 1e-12
    assert abs(reg2(theta_T) - OC) < 1e-12, "step two must restore OC at theta^T"
    assert abs(reg0(x_R) - v_R / 2) < 1e-12 and abs(reg1(x_R) - v_R / 2) < 1e-12
    assert OC1 < OC and theta_T1 < theta_T
    assert np.all(reg2(grid[grid < theta_T]) > reg0(grid[grid < theta_T]))
    assert theta_R2 < theta_R and x_R - sigma_hi * z_c < theta_T1
    # the construction, panel (b)
    assert abs(top_rate(t_c, sigma_hi) - lambda_T) < 1e-9, "the top payline must hold"
    assert abs(top_hi(t_c) - OC) < 1e-12 and abs(reg3(t_c) - OC) < 1e-12
    assert abs(reg3(theta_R3) - cost) < 1e-12
    assert OC2 < OC and t_b < t_c < theta_T, (t_b, t_c, theta_T)
    assert theta_R3 < theta_R2, "the second cut must widen the regular program again"
    assert theta_R3 > lo, "theta^R_f must stay on the merit axis"
    assert np.all(reg3(grid) > reg2(grid))

    th = np.linspace(lo, hi, 61)
    ymax = round(top(hi) + 0.08, 2)
    xmin, xmax = lo - 0.10, hi + 1.15
    # the spacings that matter are typographic, so fix them in cm and convert
    per_y, per_x = height / ymax, 14.8 / (xmax - xmin)
    depth, row_drop = -0.10 / per_y, 0.42 / per_y
    lab_x, lead_x, gap = hi + 0.12, hi + 0.02, 0.55 / per_y

    style = f"""\\pgfplotsset{{rotaxis/.style={{
  width=14.8cm, height={height}cm,
  xmin={xmin}, xmax={xmax}, ymin=0, ymax={ymax},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=7pt]}},black}},
  clip=false, axis on top,
  xlabel={{$\\theta$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=10pt}},
  tick label style={{font=\\small}}, label style={{font=\\small}},
  title style={{font=\\small}},
  ytick=\\empty, xtick=\\empty, tick style={{black}},
}}}}"""

    def frame(title, levels, marks, curves, pivot=None, bands=(), lv_gap=0.45):
        """One panel: shared axis, its own guides, curves and right-hand labels."""
        ends = stack_labels([c[2](hi) for c in curves], gap)
        lv_sorted = sorted(levels, key=lambda L: -L[0])
        lv_ys = stack_labels([L[0] for L in lv_sorted], lv_gap / per_y)
        # a label needs about 0.8cm of its own; anything closer goes a row down
        min_sep = 0.80 / per_x
        rows, taken = {}, {}
        for t, _, lab in sorted(marks):
            if not lab:
                rows[t] = depth
                continue
            for d in (depth, depth - row_drop):
                if t - taken.get(d, -1e9) >= min_sep:
                    rows[t], taken[d] = d, t
                    break
            else:
                rows[t] = depth
        return "\n".join([
            "\\begin{tikzpicture}",
            f"\\begin{{axis}}[rotaxis, title={{{title}}}]",
            *[f"  \\addplot[draw=none,fill={fill},forget plot] coordinates "
              f"{{({a:.5f},0) ({a:.5f},{ymax}) ({b:.5f},{ymax}) ({b:.5f},0)}} \\closedcycle;"
              for a, b, fill, _, _, _ in bands],
            *[f"  \\node[anchor=north,font=\\footnotesize,text={colour},align=center] "
              f"at (axis cs:{0.5 * (a + b):.4f},"
              f"{ymax - 0.015 - row * 0.55 / per_y:.4f}) {{{text}}};"
              for a, b, _, colour, text, row in bands],
            *[f"  \\draw[gray,dashed] (axis cs:{xmin},{y:.4f}) -- (axis cs:{x:.4f},{y:.4f});"
              for y, x, _ in levels],
            *[f"  \\draw[gray,thin] (axis cs:{xmin - 0.04},{ly:.4f}) -- (axis cs:{xmin},{y:.4f});"
              for (y, _, _), ly in zip(lv_sorted, lv_ys) if abs(ly - y) > 0.004],
            *[f"  \\node[anchor=east,font=\\small] at (axis cs:{xmin - 0.06},{ly:.4f}) {{{lab}}};"
              for (_, _, lab), ly in zip(lv_sorted, lv_ys)],
            *[f"  \\draw[gray,dashed] (axis cs:{t:.4f},{rows.get(t, depth):.4f}) -- "
              f"(axis cs:{t:.4f},{y:.4f});" for t, y, _ in marks],
            *[f"  \\node[anchor=north,font=\\small] at (axis cs:{t:.4f},{rows[t]:.4f}) {{{lab}}};"
              for t, _, lab in marks if lab],
            *[f"  \\addplot[{spec},very thick,smooth] coordinates {{{coords(th, f(th))}}};"
              for spec, _, f, _ in curves],
            *([f"  \\addplot[{pivot[0]},only marks,mark=*,mark size=2.2pt,forget plot] "
               f"coordinates {{({pivot[1]:.4f},{pivot[2]:.4f})}};"] if pivot else []),
            *[f"  \\draw[gray,thin] (axis cs:{lead_x},{c[2](hi):.4f}) -- "
              f"(axis cs:{lab_x - 0.02},{y:.4f});" for c, y in zip(curves, ends)],
            *[f"  \\node[{c[1]},anchor=west,font=\\small] at (axis cs:{lab_x},{y:.4f}) {{{c[3]}}};"
              for c, y in zip(curves, ends)],
            "\\end{axis}", "\\end{tikzpicture}"])

    # The states, in order:
    #   (1) initial        (2) regular noise up      (3) regular cutoff cut
    #   (4) top noise up   (5) regular cutoff cut again
    # The labels follow the notation of the text: the initial state is indexed
    # by i, the state after panel (a) by a prime, the intermediate state of
    # panel (b) by a double prime, and the final state by f.  Each x axis
    # carries only the cutoff about which a curve rotates and the marginal
    # applicants, before and after.
    lv = lambda y, x, lab: (y, x, lab)
    REG1 = "$\\theta\\mapsto v^R\\bar{F}_{\\sigma_i}(\\hat{x}^R_i|\\theta)$"
    REG2 = "$\\theta\\mapsto v^R\\bar{F}_{\\sigma_f}(\\hat{x}^R_i|\\theta)$"
    REG3 = "$\\theta\\mapsto v^R\\bar{F}_{\\sigma_f}(\\hat{x}^{R\\prime}|\\theta)$"
    REG5 = "$\\theta\\mapsto v^R\\bar{F}_{\\sigma_f}(\\hat{x}^R_f|\\theta)$"
    TOP1 = "$\\theta\\mapsto v^T\\bar{F}_{\\sigma_i}(\\hat{x}^T_i|\\theta)$"
    TOP4 = "$\\theta\\mapsto v^T\\bar{F}_{\\sigma_f}(\\hat{x}^T_f|\\theta)$"
    # the panels carry no title: their typeset captions, below, describe them
    panel_a = frame(
        "",
        [lv(OC, theta_T, "$OC_i$"), lv(OC1, theta_T1, "$OC'$"), lv(cost, hi, "$c$")],
        [(theta_R2, cost, "$\\hat{\\theta}^{R\\prime}$"),
         (theta_R, cost, "$\\hat{\\theta}^R_i$"),
         (x_R, v_R / 2, "$\\hat{x}^R_i$"),
         (theta_T1, OC1, "$\\hat{\\theta}^{T\\prime}$"),
         (theta_T, OC, "$\\hat{\\theta}^T_i$")],
        [("blue", "blue", reg0, REG1),
         ("darkgreen", "darkgreen", reg2, REG3),
         ("blue,densely dashed", "blue", reg1, REG2)],
        ("blue", x_R, v_R / 2),
        bands=[(theta_R2, theta_R, "gray!14", "black!65", "New\\\\Applicants", 0),
               (theta_R, theta_T, "blue!9", "blue!60!black", "Inframarginal\\\\Applicants", 0)])
    panel_b = frame(
        "",
        [lv(OC, theta_T, "$OC_i$"), lv(OC2, t_b, "$OC''$"), lv(cost, hi, "$c$")],
        [(theta_R3, cost, "$\\hat{\\theta}^R_f$"),
         (theta_R2, cost, "$\\hat{\\theta}^{R\\prime}$"),
         (t_b, OC2, "$\\hat{\\theta}^{T\\prime\\prime}$"),
         (t_c, OC, "$\\hat{\\theta}^T_f$"),
         (theta_T, OC, "$\\hat{\\theta}^T_i$")],
        [("red", "red", top, TOP1),
         ("red,densely dashed", "red", top_hi, TOP4),
         ("darkgreen,densely dashed", "darkgreen", reg3, REG5),
         ("darkgreen", "darkgreen", reg2, REG3)],
        lv_gap=0.60)   # OC_i over OC'' needs the room
    panel_c = frame(
        "",
        [lv(OC, theta_T, "$OC_i$"), lv(cost, hi, "$c$")],
        [(theta_R3, cost, "$\\hat{\\theta}^R_f$"),
         (theta_R, cost, "$\\hat{\\theta}^R_i$"),
         (t_c, OC, "$\\hat{\\theta}^T_f$"),
         (theta_T, OC, "$\\hat{\\theta}^T_i$")],
        [("red", "red", top, TOP1),
         ("red,densely dashed", "red", top_hi, TOP4),
         ("blue", "blue", reg0, REG1),
         ("darkgreen,densely dashed", "darkgreen", reg3, REG5)],
        bands=[(theta_R3, theta_R, "gray!14", "black!65",
                "New regular applicants", 0),
               (theta_R, t_c, "blue!9", "blue!60!black",
                "Remaining regular applicants", 1),
               (t_c, theta_T, "red!9", "red!60!black",
                "Regular to top switchers", 0)])
    # panel (a) also carries the top curve, drawn last so it sits over the bands
    panel_a = panel_a.replace(
        "\\end{axis}",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(th, top(th))}}};\n"
        f"  \\node[red,anchor=west,font=\\small] at (axis cs:{lab_x},{top(hi):.4f}) "
        f"{{{TOP1}}};\n\\end{{axis}}")

    captions = {
        "fig:noise_rot_a": (
            "The solid blue and red curves represent the initial expected value from "
            "applying to the regular and top programs, respectively. Now consider a rise in "
            "noise in the regular program from $\\sigma_i$ to $\\sigma_f$. This rotates the "
            "blue curve around the initial score cutoff $\\hat{x}^R_i$ to obtain the dashed "
            "blue curve. The opportunity cost is now $OC'$. The corresponding fall in the "
            "regular score cutoff to $\\hat{x}^{R\\prime}$ to maintain the opportunity cost "
            "at its initial value $OC_i$ raises the regular program winning probabilities, "
            "yielding the solid green curve."),
        "fig:noise_rot_b": (
            "The solid green and red curves are where we left off in Figure "
            "\\ref{fig:noise_rot_a}. Following a rise in noise in the top program from "
            "$\\sigma_i$ to $\\sigma_f$, the top program adjusts its score cutoff to "
            "$\\hat{x}^T_f$ to keep its acceptance rate equal to its payline. This yields "
            "the dashed red curve. At the regular score cutoff $\\hat{x}^{R\\prime}$, the "
            "marginal top applicant falls to $\\hat{\\theta}^{T\\prime\\prime}$ and his "
            "opportunity cost falls to $OC''$. The corresponding fall in the regular score "
            "cutoff from $\\hat{x}^{R\\prime}$ to $\\hat{x}^R_f$ raises regular winning "
            "probabilities further, yielding the dashed green curve."),
        "fig:noise_rot_c": (
            "At the initial noise level $\\sigma_i$, the solid blue and red curves represent "
            "the initial expected value from applying to the regular and top programs, "
            "respectively. The dashed green and red curves represent the expected value "
            "from applying to the regular and top programs, respectively, after the rise in "
            "noise to $\\sigma_f$ and the subsequent adjustment to keep the opportunity cost "
            "equal to its initial value $OC_i$. Notice that the dashed green curve "
            "intersects the dashed red curve at exactly $OC_i$."),
    }
    short_captions = {
        "fig:noise_rot_a": ("Higher noise in the regular program evaluation, on the merit "
                            "scale of Figure \\ref{fig:sorting}, holding the opportunity "
                            "cost of the marginal top applicant at $OC_i$."),
        "fig:noise_rot_b": ("Higher noise in the top program evaluation, starting where "
                            "Figure \\ref{fig:noise_rot_a} ends, holding the opportunity "
                            "cost of the marginal top applicant at $OC_i$."),
        "fig:noise_rot_c": ("Figures \\ref{fig:noise_rot_a} and \\ref{fig:noise_rot_b} "
                            "combined: before and after, holding the opportunity cost of "
                            "the marginal top applicant at $OC_i$ throughout."),
    }

    def figure(panel, label):
        return "\n".join([
            "\\begin{figure}[htbp]", "\\centering", panel,
            "% Shorter caption, not typeset:",
            f"% {short_captions[label]}",
            f"\\caption{{\\footnotesize {captions[label]}}}", f"\\label{{{label}}}",
            "\\end{figure}", ""])

    # Each file is self-contained (it sets the axis style itself), so the
    # three can be input in any order and in separate places.
    gen = "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
    headers = {
        "a": ("% Higher noise in the regular program evaluation (section 4.0.4).\n" + gen +
              f"% x^R_i={x_R:.4f} (pivot v^R/2={v_R / 2:.3f}), x^T_i={x_T:.4f}, "
              f"x^R'={x_R2:.4f}\n"
              f"% OC={OC:.4f} -> {OC1:.4f};  theta^T={theta_T:.4f} -> {theta_T1:.4f};  "
              f"theta^R={theta_R:.4f} -> {theta_R2:.4f}\n"),
        "b": ("% Higher noise in the top program evaluation (section 4.0.4).\n" + gen +
              f"% merit N(0,{merit_sd}^2), top payline lambda^T={lambda_T:.4f} held fixed;  "
              f"x^T={x_T:.4f} -> {x_T_f:.4f}, x^R_f={x_R3:.4f}\n"
              f"% OC={OC:.4f} -> {OC2:.4f} -> {OC:.4f};  theta^T={theta_T:.4f} -> "
              f"{t_b:.4f} -> {t_c:.4f};  theta^R'={theta_R2:.4f} -> {theta_R3:.4f}\n"),
        "c": ("% Higher noise in both programs, before and after (section 4.0.4).\n" + gen +
              f"% at OC={OC:.4f} throughout: theta^T={theta_T:.4f} -> {t_c:.4f};  "
              f"theta^R={theta_R:.4f} -> {theta_R3:.4f}\n"),
    }
    for key, panel in (("a", panel_a), ("b", panel_b), ("c", panel_c)):
        write(f"{stem}_{key}.tex", headers[key] + style + "\n\n"
              + figure(panel, f"fig:noise_rot_{key}"))
    print(f"    (a) OC {OC:.4f}->{OC1:.4f}; theta^T {theta_T:.4f}->{theta_T1:.4f}; "
          f"theta^R {theta_R:.4f}->{theta_R2:.4f}")
    print(f"    (b) OC {OC:.4f}->{OC2:.4f}; crossing {theta_T:.4f}->{t_b:.4f}->{t_c:.4f}; "
          f"theta^R' {theta_R2:.4f}->{theta_R3:.4f}")
    print(f"    (c) at OC={OC:.4f} throughout: theta^T {theta_T:.4f}->{t_c:.4f}, "
          f"theta^R {theta_R:.4f}->{theta_R3:.4f}")


# --------------------------------------------------------------------------
# Proposition 12 - the payline is restored by a transfer of winning probability
# --------------------------------------------------------------------------
class _MeritNormal:
    """Standard normal merit: mills ratio strictly decreasing and convex."""
    tag, curvature = "N(0,1)", "convex"
    lo, hi = -4.0, 4.5
    cdf = staticmethod(stats.norm.cdf)
    inv = staticmethod(stats.norm.ppf)

    @staticmethod
    def mills(t):
        return stats.norm.sf(t) / stats.norm.pdf(t)

    @staticmethod
    def between(a, b, p):
        """Quantile p of merit conditional on [a, b]: the regular pool."""
        lo_u, hi_u = stats.norm.cdf(a), stats.norm.cdf(b)
        return stats.norm.ppf(lo_u + (hi_u - lo_u) * np.asarray(p))


class _MeritTruncExp:
    """Right-truncated exponential: mills ratio strictly decreasing, concave."""
    tag, curvature = "right-truncated exponential", "concave"

    def __init__(self, rate, upper):
        self.rate, self.upper = rate, upper
        self.Z = 1.0 - np.exp(-rate * upper)
        self.lo, self.hi = 0.02, upper - 0.05

    def cdf(self, t):
        return (1.0 - np.exp(-self.rate * np.clip(t, 0.0, self.upper))) / self.Z

    def inv(self, q):
        return -np.log(1.0 - np.asarray(q) * self.Z) / self.rate

    def mills(self, t):
        return (1.0 - np.exp(-self.rate * (self.upper - t))) / self.rate

    def between(self, a, b, p):
        """Quantile p of merit conditional on [a, b]: the regular pool.

        The density exp(-rate theta) is read on [a, b] directly, so the regular
        pool may reach below 0.  This extends the support downward and leaves
        the mills ratio, and everything above the marginal top applicant,
        unchanged.
        """
        ea, eb = np.exp(-self.rate * a), np.exp(-self.rate * b)
        return -np.log(ea - np.asarray(p) * (ea - eb)) / self.rate


# --------------------------------------------------------------------------
# One QQ panel, shared by the winning probability and residual life figures
# --------------------------------------------------------------------------
def qq_panel(name, title, cx, cy, ws, gain_first, start_lab, ends=None, cap=None,
             tip_at_centroid=False):
    """A square QQ panel: before on the vertical, after on the horizontal.

    cx, cy start at the common rank 0 value; ws is the value at q*.  `ends`
    labels the two q = 1 quantiles, or is None when the support is unbounded
    and the curve leaves through the top of the frame; `cap` is then the top.
    With `tip_at_centroid` each leader points at the centroid of its region
    rather than at its widest point, which suits thin regions cut by the box.
    """
    start = cx[0]
    lo = min(cx.min(), cy.min())
    hi = max(cx.max(), cy.max())
    pad = 0.05 * (hi - lo)
    xlo, xhi = lo - pad, hi + pad
    # the 45 degree line leaves the box built by the two q = 1 quantiles at
    # the smaller of them, so the line and the shading stop there
    if cap is None:
        cap = min(cx[-1], cy[-1])
    m = cx <= ws
    # the whole curve past q*: closing the polygon at (cap, cap) traces the
    # 45 degree line up to the box, then the box edge back to the curve's end
    n = cx > ws
    first = coords(np.r_[cx[m], ws, start], np.r_[cy[m], ws, start])
    second = coords(np.r_[cx[n], cap, ws], np.r_[cy[n], cap, ws])
    f_fill = "darkgreen!20" if gain_first else "red!18"
    s_fill = "red!18" if gain_first else "darkgreen!20"
    f_txt = "darkgreen" if gain_first else "red!70!black"
    s_txt = "red!70!black" if gain_first else "darkgreen"
    f_lab, s_lab = ("gain", "loss") if gain_first else ("loss", "gain")
    gap = cx - cy                             # > 0 where the curve is below
    # the shading stops at the box top, so measure each region against the
    # 45 degree line capped at the box
    diag = np.minimum(cx, cap)
    ia = int(np.argmax(np.abs(np.where(m, diag - cy, 0.0))))
    ib = int(np.argmax(np.abs(np.where(n, diag - cy, 0.0))))
    tip = lambda i: (cx[i], 0.5 * (cy[i] + diag[i]))   # inside the shading
    if tip_at_centroid:
        def centroid(px, py):
            """Centroid of a closed polygon (shoelace formula)."""
            cr = px * np.roll(py, -1) - np.roll(px, -1) * py
            A = cr.sum() / 2
            return (((px + np.roll(px, -1)) * cr).sum() / (6 * A),
                    ((py + np.roll(py, -1)) * cr).sum() / (6 * A))
        cents = {ia: centroid(np.r_[cx[m], ws, start], np.r_[cy[m], ws, start]),
                 ib: centroid(np.r_[cx[n], cap, ws], np.r_[cy[n], cap, ws])}
        # the leader and the label follow the centroid; the side of the
        # diagonal is still read off the widest point
        tip = lambda i: cents[i]
    # push each label off its region, perpendicular to the diagonal, into the
    # empty triangle on its own side: below the line goes down-right, above
    # the line goes up-left
    off = 0.13 * (hi - lo)

    def place(idx):
        mx, my = tip(idx)
        s = 1.0 if gap[idx] > 0 else -1.0     # below the line -> down-right
        lx, ly = mx + s * off, my - s * off
        # a region near the top corner has no room on its own side of the
        # diagonal inside the box: drop the label straight below the curve
        if lx > hi - 0.10 * (hi - lo):
            lx, ly = (min(mx, hi - 0.10 * (hi - lo)),
                      (my if tip_at_centroid else cy[idx]) - 1.3 * off)
        lx = max(lx, lo + 0.08 * (hi - lo))
        ly = min(max(ly, lo + 0.06 * (hi - lo)), hi - 0.06 * (hi - lo))
        return lx, ly
    fx, fy = place(ia)
    sx, sy = place(ib)
    deg_anchor = "north west" if gain_first else "south east"
    style = f"""\\pgfplotsset{{{name}/.style={{
  width=7.3cm, height=7.3cm,
  xmin={xlo:.4f}, xmax={xhi:.4f}, ymin={xlo:.4f}, ymax={xhi:.4f},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=6pt]}},black}},
  clip=false, axis on top,
  xlabel={{$t$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=8pt}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\footnotesize}},
  title style={{font=\\small, yshift=2pt}},
  ytick=\\empty, xtick=\\empty, tick style={{black}},
}}}}"""
    end_guides = [] if ends is None else [
        "  % rank 1: the highest merit applicant",
        f"  \\draw[gray,dashed] (axis cs:{cx[-1]:.4f},{xlo:.4f}) -- "
        f"(axis cs:{cx[-1]:.4f},{cy[-1]:.4f}) -- (axis cs:{xlo:.4f},{cy[-1]:.4f});"]
    end_labels = [] if ends is None else [
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{cx[-1]:.4f},"
        f"{xlo - 0.02 * (hi - lo):.4f}) {{{ends[0]}}};",
        f"  \\node[anchor=south west,font=\\footnotesize] at (axis cs:{xlo + 0.01 * (hi - lo):.4f},"
        f"{cy[-1]:.4f}) {{{ends[1]}}};"]
    # align side-by-side panels on the corner of their axes, not on whichever
    # label hangs lowest
    body = "\n".join([
        f"\\begin{{tikzpicture}}[baseline=({name}frame.south west)]",
        f"\\begin{{axis}}[{name}, name={name}frame, title={{{title}}}]",
        "  % the gain and the loss: the area between the curve and the 45 degree line",
        f"  \\addplot[draw=none,fill={f_fill},forget plot] coordinates {{{first}}};",
        f"  \\addplot[draw=none,fill={s_fill},forget plot] coordinates {{{second}}};",
        f"  \\addplot[black,dashed,thick,samples=2,domain={xlo:.4f}:{cap:.4f}] {{x}};",
        f"  \\node[anchor={deg_anchor},font=\\footnotesize] at "
        f"(axis cs:{cap:.4f},{cap:.4f}) {{$45^{{\\circ}}$}};",
        "  % rank 0: the marginal applicant, common to both noise levels",
        f"  \\draw[gray,dashed] (axis cs:{start:.4f},{xlo:.4f}) -- "
        f"(axis cs:{start:.4f},{start:.4f}) -- (axis cs:{xlo:.4f},{start:.4f});",
        *end_guides,
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(cx, cy)}}};",
        f"  \\addplot[blue,only marks,mark=*,mark size=1.8pt,forget plot] "
        f"coordinates {{({start:.4f},{start:.4f}) ({ws:.4f},{ws:.4f})}};",
        f"  \\node[anchor=north west,font=\\footnotesize] at "
        f"(axis cs:{ws:.4f},{ws:.4f}) {{$q^{{*}}$}};",
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{start:.4f},"
        f"{xlo - 0.02 * (hi - lo):.4f}) {{{start_lab}}};",
        *end_labels,
        f"  \\node[anchor=east,font=\\footnotesize] at (axis cs:{xlo - 0.01 * (hi - lo):.4f},"
        f"{start:.4f}) {{{start_lab}}};",
        f"  \\node[{f_txt},anchor=center,font=\\footnotesize,inner sep=1.5pt] (flab) at "
        f"(axis cs:{fx:.4f},{fy:.4f}) {{{f_lab}}};",
        f"  \\draw[{f_txt},thin] (flab) -- "
        f"(axis cs:{tip(ia)[0]:.4f},{tip(ia)[1]:.4f});",
        f"  \\node[{s_txt},anchor=center,font=\\footnotesize,inner sep=1.5pt] (slab) at "
        f"(axis cs:{sx:.4f},{sy:.4f}) {{{s_lab}}};",
        f"  \\draw[{s_txt},thin] (slab) -- "
        f"(axis cs:{tip(ib)[0]:.4f},{tip(ib)[1]:.4f});",
        "\\end{axis}", "\\end{tikzpicture}"])
    return style + "\n\n" + body


class _MeritTruncExpUnit:
    """Truncated exponential on [0, 1], density proportional to exp(-kappa theta).

    Its mills ratio (1 - exp(-kappa (1 - theta))) / kappa is strictly decreasing
    for every kappa: concave for kappa > 0, convex for kappa < 0, and linear (the
    uniform, whose residual lives are a scale family) at kappa = 0.  Flipping the
    sign of kappa flips the curvature and nothing else.
    """

    def __init__(self, kappa):
        self.kappa = kappa
        self.curvature = "convex" if kappa < 0 else "concave"
        self.tag = f"truncated exponential, kappa = {kappa:g}"

    def cdf(self, t):
        return np.expm1(-self.kappa * np.clip(t, 0.0, 1.0)) / np.expm1(-self.kappa)

    def inv(self, q):
        return -np.log1p(np.asarray(q) * np.expm1(-self.kappa)) / self.kappa

    def mills(self, t):
        return -np.expm1(-self.kappa * (1.0 - t)) / self.kappa


def figure_qq_residual(fname="qq_residual.tex", kappa=3.0, sigma_1=0.02, sigma_2=0.40,
                       OC=0.40, v_T=1.0, lambda_T=0.60):
    """QQ plots of standardized residual life, convex (a) and concave (b).

    The two panels differ only in the sign of kappa, so one is the mirror of the
    other: the slopes at the origin are reciprocal and the gain and the loss swap.
    """
    Fbar, Finv = stats.norm.sf, stats.norm.ppf
    z = Finv(1 - OC / v_T)          # every marginal applicant wins with prob OC/v^T
    q = np.linspace(0.0, 1.0, 241)  # U is bounded: rank 1 is (1 - theta^T)/sigma

    uqq, meta = [], []
    for k, (G, title) in enumerate(zip(
            (_MeritTruncExpUnit(-kappa), _MeritTruncExpUnit(kappa)),
            ("(a) Convex mills ratio", "(b) Concave mills ratio"))):
        tm = np.linspace(0.02, 0.98, 400)
        r = G.mills(tm)
        assert np.all(np.diff(r) < 0), f"{G.tag}: mills ratio not decreasing"
        d2 = np.gradient(np.gradient(r, tm), tm)[8:-8]
        assert (np.all(d2 > 0) if G.curvature == "convex" else np.all(d2 < 0)), \
            f"{G.tag}: mills ratio is not {G.curvature}"

        def uquant(qs, theta_T, sigma):
            """U^{-1}_{sigma,OC}(q): standardized residual life at rank q."""
            u0 = G.cdf(theta_T)
            return (G.inv(u0 + (1 - u0) * np.asarray(qs)) - theta_T) / sigma

        def rate(theta_T, sigma):
            """T(theta^T | sigma, OC): mean winning probability among applicants."""
            return float(np.dot(_WTS, Fbar(z - uquant(_NODES, theta_T, sigma))))

        t1, t2 = (optimize.brentq(lambda t: rate(t, sg) - lambda_T, 1e-9, 1 - 1e-9,
                                  xtol=1e-14) for sg in (sigma_1, sigma_2))
        assert t2 < t1, "the marginal top applicant must fall"
        ux, uy = uquant(q, t2, sigma_2), uquant(q, t1, sigma_1)   # after, before
        assert abs(ux[0]) < 1e-12 and abs(uy[0]) < 1e-12, \
            "the marginal applicant has zero standardized residual life either way"
        gap = ux - uy
        flip = np.where(np.sign(gap[1:-1])[1:] != np.sign(gap[1:-1])[:-1])[0]
        assert len(flip) == 1, f"{G.tag}: expected one crossing, got {len(flip)}"
        jq = flip[0] + 1
        qs = q[jq] + (q[jq + 1] - q[jq]) * abs(gap[jq]) / (abs(gap[jq]) + abs(gap[jq + 1]))
        us = float(np.interp(qs, q, ux))
        # convex (concave) mills ratio: the QQ plot is convex (concave) and leaves
        # the origin below (above) the 45 degree line
        slope = np.diff(uy) / np.diff(ux)
        assert (np.all(np.diff(slope) > 0) if G.curvature == "convex"
                else np.all(np.diff(slope) < 0)), f"{G.tag}: QQ plot not {G.curvature}"
        assert (gap[1] > 0) == (G.curvature == "convex"), \
            f"{G.tag}: residual life transfer runs the wrong way"
        uqq.append(qq_panel(
            f"qqu{'b' if k == 0 else 'c'}", title, ux, uy, us,
            G.curvature == "convex", "$0$",
            ends=("$U_{\\sigma_2,OC}^{-1}(1)$", "$U_{\\sigma_1,OC}^{-1}(1)$"),
            tip_at_centroid=True))
        meta.append((G.kappa, t1, t2, qs, slope[0], ux[-1], uy[-1]))

    cap_u = ("The QQ plots $t\\mapsto U_{\\sigma_1,OC}^{-1}(U_{\\sigma_2,OC}(t))$ of "
             "standardized residual life "
             "$U_{\\sigma,OC}\\sim(\\theta-\\hat{\\theta}^T_{\\sigma,OC})/\\sigma\\mid"
             "\\theta>\\hat{\\theta}^T_{\\sigma,OC}$ under a convex (a) and a concave (b) "
             "mills ratio. Both distributions begin at zero, since the marginal applicant "
             "has no excess quality at either noise level. $G$ is exponential truncated "
             "to $[0,1]$ with density proportional to $e^{-\\kappa\\theta}$, "
             f"$\\kappa=-{kappa:g}$ in (a) and $\\kappa={kappa:g}$ in (b); $F$ standard "
             f"normal, $\\sigma_1={sigma_1:g}$, $\\sigma_2={sigma_2:g}$, "
             f"$v^T={v_T:g}$, $OC={OC:g}$, $\\lambda^T={lambda_T:g}$.")
    env_u = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(uqq[0], "fig:qq_resid_convex", frac=0.49),
        "\\hfill",
        subfigure(uqq[1], "fig:qq_resid_concave", frac=0.49),
        # the curve is the same map in both panels: one legend line serves both
        "\\par\\medskip",
        "{\\footnotesize\\tikz[baseline=-0.6ex]\\draw[blue,very thick] (0,0) -- (0.7,0);"
        "\\ $t\\mapsto U_{\\sigma_1,OC}^{-1}(U_{\\sigma_2,OC}(t))$\\qquad"
        "\\tikz[baseline=-0.6ex]\\draw[black,thick,dashed] (0,0) -- (0.7,0);"
        "\\ $45^\\circ$ line\\par}",
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {cap_u}",
        "\\caption{QQ Plot: Standardized Residual Life}", "\\label{fig:qq_resid}",
        "\\end{figure}", ""])
    write(fname,
          "% Proposition 12 - QQ plots of standardized residual life, convex and concave.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          + "".join(f"% ({'ab'[i]}) kappa = {kp:+g}: theta^T {a:.4f} -> {b:.4f}, "
                    f"q* = {c:.4f}, slope at 0 = {sl:.3f}, ends at ({xe:.3f}, {ye:.3f})\n"
                    for i, (kp, a, b, c, sl, xe, ye) in enumerate(meta))
          + env_u)


def figure_qq_top(fname="qq_top.tex", OC=0.08, lambda_T=0.40, v_T=1.0,
                  sigma_1=0.4, sigma_2=1.20, u_top=2.8):
    """One rise in noise in the top program, convex mills ratio, two QQ plots.

    (a) winning probabilities, as in panel (a) of qq_pair and the top square of
    qq_diagonal; (b) standardized residual life, from the same primitives.  A
    winning probability is the same increasing function of standardized
    residual life at both noise levels, so the two curves cross the 45 degree
    line at the same rank q*.  Merit is standard normal, so residual life is
    unbounded: panel (b) is cut at u_top and its curve leaves through the top.
    """
    Fbar, Finv = stats.norm.sf, stats.norm.ppf
    z = Finv(1 - OC / v_T)          # every marginal applicant wins with prob OC/v^T
    G = _MeritNormal()

    def uquant(qs, theta_T, sigma):
        """U_T^{-1}(q|sigma,OC): standardized residual life at rank q."""
        u = G.cdf(theta_T)
        return (G.inv(u + (1 - u) * np.asarray(qs)) - theta_T) / sigma

    def rate(theta_T, sigma):
        """T(theta^T | sigma, OC): mean winning probability among applicants."""
        return float(np.dot(_WTS, Fbar(z - uquant(_NODES, theta_T, sigma))))

    t1, t2 = (optimize.brentq(lambda t: rate(t, sg) - lambda_T, G.lo, G.hi, xtol=1e-13)
              for sg in (sigma_1, sigma_2))
    assert t2 < t1, "the marginal top applicant must fall"

    def crossing(q, x, y):
        """The one rank at which the QQ curve meets the 45 degree line."""
        d = x - y
        flips = np.where(np.sign(d)[1:] != np.sign(d)[:-1])[0]
        assert len(flips) == 1, f"expected a single crossing, got {len(flips)}"
        j = flips[0]
        return q[j] + (q[j + 1] - q[j]) * abs(d[j]) / (abs(d[j]) + abs(d[j + 1]))

    # (a) winning probabilities, on the grid of qq_pair
    qw = np.linspace(0.0, 1.0, 241)[1:-1]
    w1, w2 = Fbar(z - uquant(qw, t1, sigma_1)), Fbar(z - uquant(qw, t2, sigma_2))
    qs_w = crossing(qw, w2, w1)
    ws = float(np.interp(qs_w, qw, w2))
    assert np.mean(w1[:20] - w2[:20]) < 0, "the low ranks must gain"

    # (b) standardized residual life, up to the rank whose sigma_1 residual
    # life reaches the top of the frame
    u1 = G.cdf(t1)
    q_top = (G.cdf(t1 + sigma_1 * u_top) - u1) / (1 - u1)
    qu = np.linspace(0.0, q_top, 241)
    ux, uy = uquant(qu, t2, sigma_2), uquant(qu, t1, sigma_1)    # after, before
    assert abs(ux[0]) < 1e-12 and abs(uy[0]) < 1e-12, \
        "the marginal applicant has zero standardized residual life either way"
    assert abs(uy[-1] - u_top) < 1e-9 and ux[-1] < u_top, \
        "the curve must leave through the top of the frame"
    qs_u = crossing(qu, ux, uy)
    assert abs(qs_u - qs_w) < 2e-3, "both panels must cross at the same rank"
    us = float(np.interp(qs_u, qu, ux))
    slope = np.diff(uy) / np.diff(ux)
    assert np.all(np.diff(slope) > 0), "Lemma 1: the residual life QQ plot is convex"
    assert slope[0] < 1, "the residual life QQ plot leaves the origin below the diagonal"
    # the shading stays inside the square of the axes and the top of the frame
    for px, py in ((np.r_[ux[ux <= us], us, 0.0], np.r_[uy[ux <= us], us, 0.0]),
                   (np.r_[ux[ux > us], u_top, us], np.r_[uy[ux > us], u_top, us])):
        assert px.min() >= -1e-9 and px.max() <= u_top + 1e-9 \
            and py.min() >= -1e-9 and py.max() <= u_top + 1e-9, "shading leaves its square"

    start = OC / v_T                              # rank 0 wins with prob OC/v^T
    panel_w = qq_panel(
        "qqtw", "(a) Winning probabilities", np.r_[start, w2], np.r_[start, w1], ws,
        True, "$OC/v^T$",
        ends=("$W_T^{-1}(1|\\sigma_2,OC)$", "$W_T^{-1}(1|\\sigma_1,OC)$"))
    panel_u = qq_panel(
        "qqtu", "(b) Standardized residual life", ux, uy, us, True, "$0$",
        ends=None, cap=u_top, tip_at_centroid=True)

    def legend(curve):
        return ("\\par\\smallskip{\\footnotesize\\tikz[baseline=-0.6ex]\\draw[blue,very thick] "
                f"(0,0) -- (0.6,0);\\ ${curve}$\\par}}")
    cap_c = ("The QQ plots $t\\mapsto W_T^{-1}(W_T(t|\\sigma_2,OC)|\\sigma_1,OC)$ of "
             "winning probabilities (a) and $t\\mapsto U_T^{-1}(U_T(t|\\sigma_2,OC)|"
             "\\sigma_1,OC)$ of standardized residual life (b), for one rise in noise "
             "under a convex mills ratio. A winning probability is the same increasing "
             "function of standardized residual life at both noise levels, so both curves "
             "cross the $45^{\\circ}$ line at the same rank $q^*$. (b) is convex by Lemma "
             "\\ref{lemma:millsratio_curvature}; residual life is unbounded, so its curve "
             "leaves through the top of the frame. $G$ and $F$ standard normal, "
             f"$\\sigma_1={sigma_1:g}$, $\\sigma_2={sigma_2:g}$, $v^T={v_T:g}$, "
             f"$OC={OC:g}$, $\\lambda^T={lambda_T:g}$; (a) is the top square of Figure "
             "\\ref{fig:qq_diag_convex}.")
    caption = ("\\footnotesize Panel (a) shows the QQ plot of the winning probability "
               "distribution in the top program, and panel (b) the QQ plot of the "
               "standardized residual life distribution. In both panels, the $y$ axis "
               "corresponds to the distribution before the rise in noise from $\\sigma_1$, while "
               "the $x$ axis corresponds to the distribution after the rise in noise to "
               "$\\sigma_2>\\sigma_1$. As we can see, higher noise involves a transfer of "
               "winning rents from the best applicants to the worst when the mills ratio is "
               "convex. The opposite is true when the mills ratio is concave.")
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(panel_w + "\n" + legend(
            "t\\mapsto W_T^{-1}(W_T(t|\\sigma_2,OC)|\\sigma_1,OC)"), "fig:qq_convex", frac=0.49),
        "\\hfill",
        subfigure(panel_u + "\n" + legend(
            "t\\mapsto U_T^{-1}(U_T(t|\\sigma_2,OC)|\\sigma_1,OC)"), "fig:qq_resid", frac=0.49),
        "% Longer caption, not typeset:",
        f"% {cap_c}",
        f"\\caption{{{caption}}}", "\\label{fig:qq}", "\\end{figure}", ""])
    write(fname,
          "% Theorem prop:winner_redistribution and Lemma lemma:millsratio_curvature - one rise\n"
          "% in noise, convex mills ratio: QQ plots of winning probabilities (a) and of\n"
          "% standardized residual life (b).  Replaces qq_pair (a) and qq_residual (a).\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          f"% theta^T {t1:+.4f} -> {t2:+.4f}; q* = {qs_w:.4f} (a), {qs_u:.4f} (b); "
          f"slope of (b) at 0 = {slope[0]:.3f}; (b) cut at u = {u_top:g} (rank {q_top:.3f})\n"
          + env)
    print(f"    theta^T {t1:+.4f} -> {t2:+.4f};  q* = {qs_w:.4f} (a), {qs_u:.4f} (b)")


def figure_qq_transfer(fname="qq_transfer.tex", OC=0.08, lambda_T=0.40, v_T=1.0,
                       sigma_1=0.4, sigma_2=1.20, exp_rate=0.6, exp_upper=5.0,
                       v_R=0.40, cost=0.03, v_R_concave=0.12, cost_concave=0.01,
                       qq_diagonal_concave=False):
    """Panels (a) the fall in the marginal top applicant, (b)/(c) the QQ plots.

    Only the merit distribution differs between (b) and (c): OC, lambda^T and
    the two noise levels are held common, so the reversal of the transfer is
    attributable to the curvature of the mills ratio alone.

    qq_diagonal.tex shows only the convex staircase unless qq_diagonal_concave
    is set, which adds the concave one below it.
    """
    Fbar, Finv = stats.norm.sf, stats.norm.ppf
    z = Finv(1 - OC / v_T)          # every marginal applicant wins with prob OC/v^T

    def rate(G, theta_T, sigma):
        """T(theta^T | sigma, OC): mean winning probability among applicants."""
        u = G.cdf(theta_T)
        grid = G.inv(u + (1 - u) * _NODES)
        return float(np.dot(_WTS, Fbar(-(grid - theta_T) / sigma + z)))

    def marginal(G, sigma):
        """theta^T_{sigma,OC}: the marginal applicant that restores the payline."""
        return optimize.brentq(lambda t: rate(G, t, sigma) - lambda_T,
                               G.lo, G.hi, xtol=1e-13)

    def quantiles(G, q, theta_T, sigma):
        """W^{-1}_{sigma,OC}(q), winning probability at rank q of the pool."""
        u = G.cdf(theta_T)
        return Fbar(-(G.inv(u + (1 - u) * np.asarray(q)) - theta_T) / sigma + z)

    convex, concave = _MeritNormal(), _MeritTruncExp(exp_rate, exp_upper)
    q = np.linspace(0.0, 1.0, 241)[1:-1]
    panels, info = [], []
    for G in (convex, concave):
        # the merit distribution has the curvature the panel claims
        tm = np.linspace(G.lo + 0.3, G.hi - 0.3, 400)
        r = G.mills(tm)
        assert np.all(np.diff(r) < 0), f"{G.tag}: mills ratio not decreasing"
        d2 = np.gradient(np.gradient(r, tm), tm)[8:-8]
        assert (np.all(d2 > 0) if G.curvature == "convex" else np.all(d2 < 0)), \
            f"{G.tag}: mills ratio is not {G.curvature}"
        t1, t2 = marginal(G, sigma_1), marginal(G, sigma_2)
        assert t2 < t1, "the marginal top applicant must fall"
        w1, w2 = quantiles(G, q, t1, sigma_1), quantiles(G, q, t2, sigma_2)
        # both pools deliver the payline on average, so the QQ curve must cross
        assert abs(rate(G, t1, sigma_1) - lambda_T) < 1e-9
        assert abs(rate(G, t2, sigma_2) - lambda_T) < 1e-9
        d = w1 - w2
        flips = np.where(np.sign(d)[1:] != np.sign(d)[:-1])[0]
        assert len(flips) == 1, f"{G.tag}: expected a single crossing, got {len(flips)}"
        j = flips[0]
        qs = q[j] + (q[j + 1] - q[j]) * abs(d[j]) / (abs(d[j]) + abs(d[j + 1]))
        ws = float(np.interp(qs, q, w2))
        # convex: the low ranks gain and the high ranks lose; concave reverses it
        lo_gain = d[:20].mean() < 0
        assert lo_gain == (G.curvature == "convex"), \
            f"{G.tag}: transfer runs the wrong way"
        info.append(dict(G=G, t1=t1, t2=t2, w1=w1, w2=w2, qs=qs, ws=ws))

    # ---- figure 1 of 2: the fall in the marginal top applicant ---------------
    G = convex
    a = info[0]
    span = np.linspace(a["t2"] - 1.1, a["t1"] + 1.7, 61)
    T1 = np.array([rate(G, t, sigma_1) for t in span])
    T2 = np.array([rate(G, t, sigma_2) for t in span])
    assert np.all(np.diff(T1) < 0) and np.all(np.diff(T2) < 0), "T must decrease"
    assert np.all(T2 < T1), "higher noise must lower the acceptance rate"
    xmin_a, xmax_a = span[0] - 0.15, span[-1] + 1.8
    ymax_a = round(T1[0] + 0.06, 2)
    depth_a = -0.02 * ymax_a / 0.55
    style_a = f"""\\pgfplotsset{{qqaxis/.style={{
  width=14.8cm, height=6.0cm,
  xmin={xmin_a:.3f}, xmax={xmax_a:.3f}, ymin=0, ymax={ymax_a},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=7pt]}},black}},
  clip=false, axis on top,
  xlabel={{$\\hat{{\\theta}}^T$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=10pt}},
  tick label style={{font=\\small}}, label style={{font=\\small}},
  title style={{font=\\small}},
  ytick=\\empty, xtick=\\empty, tick style={{black}},
}}}}"""
    lab_a = span[-1] + 0.12
    body_a = "\n".join([
        "\\begin{tikzpicture}",
        "\\begin{axis}[qqaxis, title={the payline is restored by a lower marginal "
        "top applicant}]",
        f"  \\draw[gray,dashed] (axis cs:{xmin_a:.3f},{lambda_T:.4f}) -- "
        f"(axis cs:{span[-1]:.4f},{lambda_T:.4f});",
        f"  \\node[anchor=east,font=\\small] at (axis cs:{xmin_a - 0.05:.3f},"
        f"{lambda_T:.4f}) {{$\\lambda^T$}};",
        *[f"  \\draw[gray,dashed] (axis cs:{t:.4f},0) -- (axis cs:{t:.4f},{lambda_T:.4f});"
          for t in (a["t2"], a["t1"])],
        *[f"  \\node[anchor=north,font=\\small] at (axis cs:{t:.4f},{depth_a:.4f}) {{{lab}}};"
          for t, lab in ((a["t2"], "$\\hat{\\theta}^T_{\\sigma_2,OC}$"),
                         (a["t1"], "$\\hat{\\theta}^T_{\\sigma_1,OC}$"))],
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(span, T1)}}};",
        f"  \\addplot[blue,very thick,densely dashed,smooth] coordinates "
        f"{{{coords(span, T2)}}};",
        shift_arrow("blue", a["t1"] + 0.9, float(np.interp(a["t1"] + 0.9, span, T1)),
                    float(np.interp(a["t1"] + 0.9, span, T2))),
        f"  \\node[blue,anchor=west,font=\\small] at (axis cs:{lab_a:.4f},{T1[-1]:.4f}) "
        f"{{$\\hat{{\\theta}}^T\\mapsto\\mathcal{{T}}(\\hat{{\\theta}}^T|\\sigma_1,OC)$}};",
        f"  \\node[blue,anchor=west,font=\\small] at (axis cs:{lab_a:.4f},{T2[-1]:.4f}) "
        f"{{$\\hat{{\\theta}}^T\\mapsto\\mathcal{{T}}(\\hat{{\\theta}}^T|\\sigma_2,OC)$}};",
        "\\end{axis}", "\\end{tikzpicture}"])
    cap_a = ("Higher evaluation noise lowers the top program acceptance rate at every "
             "marginal top applicant, so the payline is met at a lower merit marginal "
             "top applicant.")
    env_a = "\n".join([
        "\\begin{figure}[htbp]", "\\centering", style_a, "", body_a,
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {cap_a}",
        "\\caption{}", "\\label{fig:payline_restore}", "\\end{figure}", ""])
    write("payline_restore.tex",
          "% The fall in the marginal top applicant that restores the payline.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          f"% theta^T {a['t1']:+.4f} -> {a['t2']:+.4f} at lambda^T = {lambda_T:g}\n"
          + env_a)

    # ---- figure 2 of 2: the QQ plots -----------------------------------------
    qq = []
    for k, (d, title) in enumerate(zip(info, (
            "(a) Convex mills ratio", "(b) Concave mills ratio"))):
        start = OC / v_T                          # rank 0 wins with prob OC/v^T
        qq.append(qq_panel(
            f"qq{'b' if k == 0 else 'c'}", title,
            np.r_[start, d["w2"]], np.r_[start, d["w1"]],   # x = after, y = before
            d["ws"], d["G"].curvature == "convex", "$OC/v^T$",
            ends=("$W_T^{-1}(1|\\sigma_2,OC)$", "$W_T^{-1}(1|\\sigma_1,OC)$")))

    shared = ("Both begin at $OC/v^T$, the winning probability of the marginal "
              "applicant, which is common to both noise levels. Only the merit "
              "distribution differs between this figure and its companion: $OC$, "
              "$\\lambda^T$, $\\sigma_1$ and $\\sigma_2$ are common. $F$ standard "
              f"normal, $\\sigma_1={sigma_1:g}$, $\\sigma_2={sigma_2:g}$; "
              f"$v^T={v_T:g}$, $OC={OC:g}$, $\\lambda^T={lambda_T:g}$.")
    caps = [
        ("The QQ plot $t\\mapsto W_T^{-1}(W_T(t|\\sigma_2,OC)|\\sigma_1,OC)$ of winning "
         "probabilities before and after the rise in evaluation noise, under a convex "
         "mills ratio. The curve leaves the $45^{\\circ}$ line from below and single "
         "crosses it at the pivotal rank $q^*$, so applicants ranked below $q^*$ gain "
         "and those above lose. $G$ standard normal. " + shared),
        ("The same QQ plot under a concave mills ratio: the curve leaves the "
         "$45^{\\circ}$ line from above and crosses it from above at $q^*$, so the "
         "transfer runs the other way. $G$ right-truncated exponential with rate "
         f"${exp_rate:g}$ on $[0,{exp_upper:g}]$. " + shared)]
    pair = ("The QQ plots $t\\mapsto W_T^{-1}(W_T(t|\\sigma_2,OC)|\\sigma_1,OC)$ "
            "under a convex (a) and a concave (b) mills ratio. " + caps[0].split(". ", 1)[1])
    env_p = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(qq[0], "fig:qq_convex", frac=0.49),
        "\\hfill",
        subfigure(qq[1], "fig:qq_concave", frac=0.49),
        # the curve is the same map in both panels: one legend line serves both
        "\\par\\medskip",
        "{\\footnotesize\\tikz[baseline=-0.6ex]\\draw[blue,very thick] (0,0) -- (0.7,0);"
        "\\ $t\\mapsto W_T^{-1}(W_T(t|\\sigma_2,OC)|\\sigma_1,OC)$\\qquad"
        "\\tikz[baseline=-0.6ex]\\draw[black,thick,dashed] (0,0) -- (0.7,0);"
        "\\ $45^\\circ$ line\\par}",
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {pair}",
        "\\caption{\\footnotesize Both figures show the QQ pot of the winning probability "
        "distribution in the top program. The $y$ axis corresponds to the distribution "
        "before the rise in noise $\\sigma_1$, while the $x$ axis corresponds to the "
        "distribution after the rise in noise to $\\sigma_2>\\sigma_1$. As we can see, "
        "higher noise involves a transfer of winning rents from the best applicants to the "
        "worst when the mills ratio is convex. The opposite is true when the mills ratio "
        "is concave.}",
        "\\label{fig:qq}", "\\end{figure}", ""])
    write("qq_pair.tex",
          "% Proposition 12 - QQ plots of winning probabilities, convex and concave.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          f"% (a) theta^T {info[0]['t1']:+.4f} -> {info[0]['t2']:+.4f}, q* = {info[0]['qs']:.4f}\n"
          f"% (b) theta^T {info[1]['t1']:+.4f} -> {info[1]['t2']:+.4f}, q* = {info[1]['qs']:.4f}\n"
          + env_p)

    # ---- figure 3 of 3: the QQ plots in standardized residual life ----------
    # drawn on primitives of their own, chosen for readability
    figure_qq_residual()
    c = info[0]
    Gc = c["G"]

    # ---- figure 4 of 4: both programs strung on one 45 degree line -----------
    # Each QQ plot carries the same reference line, so the two squares can be
    # placed on a single diagonal: the regular program below left, the top
    # program above right.  Drawn once per merit distribution and set on one
    # float page: each title sits in an empty quadrant of its own staircase, so
    # each picture is no taller than its two squares.
    B, SIDE = 0.4, 5.0
    b1 = (B, B + SIDE)                            # regular, below left
    b2 = (B + SIDE, B + 2 * SIDE)                 # top program, above right
    assert abs(b1[1] - b2[0]) < 1e-9, "the squares must share the corner"
    D = 0.75                                      # leader length along each axis

    def box_map(v, rng, box):
        lo_r, hi_r = rng
        return box[0] + (np.asarray(v) - lo_r) / (hi_r - lo_r) * (box[1] - box[0])

    def leader(x, ymid, ycurve, below, box, colour, text):
        """Leader from inside a region to its label, on the region's own side of
        the diagonal: below the line goes down-right, above goes up-left."""
        if not below:
            if x - D - 1.0 > box[0]:
                # up-left, but no higher than just under the top guide
                ly = min(ymid + D, box[1] - 0.4)
                return [f"  \\draw[{colour},thin] ({x - D:.2f},{ly:.2f}) -- ({x:.2f},{ymid:.2f});",
                        f"  \\node[{colour},anchor=east,font=\\small] at "
                        f"({x - D - 0.03:.2f},{ly + 0.03:.2f}) {{{text}}};"]
            # a region by the vertical axis has no room to its left inside the
            # box: lift the label straight above the curve
            return [f"  \\draw[{colour},thin] ({x:.2f},{ycurve + 1.0:.2f}) -- ({x:.2f},{ymid:.2f});",
                    f"  \\node[{colour},anchor=south,font=\\small] at "
                    f"({x:.2f},{ycurve + 1.03:.2f}) {{{text}}};"]
        if x + D + 1.0 < box[1]:
            ly = max(ymid - D, box[0] + 0.4)
            return [f"  \\draw[{colour},thin] ({x + D:.2f},{ly:.2f}) -- ({x:.2f},{ymid:.2f});",
                    f"  \\node[{colour},anchor=west,font=\\small] at "
                    f"({x + D + 0.03:.2f},{ly - 0.03:.2f}) {{{text}}};"]
        # a region at the top corner has no room to its right inside the box:
        # drop the label straight below the curve
        return [f"  \\draw[{colour},thin] ({x:.2f},{ycurve - 1.0:.2f}) -- ({x:.2f},{ymid:.2f});",
                f"  \\node[{colour},anchor=north,font=\\small] at "
                f"({x:.2f},{ycurve - 1.03:.2f}) {{{text}}};"]

    def axed(box, curve_x, curve_y, marks, guides, extras=()):
        """One QQ box: axes, dashed guides to both of them, and the curve."""
        a0, a1 = box
        out = [
            f"  \\draw[-{{Stealth[length=6pt]}},black] ({a0:.2f},{a0:.2f}) -- "
            f"({a1 + 0.45:.2f},{a0:.2f});",
            f"  \\draw[-{{Stealth[length=6pt]}},black] ({a0:.2f},{a0:.2f}) -- "
            f"({a0:.2f},{a1 + 0.45:.2f});",
            f"  \\node[anchor=west,font=\\small] at ({a1 + 0.55:.2f},{a0:.2f}) {{$t$}};"]
        for gx, gy, xlab, ylab in guides:
            if abs(gx - a0) < 1e-9 and abs(gy - a0) < 1e-9:
                # the point sits on the axes corner, so one label serves both
                out.append(f"  \\node[anchor=north west,font=\\small] at "
                           f"({gx + 0.12:.2f},{gy - 0.06:.2f}) {{{xlab}}};")
                continue
            out.append(f"  \\draw[gray,dashed] ({gx:.2f},{a0:.2f}) -- "
                       f"({gx:.2f},{gy:.2f}) -- ({a0:.2f},{gy:.2f});")
            out.append(f"  \\node[anchor=north,font=\\small] at ({gx:.2f},"
                       f"{a0 - 0.12:.2f}) {{{xlab}}};")
            out.append(f"  \\node[anchor=east,font=\\small] at ({a0 - 0.12:.2f},"
                       f"{gy:.2f}) {{{ylab}}};")
        out.append(f"  \\draw[blue,very thick] plot[smooth] coordinates "
                   f"{{{coords(curve_x, curve_y)}}};")
        return out + marks + list(extras)

    def staircase(d, v_R, cost):
        """Regular and top program QQ plots on one diagonal, for one G."""
        G = d["G"]
        assert cost < OC < v_R, "need c < OC < v^R"
        z_R = Finv(1 - cost / v_R)
        kappa = z_R - Finv(1 - OC / v_R)
        cvr, ocvr = cost / v_R, OC / v_R

        def wreg(qs, theta_T, sigma):
            """Winning probabilities among regular applicants, by rank."""
            thR = theta_T - sigma * kappa
            th = G.between(thR, theta_T, qs)
            return Fbar(-(th - thR) / sigma + z_R)

        qd = np.linspace(0.0, 1.0, 81)
        r_after = wreg(qd, d["t2"], sigma_2)      # horizontal
        r_before = wreg(qd, d["t1"], sigma_1)     # vertical
        assert abs(r_before[0] - cvr) < 1e-9 and abs(r_after[0] - cvr) < 1e-9, \
            "rank 0 wins with probability c/v^R at either noise level"
        assert abs(r_before[-1] - ocvr) < 1e-9 and abs(r_after[-1] - ocvr) < 1e-9, \
            "rank 1 wins with probability OC/v^R at either noise level"
        convex = G.curvature == "convex"
        # Proposition 13: a convex (concave) mills ratio raises (lowers) every
        # regular applicant's chances
        rgap = r_after - r_before
        assert (np.all(rgap >= -1e-12) if convex else np.all(rgap <= 1e-12)), \
            f"{G.tag}: the regular program does not move one way"
        assert np.abs(rgap).max() > 0.08 * (ocvr - cvr), "the lens is too thin to read"

        # Each square carries its own winning-probability scale, and they are
        # butted together at the marginal top applicant: the regular curve ends
        # at OC/v^R, the top curve begins at OC/v^T.  The same researcher,
        # indifferent between the programs, so the boxes share that corner.
        rng1 = (cvr, ocvr)
        rx, ry = box_map(r_after, rng1, b1), box_map(r_before, rng1, b1)
        reg_poly = coords(np.r_[rx, b1[1], b1[0]], np.r_[ry, b1[1], b1[0]])
        jr = int(np.argmax(np.abs(rx - ry)))
        r_fill = "darkgreen!20" if convex else "red!18"
        r_txt = "darkgreen" if convex else "red!70!black"

        # top program: begins at OC/v^T and crosses the diagonal at q*
        tw1, tw2 = d["w1"], d["w2"]
        rng2 = (OC / v_T, max(tw1.max(), tw2.max()))
        tx = box_map(np.r_[OC / v_T, tw2], rng2, b2)
        ty = box_map(np.r_[OC / v_T, tw1], rng2, b2)
        ws_c = float(box_map(d["ws"], rng2, b2))
        cap = min(tx[-1], ty[-1])
        mt = tx <= ws_c
        t_first = coords(np.r_[tx[mt], ws_c, b2[0]], np.r_[ty[mt], ws_c, b2[0]])
        t_second = coords(np.r_[tx[~mt], cap, ws_c], np.r_[ty[~mt], cap, ws_c])
        # the shading stays inside the square of its axes and dashed guides
        for px, py, (xl, xh, yl, yh) in (
                (np.r_[rx, b1[1], b1[0]], np.r_[ry, b1[1], b1[0]], (b1[0], b1[1], b1[0], b1[1])),
                (np.r_[tx[mt], ws_c, b2[0]], np.r_[ty[mt], ws_c, b2[0]],
                 (b2[0], tx[-1], b2[0], ty[-1])),
                (np.r_[tx[~mt], cap, ws_c], np.r_[ty[~mt], cap, ws_c],
                 (b2[0], tx[-1], b2[0], ty[-1]))):
            assert (px.min() >= xl - 1e-9 and px.max() <= xh + 1e-9
                    and py.min() >= yl - 1e-9 and py.max() <= yh + 1e-9), \
                f"{G.tag}: shading leaves its square"
        diag = np.minimum(tx, cap)
        ja = int(np.argmax(np.where(mt, np.abs(diag - ty), -1.0)))
        jb = int(np.argmax(np.where(~mt, np.abs(diag - ty), -1.0)))
        f_fill, s_fill = ("darkgreen!20", "red!18") if convex else ("red!18", "darkgreen!20")
        f_txt, s_txt = ("darkgreen", "red!70!black") if convex else ("red!70!black", "darkgreen")
        f_lab, s_lab = ("gain", "loss") if convex else ("loss", "gain")

        body = "\n".join([
            "\\begin{tikzpicture}[x=1cm,y=1cm]",
            "  % the gains and the losses, beneath the 45 degree line",
            # straight segments: the samples are dense, and smoothing would
            # round the polygons' corners past the dashed guides
            f"  \\fill[{r_fill}] plot coordinates {{{reg_poly}}};",
            f"  \\fill[{f_fill}] plot coordinates {{{t_first}}};",
            f"  \\fill[{s_fill}] plot coordinates {{{t_second}}};",
            "  % the 45 degree line runs through both squares and stops where it",
            "  % leaves the top square built by the two q = 1 quantiles",
            f"  \\draw[black,dashed,thick] ({b1[0]:.2f},{b1[0]:.2f}) -- ({cap:.2f},{cap:.2f});",
            f"  \\node[anchor={'north west' if convex else 'south east'},font=\\small] at "
            f"({cap:.2f},{cap:.2f}) {{$45^{{\\circ}}$}};",
            "  % --- regular program ---",
            *axed(b1, rx, ry,
                  [f"  \\fill[blue] ({b1[0]:.2f},{b1[0]:.2f}) circle (1.8pt) "
                   f"({b1[1]:.2f},{b1[1]:.2f}) circle (1.8pt);"],
                  [(b1[0], b1[0], "$c/v^R$", "$c/v^R$"),
                   (b1[1], b1[1], "$OC/v^R$", "$OC/v^R$")],
                  extras=leader(rx[jr], 0.5 * (rx[jr] + ry[jr]), ry[jr], rx[jr] > ry[jr],
                                b1, r_txt, "gain" if convex else "loss")),
            "  % the title sits in the empty quadrant to the right of its square",
            f"  \\node[anchor=west,font=\\small,align=center] at "
            f"({b1[1] + 0.35:.2f},{b1[0] + 0.55 * SIDE:.2f}) "
            f"{{Regular Program\\\\\\textcolor{{blue}}{{$t\\mapsto W_R^{{-1}}"
            f"(W_R(t|\\sigma_2,OC)|\\sigma_1,OC)$}}}};",
            "  % --- top program ---",
            *axed(b2, tx, ty,
                  [f"  \\fill[blue] ({b2[0]:.2f},{b2[0]:.2f}) circle (1.8pt) "
                   f"({ws_c:.2f},{ws_c:.2f}) circle (1.8pt);",
                   f"  \\node[anchor=north west,font=\\small] at ({ws_c + 0.1:.2f},"
                   f"{ws_c:.2f}) {{$q^{{*}}$}};"],
                  [(b2[0], b2[0], "$OC/v^T$", "$OC/v^T$"),
                   (float(tx[-1]), float(ty[-1]),
                    "$W_T^{-1}(1|\\sigma_2,OC)$", "$W_T^{-1}(1|\\sigma_1,OC)$")],
                  extras=leader(tx[ja], 0.5 * (ty[ja] + diag[ja]), ty[ja], diag[ja] > ty[ja],
                                b2, f_txt, f_lab)
                  + leader(tx[jb], 0.5 * (ty[jb] + diag[jb]), ty[jb], diag[jb] > ty[jb],
                           b2, s_txt, s_lab)),
            "  % the title sits in the empty quadrant to the left of its square",
            f"  \\node[anchor=east,font=\\small,align=center] at "
            f"({b2[0] - 0.35:.2f},{b2[0] + 0.45 * SIDE:.2f}) "
            f"{{Top Program\\\\\\textcolor{{blue}}{{$t\\mapsto W_T^{{-1}}"
            f"(W_T(t|\\sigma_2,OC)|\\sigma_1,OC)$}}}};",
            "\\end{tikzpicture}"])
        return body, (cvr, ocvr, rgap)

    # v^R and c enter only the regular square; the concave case takes values
    # that give its lens a readable width.  Each top square is its panel of
    # Figure qq_pair, unchanged.
    body_cx, st_cx = staircase(info[0], v_R, cost)
    body_cc, st_cc = staircase(info[1], v_R_concave, cost_concave)
    cap_d = ("Both QQ plots carry the same $45^{\\circ}$ reference line, so the two "
             "squares sit on a single diagonal. Below left, the regular program: the "
             "curve is anchored to the diagonal at both ends, since the marginal "
             "regular applicant wins with probability $c/v^R$ and the marginal top "
             "applicant with probability $OC/v^R$ whatever the noise, and it lies "
             "strictly below (above) in between, so under a convex (concave) mills "
             "ratio every regular applicant gains (loses). Above right, the top "
             "program: the curve crosses the diagonal at the pivotal rank $q^*$.")
    concave = [
        "\\par\\vspace{1.5\\baselineskip}",
        body_cc,
        f"% Concave: $G$ as in Figure \\ref{{fig:qq_concave}}, extended below 0 to "
        f"hold the regular pool; $v^R={v_R_concave:g}$, $c={cost_concave:g}$.",
        "\\caption{Concave Mills Ratio}", "\\label{fig:qq_diag_concave}"]
    env_d = "\n".join([
        "\\begin{figure}[p]", "\\centering", body_cx,
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {cap_d} Convex: $v^R={v_R:g}$, $c={cost:g}$.",
        "\\caption{Convex Mills Ratio}", "\\label{fig:qq_diag_convex}",
        *(concave if qq_diagonal_concave else []),
        "\\end{figure}", ""])
    write("qq_diagonal.tex",
          "% Regular and top program QQ plots on one 45 degree line, convex and concave.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          + "".join(f"% {nm}: regular winning probabilities {a:.4f} to {b:.4f}, "
                    f"largest move {g[np.argmax(np.abs(g))]:+.4f}\n"
                    for nm, (a, b, g) in (("convex ", st_cx), ("concave", st_cc)))
          + env_d)

    for d, lab in zip(info, ("convex ", "concave")):
        print(f"    {lab}: theta^T {d['t1']:+.4f} -> {d['t2']:+.4f};  q* = {d['qs']:.4f}"
              f"  (winning prob {d['ws']:.4f})")



# --------------------------------------------------------------------------
# The staircase of Figure qq_diagonal, after the equilibrium is re-achieved
# --------------------------------------------------------------------------
def figure_qq_equilibrium(fname="qq_equilibrium.tex", n=121):
    """Regular and top program QQ plots on one diagonal, comparing the stable
    equilibrium at sigma_1 (opportunity cost OC_1) with the stable equilibrium
    reached at sigma_2 (OC_2), under a convex and a concave mills ratio.

    Unlike Figure qq_diagonal, the opportunity cost is not held fixed.  The
    marginal top applicant wins with probability OC/v^T in the top program and
    OC/v^R in the regular one, so the top curve starts, and the regular curve
    ends, off the 45 degree line.  Each program still meets its payline before
    and after, so each curve crosses the line; here it crosses it once.  The
    primitives are those of Figure oc_noise_shift (convex) and of panel (b) of
    Figure field_pair (concave).
    """
    B, SIDE = 0.4, 5.0
    b1 = (B, B + SIDE)                            # regular, below left
    b2 = (B + SIDE, B + 2 * SIDE)                 # top program, above right
    D = 0.75                                      # leader length along each axis
    q = np.linspace(0.0, 1.0, n)

    def box_map(v, rng, box):
        lo_r, hi_r = rng
        return box[0] + (np.asarray(v) - lo_r) / (hi_r - lo_r) * (box[1] - box[0])

    def stable_crossing(mm):
        """The first crossing of lambda^R from below, stable by Definition 3."""
        def slope(d):
            return ((mm.R_of_delta(d + 1e-4) - mm.R_of_delta(d - 1e-4))
                    / (mm.oc_of_delta(d + 1e-4) - mm.oc_of_delta(d - 1e-4)))
        ups = [d for d in mm.delta_equilibria() if slope(d) > 0]
        assert ups, "the curve must cross the payline from below"
        assert all(def3_stable(mm, ups[0])), "the equilibrium must be stable"
        return ups[0]

    def winning(mm, d):
        """Winning probabilities by rank: top pool, then regular pool."""
        tT = mm.theta_T_of_delta(d)
        tR = tT - mm.sigma * d
        u0, uR = mm.Gcdf(tT), mm.Gcdf(tR)
        top = mm.Fbar(mm.z_T(d) - (mm.Ginv(u0 + (1 - u0) * q) - tT) / mm.sigma)
        reg = mm.Fbar(mm.z_R - (mm.Ginv(uR + (u0 - uR) * q) - tR) / mm.sigma)
        return top, reg, tT, tR

    def regions(x, y, cap):
        """Polygons between the curve and the 45 degree line, split where they
        cross, each with its side: True below the line (a gain at that rank)."""
        g = x - y                                 # > 0 where the curve is below
        # a curve anchored on the line starts at g = 0 up to rounding
        s = np.sign(np.where(np.abs(g) < 1e-9 * (x.max() - x.min()), 0.0, g))
        cuts = np.where(s[:-1] * s[1:] < 0)[0]
        cross = [float(x[j] + (x[j + 1] - x[j]) * g[j] / (g[j] - g[j + 1])) for j in cuts]
        bounds = [0] + [j + 1 for j in cuts] + [len(x)]
        polys = []
        for k in range(len(bounds) - 1):
            seg = slice(bounds[k], bounds[k + 1])
            # the first region closes along the rank 0 guide, the last along the
            # rank 1 guide back to the capped 45 degree line
            head = [(x[0], x[0])] if k == 0 else [(cross[k - 1], cross[k - 1])]
            tail = [(cross[k], cross[k])] if k < len(cross) else [(cap, cap)]
            pts = head + list(zip(x[seg], y[seg])) + tail
            polys.append((np.array([p[0] for p in pts]), np.array([p[1] for p in pts]),
                          bool(g[seg].mean() > 0), seg))
        return polys, cross

    def leader(x, ymid, ycurve, below, box, colour, text):
        """As in Figure qq_diagonal: a leader to the region's own side of the
        diagonal, below the line down-right, above the line up-left."""
        if not below:
            if x - D - 1.0 > box[0]:
                ly = min(ymid + D, box[1] - 0.4)
                return [f"  \\draw[{colour},thin] ({x - D:.2f},{ly:.2f}) -- ({x:.2f},{ymid:.2f});",
                        f"  \\node[{colour},anchor=east,font=\\small] at "
                        f"({x - D - 0.03:.2f},{ly + 0.03:.2f}) {{{text}}};"]
            return [f"  \\draw[{colour},thin] ({x:.2f},{ycurve + 1.0:.2f}) -- ({x:.2f},{ymid:.2f});",
                    f"  \\node[{colour},anchor=south,font=\\small] at "
                    f"({x:.2f},{ycurve + 1.03:.2f}) {{{text}}};"]
        if x + D + 1.0 < box[1]:
            ly = max(ymid - D, box[0] + 0.4)
            return [f"  \\draw[{colour},thin] ({x + D:.2f},{ly:.2f}) -- ({x:.2f},{ymid:.2f});",
                    f"  \\node[{colour},anchor=west,font=\\small] at "
                    f"({x + D + 0.03:.2f},{ly - 0.03:.2f}) {{{text}}};"]
        return [f"  \\draw[{colour},thin] ({x:.2f},{ycurve - 1.0:.2f}) -- ({x:.2f},{ymid:.2f});",
                f"  \\node[{colour},anchor=north,font=\\small] at "
                f"({x:.2f},{ycurve - 1.03:.2f}) {{{text}}};"]

    def square(box, x, y, start_lab, end_lab, title, title_at):
        """One QQ square: shading, capped 45 degree line, guides, curve, labels."""
        a0 = box[0]
        cap = float(min(x[-1], y[-1]))
        polys, cross = regions(x, y, cap)
        assert len(cross) == 1, f"expected a single crossing, got {len(cross)}"
        out = ["  % the gains and the losses, beneath the 45 degree line"]
        for px, py, below, _ in polys:
            # the shading stays inside the square of its axes and dashed guides
            assert (px.min() >= a0 - 1e-9 and px.max() <= x[-1] + 1e-9
                    and py.min() >= a0 - 1e-9 and py.max() <= y[-1] + 1e-9), \
                "shading leaves its square"
            out.append(f"  \\fill[{'darkgreen!20' if below else 'red!18'}] plot "
                       f"coordinates {{{coords(px, py)}}};")
        out += [
            "  % the 45 degree line stops where it leaves the square of the guides",
            f"  \\draw[black,dashed,thick] ({a0:.2f},{a0:.2f}) -- ({cap:.2f},{cap:.2f});",
            f"  \\draw[-{{Stealth[length=6pt]}},black] ({a0:.2f},{a0:.2f}) -- "
            f"({box[1] + 0.45:.2f},{a0:.2f});",
            f"  \\draw[-{{Stealth[length=6pt]}},black] ({a0:.2f},{a0:.2f}) -- "
            f"({a0:.2f},{box[1] + 0.45:.2f});",
            f"  \\node[anchor=west,font=\\small] at ({box[1] + 0.55:.2f},{a0:.2f}) {{$t$}};"]
        for gx, gy, (xlab, ylab) in ((x[0], y[0], start_lab), (x[-1], y[-1], end_lab)):
            if abs(gx - a0) < 1e-9 and abs(gy - a0) < 1e-9:
                # the point sits on the axes corner, so one label serves both
                out.append(f"  \\node[anchor=north west,font=\\small] at "
                           f"({gx + 0.12:.2f},{gy - 0.06:.2f}) {{{xlab}}};")
                continue
            out += [f"  \\draw[gray,dashed] ({gx:.2f},{a0:.2f}) -- ({gx:.2f},{gy:.2f}) "
                    f"-- ({a0:.2f},{gy:.2f});",
                    f"  \\node[anchor=north,font=\\small] at ({gx:.2f},{a0 - 0.12:.2f}) {{{xlab}}};",
                    f"  \\node[anchor=east,font=\\small] at ({a0 - 0.12:.2f},{gy:.2f}) {{{ylab}}};"]
        out += [f"  \\draw[blue,very thick] plot[smooth] coordinates {{{coords(x, y)}}};",
                f"  \\fill[blue] ({x[0]:.2f},{y[0]:.2f}) circle (1.8pt) "
                f"({x[-1]:.2f},{y[-1]:.2f}) circle (1.8pt) "
                f"({cross[0]:.2f},{cross[0]:.2f}) circle (1.8pt);",
                f"  \\node[anchor={'north west' if polys[0][2] else 'south east'},font=\\small] "
                f"at ({cross[0] + (0.1 if polys[0][2] else -0.1):.2f},{cross[0]:.2f}) {{$q^{{*}}$}};"]
        for px, py, below, _ in polys:
            # each leader points at the centroid of its region (shoelace formula):
            # the widest point of a region can sit on an axis
            cr = px * np.roll(py, -1) - np.roll(px, -1) * py
            area = cr.sum() / 2
            cx = ((px + np.roll(px, -1)) * cr).sum() / (6 * area)
            cy = ((py + np.roll(py, -1)) * cr).sum() / (6 * area)
            colour = "darkgreen" if below else "red!70!black"
            out += leader(cx, cy, float(np.interp(cx, x, y)), below, box, colour,
                          "gain" if below else "loss")
        out.append(title_at + f" {{{title}}};")
        return out, cross[0], [p[2] for p in polys]

    def staircase(m, sigma_2, curvature):
        m2 = replace(m, sigma=sigma_2)
        tm = np.linspace(m.theta_lo + 0.05 * m._width, m.theta_hi - 0.05 * m._width, 400)
        r = m.mills(tm)
        assert np.all(np.diff(r) < 0), "the mills ratio must be decreasing"
        assert (np.all(np.diff(r, 2) > -1e-12) if curvature == "convex"
                else np.all(np.diff(r, 2) < 1e-12)), f"the mills ratio is not {curvature}"
        d1, d2 = stable_crossing(m), stable_crossing(m2)
        oc_1, oc_2 = m.oc_of_delta(d1), m2.oc_of_delta(d2)
        assert (oc_2 < oc_1) if curvature == "convex" else (oc_2 > oc_1), \
            "the opportunity cost moves the wrong way"
        top1, reg1, tT1, tR1 = winning(m, d1)
        top2, reg2, tT2, tR2 = winning(m2, d2)
        vR, cvr = m.v_ratio, m.cost_ratio
        # rank 0 of the top pool and rank 1 of the regular pool are the marginal
        # top applicant: OC/v^T and OC/v^R, now different before and after
        assert abs(top1[0] - oc_1) < 1e-9 and abs(top2[0] - oc_2) < 1e-9
        assert abs(reg1[-1] - oc_1 / vR) < 1e-9 and abs(reg2[-1] - oc_2 / vR) < 1e-9
        assert abs(reg1[0] - cvr) < 1e-9 and abs(reg2[0] - cvr) < 1e-9, \
            "rank 0 of the regular pool wins with probability c/v^R either way"

        # one scale per square, common to both axes so the 45 degree line holds
        rng1 = (cvr, max(reg1[-1], reg2[-1]))
        lo2, hi2 = min(top1[0], top2[0]), max(top1[-1], top2[-1])
        rng2 = (lo2 - 0.06 * (hi2 - lo2), hi2)   # room for the rank 0 guides
        rx, ry = box_map(reg2, rng1, b1), box_map(reg1, rng1, b1)   # after, before
        tx, ty = box_map(top2, rng2, b2), box_map(top1, rng2, b2)
        reg_out, rq, rsides = square(
            b1, rx, ry, ("$c/v^R$", "$c/v^R$"), ("$OC_2/v^R$", "$OC_1/v^R$"),
            "Regular Program\\\\\\textcolor{blue}{$t\\mapsto W_R^{-1}"
            "(W_R(t|\\sigma_2,OC_2)|\\sigma_1,OC_1)$}",
            "  % the title sits in the empty quadrant to the right of its square\n"
            f"  \\node[anchor=west,font=\\small,align=center] at "
            f"({b1[1] + 0.35:.2f},{b1[0] + 0.55 * SIDE:.2f})")
        top_out, tq, tsides = square(
            b2, tx, ty, ("$OC_2/v^T$", "$OC_1/v^T$"),
            ("$W_T^{-1}(1|\\sigma_2,OC_2)$", "$W_T^{-1}(1|\\sigma_1,OC_1)$"),
            "Top Program\\\\\\textcolor{blue}{$t\\mapsto W_T^{-1}"
            "(W_T(t|\\sigma_2,OC_2)|\\sigma_1,OC_1)$}",
            "  % the title sits in the empty quadrant to the left of its square\n"
            f"  \\node[anchor=east,font=\\small,align=center] at "
            f"({b2[0] - 0.35:.2f},{b2[0] + 0.45 * SIDE:.2f})")
        # the 45 degree label goes on the empty side of the line's end: up-left
        # when the curve finishes below it, down-right when above
        cap2 = float(min(tx[-1], ty[-1]))
        body = "\n".join(["\\begin{tikzpicture}[x=1cm,y=1cm]",
                          "  % --- regular program ---", *reg_out,
                          "  % --- top program ---", *top_out,
                          f"  \\node[anchor={'south east' if tsides[-1] else 'north west'},"
                          f"font=\\small] at ({cap2:.2f},{cap2:.2f}) {{$45^{{\\circ}}$}};",
                          "\\end{tikzpicture}"])
        info = dict(oc_1=oc_1, oc_2=oc_2, tT=(tT1, tT2), tR=(tR1, tR2),
                    top_sides=tsides, reg_sides=rsides,
                    q_top=float(np.interp(tq, tx, q)), q_reg=float(np.interp(rq, rx, q)))
        return body, info

    parts = [staircase(FIELD_CONVEX, 2.25, "convex"),
             staircase(FIELD_CONCAVE, 0.60, "concave")]
    caps = []
    for (body, info), (m, s2, curv) in zip(parts, ((FIELD_CONVEX, 2.25, "convex"),
                                                  (FIELD_CONCAVE, 0.60, "concave"))):
        lo_top = "gain" if info["top_sides"][0] else "lose"
        lo_reg = "gain" if info["reg_sides"][0] else "lose"
        caps.append(
            f"QQ plots of winning probabilities at the stable equilibrium before the rise in "
            f"noise ($\\sigma_1={m.sigma:g}$, opportunity cost $OC_1={info['oc_1']:.4f}$) and at "
            f"the stable equilibrium reached after it ($\\sigma_2={s2:g}$, $OC_2={info['oc_2']:.4f}$), "
            f"under a {curv} mills ratio. The opportunity cost is not held fixed, so the top "
            "curve starts, and the regular curve ends, off the $45^{\\circ}$ line. Each "
            "program meets its payline before and after, so each curve crosses the line once: "
            f"low ranked top applicants {lo_top} and low ranked regular applicants {lo_reg}. "
            f"$F$ {m.noise_name}; {m.merit_name}; $\\lambda^T={m.lambda_T:g}$, "
            f"$\\lambda^R={m.lambda_R:g}$, $v^R/v^T={m.v_ratio:g}$, $c/v^R={m.cost_ratio:g}$.")
    env = "\n".join([
        "\\begin{figure}[p]", "\\centering", parts[0][0],
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {caps[0]}",
        "\\caption{Convex Mills Ratio, Before and After the New Equilibrium}",
        "\\label{fig:qq_eq_convex}",
        "\\par\\vspace{1.5\\baselineskip}",
        parts[1][0],
        f"% {caps[1]}",
        "\\caption{Concave Mills Ratio, Before and After the New Equilibrium}",
        "\\label{fig:qq_eq_concave}",
        "\\end{figure}", ""])
    write(fname,
          "% Regular and top program QQ plots, stable equilibrium before and after the\n"
          "% rise in noise (OC not held fixed), convex and concave.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          + "".join(f"% {nm}: OC {i['oc_1']:.4f} -> {i['oc_2']:.4f}; theta^T "
                    f"{i['tT'][0]:+.4f} -> {i['tT'][1]:+.4f}; theta^R {i['tR'][0]:+.4f} -> "
                    f"{i['tR'][1]:+.4f}; q* top {i['q_top']:.3f}, regular {i['q_reg']:.3f}\n"
                    for nm, (_, i) in (("convex ", parts[0]), ("concave", parts[1])))
          + env)
    for nm, (_, i) in (("convex ", parts[0]), ("concave", parts[1])):
        print(f"    {nm}: OC {i['oc_1']:.4f} -> {i['oc_2']:.4f};  q* top {i['q_top']:.3f}, "
              f"regular {i['q_reg']:.3f}")


# --------------------------------------------------------------------------
# Densities of winning probabilities, before and after the rise in noise
# --------------------------------------------------------------------------
def figure_densities(OC=0.30, lambda_T=0.45, v_T=1.0, v_R=0.80, cost=0.12,
                     sigma_1=0.18, sigma_2=0.27, beta_ab=(4, 4)):
    """One figure per noise level, each with the regular and top densities.

    Logistic noise and Beta(4,4) merit on [0,1]: log-concave, with a strictly
    decreasing convex mills ratio.  Since w = Fbar(z - (theta - theta_0)/sigma)
    is increasing in merit, the change of variable gives
        f_W(w) = g(theta(w)) / mass * sigma / f(F^{-1}(1 - w)).
    At these primitives the merit mode (0.5) lies in the regular pool before
    the rise in noise and in the top pool after it: the fall in the marginal
    top applicant moves the hump from one program to the other.
    """
    Fbar, Finv, fpdf = stats.logistic.sf, stats.logistic.ppf, stats.logistic.pdf
    G = stats.beta(*beta_ab)
    assert cost < OC < v_R < v_T, "need c < OC < v^R < v^T"
    t = np.linspace(0.03, 0.97, 400)
    r = G.sf(t) / G.pdf(t)
    assert np.all(np.diff(r) < 0) and np.all(np.diff(r, 2) > -1e-12), \
        "the merit mills ratio must be strictly decreasing and convex"
    z_T = Finv(1 - OC / v_T)
    z_R = Finv(1 - cost / v_R)
    kappa = z_R - Finv(1 - OC / v_R)
    cvr, ocvr, ocvt = cost / v_R, OC / v_R, OC / v_T

    def top_rate(theta_T, sigma):
        u = G.cdf(theta_T)
        grid = G.ppf(u + (1 - u) * _NODES)
        return float(np.dot(_WTS, Fbar(z_T - (grid - theta_T) / sigma)))

    def density(w, theta_0, z, sigma, mass):
        th = theta_0 + sigma * (z - Finv(1 - w))
        return G.pdf(th) * sigma / fpdf(Finv(1 - w)) / mass

    q = np.linspace(0.001, 0.999, 500)
    panels, Wr, Wt = [], [], []
    for sigma, tag in ((sigma_1, "before"), (sigma_2, "after")):
        theta_T = optimize.brentq(lambda s: top_rate(s, sigma) - lambda_T,
                                  1e-4, 1 - 1e-4, xtol=1e-13)
        theta_R = theta_T - sigma * kappa
        assert theta_R > 0, "the marginal regular applicant must lie in the support"
        mass_R, mass_T = G.cdf(theta_T) - G.cdf(theta_R), G.sf(theta_T)
        w_top = float(Fbar(z_T - (1 - theta_T) / sigma))   # the best applicant
        wr = np.linspace(cvr, ocvr, 500)
        wt = np.linspace(ocvt, w_top, 700)
        dr = density(wr, theta_R, z_R, sigma, mass_R)
        dt = density(wt, theta_T, z_T, sigma, mass_T)
        # genuine densities on their full supports, and the payline is the mean
        # of the top one
        assert abs(np.trapezoid(dr, wr) - 1) < 2e-3, "regular density must integrate to 1"
        assert abs(np.trapezoid(dt, wt) - 1) < 2e-3, "top density must integrate to 1"
        assert abs(np.trapezoid(wt * dt, wt) - lambda_T) < 2e-3, \
            "the payline is the mean winning probability in the top program"
        a, b = G.cdf(theta_R), G.cdf(theta_T)
        Wr.append(Fbar(z_R - (G.ppf(a + (b - a) * q) - theta_R) / sigma))
        Wt.append(Fbar(z_T - (G.ppf(b + (1 - b) * q) - theta_T) / sigma))
        panels.append(dict(tag=tag, sigma=sigma, thT=theta_T, thR=theta_R,
                           w_top=w_top, wr=wr, dr=dr, wt=wt, dt=dt))
    before, after = panels
    assert after["thT"] < before["thT"], "the marginal top applicant must fall"
    # the paper's two results hold here: every regular rank gains (Prop 11)...
    assert np.all(Wr[1] >= Wr[0] - 1e-12), "every regular applicant must gain"
    # ...and the top distribution single crosses (Prop 12)
    d = Wt[0] - Wt[1]
    assert np.sum(np.sign(d[1:]) != np.sign(d[:-1])) == 1, "top QQ must single cross"
    # and the hump moves from the regular pool to the top pool
    peak = lambda arr: int(np.argmax(arr))
    assert 5 < peak(before["dr"]) < len(before["dr"]) - 5, "regular should hump before"
    assert peak(before["dt"]) <= 5, "top should decline before"
    assert 5 < peak(after["dt"]) < len(after["dt"]) - 5, "top should hump after"
    assert peak(after["dr"]) >= len(after["dr"]) - 5, "regular should rise after"

    ymax = round(max(max(p["dr"].max(), p["dt"].max()) for p in panels) * 1.22, 2)
    # the frame follows the data: from just below c/v^R to just past the best
    # applicant, rather than the whole unit interval
    xmin = round(cvr - 0.07, 3)
    xmax = round(max(p["w_top"] for p in panels) + 0.12, 3)
    per_x = 14.8 / (xmax - xmin)
    for p in panels:
        sub = "1" if p["tag"] == "before" else "2"
        raw = [(cvr, "$c/v^R$"), (ocvt, "$OC/v^T$"), (ocvr, "$OC/v^R$"),
               (lambda_T, "$\\lambda^T$"),
               (p["w_top"], "$W_T^{-1}(1|\\sigma_" + sub + ",OC)$")]
        # a label needs about 1cm to itself; anything closer drops a row
        marks, last = [], {}
        for x, lab in raw:
            row = 0 if x - last.get(0, -9) >= 1.4 / per_x else 1
            last[row] = x
            marks.append((x, lab, row))
        depth, drop = -0.03 * ymax, 0.075 * ymax

        def region(w, dens):
            """Outline with vertical edges: a truncated density jumps at its ends."""
            return coords(np.r_[w[0], w, w[-1]], np.r_[0.0, dens, 0.0])
        style = f"""\\pgfplotsset{{dens{p['tag']}/.style={{
  width=14.8cm, height=7.2cm,
  xmin={xmin}, xmax={xmax}, ymin=0, ymax={ymax},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=7pt]}},black}},
  clip=false, axis on top,
  xlabel={{$w$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=10pt}},
  tick label style={{font=\\small}}, label style={{font=\\small}},
  title style={{font=\\small}},
  ytick=\\empty, xtick=\\empty, tick style={{black}},
}}}}"""
        title = ("winning probabilities before the rise in noise" if p["tag"] == "before"
                 else "winning probabilities after the rise in noise")
        jr, jt = int(np.argmax(p["dr"])), int(np.argmax(p["dt"]))
        body = "\n".join([
            "\\begin{tikzpicture}",
            f"\\begin{{axis}}[dens{p['tag']}, title={{{title}}}]",
            f"  \\addplot[draw=none,fill=blue!55,fill opacity=0.22,forget plot] "
            f"coordinates {{{region(p['wr'], p['dr'])}}};",
            f"  \\addplot[draw=none,fill=red!60,fill opacity=0.22,forget plot] "
            f"coordinates {{{region(p['wt'], p['dt'])}}};",
            *[f"  \\draw[gray,dashed] (axis cs:{x:.5f},{depth - r * drop:.4f}) -- "
              f"(axis cs:{x:.5f},{ymax});" for x, _, r in marks],
            *[f"  \\node[anchor=north,font=\\small] at (axis cs:{x:.5f},"
              f"{depth - r * drop:.4f}) {{{lab}}};" for x, lab, r in marks],
            f"  \\addplot[blue,very thick] coordinates {{{region(p['wr'], p['dr'])}}};",
            f"  \\addplot[red,very thick] coordinates {{{region(p['wt'], p['dt'])}}};",
            f"  \\node[blue,anchor=south,font=\\small] at "
            f"(axis cs:{0.5 * (cvr + ocvr):.4f},{p['dr'].max() + 0.03 * ymax:.4f}) "
            f"{{regular: $W_R(\\cdot|\\sigma_{sub},OC)$}};",
            f"  \\node[red,anchor=south west,font=\\small] at "
            f"(axis cs:{max(p['wt'][jt], lambda_T + 0.03):.4f},"
            f"{np.interp(max(p['wt'][jt], lambda_T + 0.03), p['wt'], p['dt']) + 0.05 * ymax:.4f}) "
            f"{{top: $W_T(\\cdot|\\sigma_{sub},OC)$}};",
            "\\end{axis}", "\\end{tikzpicture}"])
        cap = (f"Densities of winning probabilities {p['tag']} the rise in evaluation "
               f"noise. Regular applicants occupy $[c/v^R,OC/v^R]$ and top applicants "
               f"$[OC/v^T,W_T^{{-1}}(1|\\sigma_{sub},OC)]$; the two overlap on "
               f"$[OC/v^T,OC/v^R]$ because the marginal top applicant, indifferent "
               f"between the programs, wins with probability $OC/v^T$ in one and "
               f"$OC/v^R$ in the other. The payline $\\lambda^T$ is the mean of the top "
               f"density. $F$ standard logistic, $G=\\mathrm{{Beta}}({beta_ab[0]},{beta_ab[1]})$ "
               f"on $[0,1]$; $v^T={v_T:g}$, $v^R={v_R:g}$, $c={cost:g}$, $OC={OC:g}$, "
               f"$\\lambda^T={lambda_T:g}$, $\\sigma={p['sigma']:g}$.")
        env = "\n".join([
            "\\begin{figure}[htbp]", "\\centering", style, "", body,
            "% Caption kept here, deliberately not typeset - the figure carries a",
            "% number only.  To restore it, move the line below into \\caption{}.",
            f"% {cap}",
            "\\caption{}", f"\\label{{fig:density_{p['tag']}}}", "\\end{figure}", ""])
        write(f"density_{p['tag']}.tex",
              f"% Winning probability densities, {p['tag']} the rise in noise.\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              f"% sigma={p['sigma']:g}: theta^T={p['thT']:.4f}, theta^R={p['thR']:.4f}; "
              f"regular on [{cvr:.3f},{ocvr:.3f}], top on [{ocvt:.3f},{p['w_top']:.3f}]\n"
              + env)
        print(f"    {p['tag']:6s}: sigma={p['sigma']:g}  theta^T={p['thT']:.3f}  "
              f"regular peak {p['dr'].max():.2f} at w={p['wr'][jr]:.3f}, "
              f"top peak {p['dt'].max():.2f} at w={p['wt'][jt]:.3f}")


def footnote_params(m: Model):
    """Empty: a \\footnote inside a \\caption breaks caption's fragile argument.

    The parameters are written once to misc_params.tex instead, and Writeup.tex
    pulls them into a real footnote on the sentence introducing the figures.
    """
    return ""


def write_misc_params(m: Model, fname="misc_params.tex"):
    txt = ("$F$ " + m.noise_name + f", $\\sigma={m.sigma:g}$; "
           f"$G=\\mathrm{{Beta}}({m.beta_a:g},{m.beta_b:g})$ on "
           f"$[{m.theta_lo:g},{m.theta_hi:g}]$; $\\lambda^T={m.lambda_T:g}$, "
           f"$\\lambda^R={m.lambda_R:g}$, $v^R/v^T={m.v_ratio:g}$, "
           f"$c/v^R={m.cost_ratio:g}$.%")
    write(fname, txt)


def figure_misc_triptych(m: Model, deltas, idx, fname):
    """Three diagrams for ONE of the two general equilibria of Figure 7.

    (a) the top program's partial equilibrium at that equilibrium's xhat^R,
    (b) the regular program's at that equilibrium's thetahat^T,
    (c) the two best responses, with the axes centred on that intersection.

    Panels (a) and (b) are Figures 3(a) and 4(a) redrawn; panel (c) is Figure 5.
    """
    d = deltas[idx]
    other = deltas[1 - idx]
    n = idx + 1
    pad = 0.02 * (m.theta_hi - m.theta_lo)
    lo, hi = m.theta_lo + pad, m.theta_hi - pad
    grid = np.linspace(lo, hi, 200)

    tT = m.theta_T_of_delta(d)
    tR = tT - m.sigma * d
    xR = tR + m.reg_intercept
    dR = (m.R_of_delta(d + 1e-4) - m.R_of_delta(d - 1e-4)) / 2e-4

    W, H = 5.3, 5.2          # three panels across one \textwidth

    # ---------------- (a) top program ------------------------------------
    th = np.linspace(lo, hi, 170)
    S = np.array([m.x_supply_top(t) for t in th])
    D = np.array([m.x_demand_top(t, xR) for t in th])
    a_ylo, a_yhi = min(S.min(), D.min()) - 0.30, max(S.max(), D.max()) + 0.15
    a_xmin, a_xmax = lo - 0.25, hi + 0.70
    a_depth = a_ylo - 0.06
    style_a = axis_style(f"mt{n}a", width=W, height=H, xmin=a_xmin, xmax=a_xmax,
                         ymin=a_ylo, ymax=a_yhi, xlabel="$\\hat{\\theta}^T$",
                         ylabel="$\\hat{x}^T$", extra="\n  xtick=\\empty, ytick=\\empty,")
    left = (tT - lo) / (hi - lo) < 0.5
    j = int((0.80 if left else 0.16) * len(th))
    up = S[j] > D[j]
    panel_a = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        f"\\begin{{axis}}[mt{n}a, title={{(a) top program}}]",
        f"  \\draw[gray,dashed] (axis cs:{tT:.5f},{a_depth:.3f}) -- "
        f"(axis cs:{tT:.5f},{m.x_supply_top(tT):.5f});",
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{tT:.5f},"
        f"{a_depth:.3f}) {{$\\hat{{\\theta}}^{{T,*}}_{{{n}}}$}};",
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(th, S)}}};",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(th, D)}}};",
        f"  \\node[blue,anchor={'south' if up else 'north'},font=\\footnotesize] at "
        f"(axis cs:{th[j]:.4f},{S[j] + (0.14 if up else -0.14):.4f}) {{$\\hat{{x}}^T_S$}};",
        f"  \\node[red,anchor={'north' if up else 'south'},font=\\footnotesize] at "
        f"(axis cs:{th[j]:.4f},{D[j] + (-0.14 if up else 0.14):.4f}) {{$\\hat{{x}}^T_D$}};",
        "\\end{axis}", "\\end{tikzpicture}"])

    # ---------------- (b) regular program --------------------------------
    gr = np.linspace(lo, tT - 0.004, 150)
    SR = np.array([m.x_supply_reg(t, tT) for t in gr])
    edge = np.array([lo, tT + 0.10])
    RD = edge + m.reg_intercept
    b_ylo, b_yhi = min(SR.min(), RD.min()) - 0.30, max(SR.max(), RD.max()) + 0.15
    b_xmin, b_xmax = lo - 0.25, tT + 0.80
    b_depth = b_ylo - 0.06
    style_b = axis_style(f"mt{n}b", width=W, height=H, xmin=b_xmin, xmax=b_xmax,
                         ymin=b_ylo, ymax=b_yhi, xlabel="$\\hat{\\theta}^R$",
                         ylabel="$\\hat{x}^R$", extra="\n  xtick=\\empty, ytick=\\empty,")
    # label both curves on the side away from the crossing, with whichever is
    # on top placed above.  Supply cuts demand from above, so left of the root
    # blue is the upper curve and right of it red is.
    b_right = (tR - lo) / (tT + 0.10 - lo) < 0.5
    jb = int((0.80 if b_right else 0.16) * len(gr))
    x_lab = float(gr[jb])
    blue_up = not b_right          # right of the root, red is on top
    red_y = x_lab + m.reg_intercept
    panel_b = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        f"\\begin{{axis}}[mt{n}b, title={{(b) regular program}}]",
        f"  \\draw[gray,dashed] (axis cs:{tR:.5f},{b_depth:.3f}) -- "
        f"(axis cs:{tR:.5f},{xR:.5f});",
        f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{tR:.5f},"
        f"{b_depth:.3f}) {{$\\hat{{\\theta}}^{{R,*}}_{{{n}}}$}};",
        f"  \\addplot[red,very thick] coordinates {{{coords(edge, RD)}}};",
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(gr, SR)}}};",
        f"  \\node[blue,anchor={'south' if blue_up else 'north'},font=\\footnotesize] "
        f"at (axis cs:{x_lab:.4f},"
        f"{SR[jb] + (0.16 if blue_up else -0.16):.4f}) {{$\\hat{{x}}^R_S$}};",
        f"  \\node[red,anchor={'north' if blue_up else 'south'},font=\\footnotesize] "
        f"at (axis cs:{x_lab:.4f},"
        f"{red_y + (-0.16 if blue_up else 0.16):.4f}) {{$\\hat{{x}}^R_D$}};",
        "\\end{axis}", "\\end{tikzpicture}"])

    # ---------------- (c) best responses, centred on this equilibrium -----
    bxs = np.linspace(-1.55, 1.82, 150)
    blue = [(x, m.theta_top(x, grid)) for x in bxs]
    blue = [(x, t) for x, t in blue if t is not None]
    bx = np.array([p[0] for p in blue]); by = np.array([p[1] for p in blue])
    tts = np.linspace(-2.1, 2.35, 150)
    red = [(m.x_reg(t), t) for t in tts]
    red = [(x, t) for x, t in red if x is not None]
    rx = np.array([p[0] for p in red]); ry = np.array([p[1] for p in red])
    # centre the window on this intersection
    halfx = 1.30
    c_xmin, c_xmax = xR - halfx, xR + halfx
    mb = (bx >= c_xmin) & (bx <= c_xmax)
    mr = (rx >= c_xmin) & (rx <= c_xmax)
    ys = np.concatenate([by[mb], ry[mr]])
    c_ylo, c_yhi = ys.min() - 0.25, ys.max() + 0.15
    style_c = axis_style(f"mt{n}c", width=W, height=H, xmin=c_xmin - 0.10,
                         xmax=c_xmax + 0.42, ymin=c_ylo, ymax=c_yhi,
                         xlabel="$\\hat{x}^R$", ylabel="$\\hat{\\theta}^T$",
                         extra=(f"\n  xtick={{{xR:.4f}}}, "
                                f"xticklabels={{$\\hat{{x}}^{{R,*}}_{{{n}}}$}},"
                                f"\n  ytick={{{tT:.4f}}}, "
                                f"yticklabels={{$\\hat{{\\theta}}^{{T,*}}_{{{n}}}$}},"))
    jc = int(0.18 * mb.sum())
    panel_c = "\n".join([
        "\\begin{tikzpicture}[baseline=(current bounding box.north)]",
        f"\\begin{{axis}}[mt{n}c, title={{(c) best responses}}]",
        f"  \\draw[gray,dashed] (axis cs:{xR:.5f},{c_ylo:.3f}) -- "
        f"(axis cs:{xR:.5f},{tT:.5f});",
        f"  \\draw[gray,dashed] (axis cs:{c_xmin - 0.10:.3f},{tT:.5f}) -- "
        f"(axis cs:{xR:.5f},{tT:.5f});",
        f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(bx[mb], by[mb])}}};",
        f"  \\addplot[red,very thick,smooth] coordinates {{{coords(rx[mr], ry[mr])}}};",
        f"  \\node[blue,anchor=north west,font=\\footnotesize] at (axis cs:"
        f"{bx[mb][jc]:.4f},{by[mb][jc] - 0.10:.4f}) {{$\\hat{{\\theta}}^T_{{top}}$}};",
        f"  \\node[red,anchor=south east,font=\\footnotesize] at (axis cs:"
        f"{rx[mr][jc]:.4f},{ry[mr][jc] + 0.10:.4f}) {{$\\hat{{x}}^R_{{reg}}$}};",
        "\\end{axis}", "\\end{tikzpicture}"])

    word = "the stability condition of Assumption 2 holds" if dR > 0 else \
           "the stability condition of Assumption 2 fails"
    caption = (f"The equilibrium at $\\delta^*_{{{n}}}={d:.3f}$ of "
               f"Figure~\\ref{{fig:misc_acc}}, where $\\partial\\mathcal{{R}}/"
               f"\\partial\\delta={dR:+.3f}$ and so {word}. "
               f"Panel (a) is Figure~\\ref{{fig:pe_a}} at "
               f"$\\hat{{x}}^{{R,*}}_{{{n}}}={xR:.3f}$, panel (b) is "
               f"Figure~\\ref{{fig:reg_a}} at $\\hat{{\\theta}}^{{T,*}}_{{{n}}}={tT:.3f}$, "
               f"and panel (c) is Figure~\\ref{{fig:geneq}} centred on this "
               f"intersection. Both best responses are increasing here, which is how "
               f"they meet twice; the other intersection is at "
               f"$\\hat{{x}}^R={m.theta_T_of_delta(other) - m.sigma * other + m.reg_intercept:.3f}$.")
    header = (f"% Miscellaneous - the three diagrams at equilibrium {n} "
              f"(delta*={d:.4f}).\n"
              "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
              f"% theta^T*={tT:.4f}, theta^R*={tR:.4f}, xhat^R*={xR:.4f}, "
              f"dR/ddelta={dR:+.4f}\n")
    body = (subfigure(panel_a, f"fig:tri{n}_a", frac=0.32) + "\n\\hfill\n" +
            subfigure(panel_b, f"fig:tri{n}_b", frac=0.32) + "\n\\hfill\n" +
            subfigure(panel_c, f"fig:tri{n}_c", frac=0.32))
    write(fname, header + style_a + "\n" + style_b + "\n" + style_c + "\n\n" +
          figure_env(body, caption, f"fig:tri{n}"))
    print(f"    eq {n}: delta*={d:.4f} theta^T*={tT:.4f} theta^R*={tR:.4f} "
          f"xhat^R*={xR:.4f} dR/ddelta={dR:+.4f}")


# --------------------------------------------------------------------------
# --------------------------------------------------------------------------
# build targets
# --------------------------------------------------------------------------
# Shared primitives for every figure: standard logistic noise, sigma = 0.7,
# Beta(1.5, 1.5) merit on [-1.6, 1.6], lambda^T = 0.35, v^R/v^T = 0.35.
#
# The regular-program parameters differ between Figures 4 and 5 on purpose.
# (0.55, 0.40) centres Figure 4's crossing, but at those values the general
# equilibrium has composite slope -7.72 and is UNSTABLE, so Figure 5 would
# contradict itself.  (0.60, 0.30) gives composite slope -0.52.
BASE = Model(lambda_R=0.55, cost_ratio=0.40)
STABLE = replace(BASE, lambda_R=0.60, cost_ratio=0.30)

# Miscellaneous: a parameterisation with TWO general equilibria.  R(sigma, .)
# turns over here, so lambda^R is crossed twice.  A large v^R/v^T and a small
# c/v^R are what make R non-monotone; the merit law is still log-concave, and
# right truncation still preserves its IHR.
MISC = Model(noise="normal", sigma=0.50, lambda_T=0.50, lambda_R=0.20,
             v_ratio=0.60, cost_ratio=0.10,
             beta_a=2, beta_b=8, theta_lo=-2.5, theta_hi=2.5)


# Section 4.3.1: a rise in noise under a convex mills ratio, starting from an
# equilibrium that is stable in the sense of Definition 3.  Found by a search
# over convex-mills Beta merit laws.  A small v^R/v^T keeps the top program
# stable (the opportunity cost responds weakly to the marginal top applicant)
# and keeps R(OC, sigma) smooth and rising; a large v^R/v^T instead gives a
# hump, and stable crossings then survive only on a spike at the top of the
# range.  The top program is stable only above a threshold opportunity cost,
# so lambda^R places OC_1 well inside that region; OC_2 then stays stable for
# sigma_2 = 2.5 sigma_1, about the largest rise for which R(., sigma_2) still
# spans the whole range of opportunity costs.
SHIFT = Model(noise="logistic", sigma=0.90, lambda_T=0.30, lambda_R=0.57,
              v_ratio=0.35, cost_ratio=0.40,
              beta_a=2, beta_b=2, theta_lo=-2.5, theta_hi=2.5)


def _br_second(m, sigma_2):
    """Best responses at noise sigma_2 whose payline is R(OC_1, sigma_2): the
    orange dot of Figure oc_noise_shift.  Their intersection is the sigma_2
    equilibrium with opportunity cost OC_1, so it lies on the OC_1 line at
    sigma_2, and payline restoration puts it below the sigma_1 equilibrium."""
    d1 = m.delta_equilibria()[0]
    m2 = replace(m, sigma=sigma_2)
    m2 = replace(m2, lambda_R=m2.R_of_delta(d1))
    t1, t2 = m.theta_T_of_delta(d1), m2.theta_T_of_delta(d1)
    assert t2 < t1, "the sigma_2 equilibrium at OC_1 must sit below the sigma_1 one"
    x2 = m2.x_demand_reg(t2 - sigma_2 * d1)
    # after the readjustment the regular payline is back at lambda^R: the red
    # dot of Figure oc_noise_shift, with a lower opportunity cost than OC_1
    m3 = replace(m2, lambda_R=m.lambda_R)
    return dict(model=m2, x_range=(x2 - 0.08, x2 + 0.05), t_range=(1.85, 2.42),
                dot="orange", sig="\\sigma_2", lam="\\mathcal{R}(OC_1,\\sigma_2)",
                top_lab=("south east", "first"), reg_lab=("north west", "first"),
                # the two new points sit about a millimetre apart horizontally:
                # the guide of this one runs below the axis, so its label sits a
                # row below the other's
                guide=("$\\hat{x}^{R}_{\\sigma_2,OC_1}$", "$\\hat{\\theta}^{T}_{\\sigma_2,OC_1}$",
                       "north", 16),
                # short dashes set the readjusted curve apart from the long
                # dashes of the other curves at sigma_2; it runs close to them,
                # so its label sits in the empty upper left with a leader line
                third=dict(model=m3, color="red", dot="red", lam="\\lambda^R",
                           dash="dash pattern=on 2pt off 2pt",
                           lab_at=(x2 - 0.09, 2.30), lab_anchor="east", lead_t=2.30,
                           guide=("$\\hat{x}^R_2$", "$\\hat{\\theta}^T_2$", "north west")))


def _br_shift_settings(**over):
    """Settings of Figure br_shift, shared by its copies; keywords override."""
    kw = dict(x_start=0.0, fname="br_shift.tex",
        # the stable top branch runs from the top of the merit grid down to
        # the fold where it meets the unstable branch
        x_range=(0.943, 1.045), t_range=(1.85, 2.42), n_blue=160,
        # a close-up of the three equilibria: the curves run off the edges
        # of the frame, and the frame reaches right to leave an empty
        # corner for the legend below the OC_1 line
        window=(0.76, 1.30, 2.02, 2.33),
        stars=("$\\hat{x}^R_1$", "$\\hat{\\theta}^T_1$"), label="fig:br_shift",
        title="Section 4.3.1 - best responses at the initial equilibrium OC_1 of "
              "Figure oc_noise_shift.",
        cobweb=False, flow=False, eq_dot="darkgreen", sig="\\sigma_1",
        # the best responses are named in a legend to the right of the plot
        legend="inside", size=(15.5, 9.4),
        second=_br_second(SHIFT, 0.99),
        # The OC_1 line at higher noise lies parallel to the sigma_1 line and
        # further from the 45 degree line, by (sigma_2 - sigma_1)|F^{-1}(1 -
        # OC_1/v^R)|.  At the sigma_2 = 2.25 of Figure oc_noise_shift that is
        # 1.77, three times the height of this window, so this figure uses
        # the smaller rise sigma_2 = 1.1 sigma_1 = 0.99 (a shift of 0.12).
        iso_oc=((0.90, "$OC_1,\\sigma_1$", 1.0, "west", 9.9),
                (0.99, "$OC_1,\\sigma_2$", 1.0, "south", 9.9)),
        note="Best responses at the initial equilibrium $OC_1$ of Figure "
             "\\ref{fig:oc_noise_shift}, with the lines along which the marginal top "
             "applicant faces opportunity cost $OC_1$ at $\\sigma_1=0.9$ and at "
             "$\\sigma_2=0.99$ (a smaller rise than in Figure \\ref{fig:oc_noise_shift}).")
    kw.update(over)
    return kw


# Section 6.0.1, concave case: exponential merit truncated to [-2.5, 2.5] has a
# concave mills ratio and is log-concave.  Found by a search for parameters
# where higher noise lowers R(OC, sigma) and both fields' equilibria at the
# common regular payline are stable in the sense of Definition 3.
FIELD_CONVEX = SHIFT
FIELD_CONCAVE = Model(noise="logistic", sigma=0.30, lambda_T=0.30, lambda_R=0.583,
                      v_ratio=0.30, cost_ratio=0.40,
                      merit="texp", kappa=1.5, theta_lo=-2.5, theta_hi=2.5)


# --------------------------------------------------------------------------
# Appendix, Lemma lemma:rays - rays from the origin and left truncation
# --------------------------------------------------------------------------
def figure_qq_rays(fname="qq_rays.tex", a=1.0, k=2.0, s=2.0, x1=3.5, x2=1.4,
                   x_end=8.5, y_end=6.7):
    """Lemma 3 of the appendix in three panels.

    F_1 is Gamma(a) and F_2 is Weibull(k, scale s); both start at 0 and F_2 has
    an increasing hazard rate (k > 1).  Since H_2(y) = (y/s)^k, the QQ plot is
    Q = s H_1^{1/k} and the left-truncated one is Q_trunc^k = Q^k - Q(x1)^k + x2^k.
    (a) the ray from the origin through (x1, Q(x1)) crosses Q from below;
    (b) the ray through (x1, x2) crosses the truncated QQ plot from above;
    (c) both, so the truncated plot starts, and stays, below Q.
    The defaults (a = 1, k = 2) give Q = s sqrt(x): concave, so the ray in (a)
    crosses at a clear angle, and the truncated plot is the shifted root
    s sqrt(x - x1 + (x2/s)^2), which stays well above its own ray in the box.
    """
    F1 = stats.gamma(a)
    H1 = lambda x: -F1.logsf(x)
    h1 = lambda x: F1.pdf(x) / F1.sf(x)
    Q = lambda x: s * H1(x) ** (1 / k)
    dQ = lambda x: s / k * H1(x) ** (1 / k - 1) * h1(x)
    Qx1 = float(Q(x1))
    Qt = lambda x: (Q(x) ** k - Qx1 ** k + x2 ** k) ** (1 / k)
    dQt1 = s ** k / k * h1(x1) * x2 ** (1 - k)               # slope of Q_trunc at x1
    m_Q, m_T = Qx1 / x1, x2 / x1                             # the two rays' slopes

    # the lemma's hypotheses and its conclusion, for these numbers
    assert k > 1, "F_2 must have a non-decreasing hazard rate"
    assert m_Q > dQ(x1), "(a): the ray must cross Q from below at x1"
    assert dQt1 > m_T, "(b): the ray must cross the truncated plot from above at x1"
    assert x2 < Qx1, "Lemma 3: the truncated plot must start below Q"
    xs = x_end * np.linspace(1e-4, 1.0, 400) ** 2           # dense near 0, where Q is steep
    xt = np.linspace(x1, x_end, 300)
    assert np.all(Qt(xt[1:]) < Q(xt[1:])), "Lemma 2(1): it must stay below Q"
    assert np.all(Qt(xt[1:]) > m_T * xt[1:]), "(b): keep the truncated plot above its ray in the box"
    assert Q(x_end) < y_end and Qt(x_end) < y_end, "curves must end inside the box"

    def ray_end(m):
        """Where the ray of slope m leaves the box [0, x_end] x [0, y_end]."""
        return (x_end, m * x_end) if m * x_end <= y_end else (y_end / m, y_end)

    pad = 0.04
    lo_x, lo_y = -pad * x_end, -pad * y_end
    style = "\n".join(
        f"""\\pgfplotsset{{{nm}/.style={{
  width=5.2cm, height=5.2cm,
  xmin={lo_x:.4f}, xmax={x_end:.4f}, ymin={lo_y:.4f}, ymax={y_end:.4f},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=6pt]}},black}},
  clip=false, axis on top,
  xlabel={{$x$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=8pt}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\footnotesize}},
  title style={{font=\\small, yshift=2pt}},
  ytick=\\empty, xtick=\\empty, tick style={{black}},
}}}}""" for nm in ("qqra", "qqrb", "qqrc"))

    def panel(name, title, orig, trunc):
        lines = [f"\\begin{{tikzpicture}}[baseline=({name}frame.south west)]",
                 f"\\begin{{axis}}[{name}, name={name}frame, title={{{title}}}]"]
        pts = ([(Qx1, m_Q, "blue", "$Q(\\underline{x}_1)$")] if orig else []) + \
              ([(x2, m_T, "red", "$\\underline{x}_2$")] if trunc else [])
        lines.append("  % rays from the origin, stopped at the edge of the box")
        for _, m, _, _ in pts:
            ex, ey = ray_end(m)
            lines.append(f"  \\draw[black,dashed,thick] (axis cs:0,0) -- (axis cs:{ex:.4f},{ey:.4f});")
        lines.append("  % guides from the axes to the point at x1")
        top = max(y for y, *_ in pts)
        lines.append(f"  \\draw[gray,dashed] (axis cs:{x1:.4f},{lo_y:.4f}) -- (axis cs:{x1:.4f},{top:.4f});")
        for y, _, _, _ in pts:
            lines.append(f"  \\draw[gray,dashed] (axis cs:{lo_x:.4f},{y:.4f}) -- (axis cs:{x1:.4f},{y:.4f});")
        if orig:
            lines.append(f"  \\addplot[blue,very thick,smooth] coordinates {{{coords(xs, Q(xs))}}};")
        if trunc:
            lines.append(f"  \\addplot[red,very thick,smooth] coordinates {{{coords(xt, Qt(xt))}}};")
        for y, _, col, lab in pts:
            lines += [f"  \\addplot[{col},only marks,mark=*,mark size=1.8pt,forget plot] "
                      f"coordinates {{({x1:.4f},{y:.4f})}};",
                      f"  \\node[anchor=east,font=\\footnotesize] at "
                      f"(axis cs:{lo_x:.4f},{y:.4f}) {{{lab}}};"]
        lines += [f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{x1:.4f},{lo_y:.4f}) "
                  "{$\\underline{x}_1$};",
                  f"  \\node[anchor=north east,font=\\footnotesize] at (axis cs:{lo_x:.4f},{lo_y:.4f}) {{$0$}};",
                  "  % the widest y label, invisible, so the three panels are equally wide",
                  f"  \\node[anchor=east,font=\\footnotesize] at (axis cs:{lo_x:.4f},{0.5 * y_end:.4f}) "
                  "{\\phantom{$Q(\\underline{x}_1)$}};",
                  "\\end{axis}", "\\end{tikzpicture}"]
        return "\n".join(lines)

    panels = [panel("qqra", "(a) Original QQ plot", True, False),
              panel("qqrb", "(b) Left-truncated QQ plot", False, True),
              panel("qqrc", "(c) Both", True, True)]
    legend = ("{\\footnotesize\\tikz[baseline=-0.6ex]\\draw[blue,very thick] (0,0) -- (0.7,0);"
              "\\ $x\\mapsto Q(x)$\\qquad"
              "\\tikz[baseline=-0.6ex]\\draw[red,very thick] (0,0) -- (0.7,0);"
              "\\ $x\\mapsto Q(x|[\\underline{x}_i,\\infty)_{i=1,2})$\\qquad"
              "\\tikz[baseline=-0.6ex]\\draw[black,thick,dashed] (0,0) -- (0.7,0);"
              "\\ rays from the origin\\par}")
    cap = ("Lemma \\ref{lemma:rays}. (a) The ray from the origin through "
           "$(\\underline{x}_1,Q(\\underline{x}_1))$ crosses the QQ plot from below, "
           "$Q(\\underline{x}_1)/\\underline{x}_1>Q'(\\underline{x}_1)$. (b) The ray through "
           "$(\\underline{x}_1,\\underline{x}_2)$ crosses the left-truncated QQ plot from above, "
           "$Q'(\\underline{x}_1|[\\underline{x}_i,\\infty)_{i=1,2})>\\underline{x}_2/\\underline{x}_1$. "
           "(c) Since $F_2$ has an increasing hazard rate, the truncated QQ plot starts below "
           "the original, $\\underline{x}_2<Q(\\underline{x}_1)$, and by Lemma "
           "\\ref{lemma:relativeposition} stays below it for all $x\\geq\\underline{x}_1$. "
           + ("$F_1$ is standard exponential" if a == 1 else f"$F_1$ is Gamma({a:g})")
           + f" and $F_2$ Weibull with shape {k:g} and scale {s:g}; "
           f"$\\underline{{x}}_1={x1:g}$, $\\underline{{x}}_2={x2:g}$.")
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        *[subfigure(p, f"fig:qq_rays_{t}", frac=0.33) + ("\\hfill" if t != "c" else "")
          for p, t in zip(panels, "abc")],
        "\\par\\medskip", legend,
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {cap}",
        "\\caption{Rays from the Origin and Left Truncation}", "\\label{fig:qq_rays}",
        "\\end{figure}", ""])
    write(fname,
          "% Appendix, Lemma lemma:rays - rays from the origin and left truncation.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          f"% F_1 = Gamma({a:g}), F_2 = Weibull({k:g}, scale {s:g}); x1 = {x1:g}, Q(x1) = {Qx1:.4f}, "
          f"x2 = {x2:g}\n"
          f"% ray slopes: through Q {m_Q:.4f} vs Q'(x1) {float(dQ(x1)):.4f}; "
          f"through x2 {m_T:.4f} vs truncated slope {dQt1:.4f}\n"
          + style + "\n\n" + env)


# --------------------------------------------------------------------------
# Appendix, section 1.2 - Theorems 1 and 2 as QQ plots of residual lives
# --------------------------------------------------------------------------
def orders_panel(name, title, x_end, y_end, curves, end_x, end_y, below_lab):
    """A square QQ panel in the style of qq_panel, for curves running from the
    origin to the common end (x_end, y_end), with the 45 degree line capped at
    the box built by the dashed end guides.  curves: (t, q, colour, label, at).
    Each label has a leader to its curve.  With at=None it is pushed off the
    curve, away from the chord; with at=(frac, dx, dy) it sits at offset
    (dx, dy) * hi from the curve point at that fraction of t."""
    hi = max(x_end, y_end)
    pad = 0.05 * hi
    xlo = -pad
    cap = min(x_end, y_end)
    off = 0.13 * hi
    style = f"""\\pgfplotsset{{{name}/.style={{
  width=7.3cm, height=7.3cm,
  xmin={xlo:.4f}, xmax={hi + pad:.4f}, ymin={xlo:.4f}, ymax={hi + pad:.4f},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=6pt]}},black}},
  clip=false, axis on top,
  xlabel={{$t$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=8pt}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\footnotesize}},
  title style={{font=\\small, yshift=2pt}},
  ytick=\\empty, xtick=\\empty, tick style={{black}},
}}}}"""
    lines = [f"\\begin{{tikzpicture}}[baseline=({name}frame.south west)]",
             f"\\begin{{axis}}[{name}, name={name}frame, title={{{title}}}]",
             f"  \\addplot[black,dashed,thick,samples=2,domain={xlo:.4f}:{cap:.4f}] {{x}};",
             f"  \\node[anchor=south east,font=\\footnotesize] at (axis cs:{cap:.4f},{cap:.4f}) "
             "{$45^{\\circ}$};",
             "  % rank 1: the ends of both residual life distributions",
             f"  \\draw[gray,dashed] (axis cs:{x_end:.4f},{xlo:.4f}) -- "
             f"(axis cs:{x_end:.4f},{y_end:.4f}) -- (axis cs:{xlo:.4f},{y_end:.4f});"]
    for t, q, col, lab, _ in curves:
        lines.append(f"  \\addplot[{col},very thick,smooth] coordinates {{{coords(t, q)}}};")
    for t, q, col, lab, at in curves:
        if at is not None:                           # placed by hand, relative to the curve
            frac, dx, dy = at
            i = int(frac * (len(t) - 1))
            lx, ly = t[i] + dx * hi, q[i] + dy * hi
        else:
            d = q - t * y_end / x_end                # signed distance from the chord
            i = int(np.argmax(np.abs(d)))
            s = 1.0 if d[i] < 0 else -1.0            # below the chord -> down-right
            lx = min(max(t[i] + s * off, 0.10 * hi), 0.92 * hi)
            ly = min(max(q[i] - s * off, 0.06 * hi), 0.94 * hi)
        lines += [f"  \\node[{col}!80!black,anchor=center,font=\\footnotesize,inner sep=1.5pt] "
                  f"({name}{col}) at (axis cs:{lx:.4f},{ly:.4f}) {{{lab}}};",
                  f"  \\draw[{col}!80!black,thin] ({name}{col}) -- (axis cs:{t[i]:.4f},{q[i]:.4f});"]
    lines += [f"  \\addplot[black,only marks,mark=*,mark size=1.8pt,forget plot] coordinates "
              f"{{(0,0) ({x_end:.4f},{y_end:.4f})}};",
              f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:0,{xlo - 0.02 * hi:.4f}) {{$0$}};",
              f"  \\node[anchor=east,font=\\footnotesize] at (axis cs:{xlo - 0.01 * hi:.4f},0) {{$0$}};",
              f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{x_end:.4f},"
              f"{xlo - 0.02 * hi:.4f}) {{{end_x}}};",
              f"  \\node[anchor=south west,font=\\footnotesize] at (axis cs:{xlo + 0.01 * hi:.4f},"
              f"{y_end:.4f}) {{{end_y}}};",
              "\\end{axis}", "\\end{tikzpicture}",
              f"\\par\\smallskip{{\\footnotesize {below_lab}\\par}}"]
    return style + "\n\n" + "\n".join(lines)


def figure_qq_orders(fname="qq_orders.tex", kappa=4.0, th_a=(0.05, 0.45),
                     th_lc=(-1.0, 0.5), th_lx=(0.05, 2.0), delta=1.5, lomax_c=2.0):
    """Section 1.2 of the appendix: Theorem 1 (a) and Theorem 2 (b).

    (a) QQ plots t -> U_2^{-1}(U_1(t)) of the residual lives at th_1 < th_2 of an
        exponential truncated to [0, 1], density prop. to exp(-kappa theta):
        kappa < 0 has a convex mills ratio and a convex QQ plot, kappa > 0 a
        concave mills ratio and a concave QQ plot.  Same support, so both run
        from the origin to (1 - th_1, 1 - th_2).
    (b) QQ plots of the residual lives conditioned on [0, delta]: below the
        diagonal for a log-concave density (standard normal), above it for a
        log-convex one (Lomax, shape lomax_c).
    """
    # (a) Theorem 1 - mills ratio curvature
    t1, t2 = th_a
    x_end, y_end = 1 - t1, 1 - t2
    ta = np.linspace(0.0, x_end, 241)
    curves_a = []
    # "concave" goes above the end guide: below the red curve it would float
    # between the curves, and up-left of it lies the 45 degree line
    at_a = {"convex": None, "concave": (0.72, 0.0, 0.17)}
    for k, col, lab, conv in ((-kappa, "blue", "convex", True), (kappa, "red", "concave", False)):
        G = _MeritTruncExpUnit(k)
        tm = np.linspace(0.02, 0.98, 400)
        r = G.mills(tm)
        assert np.all(np.diff(r) < 0), f"{G.tag}: mills ratio not decreasing"
        d2 = np.gradient(np.gradient(r, tm), tm)[8:-8]
        assert np.all(d2 > 0) if conv else np.all(d2 < 0), f"{G.tag}: mills ratio curvature"
        u1 = (G.cdf(t1 + ta) - G.cdf(t1)) / (1 - G.cdf(t1))
        q = G.inv(G.cdf(t2) + u1 * (1 - G.cdf(t2))) - t2
        q[0], q[-1] = 0.0, y_end                     # exact ends, free of rounding
        sl = np.diff(q) / np.diff(ta)
        assert np.all(np.diff(sl) > 0) if conv else np.all(np.diff(sl) < 0), \
            f"Theorem 1: the QQ plot for {G.tag} is not {lab}"
        curves_a.append((ta, q, col, lab, at_a[lab]))

    # (b) Theorem 2 - log-concavity, residual lives conditioned on [0, delta]
    def window_qq(D, a, b, t):
        A1, B1, A2, B2 = D.cdf(a), D.cdf(a + delta), D.cdf(b), D.cdf(b + delta)
        p = (D.cdf(a + t) - A1) / (B1 - A1)
        return D.ppf(A2 + p * (B2 - A2)) - b

    tb = np.linspace(0.0, delta, 241)
    curves_b = []
    for D, (a, b), col, lab, concave in (
            (stats.norm(), th_lc, "blue", "log-concave $g$", True),
            (stats.lomax(lomax_c), th_lx, "red", "log-convex $g$", False)):
        z = np.linspace(min(a, b), max(a, b) + delta, 300)
        lg2 = np.diff(D.logpdf(z), 2)
        assert np.all(lg2 < 0) if concave else np.all(lg2 > 0), f"{lab}: wrong log-curvature"
        q = window_qq(D, a, b, tb)
        q[0], q[-1] = 0.0, delta
        inner = (q - tb)[1:-1]
        assert np.all(inner < 0) if concave else np.all(inner > 0), \
            f"Theorem 2: the {lab} QQ plot is on the wrong side of the diagonal"
        curves_b.append((tb, q, col, lab, None))

    pa = orders_panel("qqoa", "(a) Mills ratio curvature", x_end, y_end, curves_a,
                      "$U_{\\hat{\\theta}_1}^{-1}(1)$", "$U_{\\hat{\\theta}_2}^{-1}(1)$",
                      "$t\\mapsto U_{\\hat{\\theta}_2}^{-1}(U_{\\hat{\\theta}_1}(t))$")
    pb = orders_panel("qqob", "(b) Log-concavity", delta, delta, curves_b,
                      "$\\Delta$", "$\\Delta$",
                      "$t\\mapsto U_{\\hat{\\theta}_2}^{-1}\\big(U_{\\hat{\\theta}_1}(t|[0,\\Delta])"
                      "\\,\\big|\\,[0,\\Delta]\\big)$")
    cap = ("(a) Theorem 1: the QQ plot of the residual lives at $\\hat{\\theta}_1<\\hat{\\theta}_2$ "
           "is convex when the mills ratio is convex (blue) and concave when it is concave (red). "
           "$G$ is exponential truncated to $[0,1]$ with density proportional to "
           f"$e^{{-\\kappa\\theta}}$, $\\kappa=-{kappa:g}$ (blue) and $\\kappa={kappa:g}$ (red); "
           f"$\\hat{{\\theta}}_1={t1:g}$, $\\hat{{\\theta}}_2={t2:g}$. (b) Theorem 2: the QQ plot "
           "of the residual lives conditioned on $[0,\\Delta]$ lies below the diagonal when the "
           "density is log-concave (blue: standard normal, "
           f"$\\hat{{\\theta}}_1={th_lc[0]:g}$, $\\hat{{\\theta}}_2={th_lc[1]:g}$) and above it when "
           f"it is log-convex (red: Lomax with shape {lomax_c:g}, "
           f"$\\hat{{\\theta}}_1={th_lx[0]:g}$, $\\hat{{\\theta}}_2={th_lx[1]:g}$); $\\Delta={delta:g}$.")
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(pa, "fig:qq_orders_ct", frac=0.49), "\\hfill",
        subfigure(pb, "fig:qq_orders_lc", frac=0.49),
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {cap}",
        "\\caption{Stochastic Orders and QQ Plots}", "\\label{fig:qq_orders}",
        "\\end{figure}", ""])
    write(fname,
          "% Appendix, section 1.2 - Theorem 1 (mills ratio curvature) and Theorem 2\n"
          "% (log-concavity) as QQ plots of residual lives.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          + "".join(f"% (a) {lab}: kappa = {kp:+g}, slope at 0 = {(q[1] - q[0]) / (t[1] - t[0]):.3f}\n"
                    for (t, q, _, lab, _), kp in zip(curves_a, (-kappa, kappa)))
          + "".join(f"% (b) {lab}: largest gap from the diagonal {np.max(np.abs(q - t)):.3f}\n"
                    for t, q, _, lab, _ in curves_b)
          + env)


# --------------------------------------------------------------------------
# Appendix, section 1.3 - the proof sketch of Proposition 12, step by step
# --------------------------------------------------------------------------
def figure_qq_proof12(fname="qq_proof12.tex", OC=0.15, lambda_T=0.50, v_T=1.0,
                      sigma_1=0.4, sigma_2=1.2, v_R=0.40, cost=0.03,
                      top=1.5):
    """The five steps of the appendix's proof of Proposition 12, convex mills ratio.

    Everything is a QQ plot in standardized residual life above the regular
    cutoff, u = (theta - theta^R_{sigma,OC}) / sigma: U_2 (noise sigma_2) on the
    x axis, U_1 (sigma_1) or U_1^* (cutoff theta^*) on the y axis.  u = 0 is the
    marginal regular applicant and u = kappa(OC) the marginal top applicant at
    both noise levels, so [0, kappa] is the regular program and [kappa, oo) the
    top program, as in the two halves of Figure qq_diagonal.  Primitives are
    those of qq_transfer (standard normal G and F) except OC and lambda^T, raised so
    that the regular program's part of each panel is legible and the left-truncated
    plot of panel (a) visibly leaves (kappa, kappa) below the 45 degree line.  top: the
    common frame of all four panels, in multiples of kappa, so they share one scale.
    """
    G = _MeritNormal()
    Fbar, Finv = stats.norm.sf, stats.norm.ppf
    z = Finv(1 - OC / v_T)

    def rate(theta_T, sigma):
        u = G.cdf(theta_T)
        grid = G.inv(u + (1 - u) * _NODES)
        return float(np.dot(_WTS, Fbar(-(grid - theta_T) / sigma + z)))

    tT1, tT2 = (optimize.brentq(lambda t: rate(t, s) - lambda_T, G.lo, G.hi, xtol=1e-13)
                for s in (sigma_1, sigma_2))
    kap = float(Finv(1 - cost / v_R) - Finv(1 - OC / v_R))
    tR1, tR2 = tT1 - sigma_1 * kap, tT2 - sigma_2 * kap
    assert tT2 < tT1 and tR2 < tR1, "higher noise must lower both cutoffs"

    def Ucdf(u, th, s):
        return (G.cdf(th + s * np.asarray(u)) - G.cdf(th)) / (1 - G.cdf(th))

    def Uinv(p, th, s):
        return (G.inv(G.cdf(th) + np.asarray(p) * (1 - G.cdf(th))) - th) / s

    U2 = lambda u: Ucdf(u, tR2, sigma_2)
    qq = lambda u, th: Uinv(U2(u), th, sigma_1)                 # y-axis cutoff th

    def qq_left(u):                                             # both from kappa up
        A2, A1 = U2(kap), Ucdf(kap, tR1, sigma_1)
        return Uinv(A1 + (U2(u) - A2) / (1 - A2) * (1 - A1), tR1, sigma_1)

    def qq_right(u, th):                                        # both up to kappa
        return Uinv(U2(u) / U2(kap) * Ucdf(kap, th, sigma_1), th, sigma_1)

    th_star = optimize.brentq(lambda th: Ucdf(kap, th, sigma_1) - U2(kap), -5.0, 4.0,
                              xtol=1e-13)
    Qk = float(qq(kap, tR1))
    eps = 1e-6
    dQk = float(qq(kap + eps, tR1) - qq(kap - eps, tR1)) / (2 * eps)
    dQl = float(qq_left(kap + eps) - kap) / eps

    # every claim of the proof sketch, for these numbers
    ua = np.linspace(0.0, top * kap, 400)
    qa = qq(ua, tR1)
    assert np.all(np.diff(np.diff(qa) / np.diff(ua)) > 0), "Theorem 1: Q must be convex"
    assert Qk / kap <= dQk and dQl <= 1, "Step 1: the hypotheses of Lemma 3 (bracketed)"
    assert Qk < kap, "Step 1: Lemma 3 gives Q(kappa) <= kappa"
    ul = np.linspace(kap, top * kap, 300)
    ql = qq_left(ul)
    assert np.all(ql[1:] > qq(ul[1:], tR1)), "Step 1: Lemma 2(1), truncated plot above Q"
    assert th_star < tR1 and abs(qq(kap, th_star) - kap) < 1e-9, "Step 2"
    ub = np.linspace(0.0, top * kap, 300)
    assert np.all(qq(ub[1:], th_star) > qq(ub[1:], tR1)), "Step 2: Q^* lies above Q"
    ur = np.linspace(0.0, kap, 241)
    qsr, qr = qq_right(ur, th_star), qq_right(ur, tR1)
    assert np.max(np.abs(qsr - qq(ur, th_star))) < 1e-9, "Step 3: Lemma 1, Q^*_r = Q^*"
    assert np.all(qsr[1:-1] < ur[1:-1]), "Step 3: Q^*_r below the diagonal"
    assert np.all(qr[1:-1] < qsr[1:-1]), "Step 4: Theorem 2, Q_r below Q^*_r"
    assert np.all(qr[1:-1] > qq(ur[1:-1], tR1)), "Lemma 2(2): Q_r lies above Q, as kappa > Q(kappa)"

    def clip(x, y, top):
        """The curve up to where it leaves the frame through the top."""
        if not np.any(y > top):
            return x, y
        k = int(np.argmax(y > top))
        xe = x[k - 1] + (x[k] - x[k - 1]) * (top - y[k - 1]) / (y[k] - y[k - 1])
        return np.r_[x[:k], xe], np.r_[y[:k], top]

    KX, KY = "$\\kappa(OC)$", "$\\kappa(OC)$"

    def panel(name, title, top, body, yticks):
        pad = 0.05 * top
        lo = -pad
        style = f"""\\pgfplotsset{{{name}/.style={{
  width=6.7cm, height=6.7cm,
  xmin={lo:.4f}, xmax={top + pad:.4f}, ymin={lo:.4f}, ymax={top + pad:.4f},
  axis x line=bottom, axis y line=left,
  axis line style={{-{{Stealth[length=6pt]}},black}},
  clip=false, axis on top,
  xlabel={{$u$}},
  every axis x label/.style={{at={{(current axis.right of origin)}},anchor=west,xshift=8pt}},
  tick label style={{font=\\footnotesize}}, label style={{font=\\footnotesize}},
  title style={{font=\\small, yshift=2pt}},
  ytick=\\empty, xtick=\\empty, tick style={{black}},
}}}}"""
        under = [ln for ln in body if "fill=" in ln]   # shading goes beneath the 45 degree line
        body = [ln for ln in body if "fill=" not in ln]
        lines = [f"\\begin{{tikzpicture}}[baseline=({name}frame.south west)]",
                 f"\\begin{{axis}}[{name}, name={name}frame, title={{{title}}}]",
                 *under,
                 f"  \\addplot[black,dashed,thick,samples=2,domain={lo:.4f}:{top:.4f}] {{x}};",
                 f"  \\node[anchor=north west,font=\\footnotesize] at (axis cs:{top:.4f},{top:.4f}) "
                 "{$45^{\\circ}$};",
                 *body,
                 f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:0,{lo - 0.02 * top:.4f}) {{$0$}};",
                 f"  \\node[anchor=east,font=\\footnotesize] at (axis cs:{lo - 0.01 * top:.4f},0) {{$0$}};",
                 f"  \\node[anchor=north,font=\\footnotesize] at (axis cs:{kap:.4f},{lo - 0.02 * top:.4f}) {{{KX}}};",
                 *[f"  \\node[anchor=east,font=\\footnotesize] at (axis cs:{lo - 0.01 * top:.4f},{y:.4f}) {{{lab}}};"
                   for y, lab in yticks],
                 "\\end{axis}", "\\end{tikzpicture}"]
        return style + "\n\n" + "\n".join(lines)

    curve = lambda x, y, st: f"  \\addplot[{st},very thick,smooth] coordinates {{{coords(x, y)}}};"
    dot = lambda x, y, c: (f"  \\addplot[{c},only marks,mark=*,mark size=1.8pt,forget plot] "
                           f"coordinates {{({x:.4f},{y:.4f})}};")
    gv = lambda x, y, lo: f"  \\draw[gray,dashed] (axis cs:{x:.4f},{lo:.4f}) -- (axis cs:{x:.4f},{y:.4f});"
    gh = lambda y, x, lo: f"  \\draw[gray,dashed] (axis cs:{lo:.4f},{y:.4f}) -- (axis cs:{x:.4f},{y:.4f});"

    def tag(x, y, lx, ly, text, col):
        """A label at (lx, ly) with a thin leader to the curve point (x, y)."""
        nm = f"lab{zlib.crc32(repr((x, y, text)).encode()) % 10**6}"   # same name on every run
        return [f"  \\node[{col},anchor=center,font=\\footnotesize,inner sep=1.5pt] ({nm}) at "
                f"(axis cs:{lx:.4f},{ly:.4f}) {{{text}}};",
                f"  \\draw[{col},thin] ({nm}) -- (axis cs:{x:.4f},{y:.4f});"]

    # (a) Step 1 - left truncation at (kappa, kappa); Lemma 3 with the 45 degree
    # line as the ray through (kappa, kappa)
    Ta = top * kap
    lo_a = -0.05 * Ta
    m = Qk / kap
    ray_end = (Ta, m * Ta) if m * Ta <= Ta else (Ta / m, Ta)
    xa, ya = clip(ua, qa, Ta)
    xl, yl = clip(ul, ql, Ta)
    ia = int(0.88 * (len(xa) - 1))                      # labelled from above, clear of the ray
    body_a = [f"  \\draw[black,dashed,thick] (axis cs:0,0) -- (axis cs:{ray_end[0]:.4f},{ray_end[1]:.4f});",
              gv(kap, kap, lo_a), gh(kap, kap, lo_a), gh(Qk, kap, lo_a),
              curve(xa, ya, "blue"), curve(xl, yl, "red"),
              dot(kap, kap, "black"), dot(kap, Qk, "blue"),
              *tag(xa[ia], ya[ia], xa[ia] - 0.10 * Ta, ya[ia] + 0.12 * Ta, "$Q$", "blue")]
    pa = panel("qqpa", "(a) Step 1", Ta, body_a,
               [(Qk, "$Q(\\kappa(OC))$"), (kap, KY)])

    # (b) Step 2 - the cutoff theta^* that puts the QQ plot through (kappa, kappa)
    Tb = top * kap
    lo_b = -0.05 * Tb
    xb, yb = clip(ub, qq(ub, tR1), Tb)
    xs, ys = clip(ub, qq(ub, th_star), Tb)
    # both curves leave the origin, so lowering the cutoff rotates Q up about it:
    # an arc of a circle about the origin from Q to Q^* (the frame is square in
    # equal units, so the circle is round on the page)
    r_arc = 1.2 * kap

    def on_circle(th):
        """Polar angle at which the QQ plot with y-axis cutoff th meets the arc."""
        u = optimize.brentq(lambda v: v ** 2 + float(qq(v, th)) ** 2 - r_arc ** 2,
                            1e-9, r_arc)
        return np.arctan2(float(qq(u, th)), u)

    ph_Q, ph_S = on_circle(tR1), on_circle(th_star)
    assert ph_Q < ph_S < np.pi / 4 + 0.6, "Q^* must sit above Q on the arc"
    ph = np.linspace(ph_Q + 0.03, ph_S - 0.02, 40)
    # the label sits just outside the arc, in the gap above Q, right of Q^* and
    # below the 45 degree line
    ph_m = 0.5 * (ph_Q + ph_S) + 0.10
    lab_r = r_arc + 0.08 * Tb
    ib, js = int(0.85 * (len(xb) - 1)), int(0.45 * (len(xs) - 1))
    body_b = [gv(kap, kap, lo_b), gh(kap, kap, lo_b), gh(Qk, kap, lo_b),
              curve(xb, yb, "blue"), curve(xs, ys, "blue,densely dashed"),
              f"  \\addplot[blue,thick,-{{Stealth[length=5pt]}},smooth] coordinates "
              f"{{{coords(r_arc * np.cos(ph), r_arc * np.sin(ph))}}};",
              f"  \\node[blue,anchor=west,font=\\footnotesize] at (axis cs:{lab_r * np.cos(ph_m):.4f},"
              f"{lab_r * np.sin(ph_m):.4f}) {{$\\hat{{\\theta}}^R_{{\\sigma_1,OC}}\\to\\hat{{\\theta}}^*$}};",
              dot(kap, kap, "black"), dot(kap, Qk, "blue"),
              *tag(xb[ib], yb[ib], xb[ib] + 0.02 * Tb, yb[ib] - 0.16 * Tb, "$Q$", "blue"),
              *tag(xs[js], ys[js], xs[js] - 0.16 * Tb, ys[js] + 0.10 * Tb, "$Q^*$", "blue")]
    pb = panel("qqpb", "(b) Step 2", Tb, body_b,
               [(Qk, "$Q(\\kappa(OC))$"), (kap, KY)])

    # (c), (d): the right-truncated QQ plots (orange) drawn on the frame of (b),
    # with Q and Q^* faded behind them for reference
    RT, fade = "orange!90!black", "opacity=0.35"
    # Q is labelled where it is in (b); Q^* above (kappa, kappa), where it no
    # longer runs under Q^*_r, from the empty space left of it
    jS = int(np.argmin(np.abs(xs - 1.10 * kap)))
    ctx = [gv(kap, kap, lo_b), gh(kap, kap, lo_b), gh(Qk, kap, lo_b),
           curve(xb, yb, f"blue,{fade}"), curve(xs, ys, f"blue,densely dashed,{fade}"),
           *tag(xb[ib], yb[ib], xb[ib] + 0.02 * Tb, yb[ib] - 0.16 * Tb, "$Q$", "blue!50"),
           *tag(xs[jS], ys[jS], xs[jS] - 0.13 * Tb, ys[jS], "$Q^*$", "blue!50")]
    i3 = int(0.60 * (len(ur) - 1))
    mid = lambda i, q: 0.5 * (ur[i] + q[i])          # halfway to the diagonal

    # (c) Step 3 - right truncation of U_1^* and U_2 at (kappa, kappa): by Lemma 1
    # Q^*_r is Q^* itself on [0, kappa], below the diagonal by convexity
    body_c = [*ctx, curve(ur, qsr, f"{RT},densely dashed"),
              dot(kap, kap, "black"), dot(kap, Qk, f"blue,{fade}"),
              *tag(ur[i3], qsr[i3], ur[i3] - 0.10 * Tb, mid(i3, qsr) + 0.10 * Tb, "$Q^*_r$", RT)]
    pc = panel("qqpc", "(c) Step 3", Tb, body_c, [(Qk, "$Q(\\kappa(OC))$"), (kap, KY)])

    # (d) Steps 4 and 5 - right truncation of U_1 and U_2: Q_r ends at (kappa, kappa),
    # above Q (Lemma 2(2)) but below Q^*_r (Theorem 2), so below the diagonal
    i4 = int(0.85 * (len(ur) - 1))
    body_d = [*ctx, curve(ur, qsr, f"{RT},densely dashed"), curve(ur, qr, RT),
              dot(kap, kap, "black"), dot(kap, Qk, f"blue,{fade}"),
              # Q_r is labelled from the empty space right of the kappa guide
              *tag(ur[i4], qr[i4], 1.16 * kap, 0.5 * (Qk + kap), "$Q_r$", RT),
              *tag(ur[i3], qsr[i3], ur[i3] - 0.10 * Tb, mid(i3, qsr) + 0.10 * Tb, "$Q^*_r$", RT)]
    pd = panel("qqpd", "(d) Steps 4 and 5", Tb, body_d, [(Qk, "$Q(\\kappa(OC))$"), (kap, KY)])

    sw = lambda st: f"\\tikz[baseline=-0.6ex]\\draw[{st}] (0,0) -- (0.7,0);"
    legend = ("{\\footnotesize " + sw("blue,very thick")
              + "\\ $U_1$ (cutoff $\\hat{\\theta}^R_{\\sigma_1,OC}$) against $U_2$\\qquad"
              + sw("blue,very thick,densely dashed")
              + "\\ $U_1^*$ (cutoff $\\hat{\\theta}^*$) against $U_2$\\par"
              + sw("red,very thick") + "\\ left truncated at $\\kappa(OC)$\\qquad"
              + sw(f"{RT},very thick") + "\\ right truncated at $\\kappa(OC)$\\par}")
    cap = ("The proof of Proposition \\ref{prop:regwin_likelihood} under a convex mills ratio, "
           "drawn in standardized residual life $u$ above the regular cutoff: $u=0$ is the marginal "
           "regular applicant and $u=\\kappa(OC)$ the marginal top applicant. (a) $Q$ is convex "
           "(Theorem \\ref{lemma:millsratio_curvature}), so the ray through $(\\kappa(OC),Q(\\kappa(OC)))$ "
           "crosses it from above, while the left-truncated QQ plot leaves $(\\kappa(OC),\\kappa(OC))$ "
           "with slope below one, so the $45^\\circ$ line crosses it from below; by Lemma "
           "\\ref{lemma:rays}, $Q(\\kappa(OC))\\leq\\kappa(OC)$. (b) Lowering the cutoff of $U_1$ to "
           "$\\hat{\\theta}^*$ lifts $Q$ to $Q^*$, which passes through $(\\kappa(OC),\\kappa(OC))$; so "
           "$\\hat{\\theta}^*\\leq\\hat{\\theta}^R_{\\sigma_1,OC}$. (c) Right truncating $U_1^*$ and $U_2$ "
           "at $\\kappa(OC)$ leaves $Q^*$ unchanged on $[0,\\kappa(OC)]$ (Lemma \\ref{lemma:QQ_preservation}), "
           "and convexity puts it below the diagonal. (d) Right truncating $U_1$ and $U_2$ at $\\kappa(OC)$ "
           "gives $Q_r$, which ends at $(\\kappa(OC),\\kappa(OC))$ and so lies above $Q$ (Lemma "
           "\\ref{lemma:relativeposition}); by Theorem \\ref{theorem:logconcavity}, the higher cutoff "
           "puts $Q_r$ below $Q^*_r$, hence below the diagonal: regular applicants gain from the rise "
           f"in noise. $G$ and $F$ standard normal, $\\sigma_1={sigma_1:g}$, $\\sigma_2={sigma_2:g}$, "
           f"$v^T={v_T:g}$, $\\lambda^T={lambda_T:g}$, $OC={OC:g}$, $v^R={v_R:g}$, $c={cost:g}$.")
    env = "\n".join([
        "\\begin{figure}[htbp]", "\\centering",
        subfigure(pa, "fig:qq_proof12_a", frac=0.49), "\\hfill",
        subfigure(pb, "fig:qq_proof12_b", frac=0.49),
        "\\par\\bigskip",
        subfigure(pc, "fig:qq_proof12_c", frac=0.49), "\\hfill",
        subfigure(pd, "fig:qq_proof12_d", frac=0.49),
        "\\par\\medskip", legend,
        "% Caption kept here, deliberately not typeset - the figure carries a",
        "% number only.  To restore it, move the line below into \\caption{}.",
        f"% {cap}",
        "\\caption{Proving Proposition 12 under Log-Concavity}", "\\label{fig:qq_proof12}",
        "\\end{figure}", ""])
    write(fname,
          "% Appendix, section 1.3 - the proof sketch of Proposition 12, step by step.\n"
          "% GENERATED BY generate_figures.py - edit that file, not this one.\n"
          f"% theta^T {tT1:+.4f} -> {tT2:+.4f}, theta^R {tR1:+.4f} -> {tR2:+.4f}, "
          f"kappa(OC) = {kap:.4f}, theta^* = {th_star:+.4f}\n"
          f"% Q(kappa) = {Qk:.4f}; Q(kappa)/kappa = {Qk / kap:.4f} <= Q'(kappa) = {dQk:.4f}; "
          f"left-truncated slope at kappa = {dQl:.4f}\n"
          f"% largest gaps on [0, kappa]: diagonal - Q*_r {np.max(ur - qsr):.4f}, "
          f"Q*_r - Q_r {np.max(qsr - qr):.4f}\n"
          + env)


# name -> (what it is, a thunk that writes it).  Add a figure by adding a line.
TARGETS = {
    "partial_eq": (
        "Figure 3 (section 3.1.6) - partial equilibrium, top program",
        lambda: figure_partial_eq(BASE, x_R_before=-0.184, x_R_after=-0.004)),
    "regular_eq": (
        "Figure 4 (section 3.1.7) - partial equilibrium, regular program",
        lambda: figure_regular_eq(BASE, theta_T_before=0.50, theta_T_after=1.10)),
    "general_eq": (
        "Figure 5 (section 3.1.8) - the two best responses and the flow",
        lambda: figure_general_eq(STABLE, x_start=-0.78)),
    "payline_rise": (
        "Figure 6 (section 3.1.9) - a rise in lambda^R",
        lambda: figure_payline_rise(STABLE, lambda_R_hi=0.66)),
    "top_acceptance": (
        "Figure 7 (section 4.0.1) - top program acceptance rate function",
        figure_top_acceptance),
    "top_tradeoff": (
        "Figure 8 (section 4.0.1) - intensive versus extensive margin",
        figure_top_tradeoff),
    "oc_acceptance": (
        "Proposition 9 - R(OC,sigma) crossing lambda^R twice",
        lambda: figure_oc_acceptance(MISC)),
    "oc_noise_shift": (
        "Section 4.3.1 - R(OC,sigma) shifted up by higher noise (convex mills ratio)",
        lambda: figure_oc_noise_shift(SHIFT, sigma_2=2.25)),
    "br_shift": (
        "Section 4.3.1 - best responses at the initial equilibrium OC_1 of oc_noise_shift",
        lambda: figure_general_eq(SHIFT, **_br_shift_settings())),
    "br_shift_arrows": (
        "Section 4.3.1 - the same, with arrows green -> orange -> red",
        lambda: figure_general_eq(SHIFT, **_br_shift_settings(
            fname="br_shift_arrows.tex", label="fig:br_shift_arrows",
            arrows=(("first", "second"), ("second", "third")), fade=0.4))),
    "field_pair": (
        "Section 6.0.1 - regular acceptance curves of two fields, convex and concave panels",
        figure_field_pair),
    "noise_rotation": (
        "Section 4.0.4 - higher noise in the regular program evaluation",
        figure_noise_rotation),
    "qq_transfer": (
        "Proposition 12 - QQ plots of winning probabilities under higher noise",
        figure_qq_transfer),
    "qq_top": (
        "Theorem 9 - QQ plots in the top program, convex mills ratio: winning "
        "probabilities (a), standardized residual life (b)",
        figure_qq_top),
    "qq_rays": (
        "Appendix, Lemma 3 - rays from the origin and the left-truncated QQ plot",
        figure_qq_rays),
    "qq_orders": (
        "Appendix, section 1.2 - Theorems 1 and 2 as QQ plots of residual lives",
        figure_qq_orders),
    "qq_proof12": (
        "Appendix, section 1.3 - the proof sketch of Proposition 12, step by step",
        figure_qq_proof12),
    "qq_equilibrium": (
        "Scratchpad - the qq_diagonal staircase, equilibrium before vs after the rise in noise",
        figure_qq_equilibrium),
    "densities": (
        "Winning probability densities before and after the rise in noise",
        figure_densities),
    "misc_params": (
        "Miscellaneous - the parameter footnote shared by the misc figures",
        lambda: write_misc_params(MISC)),
    "misc_acceptance": (
        "Miscellaneous - R(sigma,delta) with two crossings",
        lambda: figure_misc_acceptance(MISC)),
    "misc_eq1": (
        "Miscellaneous - three diagrams at the first equilibrium",
        lambda: figure_misc_triptych(MISC, MISC.delta_equilibria(), 0,
                                     "misc_eq1.tex")),
    "misc_eq2": (
        "Miscellaneous - three diagrams at the second equilibrium",
        lambda: figure_misc_triptych(MISC, MISC.delta_equilibria(), 1,
                                     "misc_eq2.tex")),
}


# --------------------------------------------------------------------------
# incremental builds
# --------------------------------------------------------------------------
# output/build_state.json records, for each target, the fingerprint of the code
# it was built from and the files it wrote.  A plain run rebuilds only the
# targets that are new, whose code changed, or whose files are missing.
STATE = OUTDIR / "build_state.json"


def fingerprints():
    """{target: hash of the code it depends on}: its TARGETS entry and, transitively,
    every top-level function, class and constant of this file that the entry uses.
    The hash is of the syntax tree, so comments and layout do not count."""
    tree = ast.parse(Path(__file__).read_text())
    defs = {}                                  # top-level name -> its definitions
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            defs.setdefault(node.name, []).append(node)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for t in targets:
                for n in ast.walk(t):
                    if isinstance(n, ast.Name):
                        defs.setdefault(n.id, []).append(node)
    uses = lambda node: {n.id for n in ast.walk(node) if isinstance(n, ast.Name)} & defs.keys()
    entries = defs["TARGETS"][0].value
    prints = {}
    for key, entry in zip(entries.keys, entries.values):
        build = entry.elts[1]                  # (description, build): the description does not count
        deps, todo = set(), uses(build) - {"TARGETS"}
        while todo:
            name = todo.pop()
            deps.add(name)
            todo |= set().union(*(uses(d) for d in defs[name])) - deps - {"TARGETS"}
        text = ast.dump(build) + "".join(ast.dump(d) for name in sorted(deps) for d in defs[name])
        prints[key.value] = hashlib.sha256(text.encode()).hexdigest()[:16]
    return prints


def stale(state, prints):
    """Targets that are new, whose code changed, or whose files are missing."""
    return [name for name in TARGETS
            if name not in state or state[name]["fingerprint"] != prints[name]
            or not all((OUTDIR / f).exists() for f in state[name]["files"])]


def build(names, state, prints):
    """Build the targets, recording each one's fingerprint and files as soon as it is done,
    so an interrupted run keeps what it finished."""
    for name in names:
        what, thunk = TARGETS[name]
        print(f"{name}: {what}")
        t0 = time.perf_counter()
        WRITTEN.clear()
        thunk()
        for old in set(state.get(name, {}).get("files", ())) - set(WRITTEN):
            (OUTDIR / old).unlink(missing_ok=True)       # no longer written by this target
        state[name] = {"fingerprint": prints[name], "files": sorted(set(WRITTEN))}
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
        print(f"    [{time.perf_counter() - t0:.1f}s]")


def remove_dropped(state):
    """Delete the files of targets that are no longer in TARGETS."""
    dropped = set(state) - set(TARGETS)
    kept = {f for name in set(state) - dropped for f in state[name]["files"]}
    for name in sorted(dropped):
        for f in set(state.pop(name)["files"]) - kept:
            (OUTDIR / f).unlink(missing_ok=True)
            print(f"  removed output/{f} (target {name} no longer exists)")
    if dropped:
        STATE.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")


def resolve(patterns):
    """Target names matching the command line, in TARGETS order, no repeats."""
    names = []
    for pat in patterns:
        pat = pat[:-4] if pat.endswith(".tex") else pat
        hits = [n for n in TARGETS if fnmatch.fnmatchcase(n, pat)]
        if not hits:
            raise SystemExit(f"generate_figures.py: no target matches {pat!r}.  "
                             "Try --list.")
        names += [h for h in hits if h not in names]
    return names


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if {"-h", "--help"} & set(argv):
        print(__doc__.strip())
        return
    if {"-l", "--list"} & set(argv):
        width = max(len(n) for n in TARGETS)
        for name, (what, _) in TARGETS.items():
            print(f"  {name:<{width}}  {what}")
        return

    OUTDIR.mkdir(exist_ok=True)
    state = json.loads(STATE.read_text()) if STATE.exists() else {}
    prints = fingerprints()
    remove_dropped(state)
    if "--all" in argv:
        names = list(TARGETS)
    elif argv:
        names = resolve(argv)
    else:
        names = stale(state, prints)
        print(f"{len(TARGETS) - len(names)} target(s) up to date"
              + (f"; building {', '.join(names)}" if names else ""))
    started = time.perf_counter()
    build(names, state, prints)
    print(f"{len(names)} target(s) in {time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()
