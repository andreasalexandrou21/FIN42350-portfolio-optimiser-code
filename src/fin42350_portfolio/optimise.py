"""Markowitz optimisation under the portfolio constraints (BKM ch.7).

Every linear constraint has the form a'w (<= or >=) b d'w, where d = 1 for an absolute bound
(so d'w = 1 under the budget) and d = the equity indicator for a share of the equity sleeve.
"""

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

FRONTIER_POINTS = 50
BIND_TOL = 1e-6  # slack below this (in portfolio weight) counts as binding
_CLARABEL = {"tol_feas": 1e-9, "tol_gap_abs": 1e-9, "tol_gap_rel": 1e-9}
_CLASSES = ("equity", "bond", "cash", "gold")


@dataclass(frozen=True)
class Constraints:
    ids: list[str]
    rows: pd.DataFrame  # group, item, sense, bound, sign (+1 for <=, -1 for >=), sleeve
    a: np.ndarray  # (rows, assets)
    d: np.ndarray  # (rows, assets): ones, or the equity indicator for sleeve rows

    def _parts(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        r = self.rows
        return r["sign"].to_numpy(float), r["bound"].to_numpy(float), r["sleeve"].to_numpy()

    def natural(self) -> tuple[np.ndarray, np.ndarray]:
        """G, h with G w <= h, given sum w = 1."""
        s, b, sleeve = self._parts()
        g = s[:, None] * np.where(sleeve[:, None], self.a - b[:, None] * self.d, self.a)
        return g, np.where(sleeve, 0.0, s * b)

    def homogeneous(self) -> np.ndarray:
        """G with G y <= 0 for any positive multiple y of a feasible w (tangency form)."""
        s, b, _ = self._parts()
        return s[:, None] * (self.a - b[:, None] * self.d)

    def report(self, w: np.ndarray, duals: np.ndarray | None = None) -> pd.DataFrame:
        """Value (a'w / d'w), slack in portfolio weight (negative = broken) and binding flag."""
        s, b, _ = self._parts()
        aw, dw = self.a @ w, self.d @ w
        out = self.rows[["group", "item", "sense", "bound"]].copy()
        out["value"] = aw / dw
        out["slack"] = s * (b * dw - aw)
        out["binding"] = out["slack"].abs() < BIND_TOL
        out["dual"] = np.nan if duals is None else duals
        return out


def build_constraints(
    meta: pd.DataFrame, ids: list[str], bounds: dict[str, tuple[float, float]]
) -> Constraints:
    """The notes' constraints for `ids`, with sets taken from the metadata (asset class,
    sector, region) and ranges from the config."""
    m = meta.loc[ids]
    cls = m["asset_class"]
    unknown = sorted(set(cls) - set(_CLASSES))
    if unknown:
        raise ValueError(f"no constraints defined for asset classes {unknown}")
    eq = cls.eq("equity")
    if m.loc[eq, ["sector", "region"]].isna().any().any():
        raise ValueError("every equity needs a sector and a region")
    n, ones, eye = len(ids), np.ones(len(ids)), np.eye(len(ids))
    ranges = [("class", c, cls.eq(c).to_numpy(float), ones, bounds[c]) for c in _CLASSES]
    for key, c in (("stock", "equity"), ("bond_fund", "bond")):
        ranges += [(key, ids[i], eye[i], ones, bounds[key]) for i in np.flatnonzero(cls.eq(c))]
    for col in ("sector", "region"):
        for g in sorted(m.loc[eq, col].unique()):
            a = m[col].eq(g).to_numpy(float)
            ranges.append((col, g, a, eq.to_numpy(float), bounds[f"{col}_of_equity"]))

    rows, a_rows, d_rows = [], [], []
    for group, item, a, d, (lo, hi) in ranges:
        for sense, bound, sign in ((">=", lo, -1), ("<=", hi, 1)):
            rows.append((group, item, sense, bound, sign, group in ("sector", "region")))
            a_rows.append(a)
            d_rows.append(d)
    table = pd.DataFrame(rows, columns=["group", "item", "sense", "bound", "sign", "sleeve"])
    shape = (0, n)
    return Constraints(
        ids=list(ids),
        rows=table,
        a=np.array(a_rows).reshape(-1, n) if a_rows else np.zeros(shape),
        d=np.array(d_rows).reshape(-1, n) if d_rows else np.zeros(shape),
    )


@dataclass(frozen=True)
class Solution:
    name: str
    weights: pd.Series
    constraints: pd.DataFrame  # Constraints.report plus the problem's own risk/return and budget


def _solve(name: str, objective: cp.Minimize | cp.Maximize, cons: list) -> None:
    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.CLARABEL, **_CLARABEL)
    if prob.status != cp.OPTIMAL:
        raise RuntimeError(f"{name}: solver status {prob.status}")


def _feasible_set(w: cp.Variable, cons: Constraints) -> tuple[list, cp.Constraint, cp.Constraint]:
    g, h = cons.natural()
    budget = cp.sum(w) == 1
    linear = g @ w <= h
    return [budget, w >= 0, linear] if len(h) else [budget, w >= 0], budget, linear


def _variance(w: cp.Expression, cov: np.ndarray) -> cp.Expression:
    # psd_wrap: the cash row/column is zero, so cov is only PSD; check_risky_pd covers the rest.
    return cp.quad_form(w, cp.psd_wrap(cov))


def _extra(group: str, sense: str, bound: float, value: float, dual) -> dict:
    slack = bound - value if sense == "<=" else value - bound if sense == ">=" else 0.0
    return {
        "group": group,
        "item": "",
        "sense": sense,
        "bound": bound,
        "value": value,
        "slack": slack,
        "binding": abs(slack) < BIND_TOL,
        "dual": float(np.squeeze(dual)),
    }


def _solution(name, cons, w, linear, extras) -> Solution:
    duals = linear.dual_value if len(cons.rows) else None
    table = pd.concat([pd.DataFrame(extras), cons.report(w, duals)], ignore_index=True)
    return Solution(name, pd.Series(w, index=cons.ids, name=name), table)


def problem_a(mu: np.ndarray, cov: np.ndarray, w0: np.ndarray, cons: Constraints) -> Solution:
    """Problem A: max mu'w s.t. w' Sigma w <= w0' Sigma w0 and the constraints."""
    var0 = float(w0 @ cov @ w0)
    w = cp.Variable(len(mu))
    base, budget, linear = _feasible_set(w, cons)
    risk = _variance(w, cov) <= var0
    _solve("Problem A", cp.Maximize(mu @ w), [*base, risk])
    x = w.value
    extras = [
        _extra("risk: variance", "<=", var0, float(x @ cov @ x), risk.dual_value),
        _extra("budget", "=", 1.0, float(x.sum()), budget.dual_value),
    ]
    return _solution("A", cons, x, linear, extras)


def problem_b(mu: np.ndarray, cov: np.ndarray, w0: np.ndarray, cons: Constraints) -> Solution:
    """Problem B (robustness): min w' Sigma w s.t. mu'w >= mu'w0 and the constraints."""
    r0 = float(mu @ w0)
    w = cp.Variable(len(mu))
    base, budget, linear = _feasible_set(w, cons)
    ret = mu @ w >= r0
    _solve("Problem B", cp.Minimize(_variance(w, cov)), [*base, ret])
    x = w.value
    extras = [
        _extra("return", ">=", r0, float(mu @ x), ret.dual_value),
        _extra("budget", "=", 1.0, float(x.sum()), budget.dual_value),
    ]
    return _solution("B", cons, x, linear, extras)


def min_variance(cov: np.ndarray, cons: Constraints, min_return: float | None = None, mu=None):
    """Minimum-variance weights under the constraints, optionally with mu'w >= min_return."""
    w = cp.Variable(cov.shape[0])
    base, _, _ = _feasible_set(w, cons)
    if min_return is not None:
        base.append(mu @ w >= min_return)
    _solve("minimum variance", cp.Minimize(_variance(w, cov)), base)
    return w.value


def max_return(mu: np.ndarray, cons: Constraints) -> np.ndarray:
    """Maximum expected-return weights under the constraints (an LP)."""
    w = cp.Variable(len(mu))
    base, _, _ = _feasible_set(w, cons)
    _solve("maximum return", cp.Maximize(mu @ w), base)
    return w.value


def frontier(mu: np.ndarray, cov: np.ndarray, cons: Constraints) -> pd.DataFrame:
    """Constrained minimum-variance frontier (BKM ch.7): targets evenly spaced from the
    minimum-variance portfolio's return to the maximum attainable return."""
    lo = float(mu @ min_variance(cov, cons))
    hi = float(mu @ max_return(mu, cons))
    rows = []
    for target in np.linspace(lo, hi, FRONTIER_POINTS):
        w = min_variance(cov, cons, target, mu)
        rows.append([target, mu @ w, np.sqrt(w @ cov @ w), *w])
    return pd.DataFrame(rows, columns=["target", "mean", "sd", *cons.ids])


def tangency(mu: np.ndarray, cov: np.ndarray, rf: float, cons: Constraints) -> Solution:
    """Maximum Sharpe ratio portfolio under the constraints (BKM ch.7 optimal risky portfolio).

    Exact convex form: min y' Sigma y s.t. (mu - rf)'y = 1, y >= 0 and the homogenised
    constraints G y <= 0; then w = y / sum(y). Valid because every constraint is a'w <= b d'w.
    """
    y = cp.Variable(len(mu))
    cons_y = [(mu - rf) @ y == 1, y >= 0]
    if len(cons.rows):
        cons_y.append(cons.homogeneous() @ y <= 0)
    _solve("tangency", cp.Minimize(_variance(y, cov)), cons_y)
    w = y.value / y.value.sum()
    extras = [_extra("budget", "=", 1.0, float(w.sum()), np.nan)]
    table = pd.concat([pd.DataFrame(extras), cons.report(w)], ignore_index=True)
    return Solution("tangency", pd.Series(w, index=cons.ids, name="tangency"), table)


def validate_w0(w0: pd.Series, cons: Constraints) -> list[str]:
    """Every way the initial weights break the budget or a constraint (empty if none)."""
    problems = []
    if abs(w0.sum() - 1) > 1e-8:
        problems.append(f"weights sum to {w0.sum():.8f}, not 1")
    problems += [f"{sid} weight {v} is negative" for sid, v in w0[w0 < 0].items()]
    rep = cons.report(w0.to_numpy(float))
    for r in rep[rep["slack"] < -1e-8].itertuples():
        problems.append(f"{r.group} {r.item}: {r.value:.6f} breaks {r.sense} {r.bound}")
    return problems


@dataclass(frozen=True)
class Results:
    w0: Solution
    a: Solution
    b: Solution
    tangency: Solution
    frontier: pd.DataFrame
    w0_problems: list[str]


def run(mu: pd.Series, cov: pd.DataFrame, rf: float, w0: pd.Series, cons: Constraints) -> Results:
    """w0 check, Problems A and B, the frontier and the tangency portfolio."""
    ids = cons.ids
    m, s, w = mu[ids].to_numpy(float), cov.loc[ids, ids].to_numpy(float), w0[ids].to_numpy(float)
    problems = validate_w0(w0[ids], cons)
    for p in problems:
        print(f"initial portfolio: {p}")
    return Results(
        w0=Solution("w0", w0[ids].rename("w0"), cons.report(w)),
        a=problem_a(m, s, w, cons),
        b=problem_b(m, s, w, cons),
        tangency=tangency(m, s, rf, cons),
        frontier=frontier(m, s, cons),
        w0_problems=problems,
    )
