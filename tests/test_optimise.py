"""Optimiser checks: closed form, constraint feasibility, risk ceiling, tangency."""

import tomllib

import numpy as np
import pandas as pd
import pytest

from fin42350_portfolio import estimates, load, optimise, returns
from fin42350_portfolio.config import PARAMS

TOL = 1e-8


def test_two_asset_min_variance_closed_form():
    s1, s2, rho = 0.20, 0.30, 0.30
    s12 = rho * s1 * s2
    cov = np.array([[s1**2, s12], [s12, s2**2]])
    none = optimise.Constraints(
        ids=["x", "y"],
        rows=pd.DataFrame(columns=["group", "item", "sense", "bound", "sign", "sleeve"]),
        a=np.zeros((0, 2)),
        d=np.zeros((0, 2)),
    )
    w = optimise.min_variance(cov, none)
    w1 = (s2**2 - s12) / (s1**2 + s2**2 - 2 * s12)  # BKM ch.7
    assert w[0] == pytest.approx(w1, abs=1e-7)


@pytest.fixture(scope="module")
def setup():
    """Test inputs only, never outputs: mu = historical means, w0 = the notes' 80/13/4/3 split
    spread equally within each class (feasible by construction)."""
    raw = tomllib.loads(PARAMS.read_text())
    ids = raw["universe"]["series"]
    bounds = {k: tuple(v) for k, v in raw["constraints"].items() if k != "source"}
    data = returns.build(ids, raw["market"]["value"])
    mu = estimates.historical_stats(data, ids)["mean"]
    cov = estimates.covariance(data, ids)
    rf, _ = estimates.risk_free()
    meta = load.metadata()
    cls = meta.loc[ids, "asset_class"]
    share = {"equity": 0.80, "bond": 0.13, "cash": 0.04, "gold": 0.03}
    w0 = pd.Series({sid: share[c] / cls.eq(c).sum() for sid, c in cls.items()})
    cons = optimise.build_constraints(meta, ids, bounds)
    res = optimise.run(mu, cov, rf, w0, cons)
    return mu.to_numpy(), cov.to_numpy(), rf, w0.to_numpy(), cons, res


def _feasible(w, cons):
    assert w.min() >= -TOL
    assert abs(w.sum() - 1) <= TOL
    assert cons.report(w)["slack"].min() >= -TOL


def test_every_solution_meets_every_constraint_and_a_stays_under_w0_risk(setup):
    mu, cov, _, w0, cons, res = setup
    assert res.w0_problems == []
    for sol in (res.a, res.b, res.tangency):
        _feasible(sol.weights.to_numpy(), cons)
    for _, row in res.frontier.iterrows():
        _feasible(row[cons.ids].to_numpy(float), cons)
    a, b = res.a.weights.to_numpy(), res.b.weights.to_numpy()
    assert a @ cov @ a <= w0 @ cov @ w0 + TOL
    assert mu @ b >= mu @ w0 - TOL


def test_tangency_beats_every_frontier_point(setup):
    mu, cov, rf, _, _, res = setup
    w = res.tangency.weights.to_numpy()
    sharpe = (mu @ w - rf) / np.sqrt(w @ cov @ w)
    frontier = (res.frontier["mean"] - rf) / res.frontier["sd"]
    assert sharpe >= frontier.max() - 1e-6
