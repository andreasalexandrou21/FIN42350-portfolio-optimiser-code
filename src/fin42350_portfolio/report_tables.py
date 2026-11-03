"""Portfolio tables for w0, A, B and the tangency portfolio."""

import numpy as np
import pandas as pd

from fin42350_portfolio.estimates import Estimates
from fin42350_portfolio.optimise import Results


def weights(res: Results) -> pd.DataFrame:
    """One column of weights per portfolio: w0, A, B, tangency."""
    sols = (res.w0, res.a, res.b, res.tangency)
    return pd.concat([s.weights.rename(s.name) for s in sols], axis=1)


def stats(w: pd.DataFrame, est: Estimates, meta: pd.DataFrame) -> pd.DataFrame:
    """Expected return, sd, Sharpe, beta and Treynor ratio of each portfolio.

    E[r_p] = w'mu and sd = sqrt(w' Sigma w) (BKM ch.7); Sharpe = (E[r_p] - rf) / sd (BKM ch.6);
    beta_p = w'beta with the cash beta set to 0, consistent with its zero covariance (BKM ch.8);
    Treynor = (E[r_p] - rf) / beta_p (BKM ch.24).
    """
    ids = w.index
    mu = est.mu.loc[ids, "mu"]
    beta = est.betas.loc[ids, "beta"].where(meta.loc[ids, "asset_class"].ne("cash"), 0.0)
    er = w.T @ mu
    sd = pd.Series(np.sqrt(np.diag(w.T @ est.cov.loc[ids, ids] @ w)), index=w.columns)
    b = w.T @ beta
    return pd.DataFrame(
        {
            "expected_return": er,
            "sd": sd,
            "sharpe": (er - est.rf) / sd,
            "beta": b,
            "treynor": (er - est.rf) / b,
        }
    ).rename_axis("portfolio")


def group_weights(w: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Weights by asset class, and by sector and region as shares of the whole portfolio and
    of the equity sleeve (the basis of the sector and region constraints)."""
    m = meta.loc[w.index]
    eq = m["asset_class"].eq("equity")
    parts = {("class", "portfolio"): w.groupby(m["asset_class"]).sum()}
    for col in ("sector", "region"):
        s = w[eq].groupby(m.loc[eq, col]).sum()
        parts[(col, "portfolio")] = s
        parts[(col, "equity sleeve")] = s / w[eq].sum()
    return pd.concat(parts, names=["group", "basis", "item"])


def risk_contribution(w: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Each asset's share of portfolio variance, w_i (Sigma w)_i / w' Sigma w (BKM ch.7:
    portfolio variance is the weighted sum of each asset's covariance with the portfolio)."""
    sw = cov.loc[w.index, w.index] @ w
    return w * sw / (w * sw).sum()
