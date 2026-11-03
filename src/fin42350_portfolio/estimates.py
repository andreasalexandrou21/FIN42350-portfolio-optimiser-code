"""Inputs to the optimiser: risk-free rate, betas, expected returns, covariance, history."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from fin42350_portfolio import load
from fin42350_portfolio.config import Config
from fin42350_portfolio.returns import PERIODS_PER_YEAR, ReturnData


def risk_free() -> tuple[float, pd.Timestamp]:
    """Latest EUR STR observation in the data (raw/ecb/EST_WT.csv) as a decimal per year.

    The quoted ACT/360 rate is used as is (decision of 26 Sep 2026), the rf of BKM ch.9 CAPM.
    """
    wt = load.ecb_daily("EST_WT")
    return float(wt.iloc[-1]) / 100, wt.index[-1]


def betas(data: ReturnData, ids: list[str], market: str) -> pd.DataFrame:
    """OLS of each asset's excess return on the market's excess return (BKM ch.8).

    Security characteristic line R_i = alpha_i + beta_i R_M + e_i with R = r - rf per period.
    beta = S_xy / S_xx, alpha = mean(R_i) - beta mean(R_M), residual sd with n - 2 degrees of
    freedom, se(beta) = s / sqrt(S_xx), se(alpha) = s sqrt(1/n + mean(R_M)^2 / S_xx).
    alpha and residual sd are annualised (x periods, x sqrt(periods)).
    """
    ppy = PERIODS_PER_YEAR
    excess = data.returns.sub(data.rf, axis=0)
    rows = {}
    for sid in ids:
        y, x = excess[sid].to_numpy(), excess[market].to_numpy()
        n = len(y)
        dx, dy = x - x.mean(), y - y.mean()
        sxx = dx @ dx
        beta = (dx @ dy) / sxx
        alpha = y.mean() - beta * x.mean()
        resid = y - alpha - beta * x
        s = np.sqrt(resid @ resid / (n - 2))
        rows[sid] = {
            "alpha": alpha * ppy,
            "beta": beta,
            "t_alpha": alpha / (s * np.sqrt(1 / n + x.mean() ** 2 / sxx)),
            "t_beta": beta / (s / np.sqrt(sxx)),
            "r2": 1 - (resid @ resid) / (dy @ dy),
            "resid_sd": s * np.sqrt(ppy),
            "n": n,
        }
    return pd.DataFrame.from_dict(rows, orient="index").rename_axis("series_id")


def historical_stats(data: ReturnData, ids: list[str]) -> pd.DataFrame:
    """Annualised arithmetic mean, sd and Sharpe ratio of each asset's EUR returns (BKM ch.5).

    mean x periods, sd x sqrt(periods); Sharpe = mean excess return / sd of excess returns,
    both annualised the same way.
    """
    ppy = PERIODS_PER_YEAR
    r = data.returns[ids]
    excess = r.sub(data.rf, axis=0)
    return pd.DataFrame(
        {
            "mean": r.mean() * ppy,
            "sd": r.std() * np.sqrt(ppy),
            "sharpe": excess.mean() * ppy / (excess.std() * np.sqrt(ppy)),
            "n": r.count(),
        }
    ).rename_axis("series_id")


def covariance(data: ReturnData, ids: list[str]) -> pd.DataFrame:
    """Sample covariance (n - 1) of weekly EUR returns x 52; cash row and column zero.

    The input to portfolio variance w' Sigma w (BKM ch.7).
    """
    cov = data.returns[ids].cov() * PERIODS_PER_YEAR
    cash = load.metadata().loc[ids, "asset_class"].eq("cash").to_numpy()
    cov.loc[cash, :] = 0.0
    cov.loc[:, cash] = 0.0
    return cov


def check_risky_pd(cov: pd.DataFrame) -> tuple[float, float]:
    """Condition number and smallest eigenvalue of the non-cash block; raises unless PD."""
    risky = load.metadata().loc[cov.index, "asset_class"].ne("cash").to_numpy()
    eig = np.linalg.eigvalsh(cov.to_numpy()[np.ix_(risky, risky)])
    if eig[0] <= 0:
        raise ValueError(f"risky covariance block is not positive definite: min eig {eig[0]:.3e}")
    return float(eig[-1] / eig[0]), float(eig[0])


def expected_returns(cfg: Config, beta: pd.Series, hist_mean: pd.Series, rf: float) -> pd.DataFrame:
    """Annual expected return and its method for every asset in the universe.

    Equities: CAPM mu = rf + beta (E[r_M] - rf) (BKM ch.9). Bond funds: yield to maturity from
    Bond_stats (BKM ch.14); for the EUR-hedged fund the USD yield plus the covered interest
    parity hedge adjustment, y_USD + (r_EUR - r_USD) (BKM ch.23). Cash: rf. Gold: CAPM or the
    historical arithmetic mean, per the config.
    """
    meta = load.metadata().loc[cfg.universe]
    bonds = load.bond_stats()
    mu, method = {}, {}
    for sid, cls in meta["asset_class"].items():
        if cls == "equity" or (cls == "gold" and cfg.gold_method == "capm"):
            mu[sid], method[sid] = rf + beta[sid] * cfg.market_risk_premium, "CAPM"
        elif cls == "gold":
            mu[sid], method[sid] = hist_mean[sid], "historical mean"
        elif cls == "cash":
            mu[sid], method[sid] = rf, "EUR STR"
        elif cls == "bond":
            ytm = float(bonds.at[sid, "Yield to maturity (fraction)"])
            if "EUR-hedged" in bonds.at[sid, "Category"]:
                mu[sid] = ytm + (cfg.hedge_r_eur - cfg.hedge_r_usd)
                method[sid] = "USD YTM + (r_EUR - r_USD)"
            else:
                mu[sid], method[sid] = ytm, "YTM"
        else:
            raise ValueError(f"{sid}: no expected-return method for asset class {cls!r}")
    return pd.DataFrame({"mu": mu, "method": method}).rename_axis("series_id")


@dataclass(frozen=True)
class Estimates:
    rf: float
    rf_date: pd.Timestamp
    betas: pd.DataFrame
    mu: pd.DataFrame  # mu, method
    cov: pd.DataFrame
    historical: pd.DataFrame
    cond: float  # condition number of the risky covariance block
    min_eig: float


def estimate(cfg: Config, data: ReturnData) -> Estimates:
    """All optimiser inputs for the config universe from one ReturnData."""
    ids = cfg.universe
    rf, rf_date = risk_free()
    b = betas(data, ids, cfg.market)
    hist = historical_stats(data, ids)
    cov = covariance(data, ids)
    cond, min_eig = check_risky_pd(cov)
    return Estimates(
        rf=rf,
        rf_date=rf_date,
        betas=b,
        mu=expected_returns(cfg, b["beta"], hist["mean"], rf),
        cov=cov,
        historical=hist,
        cond=cond,
        min_eig=min_eig,
    )
