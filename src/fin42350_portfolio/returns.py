"""Weekly EUR returns and risk-free returns."""

from dataclasses import dataclass

import pandas as pd

from fin42350_portfolio import load

PERIODS_PER_YEAR = 52  # weekly returns
_MAX_FX_LAG_DAYS = 4  # a rate older than this before a Friday means a hole in the FX file


def _on_or_before(index: pd.DatetimeIndex, dates: pd.DatetimeIndex, what: str) -> pd.DatetimeIndex:
    """For each date, the latest date in `index` on or before it."""
    pos = index.searchsorted(dates, side="right") - 1
    if (pos < 0).any():
        raise ValueError(f"{what}: no observation on or before {dates[pos < 0][0].date()}")
    found = index[pos]
    lag = (dates - found).days
    if lag.max() > _MAX_FX_LAG_DAYS:
        worst = dates[lag.argmax()].date()
        raise ValueError(f"{what}: latest observation before {worst} is {lag.max()} days old")
    return found


def fx_per_eur(currency: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Foreign units per EUR on or before each date, same convention as the workbook's FX sheet.

    ECB reference rates; TWD = FRED TWD per USD (on the ECB date, or the latest earlier FRED
    date) x ECB USD per EUR, the workbook's method.
    """
    if currency == "EUR":
        return pd.Series(1.0, index=dates)
    if currency == "TWD":
        usd = load.ecb_daily("EXR_D_USD")
        ecb_dates = _on_or_before(usd.index, dates, "ECB USD")
        twd = load.fred_twd_per_usd()
        fred_dates = _on_or_before(twd.index, ecb_dates, "FRED DEXTAUS")
        return pd.Series(twd[fred_dates].to_numpy() * usd[ecb_dates].to_numpy(), index=dates)
    rates = load.ecb_daily(f"EXR_D_{currency}")
    used = _on_or_before(rates.index, dates, f"ECB {currency}")
    return pd.Series(rates[used].to_numpy(), index=dates)


def weekly_eur_prices(ids: list[str]) -> pd.DataFrame:
    """Weekly_local converted to EUR: P_EUR = P_local / FX (foreign units per EUR)."""
    local = load.weekly_local()[ids]
    ccy = load.metadata().loc[ids, "currency"]
    fx = {c: fx_per_eur(c, local.index) for c in ccy.unique()}
    return local / pd.DataFrame({sid: fx[ccy[sid]] for sid in ids})


def weekly_rf(dates: pd.DatetimeIndex) -> pd.Series:
    """Weekly EUR STR return from the compounded index, the RiskFree ratio method.

    rf(t) = CI(last EUR STR date on or before Friday t) / CI(same for Friday t-1) - 1.
    """
    ci = load.ecb_daily("EST_CI")
    level = ci[_on_or_before(ci.index, dates, "EUR STR index")].to_numpy()
    return pd.Series(level[1:] / level[:-1] - 1, index=dates[1:], name="rf")


def market_levels(market: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Level of `market` from the market file on the last trading day on or before each date."""
    daily = load.market_daily()
    if market not in daily.columns:
        raise ValueError(
            f"market {market!r} is not a workbook Series ID or a column of the market file "
            f"(columns: {', '.join(map(str, daily.columns))})"
        )
    s = daily[market].dropna()
    return pd.Series(s[_on_or_before(s.index, dates, market)].to_numpy(), index=dates)


def simple_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """P(t)/P(t-1) - 1 on consecutive rows; a missing price blanks both returns it touches."""
    return (prices / prices.shift(1) - 1).iloc[1:]


def gap_table(prices: pd.DataFrame) -> pd.DataFrame:
    """Every missing price as (series_id, asset, date)."""
    missing = prices.isna().stack()
    missing = missing[missing]
    out = pd.DataFrame(
        {
            "series_id": missing.index.get_level_values(1),
            "date": missing.index.get_level_values(0),
        }
    )
    out.insert(1, "asset", out["series_id"].map(load.metadata()["asset"]))
    return out.sort_values(["series_id", "date"], ignore_index=True)


@dataclass(frozen=True)
class ReturnData:
    returns: pd.DataFrame  # EUR simple returns, assets and market, common sample
    rf: pd.Series  # period risk-free return on the same dates
    gaps: pd.DataFrame  # every missing price
    dropped: pd.DatetimeIndex  # periods removed because some asset had no return
    n_periods_all: int  # return periods before dropping


def build(ids: list[str], market: str) -> ReturnData:
    """Weekly returns of `ids` plus `market` on the common sample: a period is dropped if any
    series has no return in it (decision of 26 Sep 2026), so every asset uses the same dates.

    `market` is a workbook Series ID (M01) or a column of the market file (data/README.md).
    """
    external = market not in load.metadata().index
    cols = ids if external else [*ids, market]
    prices = weekly_eur_prices(cols)
    rf = weekly_rf(prices.index)
    if external:
        prices = prices.assign(**{market: market_levels(market, prices.index)})
    rets = simple_returns(prices)

    kept = rets.dropna()
    rf = rf.reindex(kept.index)
    if rf.isna().any():
        raise ValueError(f"no risk-free return for {rf.index[rf.isna()][0].date()}")
    return ReturnData(
        returns=kept,
        rf=rf,
        gaps=gap_table(prices),
        dropped=rets.index.difference(kept.index),
        n_periods_all=len(rets),
    )
