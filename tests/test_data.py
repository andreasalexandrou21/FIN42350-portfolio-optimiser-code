"""Data loading and return construction checked against the workbook and the raw files."""

import csv
import io
import tomllib
import zipfile

import pandas as pd
import pytest

from fin42350_portfolio import load, returns
from fin42350_portfolio.config import PARAMS

UNIVERSE = tomllib.loads(PARAMS.read_text())["universe"]["series"]


def _raw_rate(member: str, date_col: str, value_col: str, day: str) -> tuple[float, str]:
    """Latest value on or before `day`, read straight from the zip with the csv module."""
    with zipfile.ZipFile(load.RAW_ZIP) as z:
        rows = list(csv.DictReader(io.TextIOWrapper(z.open(member), encoding="utf-8")))
    rows = [r for r in rows if r[date_col] <= day and r[value_col] not in ("", ".")]
    best = max(rows, key=lambda r: r[date_col])
    return float(best[value_col]), best[date_col]


@pytest.mark.parametrize("sid", ["P01", "P06", "P05"])  # USD, KRW, TWD
def test_weekly_eur_conversion_on_good_friday(sid):
    # 29 Mar 2024 is Good Friday: no ECB fixing, so the Thursday rate must be used.
    day = "2024-03-29"
    ccy = load.metadata().at[sid, "currency"]
    usd_eur, ecb_day = _raw_rate("raw/ecb/EXR_D_USD.csv", "TIME_PERIOD", "OBS_VALUE", day)
    assert ecb_day == "2024-03-28"
    if ccy == "TWD":
        twd_usd, _ = _raw_rate("raw/fred/DEXTAUS.csv", "observation_date", "DEXTAUS", ecb_day)
        fx = twd_usd * usd_eur
    else:
        fx, _ = _raw_rate(f"raw/ecb/EXR_D_{ccy}.csv", "TIME_PERIOD", "OBS_VALUE", day)
    local = load.weekly_local().at[pd.Timestamp(day), sid]
    got = returns.weekly_eur_prices([sid]).at[pd.Timestamp(day), sid]
    assert got == pytest.approx(local / fx, rel=1e-14)


def test_weekly_common_sample_and_gaps():
    data = returns.build(UNIVERSE, "M01")
    gaps = set(zip(data.gaps["series_id"], data.gaps["date"].dt.strftime("%Y-%m-%d"), strict=True))
    assert gaps == {
        ("P05", "2022-02-04"),
        ("P05", "2023-01-27"),
        ("P05", "2025-01-31"),
        ("P05", "2026-02-20"),
    }  # Lunar New Year weeks; the 28 Aug 2026 week comes from the Weekly_fill re-pull
    assert load.weekly_local().loc["2026-08-28", ["P05", "P06"]].tolist() == [2647.91, 283529.0]
    assert data.n_periods_all == 261
    assert len(data.returns) == 253
    assert data.returns.notna().all().all() and data.rf.notna().all()


def test_repull_on_another_scale_is_refused(monkeypatch):
    bad = load.weekly_fill() * 1.01
    monkeypatch.setattr(load, "weekly_fill", lambda: bad)
    with pytest.raises(ValueError, match="not the same pull"):
        load._fill_from_repull(load._series_sheet("Weekly_local").astype(float))


def test_market_file_column_gives_same_returns_as_workbook_m01(monkeypatch):
    # M01 written out as a daily market-file column must reproduce the workbook's M01 returns.
    levels = returns.weekly_eur_prices(["M01"])
    daily = levels.rename(columns={"M01": "TEST Index"}).resample("D").ffill()
    monkeypatch.setattr(load, "market_daily", lambda: daily)
    ours = returns.build(UNIVERSE, "TEST Index").returns
    book = returns.build(UNIVERSE, "M01").returns
    assert len(ours) == 253
    pd.testing.assert_frame_equal(ours.rename(columns={"TEST Index": "M01"}), book)


def test_market_file_reader(tmp_path, monkeypatch, request):
    sheet = pd.DataFrame(
        {"SXW1R Index": [100.0, "#N/A N/A", 101.0]},
        index=pd.Index(pd.to_datetime(["2026-08-27", "2026-08-28", "2026-08-31"]), name="Date"),
    )
    path = tmp_path / "FIN42350_market_test.xlsx"
    sheet.to_excel(path, sheet_name="Daily")
    monkeypatch.setattr(load, "_market_file", lambda: path)
    load.market_daily.cache_clear()
    request.addfinalizer(load.market_daily.cache_clear)
    got = load.market_daily()["SXW1R Index"]
    assert got.isna().tolist() == [False, True, False] and got.iloc[-1] == 101.0
    dates = pd.DatetimeIndex(["2026-08-28", "2026-08-31"])
    assert returns.market_levels("SXW1R Index", dates).tolist() == [100.0, 101.0]
