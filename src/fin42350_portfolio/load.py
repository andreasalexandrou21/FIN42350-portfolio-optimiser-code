"""Read the input workbooks and the raw zip. Read-only: nothing here writes to data/.

Sheets are located by name and rows by their column A label (Series ID, ...),
never by fixed cell addresses.
"""

import io
import re
import zipfile
from datetime import datetime
from functools import cache
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
PRICE_WB = DATA / "FIN42350_price_history_2026-09-24.xlsx"
SELECTION_WB = DATA / "FIN42350_asset_selection_2026-09-24.xlsx"
RAW_ZIP = DATA / "FIN42350_price_history_raw_2026-09-24.zip"
MARKET_GLOB = "FIN42350_market_*.xlsx"  # Bloomberg pull of the market index, see data/README.md

_SHEETS = [
    "Returns_EUR",
    "Weekly_local",
    "Dates_weekly",
    "Bond_stats",
]


@cache
def _price_sheets() -> dict[str, pd.DataFrame]:
    # One read of all needed sheets, cached: the workbook is the slow part.
    return pd.read_excel(PRICE_WB, sheet_name=_SHEETS, header=None)


def raw_sheet(name: str) -> pd.DataFrame:
    """A price-history sheet as read, no header (row and column positions start at 0)."""
    return _price_sheets()[name]


def _is_date(v: object) -> bool:
    return isinstance(v, datetime)


def _label_row(raw: pd.DataFrame, label: str) -> int:
    hits = raw.index[raw[0] == label]
    if len(hits) != 1:
        raise ValueError(f"expected one row labelled {label!r} in column A, found {len(hits)}")
    return int(hits[0])


def _series_sheet(name: str) -> pd.DataFrame:
    """Date-indexed body of a sheet in the Series ID layout, one column per Series ID."""
    raw = raw_sheet(name)
    ids = raw.iloc[_label_row(raw, "Series ID"), 1:]
    rows = raw[0].map(_is_date)
    body = raw.loc[rows, 1:]
    body.index = pd.DatetimeIndex(raw.loc[rows, 0], name="date")
    body.columns = pd.Index(ids.to_list(), name="series_id")
    if not body.columns.is_unique:
        raise ValueError(f"{name}: duplicate Series IDs")
    return body


@cache
def weekly_local() -> pd.DataFrame:
    """Weekly total return index in local currency, one row per Friday.

    Korean and Taiwanese values carry the following Saturday's label in Dates_weekly but already
    sit on the Friday row; this checks that every label is the row's Friday or the day after.
    """
    prices = _series_sheet("Weekly_local").astype(float)
    labels = _series_sheet("Dates_weekly").apply(pd.to_datetime)
    lag = labels.sub(pd.Series(labels.index, index=labels.index), axis=0)
    bad = labels.notna() & ~lag.isin([pd.Timedelta(0), pd.Timedelta(days=1)])
    if bad.to_numpy().any():
        raise ValueError(f"Dates_weekly labels off the Friday row: {bad.stack()[lambda s: s]}")
    if (prices.index.dayofweek != 4).any():
        raise ValueError("Weekly_local rows are not all Fridays")
    return _fill_from_repull(prices)


def _fill_from_repull(prices: pd.DataFrame) -> pd.DataFrame:
    """Fill blanks in Weekly_local from the market file's Weekly_fill re-pull.

    The re-pull is the same field and start date, so every week both have must agree exactly;
    that is checked first. Only weeks the re-pull has are filled (the 28 Aug 2026 week cut off
    by the original end date, workbook Notes point 3). Weeks with no trading stay blank.
    """
    fill = weekly_fill().reindex(prices.index)
    old = prices[fill.columns]
    both = old.notna() & fill.notna()
    worst = (fill / old - 1).abs().where(both).max().max()
    if worst > 1e-12:
        raise ValueError(f"Weekly_fill differs from Weekly_local by {worst:.2e}: not the same pull")
    return prices.assign(**old.fillna(fill))


def _market_file() -> Path:
    hits = sorted(DATA.glob(MARKET_GLOB))
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected one data/{MARKET_GLOB}, found {len(hits)}: {[h.name for h in hits]}"
        )
    return hits[0]


@cache
def weekly_fill() -> pd.DataFrame:
    """Sheet Weekly_fill of the market file: TOT_RETURN_INDEX_GROSS_DVDS, Per=W, 27 Aug 2021 to
    29 Aug 2026. Row 1 holds each Bloomberg ticker above its date column, values in the next
    column. Returned indexed by Friday (labels are the Friday or the following Saturday), one
    column per Series ID."""
    raw = pd.read_excel(_market_file(), sheet_name="Weekly_fill", header=None)
    sid_of = pd.Series(metadata().index, index=metadata()["ticker"])
    cols = {}
    for c in raw.columns[raw.iloc[0].notna()]:
        body = raw.iloc[1:, [c, c + 1]].dropna()
        labels = pd.DatetimeIndex(body[c])
        if not labels.dayofweek.isin([4, 5]).all():
            raise ValueError(f"Weekly_fill {raw.iat[0, c]}: labels must be Fridays or Saturdays")
        fridays = labels - pd.to_timedelta(labels.dayofweek - 4, unit="D")
        sid = sid_of[str(raw.iat[0, c]).removesuffix(" Equity")]
        cols[sid] = pd.Series(body[c + 1].astype(float).to_numpy(), index=fridays)
    return pd.DataFrame(cols)


@cache
def market_daily() -> pd.DataFrame:
    """Daily EUR index levels from the market file, sheet Daily: column A dates, one column per
    index headed by its Bloomberg ticker. Non-numeric cells (Bloomberg #N/A) become blanks."""
    df = pd.read_excel(_market_file(), sheet_name="Daily", index_col=0)
    df = df[df.index.map(_is_date)]
    df.index = pd.DatetimeIndex(df.index, name="date")
    return df.apply(pd.to_numeric, errors="coerce").sort_index()


def _asset_class(label: str) -> str:
    if label == "Equity":
        return "equity"
    if label == "Index":
        return "index"
    kind = label.removeprefix("Non-equity: ")
    if kind.startswith("Cash"):
        return "cash"
    if kind.startswith("Gold"):
        return "gold"
    if "bonds" in kind or "Treasuries" in kind:
        return "bond"
    raise ValueError(f"unknown asset class label: {label!r}")


@cache
def metadata() -> pd.DataFrame:
    """One row per Series ID: asset, ticker, currency, ISIN, asset_class, sector, region.

    Sector and region come from the Picks sheet of the selection workbook (joined on ticker),
    so they are blank for reserves, non-equity funds and the market index.
    """
    raw = raw_sheet("Returns_EUR")
    fields = {
        "asset": "Asset",
        "ticker": "Ticker (selection workbook)",
        "currency": "Currency",
        "isin": "ISIN",
        "class_label": "Asset class",
    }
    ids = raw.iloc[_label_row(raw, "Series ID"), 1:].to_list()
    meta = pd.DataFrame(
        {k: raw.iloc[_label_row(raw, v), 1:].to_list() for k, v in fields.items()},
        index=pd.Index(ids, name="series_id"),
    )
    meta["asset_class"] = meta["class_label"].map(_asset_class)
    picks = pd.read_excel(SELECTION_WB, sheet_name="Picks").dropna(subset=["Ticker"])
    picks = picks.set_index("Ticker")
    meta["sector"] = meta["ticker"].map(picks["Sector"])
    meta["region"] = meta["ticker"].map(picks["Region"])
    return meta.drop(columns="class_label")


def _table(sheet: str, first_header: str) -> pd.DataFrame:
    """Rows below the header row whose column A reads `first_header`, headers as columns."""
    raw = raw_sheet(sheet)
    hdr = _label_row(raw, first_header)
    body = raw.iloc[hdr + 1 :].copy()
    body.columns = raw.iloc[hdr].to_list()
    return body


@cache
def bond_stats() -> pd.DataFrame:
    """Bond_stats rows (one per bond fund), indexed by Series ID."""
    t = _table("Bond_stats", "Series ID")
    t = t[
        t["Series ID"].map(lambda v: isinstance(v, str) and re.fullmatch(r"N\d\d", v) is not None)
    ]
    return t.set_index("Series ID")


def _raw_csv(member: str, **kwargs: object) -> pd.DataFrame:
    with zipfile.ZipFile(RAW_ZIP) as z:
        return pd.read_csv(io.BytesIO(z.read(member)), **kwargs)


@cache
def ecb_daily(key: str) -> pd.Series:
    """Daily ECB series from raw/ecb/<key>.csv, e.g. EXR_D_USD (USD per EUR), EST_CI, EST_WT."""
    df = _raw_csv(f"raw/ecb/{key}.csv", usecols=["TIME_PERIOD", "OBS_VALUE"])
    s = df.set_index(pd.DatetimeIndex(df["TIME_PERIOD"], name="date"))["OBS_VALUE"]
    s = s.astype(float).dropna().sort_index()
    if s.empty:
        raise ValueError(f"raw/ecb/{key}.csv has no observations")
    return s


@cache
def fred_twd_per_usd() -> pd.Series:
    """Daily FRED DEXTAUS (TWD per USD), the workbook's source for TWD (ECB has no TWD rate)."""
    df = _raw_csv("raw/fred/DEXTAUS.csv")
    s = df.set_index(pd.DatetimeIndex(df["observation_date"], name="date"))["DEXTAUS"]
    return s.astype(float).dropna().sort_index()
