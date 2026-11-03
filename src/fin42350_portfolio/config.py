"""Parameters from config/. Every value needs a source; a blank stops the run and names it."""

import tomllib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
PARAMS = CONFIG_DIR / "params.toml"
WEIGHTS_FILE = "initial_weights.csv"  # read from the same folder as params.toml

# Group-chosen fields, checked in this order. None = a number, else the allowed values.
FIELDS: dict[str, tuple[str, ...] | None] = {
    "market_risk_premium": None,
    "gold_method": ("capm", "historical"),
    "hedge_r_eur": None,
    "hedge_r_usd": None,
}
BOUNDS = (
    "equity",
    "bond",
    "cash",
    "gold",
    "stock",
    "bond_fund",
    "sector_of_equity",
    "region_of_equity",
)


class ConfigError(ValueError):
    """A config value or its source is missing or invalid."""


@dataclass(frozen=True)
class Config:
    market_risk_premium: float
    gold_method: str
    hedge_r_eur: float
    hedge_r_usd: float
    bounds: dict[str, tuple[float, float]]
    universe: list[str]
    market: str
    value_eur: float
    initial_weights: pd.Series
    sources: dict[str, str]
    path: Path


def _is_number(v: object) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool)


def _source(raw: dict, section: str, where: str) -> str:
    source = raw.get(section, {}).get("source", "")
    if not isinstance(source, str) or not source.strip():
        raise ConfigError(f"{where}: [{section}] source is missing")
    return source.strip()


def load_config(path: Path | str = PARAMS) -> Config:
    """Read and validate params.toml and the initial_weights.csv next to it."""
    path = Path(path)
    with path.open("rb") as f:
        raw = tomllib.load(f)
    where = str(path)
    values: dict[str, object] = {}
    sources: dict[str, str] = {}

    for name, allowed in FIELDS.items():
        value = raw.get(name, {}).get("value", "")
        if value == "":
            raise ConfigError(f"{where}: [{name}] value is missing")
        if allowed is None and not _is_number(value):
            raise ConfigError(f"{where}: [{name}] value must be a number, got {value!r}")
        if allowed is not None and value not in allowed:
            raise ConfigError(f"{where}: [{name}] value must be one of {allowed}, got {value!r}")
        values[name] = float(value) if allowed is None else value
        sources[name] = _source(raw, name, where)

    market = raw.get("market", {}).get("value", "")
    if not isinstance(market, str) or not market.strip():
        raise ConfigError(f"{where}: [market] value is missing")
    sources["market"] = _source(raw, "market", where)

    sources["constraints"] = _source(raw, "constraints", where)
    bounds = {}
    for name in BOUNDS:
        pair = raw["constraints"].get(name)
        ok = isinstance(pair, list) and len(pair) == 2 and all(map(_is_number, pair))
        if not ok or pair[0] > pair[1]:
            raise ConfigError(f"{where}: [constraints] {name} must be [lower, upper], got {pair!r}")
        bounds[name] = (float(pair[0]), float(pair[1]))

    sources["universe"] = _source(raw, "universe", where)
    universe = raw["universe"].get("series", [])
    if not universe or not all(isinstance(s, str) and s for s in universe):
        raise ConfigError(f"{where}: [universe] needs a list of series")

    sources["portfolio"] = _source(raw, "portfolio", where)
    value_eur = raw["portfolio"].get("value_eur", "")
    if not _is_number(value_eur):
        raise ConfigError(f"{where}: [portfolio] value_eur is missing")

    sources["initial_weights"] = _source(raw, "initial_weights", where)
    weights = _load_weights(path.parent / WEIGHTS_FILE, universe)

    return Config(
        **values,
        bounds=bounds,
        universe=list(universe),
        market=market.strip(),
        value_eur=float(value_eur),
        initial_weights=weights,
        sources=sources,
        path=path,
    )


def _load_weights(path: Path, universe: list[str]) -> pd.Series:
    df = pd.read_csv(path, dtype={"series_id": str})
    ids = df["series_id"].to_list()
    missing, extra = set(universe) - set(ids), set(ids) - set(universe)
    if missing or extra or len(ids) != len(set(ids)):
        raise ConfigError(
            f"{path}: series_id must list each universe asset once "
            f"(missing {sorted(missing)}, extra {sorted(extra)})"
        )
    blank = df.loc[df["weight"].isna(), "series_id"].to_list()
    if blank:
        raise ConfigError(f"{path}: weight is missing for {', '.join(blank)}")
    weight = pd.to_numeric(df["weight"], errors="coerce")
    if weight.isna().any():
        bad = df.loc[weight.isna(), "series_id"].to_list()
        raise ConfigError(f"{path}: weight is not a number for {', '.join(bad)}")
    return pd.Series(weight.to_numpy(), index=ids, name="w0").reindex(universe)
