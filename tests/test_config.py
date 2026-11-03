"""Config validation: a blank value or a blank source must stop the run."""

import tomllib

import pytest

from fin42350_portfolio import config
from fin42350_portfolio.config import PARAMS, ConfigError, load_config

TEMPLATE = PARAMS.read_text()
SPEC = TEMPLATE[TEMPLATE.index("[constraints]") :]  # prefilled sections, used as they are

# Placeholder values only to exercise validation; never used for any result.
GROUP = """
[market_risk_premium]
value = 0.0
source = "test"
[gold_method]
value = "capm"
source = "test"
[hedge_r_eur]
value = 0.0
source = "test"
[hedge_r_usd]
value = 0.0
source = "test"
[market]
value = "M01"
source = "test"
[initial_weights]
source = "test"
"""


def _write(tmp_path, group=GROUP, blank_weight_for=None):
    (tmp_path / "params.toml").write_text(group + SPEC)
    ids = tomllib.loads(SPEC)["universe"]["series"]
    rows = [f"{s},{'' if s == blank_weight_for else 1 / len(ids)}" for s in ids]
    (tmp_path / config.WEIGHTS_FILE).write_text("series_id,weight\n" + "\n".join(rows) + "\n")
    return tmp_path / "params.toml"


def test_repo_config_is_complete():
    cfg = load_config()
    assert cfg.initial_weights.sum() == pytest.approx(1, abs=1e-8)


def test_filled_config_loads(tmp_path):
    cfg = load_config(_write(tmp_path))
    assert cfg.bounds["stock"] == (0.01, 0.10)
    assert len(cfg.initial_weights) == 35 and cfg.market == "M01"


def test_missing_value_raises(tmp_path):
    group = GROUP.replace('value = "capm"', 'value = ""')
    with pytest.raises(ConfigError, match=r"\[gold_method\] value is missing"):
        load_config(_write(tmp_path, group))


def test_missing_source_raises(tmp_path):
    group = GROUP.replace('value = 0.0\nsource = "test"', 'value = 0.0\nsource = ""', 1)
    with pytest.raises(ConfigError, match=r"\[market_risk_premium\] source is missing"):
        load_config(_write(tmp_path, group))


def test_missing_initial_weight_raises(tmp_path):
    with pytest.raises(ConfigError, match="weight is missing for N05"):
        load_config(_write(tmp_path, blank_weight_for="N05"))
