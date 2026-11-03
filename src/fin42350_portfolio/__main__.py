"""Full pipeline: uv run python -m fin42350_portfolio [--config path]."""

import argparse
import sys
from pathlib import Path

import pandas as pd

from fin42350_portfolio import estimates, export, load, optimise, report_tables, returns
from fin42350_portfolio.config import PARAMS, ConfigError, load_config

OUTPUTS = load.ROOT / "outputs"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m fin42350_portfolio", description=__doc__)
    parser.add_argument("--config", type=Path, default=PARAMS, help="path to params.toml")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.config)
    except ConfigError as e:
        sys.exit(f"Config not ready: {e}")

    data = returns.build(cfg.universe, cfg.market)
    print("Missing weekly prices:")
    print(data.gaps.to_string(index=False) if len(data.gaps) else "none")
    print(f"{len(data.returns)} of {data.n_periods_all} return periods kept (common sample)")

    est = estimates.estimate(cfg, data)
    print(f"rf = {est.rf:.5f} (EUR STR on {est.rf_date.date()})")
    print(f"risky covariance: condition number {est.cond:.1f}, min eigenvalue {est.min_eig:.3e}")

    meta = load.metadata()
    cons = optimise.build_constraints(meta, cfg.universe, cfg.bounds)
    res = optimise.run(est.mu["mu"], est.cov, est.rf, cfg.initial_weights, cons)

    with pd.option_context("display.float_format", "{:.4f}".format):
        print(report_tables.stats(report_tables.weights(res), est, meta).to_string())
    for path in export.write_all(OUTPUTS, cfg, data, est, res):
        print(f"wrote {path.relative_to(load.ROOT)}")


if __name__ == "__main__":
    main()
