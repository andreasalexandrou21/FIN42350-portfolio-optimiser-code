"""Results workbook and charts for the lecturer (outputs/)."""

import platform
import subprocess
from datetime import datetime
from pathlib import Path

import cvxpy as cp
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure  # no pyplot: files only, no GUI backend on any OS
from matplotlib.ticker import MaxNLocator, PercentFormatter
from matplotlib.transforms import Bbox

from fin42350_portfolio import load, optimise, report_tables
from fin42350_portfolio.config import FIELDS, Config
from fin42350_portfolio.estimates import Estimates
from fin42350_portfolio.returns import ReturnData

# UCD brand colours (navy, blue, gold, green) from UCD's web CSS. A, the main result, is navy;
# w0 is the grey baseline it is compared with.
NAVY, BLUE, GOLD, GREEN = "#004377", "#007db8", "#fad239", "#03884a"
GOLD_DARK = "#b8930a"  # brand gold is too pale for thin lines on white
COLOURS = {"w0": "#8c8c8c", "A": NAVY, "B": BLUE, "tangency": GOLD}
INK, MUTED, GRID = "#1f1f1e", "#898781", "#e6e5e0"
# Assets on the frontier chart: hollow markers, so the filled portfolio points stand out.
ASSET_STYLE = {
    "equity": ("o", MUTED, "stocks"),
    "bond": ("s", GREEN, "bond funds"),
    "cash": ("^", NAVY, "cash"),
    "gold": ("D", GOLD_DARK, "gold"),
}
# Asset label positions tried in order (offset in points, alignment); the first free one wins.
LABEL_SPOTS = [(4, 3, "left"), (-4, 3, "right"), (4, -9, "left"), (-4, -9, "right")]
# Chart labels where the full asset name is too long.
SHORT = {
    "P03": "ASML",
    "P06": "Samsung",
    "P08": "J&J",
    "P09": "Essilor",
    "P11": "WuXi",
    "P17": "Alibaba",
    "P18": "Toyota",
    "P23": "Shenhua",
    "P25": "Berkshire",
    "P26": "JPMorgan",
    "P29": "CCB",
    "P30": "China Life",
    "N01": "Bunds",
    "N03": "EUR corp bonds",
    "N05": "US Treasuries (EUR-hedged)",
    "N07": "Cash",
    "N09": "Gold",
}
# Label placement: A sits above w0 (same risk), B left of w0 (same return), tangency between.
LABEL_AT = {
    "w0": (8, -3, "left"),
    "A": (8, -3, "left"),
    "B": (-8, -9, "right"),
    "tangency": (-8, 5, "right"),
}


def _git_commit() -> str:
    def git(*args: str) -> str:
        out = subprocess.run(["git", *args], cwd=load.ROOT, capture_output=True, text=True)
        return out.stdout.strip() if out.returncode == 0 else ""

    commit = git("rev-parse", "--short", "HEAD") or "unknown"
    dirty = git("status", "--porcelain", "--", ".", ":!outputs")  # results don't count
    return f"{commit} (uncommitted changes)" if dirty else commit


def _blocks(writer: pd.ExcelWriter, sheet: str, blocks: list[tuple[str, pd.DataFrame]]) -> None:
    """Titled tables one below the other on one sheet."""
    row = 0
    for title, df in blocks:
        pd.DataFrame([[title]]).to_excel(
            writer, sheet_name=sheet, startrow=row, header=False, index=False
        )
        df.to_excel(writer, sheet_name=sheet, startrow=row + 1)
        row += len(df) + df.columns.nlevels + 3


def _parameters(cfg: Config, est: Estimates) -> pd.DataFrame:
    s = cfg.sources
    rows = [
        (
            "risk_free_rate",
            est.rf,
            f"raw/ecb/EST_WT.csv, EUR STR on {est.rf_date.date()}, quoted ACT/360 rate",
        )
    ]
    rows += [(name, getattr(cfg, name), s[name]) for name in FIELDS]
    rows += [(f"constraint {k}", str(list(v)), s["constraints"]) for k, v in cfg.bounds.items()]
    rows += [
        ("universe", ", ".join(cfg.universe), s["universe"]),
        ("market", cfg.market, s["market"]),
        ("value_eur", cfg.value_eur, s["portfolio"]),
        ("initial_weights", "config/initial_weights.csv", s["initial_weights"]),
    ]
    return pd.DataFrame(rows, columns=["parameter", "value", "source"]).set_index("parameter")


def _run_info(cfg: Config, data: ReturnData, est: Estimates, res: optimise.Results) -> pd.Series:
    info = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": _git_commit(),
        "config": str(cfg.path),
        "returns": "weekly, Friday to Friday",
        "first_return": data.returns.index[0].date(),
        "last_return": data.returns.index[-1].date(),
        "periods_used": len(data.returns),
        "periods_before_dropping": data.n_periods_all,
        "periods_dropped": ", ".join(str(d.date()) for d in data.dropped) or "none",
        "risk_free_rate": est.rf,
        "risk_free_date": est.rf_date.date(),
        "risky_cov_condition_number": est.cond,
        "risky_cov_min_eigenvalue": est.min_eig,
        "solver": f"cvxpy {cp.__version__}, Clarabel",
        "python": platform.python_version(),
        "initial_portfolio_problems": "; ".join(res.w0_problems) or "none",
    }
    for name in FIELDS:
        info[f"config {name}"] = f"{getattr(cfg, name)} (source: {cfg.sources[name]})"
    return pd.Series(info, name="value").rename_axis("item")


def _style(ax: Axes) -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=INK, labelsize=8)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def _short(meta: pd.DataFrame) -> pd.Series:
    return pd.Series({i: SHORT.get(i, a) for i, a in meta["asset"].items()})


def _save(fig: Figure, path: Path) -> list[Path]:
    """PNG for viewing, PDF (vector) for the report."""
    fig.savefig(path, dpi=200)
    fig.savefig(path.with_suffix(".pdf"))
    return [path, path.with_suffix(".pdf")]


def _frontier_chart(
    path: Path, est: Estimates, w: pd.DataFrame, frontier: pd.DataFrame, meta: pd.DataFrame
) -> list[Path]:
    """Frontier, capital allocation line through the tangency portfolio, assets by class."""
    ids = w.index
    mu, cov = est.mu.loc[ids, "mu"], est.cov.loc[ids, ids]
    st = pd.DataFrame(
        {"sd": [(w[c] @ cov @ w[c]) ** 0.5 for c in w], "mean": [w[c] @ mu for c in w]},
        index=w.columns,
    )
    fig = Figure(figsize=(9, 6), layout="tight")
    ax = fig.subplots()
    _style(ax)

    asset_sd = pd.Series([cov.at[i, i] ** 0.5 for i in ids], index=ids)
    names = _short(meta)
    for cls, (marker, colour, label) in ASSET_STYLE.items():
        sel = meta["asset_class"].eq(cls)
        ax.scatter(
            asset_sd[sel],
            mu[sel],
            s=22,
            marker=marker,
            facecolor="white",
            edgecolor=colour,
            linewidth=1.2,
            label=label,
            zorder=2,
        )
    # Capital allocation line (BKM ch.7): rf through the tangency portfolio.
    t = st.loc["tangency"]
    slope = (t["mean"] - est.rf) / t["sd"]
    x_end = frontier["sd"].max() * 1.15
    ax.plot(
        [0, x_end],
        [est.rf, est.rf + slope * x_end],
        color=GOLD_DARK,
        linewidth=1.5,
        linestyle="--",
        label="capital allocation line",
        zorder=3,
    )
    ax.plot(frontier["sd"], frontier["mean"], color=NAVY, linewidth=2.2, label="frontier", zorder=3)

    for name, row in st.iterrows():
        ax.scatter(
            row["sd"],
            row["mean"],
            s=80 if name == "A" else 60,
            color=COLOURS[name],
            edgecolor=INK if name == "tangency" else "white",
            linewidth=1 if name == "tangency" else 1.5,
            label=name,
            zorder=5,
        )
        dx, dy, ha = LABEL_AT[name]
        ax.annotate(
            name,
            (row["sd"], row["mean"]),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            fontsize=9,
            fontweight="bold" if name == "A" else "normal",
            color=INK,
            zorder=6,
        )
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.set_xlim(left=-0.01)
    ax.set_xlabel("Standard deviation (annual)", color=INK)
    ax.set_ylabel("Expected return (annual)", color=INK)
    ax.set_title("Constrained efficient frontier", color=NAVY, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="upper left", ncol=2)
    _label_assets(ax, asset_sd, mu, names)
    return _save(fig, path)


def _label_assets(ax: Axes, x: pd.Series, y: pd.Series, names: pd.Series) -> None:
    """Name each asset point at the first spot in LABEL_SPOTS that overlaps no other label or
    point (last spot if none is free). Called after everything else is drawn."""
    fig = ax.get_figure()
    fig.draw_without_rendering()  # final layout, so pixel positions below are the saved ones
    pts = ax.transData.transform(list(zip(x, y, strict=True)))
    taken = [Bbox.from_bounds(px - 3, py - 3, 6, 6) for px, py in pts]
    for k, i in enumerate(x.index):
        own = taken[k]
        for spot in LABEL_SPOTS:
            dx, dy, ha = spot
            ann = ax.annotate(
                names[i],
                (x[i], y[i]),
                xytext=(dx, dy),
                textcoords="offset points",
                ha=ha,
                fontsize=6,
                color=MUTED,
            )
            box = ann.get_window_extent()
            free = not any(box.overlaps(b) for b in taken if b is not own)
            if free or spot == LABEL_SPOTS[-1]:
                taken.append(box)
                break
            ann.remove()


def _groups(meta: pd.DataFrame) -> list[tuple[str, list[str]]]:
    """Assets in chart order: one group per sector, then bonds, cash and gold."""
    out = []
    eq = meta[meta["asset_class"].eq("equity")]
    for sector in eq["sector"].unique():
        out.append((sector, eq.index[eq["sector"].eq(sector)].to_list()))
    for cls, title in (("bond", "Bonds"), ("cash", "Cash"), ("gold", "Gold")):
        out.append((title, meta.index[meta["asset_class"].eq(cls)].to_list()))
    return out


def _paired_chart(path: Path, df: pd.DataFrame, meta: pd.DataFrame, title: str) -> list[Path]:
    """w0 and A side by side for every asset, grouped under sector and asset-class headings."""
    names = _short(meta)
    fig = Figure(figsize=(7.5, 11), layout="constrained")
    ax = fig.subplots()
    _style(ax)
    ax.grid(axis="y", visible=False)
    y, ticks, labels = 0.0, [], []
    for group, ids in _groups(meta):
        ax.text(
            0,
            y,
            group,
            fontsize=8.5,
            fontweight="bold",
            color=NAVY,
            va="center",
            transform=ax.get_yaxis_transform(),
        )
        y += 1
        for i in ids:
            for name, off in (("w0", -0.2), ("A", 0.2)):
                ax.barh(
                    y + off,
                    df.at[i, name],
                    height=0.38,
                    color=COLOURS[name],
                    label=None if ticks else name,
                )
            ticks.append(y)
            labels.append(f"{i} {names[i]}")
            y += 1
        y += 0.4
    ax.set_yticks(ticks, labels, fontsize=7)
    ax.set_ylim(y - 0.4, -0.8)
    ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, steps=[1, 2, 5, 10]))
    ax.xaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax.xaxis.tick_top()
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title(title, color=NAVY, loc="left", fontweight="bold")
    return _save(fig, path)


def write_all(
    out_dir: Path,
    cfg: Config,
    data: ReturnData,
    est: Estimates,
    res: optimise.Results,
) -> list[Path]:
    """outputs/results.xlsx plus the frontier, weights and risk-contribution charts."""
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = load.metadata().loc[cfg.universe]
    w = report_tables.weights(res)
    xlsx = out_dir / "results.xlsx"

    assets = meta[["asset", "asset_class", "currency", "sector", "region"]]
    inputs = assets.join(est.mu).join(est.betas).join(est.historical.add_prefix("hist_"))
    constraints = pd.concat(
        {s.name: s.constraints for s in (res.w0, res.a, res.b, res.tangency)},
        names=["portfolio", "row"],
    )
    with pd.ExcelWriter(xlsx, engine="openpyxl") as xl:
        _blocks(xl, "Cov", [("Weekly EUR returns, sample covariance x52", est.cov)])
        _blocks(xl, "Inputs", [("Assets", inputs), ("Parameters", _parameters(cfg, est))])
        _blocks(
            xl,
            "Weights",
            [("Weights", assets[["asset"]].join(w)), ("EUR amounts", w * cfg.value_eur)],
        )
        _blocks(
            xl,
            "Stats",
            [
                ("Portfolio statistics", report_tables.stats(w, est, meta)),
                ("Group weights", report_tables.group_weights(w, meta)),
                ("Share of portfolio variance", report_tables.risk_contribution(w, est.cov)),
            ],
        )
        constraints.to_excel(xl, sheet_name="Constraints")
        res.frontier.to_excel(xl, sheet_name="Frontier", index_label="point")
        dropped = pd.DataFrame({"date": data.dropped.date})
        _blocks(
            xl,
            "Data_gaps",
            [("Missing prices", data.gaps), ("Periods dropped (common sample)", dropped)],
        )
        _run_info(cfg, data, est, res).to_excel(xl, sheet_name="Run_info")

    rc = report_tables.risk_contribution(w, est.cov)
    return [
        xlsx,
        *_frontier_chart(out_dir / "frontier.png", est, w, res.frontier, meta),
        *_paired_chart(out_dir / "weights.png", w, meta, "Weights, w0 vs A"),
        *_paired_chart(
            out_dir / "risk_contribution.png",
            rc,
            meta,
            "Share of portfolio variance, w0 vs A",
        ),
    ]
