"""Test whether simulated high-compute signals appear before earnings day.

The event-day result alone cannot establish when information entered prices.
This runner regresses formation-to-announcement market-adjusted returns on two
pre-announcement signals: H0-L0 (additional public-information processing
before observing the current price) and H1-L1 (additional processing after the
price snapshot).  A zero coefficient is deliberately interpreted as ambiguous,
not proof of immediate price incorporation.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from step2_market_expectations import normal_two_sided_pvalue, two_way_cluster_covariance, winsorize_by_year


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel_pre_returns.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "output/step2"


def prepare_panel(path: Path, lower: float, upper: float) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing pre-announcement panel: {path}")
    panel = pd.read_parquet(path).copy()
    required = {
        "permno", "formation_date", "announcement_date", "prediction_L0", "prediction_L1", "prediction_H0",
        "prediction_H1", "log_market_equity_lag1", "analyst_coverage", "pre_trading_days",
        "pre_car_market_adjusted", "pre_car_market_adjusted_per_day", "pre_car_market_adjusted_early20",
        "pre_car_market_adjusted_late20", "pre_complete_early20", "pre_complete_late20",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Missing required pre-announcement columns: {', '.join(missing)}")
    panel["formation_date"] = pd.to_datetime(panel.formation_date, errors="coerce")
    panel["announcement_date"] = pd.to_datetime(panel.announcement_date, errors="coerce")
    panel["announcement_year"] = panel.announcement_date.dt.year
    numeric = [
        "permno", "prediction_L0", "prediction_L1", "prediction_H0", "prediction_H1",
        "log_market_equity_lag1", "analyst_coverage", "pre_trading_days", "pre_car_market_adjusted",
        "pre_car_market_adjusted_per_day", "pre_car_market_adjusted_early20", "pre_car_market_adjusted_late20",
    ]
    for column in numeric:
        panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["q_H_public"] = panel.prediction_H0 - panel.prediction_L0
    panel["q_H_post_price"] = panel.prediction_H1 - panel.prediction_L1
    panel["log1p_analyst_coverage"] = np.log1p(panel.analyst_coverage.clip(lower=0))
    panel["pre_days_sq"] = panel.pre_trading_days**2
    panel["size_middle"] = panel.get("size_tercile", pd.Series(index=panel.index, dtype="object")).eq("middle").astype(float)
    panel["size_high"] = panel.get("size_tercile", pd.Series(index=panel.index, dtype="object")).eq("high").astype(float)
    panel["coverage_middle"] = panel.get(
        "analyst_coverage_tercile", pd.Series(index=panel.index, dtype="object")
    ).eq("middle").astype(float)
    panel["coverage_high"] = panel.get(
        "analyst_coverage_tercile", pd.Series(index=panel.index, dtype="object")
    ).eq("high").astype(float)
    for column in [
        "pre_car_market_adjusted_per_day", "pre_car_market_adjusted_early20", "pre_car_market_adjusted_late20",
    ]:
        panel[f"{column}_w"] = winsorize_by_year(panel[column], panel.announcement_year, lower, upper)
    return panel


def demean_by_formation_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    return frame[columns] - frame.groupby("formation_date", observed=True)[columns].transform("mean")


def estimate(
    panel: pd.DataFrame,
    outcome: str,
    signal: str,
    complete_flag: str | None,
    include_horizon_controls: bool,
) -> dict[str, object]:
    controls = [
        "log_market_equity_lag1", "log1p_analyst_coverage", "size_middle", "size_high", "coverage_middle", "coverage_high",
    ]
    if include_horizon_controls:
        controls += ["pre_trading_days", "pre_days_sq"]
    terms = [signal, *controls]
    needed = [outcome, "permno", "formation_date", *terms]
    sample = panel[needed].dropna().copy()
    if complete_flag is not None:
        sample = sample.loc[panel.loc[sample.index, complete_flag].fillna(False)].copy()
    if outcome.endswith("per_day_w"):
        sample = sample.loc[panel.loc[sample.index, "pre_trading_days"].gt(0)].copy()
    demeaned = demean_by_formation_date(sample, [outcome, *terms])
    y = demeaned[outcome].to_numpy(dtype=float)
    x = demeaned[terms].to_numpy(dtype=float)
    beta = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    residual = y - x @ beta
    covariance = two_way_cluster_covariance(x, residual, sample.permno, sample.formation_date)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    z = beta / se
    within_tss = float(y @ y)
    within_r2 = float(1.0 - (residual @ residual) / within_tss) if within_tss > 0 else np.nan
    return {
        "outcome": outcome,
        "signal": signal,
        "n_obs": len(sample),
        "n_firms": int(sample.permno.nunique()),
        "n_formation_dates": int(sample.formation_date.nunique()),
        "within_r2": within_r2,
        "coefficient": beta[0],
        "std_error_two_way_cluster": se[0],
        "p_value": normal_two_sided_pvalue(z[0]),
        "ci_low": beta[0] - 1.96 * se[0],
        "ci_high": beta[0] + 1.96 * se[0],
    }


def run(args: argparse.Namespace) -> None:
    panel = prepare_panel(args.panel, args.winsor_lower, args.winsor_upper)
    outcomes = [
        ("pre_car_market_adjusted_per_day_w", None, True),
        ("pre_car_market_adjusted_early20_w", "pre_complete_early20", False),
        ("pre_car_market_adjusted_late20_w", "pre_complete_late20", False),
    ]
    signals = ["q_H_public", "q_H_post_price"]
    results = [
        estimate(panel, outcome, signal, flag, horizon)
        for outcome, flag, horizon in outcomes
        for signal in signals
    ]
    output = pd.DataFrame(results)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "preannouncement_signal_returns.csv", index=False)
    report = "# Step 2 pre-announcement return tests\n\n"
    report += "The coefficients relate pre-announcement market-adjusted returns to pre-specified model signals. "
    report += "A positive coefficient is consistent with gradual pre-announcement incorporation; a null coefficient is ambiguous. "
    report += "Formation-date fixed effects and two-way PERMNO/formation-date clustered standard errors are used.\n\n"
    report += "| Outcome | Signal | Coefficient | Two-way SE | 95% CI | p-value | N |\n"
    report += "| --- | --- | ---: | ---: | ---: | ---: | ---: |\n"
    for row in output.itertuples(index=False):
        report += (
            f"| {row.outcome} | {row.signal} | {row.coefficient:.6f} | "
            f"{row.std_error_two_way_cluster:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | "
            f"{row.p_value:.4g} | {row.n_obs:,} |\n"
        )
    (args.output_dir / "step2_preannouncement_report.md").write_text(report)
    print(output.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
