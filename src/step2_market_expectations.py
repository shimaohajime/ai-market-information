"""Estimate the pre-specified Step 2 market-expectations tests.

The four Step 1 predictions are strictly pre-announcement, out-of-sample
objects.  Step 2 uses earnings-announcement CARs to ask which simulated
expectation best resembles the expectation embedded in the market price.

The two main decompositions are:

    u_L1 = SUE - prediction_L1
    p_L  = prediction_L1 - prediction_L0

and

    u_H1 = SUE - prediction_H1
    q_H  = prediction_H1 - prediction_L1.

If the price-added low-compute prediction is already incorporated before the
announcement, CAR should load on u_L1 but not p_L.  If the high-compute
increment is also incorporated, CAR should load on u_H1 but not q_H.  The
comparison is about simulated information-processing technologies; it is not a
test of actual AI adoption or evidence that q_H is literally private
information.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "output/step2"


PRIMARY_OUTCOME = "car_market_adjusted_0_p1"
ROBUST_OUTCOMES = ["car_market_adjusted_m1_p1", "bhar_market_adjusted_0_p1"]
CONTROL_COLUMNS = ["log_market_equity_lag1", "log1p_analyst_coverage"]


def winsorize_by_year(values: pd.Series, years: pd.Series, lower: float, upper: float) -> pd.Series:
    """Winsorize within announcement year, retaining a transparent deterministic rule."""
    result = values.astype(float).copy()
    for _, index in result.groupby(years).groups.items():
        valid = result.loc[index].dropna()
        if len(valid) < 10:
            continue
        lo, hi = valid.quantile([lower, upper])
        result.loc[index] = result.loc[index].clip(lo, hi)
    return result


def prepare_analysis_sample(
    panel_path: Path, lower: float, upper: float, sample_flag: str | None = None
) -> pd.DataFrame:
    """Construct only ex-ante Step 1 signals and a complete primary event window."""
    if not panel_path.exists():
        raise FileNotFoundError(
            f"Missing Step 2 event panel at {panel_path}. Run `python3 src/step2_event_panel.py build` first."
        )
    panel = pd.read_parquet(panel_path).copy()
    required = {
        "permno", "announcement_date", "event_trade_date", "next_sue", "prediction_L0", "prediction_L1",
        "prediction_H0", "prediction_H1", "complete_0_p1", "complete_m1_p1", PRIMARY_OUTCOME,
        "log_market_equity_lag1", "analyst_coverage",
    }
    missing = sorted(required.difference(panel.columns))
    if missing:
        raise ValueError(f"Event panel is missing required columns: {', '.join(missing)}")
    if sample_flag is not None:
        if sample_flag not in panel:
            raise ValueError(f"Requested sample flag is absent from the event panel: {sample_flag}")
        panel = panel.loc[panel[sample_flag].fillna(False).astype(bool)].copy()
    for column in ["announcement_date", "event_trade_date"]:
        panel[column] = pd.to_datetime(panel[column], errors="coerce")
    numeric = [
        "permno", "next_sue", "prediction_L0", "prediction_L1", "prediction_H0", "prediction_H1",
        "log_market_equity_lag1", "analyst_coverage", PRIMARY_OUTCOME, *ROBUST_OUTCOMES,
    ]
    for column in numeric:
        if column in panel:
            panel[column] = pd.to_numeric(panel[column], errors="coerce")
    panel["announcement_year"] = panel.announcement_date.dt.year
    panel["sue_for_event_test"] = winsorize_by_year(panel.next_sue, panel.announcement_year, lower, upper)
    panel["car_market_adjusted_0_p1_w"] = winsorize_by_year(
        panel[PRIMARY_OUTCOME], panel.announcement_year, lower, upper
    )
    if "car_market_adjusted_m1_p1" in panel:
        panel["car_market_adjusted_m1_p1_w"] = winsorize_by_year(
            panel["car_market_adjusted_m1_p1"], panel.announcement_year, lower, upper
        )
    if "bhar_market_adjusted_0_p1" in panel:
        panel["bhar_market_adjusted_0_p1_w"] = winsorize_by_year(
            panel["bhar_market_adjusted_0_p1"], panel.announcement_year, lower, upper
        )
    # Keep predictions in their OOS scale.  Winsorizing SUE alone preserves
    # the exact identities u_L1 = u_H1 + q_H and u_L0 = u_L1 + p_L.
    panel["u_L1"] = panel.sue_for_event_test - panel.prediction_L1
    panel["p_L"] = panel.prediction_L1 - panel.prediction_L0
    panel["u_H0"] = panel.sue_for_event_test - panel.prediction_H0
    panel["u_H1"] = panel.sue_for_event_test - panel.prediction_H1
    panel["p_H"] = panel.prediction_H1 - panel.prediction_H0
    panel["q_H_public"] = panel.prediction_H0 - panel.prediction_L0
    panel["q_H_post_price"] = panel.prediction_H1 - panel.prediction_L1
    panel["log1p_analyst_coverage"] = np.log1p(panel.analyst_coverage.clip(lower=0.0))
    panel["size_middle"] = panel.get("size_tercile", pd.Series(index=panel.index, dtype="object")).eq("middle").astype(float)
    panel["size_high"] = panel.get("size_tercile", pd.Series(index=panel.index, dtype="object")).eq("high").astype(float)
    panel["coverage_middle"] = panel.get(
        "analyst_coverage_tercile", pd.Series(index=panel.index, dtype="object")
    ).eq("middle").astype(float)
    panel["coverage_high"] = panel.get(
        "analyst_coverage_tercile", pd.Series(index=panel.index, dtype="object")
    ).eq("high").astype(float)
    return panel


def demean_by_date(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Absorb event-trading-date fixed effects before OLS."""
    return frame[columns] - frame.groupby("event_trade_date", observed=True)[columns].transform("mean")


def cluster_meat(x: np.ndarray, residual: np.ndarray, labels: pd.Series) -> tuple[np.ndarray, int]:
    """One component of the Cameron-Gelbach-Miller two-way cluster covariance."""
    score = x * residual[:, None]
    groups = pd.Series(labels.to_numpy()).astype("category")
    codes = groups.cat.codes.to_numpy()
    n_groups = int(len(groups.cat.categories))
    grouped_scores = np.zeros((n_groups, score.shape[1]))
    np.add.at(grouped_scores, codes, score)
    meat = grouped_scores.T @ grouped_scores
    return meat, n_groups


def two_way_cluster_covariance(x: np.ndarray, residual: np.ndarray, firm: pd.Series, date: pd.Series) -> np.ndarray:
    """Two-way clustered HC covariance, with standard small-sample corrections."""
    n_obs, n_terms = x.shape
    bread = np.linalg.pinv(x.T @ x)
    firm_meat, n_firm = cluster_meat(x, residual, firm)
    date_meat, n_date = cluster_meat(x, residual, date)
    pair = firm.astype(str) + "|" + date.astype(str)
    pair_meat, n_pair = cluster_meat(x, residual, pair)

    base = (n_obs - 1.0) / max(n_obs - n_terms, 1)
    firm_correction = base * n_firm / max(n_firm - 1.0, 1.0)
    date_correction = base * n_date / max(n_date - 1.0, 1.0)
    pair_correction = base * n_pair / max(n_pair - 1.0, 1.0)
    meat = firm_correction * firm_meat + date_correction * date_meat - pair_correction * pair_meat
    covariance = bread @ meat @ bread
    return (covariance + covariance.T) / 2.0


def normal_two_sided_pvalue(statistic: float) -> float:
    if not np.isfinite(statistic):
        return np.nan
    return math.erfc(abs(statistic) / math.sqrt(2.0))


def estimate_specification(
    frame: pd.DataFrame,
    outcome: str,
    complete_column: str,
    residual_signal: str,
    incremental_signal: str,
    label: str,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Estimate one CAR specification with date FE and two-way clustered SEs."""
    controls = [*CONTROL_COLUMNS, "size_middle", "size_high", "coverage_middle", "coverage_high"]
    terms = [residual_signal, incremental_signal, *controls]
    needed = [outcome, complete_column, "permno", "event_trade_date", *terms]
    sample = frame.loc[frame[complete_column].fillna(False), needed].dropna().copy()
    if sample.empty:
        raise ValueError(f"No usable observations for {label} with outcome {outcome}.")
    demeaned = demean_by_date(sample, [outcome, *terms])
    y = demeaned[outcome].to_numpy(dtype=float)
    x = demeaned[terms].to_numpy(dtype=float)
    coefficient = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    residual = y - x @ coefficient
    covariance = two_way_cluster_covariance(x, residual, sample.permno, sample.event_trade_date)
    variance = np.clip(np.diag(covariance), 0.0, None)
    standard_error = np.sqrt(variance)
    with np.errstate(divide="ignore", invalid="ignore"):
        statistic = coefficient / standard_error
    pvalue = np.array([normal_two_sided_pvalue(value) for value in statistic])
    within_tss = float(np.dot(y, y))
    within_r2 = float(1.0 - np.dot(residual, residual) / within_tss) if within_tss > 0 else np.nan

    beta, delta = coefficient[:2]
    contrast = delta - beta
    contrast_variance = covariance[1, 1] + covariance[0, 0] - 2.0 * covariance[0, 1]
    contrast_se = math.sqrt(max(float(contrast_variance), 0.0))
    contrast_statistic = contrast / contrast_se if contrast_se > 0 else np.nan
    summary = {
        "specification": label,
        "outcome": outcome,
        "n_obs": len(sample),
        "n_firms": int(sample.permno.nunique()),
        "n_event_dates": int(sample.event_trade_date.nunique()),
        "within_r2": within_r2,
        "beta_residual": beta,
        "se_beta_residual": standard_error[0],
        "p_beta_residual": pvalue[0],
        "delta_increment": delta,
        "se_delta_increment": standard_error[1],
        "p_delta_increment": pvalue[1],
        "delta_minus_beta": contrast,
        "se_delta_minus_beta": contrast_se,
        "p_delta_equals_beta": normal_two_sided_pvalue(contrast_statistic),
        "increment_over_residual": delta / beta if abs(beta) > 1e-12 else np.nan,
    }
    detail = pd.DataFrame({
        "specification": label,
        "outcome": outcome,
        "term": terms,
        "coefficient": coefficient,
        "std_error_two_way_cluster": standard_error,
        "z_stat": statistic,
        "p_value_normal": pvalue,
    })
    return summary, detail


def estimate_all(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the two pre-specified decompositions on the primary and robust outcomes."""
    designs = [
        ("low_compute_price_signal", "u_L1", "p_L"),
        ("high_compute_price_signal", "u_H1", "p_H"),
        ("high_compute_public_signal", "u_H0", "q_H_public"),
        ("high_compute_post_price_signal", "u_H1", "q_H_post_price"),
    ]
    outcomes = [
        ("car_market_adjusted_0_p1_w", "complete_0_p1"),
        ("car_market_adjusted_m1_p1_w", "complete_m1_p1"),
        ("bhar_market_adjusted_0_p1_w", "complete_0_p1"),
    ]
    summaries: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for outcome, complete_column in outcomes:
        for label, residual_signal, incremental_signal in designs:
            summary, detail = estimate_specification(
                frame, outcome, complete_column, residual_signal, incremental_signal, label
            )
            summaries.append(summary)
            details.append(detail)
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def estimate_stability_splits(frame: pd.DataFrame) -> pd.DataFrame:
    """Repeat the primary specification in two transparent calendar subsamples."""
    periods = {
        "2018_2021": [2018, 2019, 2020, 2021],
        "2022_2025": [2022, 2023, 2024, 2025],
    }
    designs = [
        ("low_compute_price_signal", "u_L1", "p_L"),
        ("high_compute_price_signal", "u_H1", "p_H"),
        ("high_compute_public_signal", "u_H0", "q_H_public"),
        ("high_compute_post_price_signal", "u_H1", "q_H_post_price"),
    ]
    output: list[dict[str, object]] = []
    for period, years in periods.items():
        subset = frame.loc[frame.announcement_year.isin(years)].copy()
        for label, residual_signal, incremental_signal in designs:
            result, _ = estimate_specification(
                subset,
                "car_market_adjusted_0_p1_w",
                "complete_0_p1",
                residual_signal,
                incremental_signal,
                label,
            )
            result["period"] = period
            output.append(result)
    return pd.DataFrame(output)


def write_report(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    output_path: Path,
    lower: float,
    upper: float,
    sample_flag: str | None,
) -> None:
    primary = summary.loc[summary.outcome.eq("car_market_adjusted_0_p1_w")].copy()
    equations = {
        "low_compute_price_signal": "CAR = beta * (SUE - L1) + delta * (L1 - L0) + controls + event-date FE",
        "high_compute_price_signal": "CAR = beta * (SUE - H1) + delta * (H1 - H0) + controls + event-date FE",
        "high_compute_public_signal": "CAR = beta * (SUE - H0) + delta * (H0 - L0) + controls + event-date FE",
        "high_compute_post_price_signal": "CAR = beta * (SUE - H1) + delta * (H1 - L1) + controls + event-date FE",
    }
    primary_lines = "\n\n".join(
        f"### {row.specification}\n\n`{equations[row.specification]}`\n\n"
        f"- `beta` = {row.beta_residual:.6f} (SE {row.se_beta_residual:.6f})\n"
        f"- `delta` = {row.delta_increment:.6f} (SE {row.se_delta_increment:.6f})\n"
        f"- p-value for `delta = beta` = {row.p_delta_equals_beta:.4g}"
        for row in primary.itertuples(index=False)
    )
    stability_lines = "\n".join(
        f"- {row.specification}, {row.period}: delta = {row.delta_increment:.6f} "
        f"(SE {row.se_delta_increment:.6f}); p(delta = beta) = {row.p_delta_equals_beta:.4g}"
        for row in stability.itertuples(index=False)
    )
    sample_note = "Full complete-window sample" if sample_flag is None else f"Sample flag: `{sample_flag}`"
    report = f"""# Step 2: market-expectations event study

The primary outcome is winsorized market-adjusted CAR[0,+1]. The sample uses
only complete event windows and Step 1's out-of-sample predictions. SUE and
CAR are winsorized within announcement year at [{lower:.0%}, {upper:.0%}].
Predictions are not winsorized, preserving the signal decompositions exactly.
{sample_note}.

All regressions absorb event-trading-date fixed effects, control for formation
size and analyst coverage, and report two-way clustered standard errors by
PERMNO and event date.

## Primary decompositions

{primary_lines}

## Calendar-subsample stability

{stability_lines}

`delta` near zero means the corresponding incremental model signal does not
move prices at the announcement conditional on the residual. `delta` near
`beta` means it moves announcement CAR one-for-one with residual SUE news.
This is an association-based market-expectations test, not a causal AI-adoption
estimate and not proof that a simulated high-compute signal is private.
"""
    output_path.write_text(report)


def run(args: argparse.Namespace) -> None:
    frame = prepare_analysis_sample(args.panel, args.winsor_lower, args.winsor_upper, args.sample_flag)
    summary, detail = estimate_all(frame)
    stability = estimate_stability_splits(frame)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.output_dir / f"{args.prefix}_summary.csv", index=False)
    detail.to_csv(args.output_dir / f"{args.prefix}_coefficients.csv", index=False)
    stability.to_csv(args.output_dir / f"{args.prefix}_stability.csv", index=False)
    write_report(
        summary, stability, args.output_dir / f"{args.prefix}_report.md",
        args.winsor_lower, args.winsor_upper, args.sample_flag,
    )
    print(summary.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--prefix", default="market_expectations")
    parser.add_argument("--sample-flag", default=None)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
