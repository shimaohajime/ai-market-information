"""Test whether high-compute earnings signals are associated with later SUEs.

An event-day coefficient below the residual-SUE coefficient can arise because a
signal was processed before the announcement, but also because it forecasts a
less valuation-relevant, transitory earnings component.  This runner reports
the association of high-compute increments formed before announcement t with
the realized SUE at t through t+4.  Later event dates must follow a plausible
quarterly sequence (45--180 days between adjacent announcements).

The analysis is an earnings-persistence diagnostic, not a direct estimate of
cash-flow persistence or a definitive earnings-response-coefficient test.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from step2_market_expectations import (
    CONTROL_COLUMNS,
    demean_by_date,
    normal_two_sided_pvalue,
    prepare_analysis_sample,
    two_way_cluster_covariance,
    winsorize_by_year,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel.parquet"
DEFAULT_RDQ_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel_rdq.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "output/step2"
CONTROLS = [*CONTROL_COLUMNS, "size_middle", "size_high", "coverage_middle", "coverage_high"]
HORIZONS = range(0, 5)
MIN_QUARTER_GAP_DAYS = 45
MAX_QUARTER_GAP_DAYS = 180


def attach_sue_leads(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach future observed SUEs along plausible consecutive quarterly events."""
    panel = frame.copy()
    panel["formation_date"] = pd.to_datetime(panel["formation_date"], errors="coerce")
    panel["announcement_date"] = pd.to_datetime(panel["announcement_date"], errors="coerce")
    if panel.duplicated(["permno", "announcement_date"]).any():
        raise ValueError("Expected a unique PERMNO/announcement-date event panel for the persistence test")
    panel = panel.sort_values(["permno", "announcement_date"]).reset_index(drop=True)
    # Winsorize unique announced SUEs before creating leads, so an earnings
    # event is not reweighted simply because it is reachable from several
    # prior horizons.
    panel["announcement_year"] = panel.announcement_date.dt.year.astype(int)
    panel["next_sue_w_event"] = winsorize_by_year(
        panel.next_sue, panel.announcement_year, lower=0.01, upper=0.99
    )
    grouped = panel.groupby("permno", sort=False)
    next_gap_days = grouped.announcement_date.shift(-1).sub(panel.announcement_date).dt.days
    valid_step = next_gap_days.between(MIN_QUARTER_GAP_DAYS, MAX_QUARTER_GAP_DAYS)
    pieces: list[pd.DataFrame] = []
    for horizon in HORIZONS:
        target_date = grouped.announcement_date.shift(-horizon)
        target_sue = grouped.next_sue_w_event.shift(-horizon)
        valid_path = pd.Series(True, index=panel.index) if horizon == 0 else pd.Series(True, index=panel.index)
        for step in range(horizon):
            valid_path &= valid_step.groupby(panel.permno, sort=False).shift(-step).fillna(False)
        selected = panel.loc[valid_path & target_date.notna()].copy()
        selected["horizon"] = horizon
        selected["target_announcement_date"] = target_date.loc[selected.index]
        selected["target_sue"] = target_sue.loc[selected.index]
        selected["target_gap_days"] = selected.target_announcement_date.sub(selected.announcement_date).dt.days
        pieces.append(selected)
    result = pd.concat(pieces, ignore_index=True)
    if result.empty:
        raise ValueError("No valid quarterly event sequences for persistence analysis")
    result["target_announcement_year"] = result.target_announcement_date.dt.year.astype(int)
    result["target_sue_w"] = result.target_sue
    return result


def absorb_fixed_effects(frame: pd.DataFrame, columns: list[str], fixed_effects: str) -> pd.DataFrame:
    """Absorb formation-date FE, optionally adding firm FE by alternating projections."""
    values = frame[columns].astype(float).copy()
    if fixed_effects == "formation_date":
        return values - values.groupby(frame.formation_date, observed=True).transform("mean")
    if fixed_effects != "firm_and_formation_date":
        raise ValueError(f"Unknown fixed-effects specification: {fixed_effects}")
    for _ in range(100):
        previous = values.to_numpy(copy=True)
        values -= values.groupby(frame.permno, observed=True).transform("mean")
        values -= values.groupby(frame.formation_date, observed=True).transform("mean")
        if np.max(np.abs(values.to_numpy() - previous)) < 1e-11:
            break
    return values


def estimate(
    frame: pd.DataFrame, horizon: int, label: str, residual_signal: str,
    incremental_signal: str, fixed_effects: str,
) -> dict[str, object]:
    terms = [residual_signal, incremental_signal, *CONTROLS]
    needed = ["permno", "formation_date", "target_announcement_date", "target_gap_days", "target_sue_w", *terms]
    sample = frame.loc[frame.horizon.eq(horizon), needed].dropna().copy()
    if sample.empty:
        raise ValueError(f"No usable observations for {label} at horizon {horizon}")
    demeaned = absorb_fixed_effects(sample, ["target_sue_w", *terms], fixed_effects)
    y = demeaned.target_sue_w.to_numpy(dtype=float)
    x = demeaned[terms].to_numpy(dtype=float)
    coefficient = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    residual = y - x @ coefficient
    covariance = two_way_cluster_covariance(x, residual, sample.permno, sample.formation_date)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    beta, delta = coefficient[:2]
    beta_se, delta_se = se[:2]
    difference = delta - beta
    diff_var = covariance[1, 1] + covariance[0, 0] - 2.0 * covariance[0, 1]
    diff_se = math.sqrt(max(float(diff_var), 0.0))
    signal_sd = float(sample[incremental_signal].std(ddof=1))
    return {
        "display": label,
        "fixed_effects": fixed_effects,
        "horizon": horizon,
        "n_obs": len(sample),
        "n_firms": int(sample.permno.nunique()),
        "n_formation_dates": int(sample.formation_date.nunique()),
        "median_target_gap_days": float(sample.target_gap_days.median()),
        "residual_signal": residual_signal,
        "incremental_signal": incremental_signal,
        "beta_residual": float(beta),
        "se_beta_residual": float(beta_se),
        "p_beta_zero": normal_two_sided_pvalue(beta / beta_se) if beta_se > 0 else np.nan,
        "delta_increment": float(delta),
        "se_delta_increment": float(delta_se),
        "ci_low": float(delta - 1.96 * delta_se),
        "ci_high": float(delta + 1.96 * delta_se),
        "p_delta_zero": normal_two_sided_pvalue(delta / delta_se) if delta_se > 0 else np.nan,
        "delta_minus_beta": float(difference),
        "se_delta_minus_beta": diff_se,
        "p_delta_equals_beta": normal_two_sided_pvalue(difference / diff_se) if diff_se > 0 else np.nan,
        "signal_sd": signal_sd,
        "target_sue_change_per_1sd_signal": float(delta * signal_sd),
        "within_r2": float(1.0 - (residual @ residual) / (y @ y)) if (y @ y) > 0 else np.nan,
    }


def estimate_all(frame: pd.DataFrame, sample_name: str, fixed_effects: str) -> pd.DataFrame:
    designs = [
        ("high public compute", "u_H0", "q_H_public"),
        ("high post-price compute", "u_H1", "q_H_post_price"),
    ]
    rows = []
    for label, residual_signal, incremental_signal in designs:
        for horizon in HORIZONS:
            result = estimate(frame, horizon, label, residual_signal, incremental_signal, fixed_effects)
            result["sample"] = sample_name
            rows.append(result)
    output = pd.DataFrame(rows)
    h0 = output.loc[output.horizon.eq(0), ["sample", "display", "fixed_effects", "delta_increment"]].rename(
        columns={"delta_increment": "delta_h0"}
    )
    output = output.merge(h0, on=["sample", "display", "fixed_effects"], how="left")
    output["delta_share_of_h0"] = output.delta_increment / output.delta_h0
    return output


def sequencing_diagnostics(frame: pd.DataFrame, sample_name: str) -> pd.DataFrame:
    return frame.groupby("horizon", observed=True).agg(
        n_pairs=("permno", "size"),
        n_firms=("permno", "nunique"),
        median_gap_days=("target_gap_days", "median"),
        p05_gap_days=("target_gap_days", lambda values: values.quantile(0.05)),
        p95_gap_days=("target_gap_days", lambda values: values.quantile(0.95)),
    ).reset_index().assign(sample=sample_name)


def write_report(results: pd.DataFrame, diagnostics: pd.DataFrame, output: Path) -> None:
    report = [
        "# E1 earnings-persistence diagnostic",
        "",
        "For each current event t, this diagnostic links the pre-announcement high-compute increment to observed standardized earnings surprises at t through t+4. "
        "Successive announcements must be 45--180 days apart. It reports both formation-date FE and the more stringent firm-plus-formation-date FE specification, with two-way PERMNO/formation-date clustered standard errors.",
        "",
        "The formation-date FE result captures cross-sectional and within-firm persistence. The firm-plus-formation-date FE result isolates within-firm time-series persistence and is the more stringent diagnostic against a purely transitory earnings-component explanation.",
        "",
        "| Sample | Fixed effects | Display | Horizon | N | Median days | Delta | Two-way SE | 95% CI | p: delta=0 | p: delta=beta | SUE change per 1 SD signal | Delta / Delta(h=0) |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in results.itertuples(index=False):
        report.append(
            f"| {row.sample} | {row.fixed_effects} | {row.display} | {row.horizon} | {row.n_obs} | {row.median_target_gap_days:.0f} | "
            f"{row.delta_increment:.6f} | {row.se_delta_increment:.6f} | [{row.ci_low:.6f}, {row.ci_high:.6f}] | "
            f"{row.p_delta_zero:.4g} | {row.p_delta_equals_beta:.4g} | {row.target_sue_change_per_1sd_signal:.6f} | {row.delta_share_of_h0:.3f} |"
        )
    report.extend([
        "",
        "## Event-sequence diagnostics",
        "",
        "| Sample | Horizon | Pairs | Firms | Median days | 5th percentile | 95th percentile |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ])
    for row in diagnostics.itertuples(index=False):
        report.append(
            f"| {row.sample} | {row.horizon} | {row.n_pairs} | {row.n_firms} | {row.median_gap_days:.0f} | "
            f"{row.p05_gap_days:.0f} | {row.p95_gap_days:.0f} |"
        )
    (output / "step2_earnings_persistence_report.md").write_text("\n".join(report) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--rdq-panel", type=Path, default=DEFAULT_RDQ_PANEL)
    parser.add_argument("--rdq-flag", default="rdq_match_within_1_calendar_day")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--winsor-lower", type=float, default=0.01)
    parser.add_argument("--winsor-upper", type=float, default=0.99)
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    full_base = prepare_analysis_sample(args.panel, args.winsor_lower, args.winsor_upper)
    rdq_base = prepare_analysis_sample(args.rdq_panel, args.winsor_lower, args.winsor_upper)
    full = attach_sue_leads(full_base)
    rdq = attach_sue_leads(rdq_base)
    rdq = rdq.loc[rdq[args.rdq_flag].fillna(False).astype(bool)].copy()
    results = pd.concat([
        estimate_all(full, "full event sample", "formation_date"),
        estimate_all(full, "full event sample", "firm_and_formation_date"),
        estimate_all(rdq, "RDQ agreement <= 1 day", "formation_date"),
        estimate_all(rdq, "RDQ agreement <= 1 day", "firm_and_formation_date"),
    ], ignore_index=True)
    diagnostics = pd.concat([
        sequencing_diagnostics(full, "full event sample"),
        sequencing_diagnostics(rdq, "RDQ agreement <= 1 day"),
    ], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(args.output_dir / "step2_earnings_persistence.csv", index=False)
    diagnostics.to_csv(args.output_dir / "step2_earnings_persistence_sequence_diagnostics.csv", index=False)
    write_report(results, diagnostics, args.output_dir)
    print(results.to_string(index=False), flush=True)
    print(diagnostics.to_string(index=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
