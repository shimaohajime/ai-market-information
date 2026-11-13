"""Estimate the Step 2 signal decomposition in one linearly independent basis.

The four displayed increments are not four independent sources of evidence:

    q_H_post = q_H_public + p_H - p_L.

This runner therefore estimates a single conditional event-time model with the
independent basis (p_L, p_H, q_H_public), and re-estimates its algebraically
equivalent q_H_post representation.  It reports both displays, their common
two-way clustered inference, zero restrictions, equality-to-residual tests,
and one-standard-deviation CAR effects.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from step2_market_expectations import (
    CONTROL_COLUMNS,
    PRIMARY_OUTCOME,
    demean_by_date,
    normal_two_sided_pvalue,
    prepare_analysis_sample,
    two_way_cluster_covariance,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel.parquet"
DEFAULT_RDQ_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel_rdq.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "output/step2"
CONTROLS = [*CONTROL_COLUMNS, "size_middle", "size_high", "coverage_middle", "coverage_high"]


def fit(frame: pd.DataFrame, signal_basis: list[str]) -> dict[str, object]:
    terms = ["u_H1", *signal_basis, *CONTROLS]
    needed = ["permno", "event_trade_date", PRIMARY_OUTCOME + "_w", "complete_0_p1", *terms]
    sample = frame.loc[frame.complete_0_p1.fillna(False), needed].dropna().copy()
    if sample.empty:
        raise ValueError("No complete event-window observations in the integrated sample")
    demeaned = demean_by_date(sample, [PRIMARY_OUTCOME + "_w", *terms])
    y = demeaned[PRIMARY_OUTCOME + "_w"].to_numpy(dtype=float)
    x = demeaned[terms].to_numpy(dtype=float)
    coefficient = np.linalg.pinv(x.T @ x) @ (x.T @ y)
    residual = y - x @ coefficient
    covariance = two_way_cluster_covariance(x, residual, sample.permno, sample.event_trade_date)
    se = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        z = coefficient / se
    return {
        "sample": sample,
        "terms": terms,
        "coefficient": coefficient,
        "covariance": covariance,
        "se": se,
        "z": z,
        "residual": residual,
        "fitted": x @ coefficient,
        "within_r2": float(1.0 - (residual @ residual) / (y @ y)),
    }


def contrast_stats(result: dict[str, object], term: str) -> dict[str, float]:
    terms = result["terms"]
    coefficient = result["coefficient"]
    covariance = result["covariance"]
    se = result["se"]
    index = terms.index(term)
    beta_index = terms.index("u_H1")
    difference = coefficient[index] - coefficient[beta_index]
    difference_variance = (
        covariance[index, index] + covariance[beta_index, beta_index] - 2.0 * covariance[index, beta_index]
    )
    difference_se = math.sqrt(max(float(difference_variance), 0.0))
    return {
        "coefficient": float(coefficient[index]),
        "std_error_two_way_cluster": float(se[index]),
        "z_zero": float(coefficient[index] / se[index]) if se[index] > 0 else np.nan,
        "p_zero": normal_two_sided_pvalue(float(coefficient[index] / se[index])) if se[index] > 0 else np.nan,
        "ci_low": float(coefficient[index] - 1.96 * se[index]),
        "ci_high": float(coefficient[index] + 1.96 * se[index]),
        "delta_minus_beta": float(difference),
        "se_delta_minus_beta": difference_se,
        "p_delta_equals_beta": normal_two_sided_pvalue(difference / difference_se) if difference_se > 0 else np.nan,
    }


def result_table(public: dict[str, object], post: dict[str, object], sample_name: str) -> pd.DataFrame:
    public_sample = public["sample"]
    rows: list[dict[str, object]] = []
    beta_index = public["terms"].index("u_H1")
    beta = float(public["coefficient"][beta_index])
    beta_se = float(public["se"][beta_index])
    rows.append({
        "sample": sample_name,
        "display": "high-model residual",
        "signal": "u_H1 = SUE - H1",
        "basis": "common residual",
        "is_linearly_independent": True,
        "n_obs": len(public_sample),
        "n_firms": int(public_sample.permno.nunique()),
        "n_event_dates": int(public_sample.event_trade_date.nunique()),
        "coefficient": beta,
        "std_error_two_way_cluster": beta_se,
        "ci_low": beta - 1.96 * beta_se,
        "ci_high": beta + 1.96 * beta_se,
        "p_zero": normal_two_sided_pvalue(beta / beta_se) if beta_se > 0 else np.nan,
        "p_delta_equals_beta": np.nan,
        "signal_sd": float(public_sample["u_H1"].std(ddof=1)),
        "car_change_per_1sd": beta * float(public_sample["u_H1"].std(ddof=1)),
        "within_r2": public["within_r2"],
        "note": "Common residual benchmark.",
    })
    displays = [
        ("low price", "p_L = L1 - L0", "p_L", public, "independent basis"),
        ("high price", "p_H = H1 - H0", "p_H", public, "independent basis"),
        ("high public compute", "q_H_public = H0 - L0", "q_H_public", public, "independent basis"),
        ("high post-price compute", "q_H_post = H1 - L1", "q_H_post_price", post, "equivalent reparameterization"),
    ]
    for display, label, signal, result, basis in displays:
        stats = contrast_stats(result, signal)
        signal_sd = float(result["sample"][signal].std(ddof=1))
        rows.append({
            "sample": sample_name,
            "display": display,
            "signal": label,
            "basis": basis,
            "is_linearly_independent": signal != "q_H_post_price",
            "n_obs": len(result["sample"]),
            "n_firms": int(result["sample"].permno.nunique()),
            "n_event_dates": int(result["sample"].event_trade_date.nunique()),
            **stats,
            "signal_sd": signal_sd,
            "car_change_per_1sd": stats["coefficient"] * signal_sd,
            "within_r2": result["within_r2"],
            "note": "Not an additional degree of freedom; q_H_post = q_H_public + p_H - p_L."
                    if signal == "q_H_post_price" else "Joint partial loading conditional on the other independent increments.",
        })
    return pd.DataFrame(rows)


def reparameterization_check(
    public: dict[str, object], post: dict[str, object], sample_name: str, signal_identity_error: float
) -> pd.DataFrame:
    public_terms = public["terms"]
    post_terms = post["terms"]
    public_coef = dict(zip(public_terms, public["coefficient"], strict=True))
    post_coef = dict(zip(post_terms, post["coefficient"], strict=True))
    mapped = {
        "u_H1": public_coef["u_H1"],
        "p_L": public_coef["p_L"] + public_coef["q_H_public"],
        "p_H": public_coef["p_H"] - public_coef["q_H_public"],
        "q_H_post_price": public_coef["q_H_public"],
        **{term: public_coef[term] for term in CONTROLS},
    }
    return pd.DataFrame([{
        "sample": sample_name,
        "n_obs": len(public["sample"]),
        "max_abs_signal_identity_error": signal_identity_error,
        "max_abs_fitted_value_difference": float(np.max(np.abs(public["fitted"] - post["fitted"]))),
        "max_abs_coefficient_mapping_difference": float(max(abs(post_coef[key] - value) for key, value in mapped.items())),
        "identity": "q_H_post_price = q_H_public + p_H - p_L",
    }])


def estimate_panel(panel: Path, sample_flag: str | None, sample_name: str, lower: float, upper: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = prepare_analysis_sample(panel, lower, upper, sample_flag)
    signal_identity_error = float(np.nanmax(np.abs(frame.q_H_post_price - (frame.q_H_public + frame.p_H - frame.p_L))))
    public = fit(frame, ["p_L", "p_H", "q_H_public"])
    post = fit(frame, ["p_L", "p_H", "q_H_post_price"])
    return result_table(public, post, sample_name), reparameterization_check(
        public, post, sample_name, signal_identity_error
    )


def write_report(table: pd.DataFrame, check: pd.DataFrame, output: Path) -> None:
    show = table.loc[table.display.ne("high-model residual")].copy()
    report = [
        "# Step 2 integrated signal decomposition",
        "",
        "The primary CAR[0,+1] model is estimated jointly as `CAR = beta*u_H1 + theta_L*p_L + theta_H*p_H + theta_Q*q_H_public + controls + event-date FE`. "
        "It uses two-way PERMNO/event-date clustered standard errors. `q_H_post` is shown through the exact equivalent basis, rather than counted as a fourth independent signal.",
        "",
        "| Sample | Signal | Coefficient | Two-way SE | 95% CI | p: coefficient=0 | p: coefficient=beta | CAR per 1 SD | Basis |",
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for row in show.itertuples(index=False):
        report.append(
            f"| {row.sample} | {row.signal} | {row.coefficient:.6f} | {row.std_error_two_way_cluster:.6f} | "
            f"[{row.ci_low:.6f}, {row.ci_high:.6f}] | {row.p_zero:.4g} | {row.p_delta_equals_beta:.4g} | "
            f"{row.car_change_per_1sd:.6f} | {row.basis} |"
        )
    check_lines = [
        "| Sample | N | Max signal-identity error | Max fitted-value difference | Max coefficient-mapping difference | Identity |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ]
    check_lines.extend(
        f"| {row.sample} | {row.n_obs} | {row.max_abs_signal_identity_error:.3e} | "
        f"{row.max_abs_fitted_value_difference:.3e} | {row.max_abs_coefficient_mapping_difference:.3e} | {row.identity} |"
        for row in check.itertuples(index=False)
    )
    report.extend([
        "",
        "## Reparameterization check",
        "",
        *check_lines,
        "",
        "A near-zero fitted-value and coefficient-mapping discrepancy verifies that the public- and post-price displays span the same conditional regression. "
        "The joint coefficients are partial loadings; they complement, rather than replace, the original one-increment event-study specifications.",
    ])
    (output / "step2_integrated_decomposition_report.md").write_text("\n".join(report) + "\n")


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
    full_table, full_check = estimate_panel(args.panel, None, "full event sample", args.winsor_lower, args.winsor_upper)
    rdq_table, rdq_check = estimate_panel(
        args.rdq_panel, args.rdq_flag, "RDQ agreement <= 1 day", args.winsor_lower, args.winsor_upper
    )
    table = pd.concat([full_table, rdq_table], ignore_index=True)
    check = pd.concat([full_check, rdq_check], ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output_dir / "step2_integrated_decomposition.csv", index=False)
    check.to_csv(args.output_dir / "step2_integrated_decomposition_identity_check.csv", index=False)
    write_report(table, check, args.output_dir)
    print(table.to_string(index=False), flush=True)
    print(check.to_string(index=False), flush=True)


if __name__ == "__main__":
    run(parse_args())
