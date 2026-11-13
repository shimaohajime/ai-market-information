"""Pre-API audit for a point-in-time public analyst-revision signal.

This is a separate, frozen treatment after the raw-price public-signal gate
failed. It uses only I/B/E/S summary records publicly available on or before
each formation date. The target, identifiers, dates, and raw estimates remain
local; only aggregate audit results are released.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from step4_phase0 import (
    REPO_ROOT,
    attach_sic,
    dossier_payload,
    eligible_rows,
    read_config,
    resolve_repo_path,
)
from step4_holdout_public_signal_audit import (
    PUBLIC_CATEGORICAL_FIELDS,
    PUBLIC_NUMERIC_FIELDS,
    bootstrap_contrast,
    choose_alpha,
    markdown_table,
    model_pipeline,
    select_training,
    training_config,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_holdout_analyst_signal_audit_v1.json"
ANALYST_FEATURES = [
    "analyst_estimate_age_days",
    "analyst_horizon_days",
    "analyst_log_coverage",
    "analyst_net_revision_share",
    "analyst_relative_dispersion",
    "analyst_revision_28d",
    "analyst_coverage_change_28d",
    "analyst_revision_available",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def public_base_features(row: Any, base_config: dict[str, Any], label: str) -> dict[str, Any]:
    work = pd.Series(row._asdict()) if hasattr(row, "_asdict") else row.copy()
    work["dossier_id"] = label
    public = dossier_payload(work, base_config)["public_information"]
    accounting = public["accounting_ratios"]
    return {
        "financial_statement_age_months": float(public["financial_statement_age_months"]),
        **{field: float(accounting[field]) for field in PUBLIC_NUMERIC_FIELDS if field != "financial_statement_age_months"},
        "industry_bucket": str(public["industry_bucket"]),
        "company_scale_bucket": str(public["company_scale_bucket"]),
        "analyst_coverage_bucket": str(public["analyst_coverage_bucket"]),
    }


def read_stats(path: Path, cusips: set[str], signal: dict[str, Any]) -> pd.DataFrame:
    usecols = [
        "CUSIP", "STATPERS", "MEASURE", "FISCALP", "FPI",
        "NUMEST", "NUMUP", "NUMDOWN", "MEANEST", "STDEV", "FPEDATS",
    ]
    pieces: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        path,
        usecols=usecols,
        dtype={"CUSIP": "string"},
        parse_dates=["STATPERS", "FPEDATS"],
        chunksize=500_000,
        low_memory=False,
    ):
        chunk["cusip8"] = chunk.CUSIP.astype(str).str[:8]
        keep = (
            chunk.cusip8.isin(cusips)
            & chunk.MEASURE.eq(str(signal["measure"]))
            & chunk.FISCALP.eq(str(signal["fiscal_period"]))
            & pd.to_numeric(chunk.FPI, errors="coerce").eq(int(signal["fpi"]))
        )
        value = chunk.loc[keep].copy()
        if not value.empty:
            pieces.append(value)
    if not pieces:
        raise ValueError("No I/B/E/S summary rows match the requested candidate CUSIPs and signal definition.")
    result = pd.concat(pieces, ignore_index=True)
    for field in ["NUMEST", "NUMUP", "NUMDOWN", "MEANEST", "STDEV"]:
        result[field] = pd.to_numeric(result[field], errors="coerce")
    return result


def analyst_snapshots(
    observations: pd.DataFrame, stats: pd.DataFrame, signal: dict[str, Any], id_column: str,
) -> tuple[pd.DataFrame, dict[str, int]]:
    required = [id_column, "cusip8", "formation_date", "next_announcement_date"]
    missing = set(required).difference(observations.columns)
    if missing:
        raise ValueError(f"Observations are missing analyst-snapshot fields: {sorted(missing)}")
    frame = observations[required].copy()
    frame["formation_date"] = pd.to_datetime(frame.formation_date)
    frame["next_announcement_date"] = pd.to_datetime(frame.next_announcement_date)
    merged = frame.merge(stats, on="cusip8", how="left", validate="many_to_many")
    merged = merged.loc[merged.STATPERS.le(merged.formation_date)].copy()
    merged["estimate_age_days"] = (merged.formation_date - merged.STATPERS).dt.days
    merged["announcement_minus_fiscal_end_days"] = (merged.next_announcement_date - merged.FPEDATS).dt.days
    merged["fiscal_end_minus_formation_days"] = (merged.FPEDATS - merged.formation_date).dt.days
    eligible = merged.loc[
        merged.estimate_age_days.between(0, int(signal["maximum_statpers_age_days"]))
        & merged.announcement_minus_fiscal_end_days.between(
            int(signal["minimum_announcement_minus_fiscal_end_days"]),
            int(signal["maximum_announcement_minus_fiscal_end_days"]),
        )
        & merged.fiscal_end_minus_formation_days.between(
            int(signal["minimum_fiscal_end_minus_formation_days"]),
            int(signal["maximum_fiscal_end_minus_formation_days"]),
        )
    ].copy()
    latest = eligible.sort_values([id_column, "STATPERS", "FPEDATS"]).groupby(id_column, as_index=False).tail(1).copy()
    rows: list[dict[str, Any]] = []
    lookback = int(signal["revision_lookback_days"])
    floor = float(signal["revision_scale_floor"])
    for value in latest.itertuples(index=False):
        prior = merged.loc[
            merged[id_column].eq(getattr(value, id_column))
            & merged.FPEDATS.eq(value.FPEDATS)
            & merged.STATPERS.le(value.STATPERS - pd.Timedelta(days=lookback))
        ].sort_values("STATPERS")
        previous = prior.iloc[-1] if not prior.empty else None
        mean = float(value.MEANEST)
        coverage = float(value.NUMEST) if pd.notna(value.NUMEST) else 0.0
        stdev = float(value.STDEV) if pd.notna(value.STDEV) else math.nan
        if previous is None or pd.isna(previous.MEANEST):
            revision = math.nan
            coverage_change = math.nan
            revision_available = 0.0
        else:
            previous_mean = float(previous.MEANEST)
            revision = (mean - previous_mean) / max(abs(previous_mean), floor)
            coverage_change = coverage - float(previous.NUMEST or 0.0)
            revision_available = 1.0
        rows.append({
            id_column: str(getattr(value, id_column)),
            "analyst_estimate_age_days": float(value.estimate_age_days),
            "analyst_horizon_days": float(value.announcement_minus_fiscal_end_days),
            "analyst_log_coverage": math.log1p(max(coverage, 0.0)),
            "analyst_net_revision_share": (
                (float(value.NUMUP or 0.0) - float(value.NUMDOWN or 0.0)) / max(coverage, 1.0)
            ),
            "analyst_relative_dispersion": stdev / max(abs(mean), floor) if math.isfinite(stdev) else math.nan,
            "analyst_revision_28d": revision,
            "analyst_coverage_change_28d": coverage_change,
            "analyst_revision_available": revision_available,
        })
    output = pd.DataFrame(rows)
    diagnostics = {
        "observations": int(len(observations)),
        "with_qualified_snapshot": int(len(output)),
        "without_qualified_snapshot": int(len(observations) - len(output)),
        "with_prior_28_day_estimate": int(output.analyst_revision_available.sum()) if not output.empty else 0,
    }
    return output, diagnostics


def training_data(
    base_config: dict[str, Any], config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, int]]:
    train_config = training_config(base_config, config)
    panel = pd.read_parquet(resolve_repo_path(base_config["panel_path"]))
    eligible, _ = eligible_rows(panel, train_config)
    eligible = attach_sic(eligible, resolve_repo_path(base_config["ccm_link_path"]))
    selected = select_training(eligible, config).reset_index(drop=True)
    selected["training_id"] = [f"T{number:04d}" for number in range(1, len(selected) + 1)]
    selected["cusip8"] = selected.cusip.astype(str).str[:8]
    stats = read_stats(resolve_repo_path(config["ibes_stats_path"]), set(selected.cusip8), config["analyst_signal"])
    snapshots, diagnostic = analyst_snapshots(selected, stats, config["analyst_signal"], "training_id")
    data = selected.merge(snapshots, on="training_id", how="inner", validate="one_to_one")
    base_rows = [
        {"training_id": str(row.training_id), **public_base_features(row, base_config, str(row.training_id))}
        for row in data.itertuples(index=False)
    ]
    base_frame = pd.DataFrame(base_rows)
    overlapping_public_columns = [
        column for column in base_frame.columns if column != "training_id" and column in data.columns
    ]
    data = data.drop(columns=overlapping_public_columns).merge(
        base_frame, on="training_id", validate="one_to_one"
    )
    if len(data) < int(config["training"]["minimum_qualified_training_dossiers"]):
        raise ValueError("I/B/E/S analyst-signal coverage is below the pre-specified training minimum.")
    return data, diagnostic


def holdout_data(
    config: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, int]]:
    root = resolve_repo_path(config["holdout_final_root"])
    outcome = pd.read_parquet(root / config["holdout_outcome_filename"])
    outcome["dossier_id"] = outcome.dossier_id.astype(str)
    candidates = pd.read_parquet(resolve_repo_path(config["holdout_candidate_key"]))
    candidates["dossier_id"] = candidates.dossier_id.astype(str)
    candidates["cusip8"] = candidates.cusip.astype(str).str[:8]
    stats = read_stats(resolve_repo_path(config["ibes_stats_path"]), set(candidates.cusip8), config["analyst_signal"])
    snapshots, diagnostic = analyst_snapshots(candidates, stats, config["analyst_signal"], "dossier_id")
    if len(snapshots) != len(candidates):
        raise ValueError("Every fixed holdout dossier must have a qualified analyst snapshot.")
    dossiers = {str(value["dossier_id"]): value for value in read_jsonl(root / config["holdout_dossier_filename"])}
    base_rows = [
        {"dossier_id": dossier_id, **{
            "financial_statement_age_months": float(dossier["public_information"]["financial_statement_age_months"]),
            **{field: float(dossier["public_information"]["accounting_ratios"][field]) for field in PUBLIC_NUMERIC_FIELDS if field != "financial_statement_age_months"},
            "industry_bucket": str(dossier["public_information"]["industry_bucket"]),
            "company_scale_bucket": str(dossier["public_information"]["company_scale_bucket"]),
            "analyst_coverage_bucket": str(dossier["public_information"]["analyst_coverage_bucket"]),
        }}
        for dossier_id, dossier in dossiers.items()
    ]
    genuine = (
        outcome[["dossier_id", "theta_ex_ante"]]
        .merge(pd.DataFrame(base_rows), on="dossier_id", validate="one_to_one")
        .merge(snapshots, on="dossier_id", validate="one_to_one")
    )
    mapping = pd.read_csv(root / config["holdout_placebo_mapping_filename"])
    mapping["dossier_id"] = mapping.dossier_id.astype(str)
    mapping["donor_dossier_id"] = mapping.donor_dossier_id.astype(str)
    donor = snapshots.rename(columns={"dossier_id": "donor_dossier_id"})
    matched_features = mapping[["dossier_id", "donor_dossier_id"]].merge(
        donor, on="donor_dossier_id", validate="many_to_one"
    ).drop(columns="donor_dossier_id")
    matched = (
        outcome[["dossier_id", "theta_ex_ante"]]
        .merge(pd.DataFrame(base_rows), on="dossier_id", validate="one_to_one")
        .merge(matched_features, on="dossier_id", validate="one_to_one")
    )
    return genuine, matched, diagnostic


def plan(config: dict[str, Any]) -> dict[str, Any]:
    years = [int(value) for value in config["training"]["formation_years"]]
    return {
        "api_calls_made": 0,
        "training_dossiers": len(years) * int(config["training"]["candidates_per_year"]),
        "holdout_dossiers_expected": 50,
        "signal": "latest pre-formation FPI=6 public quarterly consensus and 28-day revision",
        "gate_contrasts": [
            "genuine_analyst_signal_minus_no_analyst_squared_error",
            "genuine_analyst_signal_minus_matched_analyst_signal_squared_error",
        ],
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    base_config = read_config(resolve_repo_path(config["holdout_selection_config"]))
    train, train_diagnostic = training_data(base_config, config)
    genuine, matched, holdout_diagnostic = holdout_data(config)
    numeric_base = list(PUBLIC_NUMERIC_FIELDS)
    categorical = list(PUBLIC_CATEGORICAL_FIELDS)
    numeric_analyst = [*numeric_base, *ANALYST_FEATURES]
    alphas = [float(value) for value in config["ridge_alpha_grid"]]
    base_alpha = choose_alpha(train, numeric_base, categorical, alphas)
    analyst_alpha = choose_alpha(train, numeric_analyst, categorical, alphas)
    base_model = model_pipeline(numeric_base, categorical, base_alpha)
    analyst_model = model_pipeline(numeric_analyst, categorical, analyst_alpha)
    base_model.fit(train[numeric_base + categorical], train.theta_ex_ante)
    analyst_model.fit(train[numeric_analyst + categorical], train.theta_ex_ante)
    target = genuine.theta_ex_ante.to_numpy(dtype=float)
    predicted_no_analyst = base_model.predict(genuine[numeric_base + categorical])
    predicted_genuine = analyst_model.predict(genuine[numeric_analyst + categorical])
    predicted_matched = analyst_model.predict(matched[numeric_analyst + categorical])
    cells = pd.DataFrame({
        "dossier_id": genuine.dossier_id,
        "theta_ex_ante": target,
        "squared_error_no_analyst": (predicted_no_analyst - target)**2,
        "squared_error_genuine_analyst": (predicted_genuine - target)**2,
        "squared_error_matched_analyst": (predicted_matched - target)**2,
    })
    contrasts = []
    for number, (label, left, right) in enumerate([
        ("genuine_analyst_signal_minus_no_analyst_squared_error", "squared_error_genuine_analyst", "squared_error_no_analyst"),
        ("genuine_analyst_signal_minus_matched_analyst_signal_squared_error", "squared_error_genuine_analyst", "squared_error_matched_analyst"),
    ]):
        contrasts.append({
            "contrast": label,
            **bootstrap_contrast(
                cells[left] - cells[right],
                int(config["analysis"]["bootstrap_repetitions"]),
                int(config["analysis"]["bootstrap_seed"]) + number,
            ),
        })
    contrast_frame = pd.DataFrame(contrasts)
    condition_frame = pd.DataFrame([
        {"condition": "no_analyst_signal", "dossiers": len(cells), "mean_squared_error": float(cells.squared_error_no_analyst.mean())},
        {"condition": "genuine_analyst_signal", "dossiers": len(cells), "mean_squared_error": float(cells.squared_error_genuine_analyst.mean())},
        {"condition": "matched_analyst_signal", "dossiers": len(cells), "mean_squared_error": float(cells.squared_error_matched_analyst.mean())},
    ])
    gate = bool((contrast_frame.ci_95_upper < 0).all())
    interim = resolve_repo_path(config["interim_root"])
    output = resolve_repo_path(config["output_root"])
    interim.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    train.to_parquet(interim / "analyst_signal_training_private.parquet", index=False)
    cells.to_parquet(interim / "analyst_signal_holdout_cells_private.parquet", index=False)
    condition_frame.to_csv(output / "step4_holdout_analyst_signal_condition_summary.csv", index=False)
    contrast_frame.to_csv(output / "step4_holdout_analyst_signal_gate_contrasts.csv", index=False)
    summary = {
        **plan(config),
        "selected_ridge_alpha_accounting_only": base_alpha,
        "selected_ridge_alpha_accounting_plus_analyst_signal": analyst_alpha,
        "training_snapshot_diagnostic": train_diagnostic,
        "holdout_snapshot_diagnostic": holdout_diagnostic,
        "researcher_side_analyst_signal_gate_passed": gate,
        "gate_rule": "Every genuine-minus-comparator squared-error contrast has bootstrap upper CI below zero.",
        "contrast_results": contrast_frame.to_dict(orient="records"),
    }
    (output / "step4_holdout_analyst_signal_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = "\n".join([
        "# Step 4 holdout analyst-signal audit",
        "",
        "This is a non-LLM pre-API gate for a new, separately specified public analyst-revision treatment.",
        "",
        "## Condition-level holdout squared error",
        "",
        markdown_table(condition_frame),
        "",
        "## Genuine-signal contrasts",
        "",
        "Negative values favor the genuine target-firm analyst signal.",
        "",
        markdown_table(contrast_frame),
        "",
        f"Researcher-side gate passed: {gate}.",
    ])
    (output / "step4_holdout_analyst_signal_audit_report.md").write_text(report)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "run"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_config(args.config)
    print(json.dumps(plan(config) if args.command == "plan" else run(config), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
