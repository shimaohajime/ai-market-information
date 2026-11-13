"""Analyze the private-clue precision and demand supplement for E2.

The 0.7-SD and 1.4-SD conditions are combined only after validation. Dossiers—not repetitions or API calls—are the
inference unit.  Raw response files remain ignored local artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from step4_phase0 import REPO_ROOT, read_config, resolve_repo_path


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_task2_precision_v1.json"
RAW_NAME = "step4_v3_raw_responses.jsonl"
MODELS = ["gpt6", "gpt52"]
CONDITIONS = ["no_price", "informative_price_path", "matched_placebo_path"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_usable_records(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Use the first valid completion per scenario; preserve errors in raw audit."""
    raw = read_jsonl(path)
    by_id: dict[str, dict[str, Any]] = {}
    for record in raw:
        scenario_id = str(record["scenario_id"])
        current = by_id.get(scenario_id)
        usable = record.get("status") == "completed" and record.get("parse_status") == "valid"
        current_usable = current is not None and current.get("status") == "completed" and current.get("parse_status") == "valid"
        if current is None or (usable and not current_usable):
            by_id[scenario_id] = record
    return pd.DataFrame(by_id.values()), {
        "raw_rows": len(raw),
        "unique_scenarios": len(by_id),
        "raw_duplicate_rows": len(raw) - len(by_id),
        "usable_scenarios": int(sum(value.get("status") == "completed" and value.get("parse_status") == "valid" for value in by_id.values())),
    }


def bootstrap(values: pd.Series, repetitions: int, seed: int) -> tuple[float, float, float]:
    sample = values.dropna().to_numpy(dtype=float)
    if len(sample) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = sample[rng.integers(0, len(sample), size=(repetitions, len(sample)))].mean(axis=1)
    return float(sample.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def contrast_row(label: str, values: pd.Series, repetitions: int, seed: int) -> dict[str, Any]:
    mean, lower, upper = bootstrap(values, repetitions, seed)
    return {
        "contrast": label,
        "dossiers": int(values.notna().sum()),
        "mean": mean,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
        "negative_dossiers": int((values < 0).sum()),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_float_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    columns = [str(value) for value in printable.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in printable.itertuples(index=False, name=None))
    return "\n".join(lines)


def actual_cost(frame: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    def tokens(value: Any, key: str) -> float:
        return float(value.get(key, 0) or 0) if isinstance(value, dict) else 0.0

    costs: dict[str, float] = {}
    for model_key, group in frame.groupby("model_key"):
        model = config["models"][str(model_key)]
        input_tokens = sum(tokens(value, "input_tokens") for value in group.usage)
        output_tokens = sum(tokens(value, "output_tokens") for value in group.usage)
        costs[str(model_key)] = (
            input_tokens * float(model["input_usd_per_million_tokens"])
            + output_tokens * float(model["output_usd_per_million_tokens"])
        ) / 1_000_000
    return costs


def load_records(config: dict[str, Any], supplement_subdir: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    base_root = REPO_ROOT / "output" / "step4_v3_task12"
    supplement_root = resolve_repo_path(config["output_root"]) / supplement_subdir
    paths = {
        "base_gpt6": base_root / "task2_mechanism_gpt6_v1" / RAW_NAME,
        "base_gpt52": base_root / "task2_mechanism_gpt52_v1" / RAW_NAME,
        "supplement_gpt6": supplement_root / "task2_precision_gpt6_v1" / RAW_NAME,
        "supplement_gpt52": supplement_root / "task2_precision_gpt52_v1" / RAW_NAME,
    }
    missing = [str(value) for value in paths.values() if not value.exists()]
    if missing:
        raise FileNotFoundError("Required precision raw output is missing:\n" + "\n".join(missing))
    usable: list[pd.DataFrame] = []
    raw: list[pd.DataFrame] = []
    audit: dict[str, Any] = {}
    for label, path in paths.items():
        frame, item_audit = first_usable_records(path)
        frame["source"] = label
        usable.append(frame)
        raw_frame = pd.DataFrame(read_jsonl(path))
        raw_frame["source"] = label
        raw.append(raw_frame)
        audit[label] = item_audit
    return pd.concat(usable, ignore_index=True), pd.concat(raw, ignore_index=True), audit


def build_dossier_panel(records: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    frame = records.loc[records.parse_status.eq("valid")].copy()
    frame = frame.loc[frame.task.eq("mechanism")].copy()
    frame["signal_noise_sd"] = pd.to_numeric(frame.signal_noise_sd, errors="coerce")
    frame["signal_value"] = pd.to_numeric(frame.signal_value, errors="coerce")
    frame["forecast_mean"] = pd.to_numeric(frame.forecast_mean, errors="coerce")
    frame["forecast_sd"] = pd.to_numeric(frame.forecast_sd, errors="coerce")
    demand = config["demand_mapping"]
    gamma = float(demand["gamma"])
    price = float(demand["claim_price"])
    risk_free = float(demand["risk_free_gross_return"])
    idiosyncratic_variance = float(demand["idiosyncratic_payoff_noise_variance"])
    frame["synthetic_demand"] = (frame.forecast_mean - risk_free * price) / (gamma * (frame.forecast_sd.pow(2) + idiosyncratic_variance))
    identifiers = ["dossier_id", "replicate_id", "model_key", "information_condition", "signal_noise_sd"]
    cells = frame.groupby(identifiers).size()
    if not cells.eq(2).all():
        raise ValueError("Each dossier/repetition/model/condition/noise cell must have exactly two valid clue values.")
    means = frame.pivot(index=identifiers, columns="signal_value", values="forecast_mean")
    demands = frame.pivot(index=identifiers, columns="signal_value", values="synthetic_demand")
    if not {-0.8, 0.8}.issubset(means.columns) or not {-0.8, 0.8}.issubset(demands.columns):
        raise ValueError("Both pre-specified clue values are required in every cell.")
    panel = pd.DataFrame({
        "forecast_mean_slope": (means[0.8] - means[-0.8]) / 1.6,
        "risk_adjusted_demand_slope": (demands[0.8] - demands[-0.8]) / 1.6,
    }).reset_index()
    collapsed = (
        panel.groupby(["dossier_id", "model_key", "information_condition", "signal_noise_sd"], as_index=False)
        .agg(forecast_mean_slope=("forecast_mean_slope", "mean"), risk_adjusted_demand_slope=("risk_adjusted_demand_slope", "mean"), repetitions=("replicate_id", "size"))
    )
    if not collapsed.repetitions.eq(3).all():
        raise ValueError("Each dossier/model/condition/noise cell must contain exactly three repetitions.")
    return collapsed


def summarize(panel: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    repetitions = int(config["analysis"]["bootstrap_repetitions"])
    seed = int(config["analysis"]["bootstrap_seed"])
    levels = (
        panel.groupby(["model_key", "information_condition", "signal_noise_sd"], as_index=False)
        .agg(
            dossiers=("dossier_id", "size"),
            forecast_mean_slope=("forecast_mean_slope", "mean"),
            demand_slope=("risk_adjusted_demand_slope", "mean"),
            forecast_slope_sd=("forecast_mean_slope", "std"),
            demand_slope_sd=("risk_adjusted_demand_slope", "std"),
        )
        .sort_values(["model_key", "information_condition", "signal_noise_sd"])
    )
    precision_rows: list[dict[str, Any]] = []
    for (model_key, condition), group in panel.groupby(["model_key", "information_condition"]):
        for metric in ["forecast_mean_slope", "risk_adjusted_demand_slope"]:
            wide = group.pivot(index="dossier_id", columns="signal_noise_sd", values=metric)
            if not {0.7, 1.4}.issubset(wide.columns):
                raise ValueError("Both clue noise levels must be present for the precision contrast.")
            row = contrast_row(f"1.4_minus_0.7_{metric}", wide[1.4] - wide[0.7], repetitions, seed + len(precision_rows))
            row.update({"model_key": model_key, "information_condition": condition, "outcome": metric})
            precision_rows.append(row)
    precision = pd.DataFrame(precision_rows)
    gap_rows: list[dict[str, Any]] = []
    for noise, group in panel.groupby("signal_noise_sd"):
        for metric in ["forecast_mean_slope", "risk_adjusted_demand_slope"]:
            wide = group.pivot(index=["dossier_id", "information_condition"], columns="model_key", values=metric).reset_index()
            gap = wide.pivot(index="dossier_id", columns="information_condition", values="gpt6").sub(
                wide.pivot(index="dossier_id", columns="information_condition", values="gpt52")
            )
            for label, left, right in [
                ("genuine_minus_placebo_model_gap", "informative_price_path", "matched_placebo_path"),
                ("genuine_minus_no_price_model_gap", "informative_price_path", "no_price"),
            ]:
                row = contrast_row(label, gap[left] - gap[right], repetitions, seed + 100 + len(gap_rows))
                row.update({"signal_noise_sd": noise, "outcome": metric})
                gap_rows.append(row)
    return levels, precision, pd.DataFrame(gap_rows)


def oos_distribution_calibration(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, dict[str, int]]]:
    """Diagnostic calibration check on the pre-existing public-only OOS task.

    It is intentionally reported separately: twenty development dossiers are
    not enough to calibrate predictive distributions precisely, but a severely
    under-dispersed reported SD would rule out a quantitative demand claim.
    """
    root = REPO_ROOT / "output" / "step4_v3_task12"
    paths = {
        "public_gpt6": root / "task1_public_gpt6_v1" / RAW_NAME,
        "public_gpt52": root / "task1_public_gpt52_v1" / RAW_NAME,
        "public_placebo_gpt6": root / "task1_public_placebo_gpt6_v1" / RAW_NAME,
        "public_placebo_gpt52": root / "task1_public_placebo_gpt52_v1" / RAW_NAME,
    }
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required public-forecast raw output is missing:\n" + "\n".join(missing))
    frames: list[pd.DataFrame] = []
    audit: dict[str, dict[str, int]] = {}
    for label, path in paths.items():
        frame, item_audit = first_usable_records(path)
        frames.append(frame)
        audit[label] = item_audit
    frame = pd.concat(frames, ignore_index=True)
    frame = frame.loc[frame.task.eq("public_forecast") & frame.parse_status.eq("valid")].copy()
    frame["forecast_mean"] = pd.to_numeric(frame.forecast_mean, errors="coerce")
    frame["forecast_sd"] = pd.to_numeric(frame.forecast_sd, errors="coerce")
    if frame.forecast_sd.le(0).any() or frame.forecast_sd.isna().any():
        raise ValueError("Public-forecast SD must be present and positive for calibration.")
    repetitions = frame.groupby(["dossier_id", "model_key", "information_condition"]).size()
    if not repetitions.eq(3).all():
        raise ValueError("Public-forecast calibration requires three valid repetitions per cell.")
    outcome = pd.read_parquet(resolve_repo_path(config["interim_root"]) / "step4_v3_outcome_key.parquet")
    collapsed = (
        frame.groupby(["dossier_id", "model_key", "information_condition"], as_index=False)
        .agg(forecast_mean=("forecast_mean", "mean"), forecast_sd=("forecast_sd", "mean"))
        .merge(outcome[["dossier_id", "next_sue"]], on="dossier_id", validate="many_to_one")
    )
    collapsed["standardized_error"] = (collapsed.forecast_mean - collapsed.next_sue) / collapsed.forecast_sd
    collapsed["covered_80"] = collapsed.standardized_error.abs().le(1.2815515655)
    collapsed["covered_95"] = collapsed.standardized_error.abs().le(1.9599639845)
    summary = (
        collapsed.groupby(["model_key", "information_condition"], as_index=False)
        .agg(
            dossiers=("dossier_id", "size"),
            mean_forecast_sd=("forecast_sd", "mean"),
            mean_absolute_standardized_error=("standardized_error", lambda value: value.abs().mean()),
            empirical_80_coverage=("covered_80", "mean"),
            empirical_95_coverage=("covered_95", "mean"),
        )
        .sort_values(["model_key", "information_condition"])
    )
    return summary, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--supplement-subdir", default=".")
    parser.add_argument("--analysis-subdir", default="analysis_v1")
    args = parser.parse_args()
    config = read_config(args.config)
    records, raw, audit = load_records(config, args.supplement_subdir)
    panel = build_dossier_panel(records, config)
    levels, precision, gaps = summarize(panel, config)
    calibration, calibration_audit = oos_distribution_calibration(config)
    output = resolve_repo_path(config["output_root"]) / args.analysis_subdir
    output.mkdir(parents=True, exist_ok=True)
    levels.to_csv(output / "task2_precision_levels.csv", index=False)
    precision.to_csv(output / "task2_precision_contrasts.csv", index=False)
    gaps.to_csv(output / "task2_precision_model_gap_contrasts.csv", index=False)
    panel.to_csv(output / "task2_precision_dossier_panel.csv", index=False)
    calibration.to_csv(output / "task2_precision_oos_calibration_diagnostic.csv", index=False)
    new_raw = raw.loc[raw.source.str.startswith("supplement_")].copy()
    costs = actual_cost(new_raw, config)
    result = {
        "raw_audit": audit,
        "new_api_cost_usd": costs,
        "new_api_cost_usd_total": float(sum(costs.values())),
        "levels": levels.to_dict(orient="records"),
        "precision_contrasts": precision.to_dict(orient="records"),
        "model_gap_contrasts": gaps.to_dict(orient="records"),
        "oos_distribution_calibration_diagnostic": calibration.to_dict(orient="records"),
        "oos_calibration_raw_audit": calibration_audit,
    }
    (output / "task2_precision_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = "\n".join([
        "# E2 private-clue precision supplement",
        "",
        "This is a post-result behavioral-validation supplement. Inference uses 20 dossiers after averaging three repetitions within dossier.",
        "",
        "## Sensitivity levels",
        "",
        markdown_table(levels),
        "",
        "## Noise 1.4 minus 0.7 contrasts",
        "",
        markdown_table(precision),
        "",
        "## Price-condition model-gap contrasts",
        "",
        markdown_table(gaps),
        "",
        "## OOS distribution-calibration diagnostic",
        "",
        "This uses the pre-existing 20-dossier public-only development task. It is a diagnostic, not a new holdout estimate. Nominal 80% and 95% normal-interval coverage should be interpreted against the reported empirical coverage below.",
        "",
        markdown_table(calibration),
        "",
        f"New recorded API cost: ${sum(costs.values()):.4f}.",
    ])
    (output / "task2_precision_report.md").write_text(report)
    print(json.dumps({"output": str(output), "new_api_cost_usd_total": sum(costs.values())}, allow_nan=False))


if __name__ == "__main__":
    main()
