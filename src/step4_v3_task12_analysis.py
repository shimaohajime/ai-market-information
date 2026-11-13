"""Aggregate the locked Step 4 Task 1--2 development experiment.

Raw Responses API outputs remain local.  This module writes only aggregate
tables and an auditable Markdown report.  Dossier, rather than API response or
repeat, is the inference unit throughout.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from step4_phase0 import REPO_ROOT, read_config, resolve_repo_path


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_v3_task12.json"
RAW_NAME = "step4_v3_raw_responses.jsonl"
MODEL_ORDER = ["gpt6", "gpt52"]
CONDITION_ORDER = ["no_price", "informative_price_path", "matched_placebo_path"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def first_draws(path: Path) -> tuple[pd.DataFrame, dict[str, int]]:
    """Keep the first record per pre-specified scenario while retaining raw audit files."""
    raw = read_jsonl(path)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for record in raw:
        scenario_id = str(record["scenario_id"])
        if scenario_id not in seen:
            records.append(record)
            seen.add(scenario_id)
    return pd.DataFrame(records), {
        "raw_rows": len(raw),
        "unique_pre_specified_scenarios": len(records),
        "duplicate_raw_rows_ignored": len(raw) - len(records),
    }


def bootstrap_mean(values: pd.Series | np.ndarray, repetitions: int, seed: int) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return math.nan, math.nan, math.nan
    rng = np.random.default_rng(seed)
    draws = values[rng.integers(0, len(values), size=(repetitions, len(values)))].mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def ci_row(metric: str, values: pd.Series, repetitions: int, seed: int, direction: str | None = None) -> dict[str, Any]:
    average, lower, upper = bootstrap_mean(values, repetitions, seed)
    result = {
        "metric": metric,
        "n_dossiers": int(values.notna().sum()),
        "mean": average,
        "ci_95_lower": lower,
        "ci_95_upper": upper,
    }
    if direction == "negative":
        result["negative_dossiers"] = int((values < 0).sum())
    return result


def actual_cost(records: pd.DataFrame, config: dict[str, Any]) -> dict[str, float]:
    costs: dict[str, float] = {}
    for model_key, frame in records.groupby("model_key"):
        input_tokens = sum(float((value or {}).get("input_tokens", 0)) for value in frame.usage)
        output_tokens = sum(float((value or {}).get("output_tokens", 0)) for value in frame.usage)
        rates = config["models"][str(model_key)]
        costs[str(model_key)] = (
            input_tokens * float(rates["input_usd_per_million_tokens"])
            + output_tokens * float(rates["output_usd_per_million_tokens"])
        ) / 1_000_000
    return costs


def competence_summary(records: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = records.loc[records.task.eq("competence")].copy()
    tolerance = float(config["execution"]["answer_tolerance"])
    frame["absolute_error"] = (pd.to_numeric(frame.answer, errors="coerce") - frame.expected_numeric_answer).abs()
    frame["within_tolerance"] = frame.absolute_error.le(tolerance)
    summary = (
        frame.groupby(["model_key", "check_type", "information_condition"], as_index=False)
        .agg(
            expected_count=("scenario_id", "size"),
            valid_responses=("parse_status", lambda value: int((value == "valid").sum())),
            accuracy_at_tolerance=("within_tolerance", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
            median_absolute_error=("absolute_error", "median"),
        )
        .sort_values(["check_type", "information_condition", "model_key"])
    )
    overall = (
        frame.groupby("model_key", as_index=False)
        .agg(
            expected_count=("scenario_id", "size"),
            valid_responses=("parse_status", lambda value: int((value == "valid").sum())),
            accuracy_at_tolerance=("within_tolerance", "mean"),
            mean_absolute_error=("absolute_error", "mean"),
        )
        .sort_values("model_key")
    )
    return summary, {"overall": overall.to_dict(orient="records"), "tolerance": tolerance}


def public_forecast_summary(records: pd.DataFrame, outcome_key: pd.DataFrame, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = records.loc[records.task.eq("public_forecast")].copy()
    frame = frame.loc[frame.parse_status.eq("valid")].copy()
    expected = 3
    completed = frame.groupby(["dossier_id", "model_key", "information_condition"]).size()
    if not completed.eq(expected).all():
        raise ValueError("Public-only forecasts are incomplete: expected three valid repetitions per dossier/model/condition.")
    collapsed = (
        frame.groupby(["dossier_id", "model_key", "information_condition"], as_index=False)
        .agg(forecast_mean=("forecast_mean", "mean"), forecast_sd=("forecast_sd", "mean"), repetitions=("scenario_id", "size"))
        .merge(outcome_key[["dossier_id", "next_sue"]], on="dossier_id", validate="many_to_one")
    )
    collapsed["error"] = collapsed.forecast_mean - collapsed.next_sue
    collapsed["absolute_error"] = collapsed.error.abs()
    collapsed["squared_error"] = collapsed.error.pow(2)
    collapsed["normal_nll"] = 0.5 * (
        math.log(2.0 * math.pi) + 2.0 * np.log(collapsed.forecast_sd) + collapsed.squared_error / collapsed.forecast_sd.pow(2)
    )
    summary = (
        collapsed.groupby(["model_key", "information_condition"], as_index=False)
        .agg(
            dossiers=("dossier_id", "size"),
            mae=("absolute_error", "mean"),
            rmse=("squared_error", lambda value: float(np.sqrt(value.mean()))),
            mean_normal_nll=("normal_nll", "mean"),
            mean_forecast_sd=("forecast_sd", "mean"),
        )
        .sort_values(["information_condition", "model_key"])
    )
    comparisons: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        subset = collapsed.loc[collapsed.information_condition.eq(condition)]
        for metric in ["absolute_error", "squared_error", "normal_nll"]:
            pivot = subset.pivot(index="dossier_id", columns="model_key", values=metric)
            comparison = (pivot["gpt6"] - pivot["gpt52"]).dropna()
            row = ci_row(f"gpt6_minus_gpt52_{metric}", comparison, repetitions, seed + len(comparisons), "negative")
            row["information_condition"] = condition
            row["comparison_type"] = "gpt6_minus_gpt52"
            comparisons.append(row)
    for model_key in MODEL_ORDER:
        subset = collapsed.loc[collapsed.model_key.eq(model_key)]
        for metric in ["absolute_error", "squared_error", "normal_nll"]:
            pivot = subset.pivot(index="dossier_id", columns="information_condition", values=metric)
            for label, left, right in [
                ("informative_minus_no_price", "informative_price_path", "no_price"),
                ("informative_minus_matched_placebo", "informative_price_path", "matched_placebo_path"),
                ("matched_placebo_minus_no_price", "matched_placebo_path", "no_price"),
            ]:
                if left not in pivot.columns or right not in pivot.columns:
                    continue
                price_increment = (pivot[left] - pivot[right]).dropna()
                row = ci_row(f"{label}_{metric}", price_increment, repetitions, seed + len(comparisons), "negative")
                row["information_condition"] = label
                row["comparison_type"] = f"within_{model_key}"
                comparisons.append(row)
    return summary, pd.DataFrame(comparisons)


def mechanism_summary(records: pd.DataFrame, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = records.loc[records.task.eq("mechanism")].copy()
    frame = frame.loc[frame.parse_status.eq("valid")].copy()
    expected = 2
    cells = frame.groupby(["dossier_id", "replicate_id", "model_key", "information_condition"]).size()
    if not cells.eq(expected).all():
        raise ValueError("Mechanism grid is incomplete: expected two clue values per dossier/repeat/model/condition.")
    clue = frame.pivot(
        index=["dossier_id", "replicate_id", "model_key", "information_condition"],
        columns="signal_value",
        values="forecast_mean",
    )
    if -0.8 not in clue.columns or 0.8 not in clue.columns:
        raise ValueError("Mechanism grid is missing one of the two fixed private-clue values.")
    beta = ((clue[0.8] - clue[-0.8]) / 1.6).rename("private_clue_slope").reset_index()
    beta_dossier = (
        beta.groupby(["dossier_id", "model_key", "information_condition"], as_index=False)
        .agg(private_clue_slope=("private_clue_slope", "mean"), repetitions=("replicate_id", "size"))
    )
    expected_repetitions = 3
    if not beta_dossier.repetitions.eq(expected_repetitions).all():
        raise ValueError("Mechanism grid is incomplete: expected three repetitions per dossier/model/condition.")
    slope_summary = (
        beta_dossier.groupby(["model_key", "information_condition"], as_index=False)
        .agg(dossiers=("dossier_id", "size"), mean_private_clue_slope=("private_clue_slope", "mean"), sd_across_dossiers=("private_clue_slope", "std"))
        .sort_values(["information_condition", "model_key"])
    )
    gaps = beta_dossier.pivot(index=["dossier_id", "information_condition"], columns="model_key", values="private_clue_slope").reset_index()
    gaps["gpt6_minus_gpt52"] = gaps["gpt6"] - gaps["gpt52"]
    gap_summary_rows: list[dict[str, Any]] = []
    for condition in CONDITION_ORDER:
        subset = gaps.loc[gaps.information_condition.eq(condition), "gpt6_minus_gpt52"]
        if not subset.empty:
            row = ci_row("gpt6_minus_gpt52_private_clue_slope", subset, repetitions, seed + len(gap_summary_rows), "negative")
            row["information_condition"] = condition
            gap_summary_rows.append(row)
    gap_summary = pd.DataFrame(gap_summary_rows)
    gap_wide = gaps.pivot(index="dossier_id", columns="information_condition", values="gpt6_minus_gpt52")
    contrasts: list[dict[str, Any]] = []
    for label, left, right in [
        ("primary_informative_minus_placebo", "informative_price_path", "matched_placebo_path"),
        ("supplement_informative_minus_no_price", "informative_price_path", "no_price"),
    ]:
        if left in gap_wide.columns and right in gap_wide.columns:
            row = ci_row(label, (gap_wide[left] - gap_wide[right]).dropna(), repetitions, seed + 100 + len(contrasts), "negative")
            contrasts.append(row)
    return slope_summary, gap_summary, pd.DataFrame(contrasts)


def markdown_table(frame: pd.DataFrame, digits: int = 4) -> str:
    if frame.empty:
        return "_No rows._"
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_float_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: f"{value:.{digits}f}" if pd.notna(value) else "")
    columns = [str(column) for column in printable.columns]
    def cell(value: Any) -> str:
        if pd.isna(value):
            return ""
        return str(value).replace("|", "\\|").replace("\n", " ")
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    lines.extend("| " + " | ".join(cell(value) for value in row) + " |" for row in printable.itertuples(index=False, name=None))
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-subdir", default="analysis_v1")
    args = parser.parse_args()
    config = read_config(args.config)
    root = resolve_repo_path(config["output_root"])
    inputs = {
        "competence_gpt6": root / "task1_competence_gpt6_v1" / RAW_NAME,
        "competence_gpt52": root / "task1_competence_gpt52_v1" / RAW_NAME,
        "public_gpt6": root / "task1_public_gpt6_v1" / RAW_NAME,
        "public_gpt52": root / "task1_public_gpt52_v1" / RAW_NAME,
        "public_placebo_gpt6": root / "task1_public_placebo_gpt6_v1" / RAW_NAME,
        "public_placebo_gpt52": root / "task1_public_placebo_gpt52_v1" / RAW_NAME,
        "mechanism_gpt6": root / "task2_mechanism_gpt6_v1" / RAW_NAME,
        "mechanism_gpt52": root / "task2_mechanism_gpt52_v1" / RAW_NAME,
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Task 1--2 raw outputs are missing:\n" + "\n".join(missing))
    frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    audit: dict[str, dict[str, int]] = {}
    for label, path in inputs.items():
        frame, audit_row = first_draws(path)
        frames.append(frame)
        raw_frames.append(pd.DataFrame(read_jsonl(path)))
        audit[label] = audit_row
    records = pd.concat(frames, ignore_index=True)
    raw_records = pd.concat(raw_frames, ignore_index=True)
    output = root / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    reps = int(config["analysis"]["bootstrap_repetitions"])
    seed = int(config["analysis"]["bootstrap_seed"])
    competence, competence_meta = competence_summary(records, config)
    outcome_key = pd.read_parquet(resolve_repo_path(config["interim_root"]) / "step4_v3_outcome_key.parquet")
    forecast, forecast_comparisons = public_forecast_summary(records, outcome_key, reps, seed)
    slopes, gaps, contrasts = mechanism_summary(records, reps, seed)
    competence.to_csv(output / "task1_competence_aggregate.csv", index=False)
    forecast.to_csv(output / "task1_public_forecast_aggregate.csv", index=False)
    forecast_comparisons.to_csv(output / "task1_public_forecast_model_comparisons.csv", index=False)
    slopes.to_csv(output / "task2_private_clue_slopes.csv", index=False)
    gaps.to_csv(output / "task2_model_gap_by_information_condition.csv", index=False)
    contrasts.to_csv(output / "task2_informative_placebo_contrasts.csv", index=False)
    # Cost accounting includes every completed raw call, including any
    # accidentally duplicated request; analytical estimands above do not.
    costs = actual_cost(raw_records, config)
    result = {
        "raw_audit": audit,
        "actual_api_cost_usd_by_model": costs,
        "actual_api_cost_usd_total": float(sum(costs.values())),
        "competence_overall": competence_meta["overall"],
        "task1_public_forecast": forecast.to_dict(orient="records"),
        "task1_model_comparisons": forecast_comparisons.to_dict(orient="records"),
        "task2_model_gaps": gaps.to_dict(orient="records"),
        "task2_contrasts": contrasts.to_dict(orient="records"),
    }
    (output / "step4_v3_task12_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = "\n".join([
        "# Step 4 Task 1--2 development results",
        "",
        "This report uses the 20 locked development dossiers. Repetitions are averaged within dossier; all bootstrap intervals resample dossiers. It is not a new-firm holdout result.",
        "",
        "## Raw-output audit",
        "",
        "```json",
        json.dumps(audit, indent=2),
        "```",
        "",
        f"Actual recorded API cost: ${sum(costs.values()):.4f} (GPT-6 ${costs.get('gpt6', 0.0):.4f}; GPT-5.2 ${costs.get('gpt52', 0.0):.4f}).",
        "",
        "## Task 1: deterministic public-information competence",
        "",
        f"Tolerance: {competence_meta['tolerance']:.4f}.",
        "",
        markdown_table(competence),
        "",
        "## Task 1: public-only next-surprise forecasts",
        "",
        markdown_table(forecast),
        "",
        "Paired comparison is GPT-6 minus GPT-5.2; negative values favor GPT-6 for each loss.",
        "",
        markdown_table(forecast_comparisons),
        "",
        "## Task 2: private-clue slopes",
        "",
        markdown_table(slopes),
        "",
        markdown_table(gaps),
        "",
        "The primary contrast is the informative-path model gap minus the matched-placebo-path model gap. The preregistered directional prediction is negative.",
        "",
        markdown_table(contrasts),
        "",
    ])
    (output / "step4_v3_task12_report.md").write_text(report)
    print(json.dumps({"output": str(output), "actual_api_cost_usd_total": sum(costs.values()), "task2_primary": contrasts.iloc[0].to_dict() if not contrasts.empty else None}, allow_nan=False))


if __name__ == "__main__":
    main()
