"""Analyze the locked 50-dossier Step 4 behavioral replication.

This deliberately reads only the Task 2 raw outputs produced under
step4_holdout_behavioral_replication_v1.  It writes aggregate-only results;
no dossier-level response, target, or identifier is exported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from step4_phase0 import REPO_ROOT, read_config, resolve_repo_path
from step4_v3_task12_analysis import actual_cost, ci_row, first_draws, markdown_table
from step4_v3_task12_runner import RAW_NAME


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_holdout_behavioral_replication_v1.json"
EXPECTED_CONDITIONS = ["no_price", "informative_price_path"]
EXPECTED_MODELS = ["gpt6", "gpt52"]


def summary(records: pd.DataFrame, repetitions: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frame = records.loc[(records.task == "mechanism") & (records.parse_status == "valid")].copy()
    if frame.empty:
        raise ValueError("No valid Task 2 records found.")
    cells = frame.groupby(["dossier_id", "replicate_id", "model_key", "information_condition"]).size()
    if not cells.eq(2).all():
        raise ValueError("Incomplete mechanism grid: each dossier/repetition/model/context cell needs two clues.")
    clue = frame.pivot(
        index=["dossier_id", "replicate_id", "model_key", "information_condition"],
        columns="signal_value",
        values="forecast_mean",
    )
    if not {-0.8, 0.8}.issubset(clue.columns):
        raise ValueError("Both fixed private clues are required.")
    beta = ((clue[0.8] - clue[-0.8]) / 1.6).rename("private_clue_slope").reset_index()
    beta_dossier = beta.groupby(["dossier_id", "model_key", "information_condition"], as_index=False).agg(
        private_clue_slope=("private_clue_slope", "mean"), repetitions=("replicate_id", "size")
    )
    if not beta_dossier.repetitions.eq(3).all():
        raise ValueError("Incomplete mechanism grid: each dossier/model/context needs three repetitions.")
    observed_conditions = set(beta_dossier.information_condition)
    observed_models = set(beta_dossier.model_key)
    if observed_conditions != set(EXPECTED_CONDITIONS) or observed_models != set(EXPECTED_MODELS):
        raise ValueError(f"Unexpected analyzed design: conditions={observed_conditions}, models={observed_models}.")
    slopes = beta_dossier.groupby(["model_key", "information_condition"], as_index=False).agg(
        dossiers=("dossier_id", "size"),
        mean_private_clue_slope=("private_clue_slope", "mean"),
        sd_across_dossiers=("private_clue_slope", "std"),
    ).sort_values(["information_condition", "model_key"])
    wide = beta_dossier.pivot(index=["dossier_id", "information_condition"], columns="model_key", values="private_clue_slope")
    wide["gpt6_minus_gpt52"] = wide["gpt6"] - wide["gpt52"]
    gap_rows: list[dict[str, Any]] = []
    for offset, condition in enumerate(EXPECTED_CONDITIONS):
        values = wide.xs(condition, level="information_condition")["gpt6_minus_gpt52"]
        row = ci_row("gpt6_minus_gpt52_private_clue_slope", values, repetitions, seed + offset, "negative")
        row["information_condition"] = condition
        gap_rows.append(row)
    gaps = pd.DataFrame(gap_rows)
    gaps_by_dossier = wide["gpt6_minus_gpt52"].unstack("information_condition")
    primary = ci_row(
        "rich_context_gap_minus_no_price_gap",
        gaps_by_dossier["informative_price_path"] - gaps_by_dossier["no_price"],
        repetitions,
        seed + 100,
        "negative",
    )
    contrasts = pd.DataFrame([primary])
    return slopes, gaps, contrasts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-subdir", default="analysis_v1")
    args = parser.parse_args()
    config = read_config(args.config)
    root = resolve_repo_path(config["output_root"])
    inputs = {
        key: root / f"task2_mechanism_{key}_v1" / RAW_NAME
        for key in EXPECTED_MODELS
    }
    missing = [str(path) for path in inputs.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Required Task 2 raw outputs are missing:\n" + "\n".join(missing))
    first_frames: list[pd.DataFrame] = []
    raw_frames: list[pd.DataFrame] = []
    audit: dict[str, dict[str, int]] = {}
    for key, path in inputs.items():
        frame, audit_row = first_draws(path)
        first_frames.append(frame)
        raw_frames.append(pd.DataFrame([json.loads(line) for line in path.read_text().splitlines() if line.strip()]))
        audit[key] = audit_row
    records = pd.concat(first_frames, ignore_index=True)
    raw_records = pd.concat(raw_frames, ignore_index=True)
    reps = int(config["analysis"]["bootstrap_repetitions"])
    seed = int(config["analysis"]["bootstrap_seed"])
    slopes, gaps, contrasts = summary(records, reps, seed)
    output = root / args.output_subdir
    output.mkdir(parents=True, exist_ok=True)
    slopes.to_csv(output / "task2_private_clue_slopes_aggregate.csv", index=False)
    gaps.to_csv(output / "task2_model_gap_by_context_aggregate.csv", index=False)
    contrasts.to_csv(output / "task2_context_amplification_aggregate.csv", index=False)
    costs = actual_cost(raw_records, config)
    result = {
        "purpose": "Independent behavioral replication; not an issuer-specific public-signal forecasting test.",
        "raw_audit": audit,
        "actual_api_cost_usd_by_model": costs,
        "actual_api_cost_usd_total": float(sum(costs.values())),
        "private_clue_slopes": slopes.to_dict(orient="records"),
        "model_gaps": gaps.to_dict(orient="records"),
        "primary_context_amplification": contrasts.to_dict(orient="records"),
    }
    (output / "step4_holdout_behavioral_replication_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = "\n".join([
        "# Step 4 independent behavioral replication",
        "",
        "This is a fixed, independent 50-dossier Task 2 behavioral replication. It tests private-clue substitution in a richer anonymous public financial context. It does not test or establish issuer-specific predictive content for next earnings surprises.",
        "",
        "## Raw-output audit",
        "",
        "```json",
        json.dumps(audit, indent=2),
        "```",
        "",
        f"Actual recorded API cost: ${sum(costs.values()):.4f} (GPT-6 ${costs.get('gpt6', 0.0):.4f}; GPT-5.2 ${costs.get('gpt52', 0.0):.4f}).",
        "",
        "## Private-clue slopes",
        "",
        markdown_table(slopes),
        "",
        "## Capability gaps",
        "",
        "Each gap is GPT-6.1 Sol minus the fixed GPT-5.2 snapshot. Negative values mean the stronger model is less responsive to the same synthetic private clue.",
        "",
        markdown_table(gaps),
        "",
        "## Primary context-amplification contrast",
        "",
        "This is the rich-context capability gap minus the no-price capability gap. The directional prediction is negative.",
        "",
        markdown_table(contrasts),
        "",
    ])
    (output / "step4_holdout_behavioral_replication_report.md").write_text(report)
    print(json.dumps({"output": str(output), "primary": contrasts.iloc[0].to_dict()}, allow_nan=False))


if __name__ == "__main__":
    main()
