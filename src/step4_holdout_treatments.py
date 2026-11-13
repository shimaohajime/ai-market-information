"""Construct blinded real-data public-information conditions for the Step 4 holdout.

The source files are local and time-safe. This program makes no model or
network calls. It constructs equal-format genuine, cross-issuer, and
time-permuted price paths; it never exposes identities, dates, or outcomes in
the prompt-ready dossier file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from step4_phase0 import REPO_ROOT, read_config, resolve_repo_path
from step4_v3_task12_build import (
    PROHIBITED_FIELDS,
    build_checks,
    choose_donors,
    compound,
    placebo_price_information,
    price_without_summary,
    scalar,
    verify_no_prohibited,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_holdout_treatments_v1.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_seed(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def validate_price(price: dict[str, Any]) -> None:
    weekly = price["weekly_history"]
    daily = price["recent_daily_history"]
    if len(weekly) != 52 or len(daily) != 60:
        raise ValueError("A holdout treatment requires exactly 52 weekly and 60 daily rows.")
    if [row["period_index"] for row in weekly] != list(range(-52, 0)):
        raise ValueError("Weekly period indexes are not the expected anonymous sequence.")
    if [row["trading_day_index"] for row in daily] != list(range(-60, 0)):
        raise ValueError("Daily indexes are not the expected anonymous sequence.")


def permuted_price_information(target: dict[str, Any], permutation_seed: str) -> dict[str, Any]:
    """Destroy abnormal-return time order while preserving local marginals.

    The first forty displayed weeks and final sixty displayed days are distinct
    non-overlapping source segments. Their abnormal-return values are permuted
    within segment, while target market returns remain at their original
    positions. The final twelve weekly rows are regenerated from the permuted
    daily sequence, preserving daily/weekly consistency.
    """
    source = target["rich_price_information"]
    validate_price(source)
    weekly_source = source["weekly_history"]
    daily_source = source["recent_daily_history"]
    rng = np.random.default_rng(stable_seed(permutation_seed))
    weekly_abnormal = np.asarray([float(row["abnormal_return"]) for row in weekly_source[:40]], dtype=float)
    daily_abnormal = np.asarray([float(row["abnormal_return"]) for row in daily_source], dtype=float)
    weekly_abnormal = weekly_abnormal[rng.permutation(len(weekly_abnormal))]
    daily_abnormal = daily_abnormal[rng.permutation(len(daily_abnormal))]

    first_weekly: list[tuple[float, float, int]] = []
    for source_row, abnormal in zip(weekly_source[:40], weekly_abnormal, strict=True):
        market = float(source_row["market_return"])
        stock = market + float(abnormal)
        if stock <= -1.0:
            raise ValueError("Permuted weekly stock return is invalid.")
        first_weekly.append((stock, market, int(source_row["period_index"])))

    daily_values: list[tuple[float, float, int]] = []
    for source_row, abnormal in zip(daily_source, daily_abnormal, strict=True):
        market = float(source_row["market_return"])
        stock = market + float(abnormal)
        if stock <= -1.0:
            raise ValueError("Permuted daily stock return is invalid.")
        daily_values.append((stock, market, int(source_row["trading_day_index"])))

    weekly_values = first_weekly.copy()
    for number in range(12):
        block = daily_values[number * 5:(number + 1) * 5]
        weekly_values.append((
            compound([value[0] for value in block]),
            compound([value[1] for value in block]),
            int(weekly_source[40 + number]["period_index"]),
        ))
    if len(weekly_values) != 52:
        raise AssertionError("Time-permuted path has the wrong weekly length.")

    gross = 1.0
    weekly: list[dict[str, float | int]] = []
    for stock, market, index in weekly_values:
        gross *= 1.0 + stock
        weekly.append({
            "period_index": index,
            "stock_return": scalar(stock),
            "market_return": scalar(market),
            "abnormal_return": scalar(stock - market),
            "rebased_price_index": scalar(100.0 * gross),
        })
    gross_before_daily = float(np.prod([1.0 + value[0] for value in weekly_values[:40]]))
    gross = gross_before_daily
    daily: list[dict[str, float | int]] = []
    for stock, market, index in daily_values:
        gross *= 1.0 + stock
        daily.append({
            "trading_day_index": index,
            "stock_return": scalar(stock),
            "market_return": scalar(market),
            "abnormal_return": scalar(stock - market),
            "rebased_price_index": scalar(100.0 * gross),
        })
    if abs(float(weekly[-1]["rebased_price_index"]) - float(daily[-1]["rebased_price_index"])) > 0.001:
        raise AssertionError("Time-permuted weekly and daily endpoint indexes disagree.")
    return {
        "price_information_version": "v2_daily_path_anonymous",
        "timing": "All price-derived values end on the last CRSP trading day strictly before the information cutoff.",
        "price_index_definition": "A split-adjusted return-implied index rebased to 100 at the start of the 260-trading-day history. It is not a dollar price.",
        "weekly_history": weekly,
        "recent_daily_history": daily,
    }


def build(config: dict[str, Any]) -> dict[str, Any]:
    source_root = resolve_repo_path(config["source_interim_root"])
    source_dossiers = {
        str(item["dossier_id"]): item
        for item in read_jsonl(source_root / "step4_price_v2_dossiers.jsonl")
    }
    candidates = pd.read_parquet(resolve_repo_path(config["source_candidate_key"])).copy()
    candidates["dossier_id"] = candidates.dossier_id.astype(str)
    if len(source_dossiers) != len(candidates) or set(source_dossiers) != set(candidates.dossier_id):
        raise ValueError("Rich-price dossiers and holdout candidate key are not aligned.")
    mapping = choose_donors(candidates, source_dossiers, config)
    matched = {
        row.dossier_id: placebo_price_information(source_dossiers[row.dossier_id], source_dossiers[row.donor_dossier_id])
        for row in mapping.itertuples(index=False)
    }
    permuted = {
        dossier_id: permuted_price_information(
            source,
            f"{config['placebo']['time_permutation_seed']}|{dossier_id}",
        )
        for dossier_id, source in source_dossiers.items()
    }
    built: list[dict[str, Any]] = []
    for dossier_id in sorted(source_dossiers):
        source = source_dossiers[dossier_id]
        payload = {
            "dossier_id": dossier_id,
            "dossier_version": "holdout_v1_four_condition_anonymous",
            "public_information": source["public_information_without_price"],
            "informative_price_information": price_without_summary(source["rich_price_information"]),
            "matched_placebo_price_information": matched[dossier_id],
            "time_permuted_price_information": permuted[dossier_id],
        }
        verify_no_prohibited(payload, f"holdout treatment dossier {dossier_id}")
        built.append(payload)
    checks = build_checks({item["dossier_id"]: item for item in built})

    interim = resolve_repo_path(config["interim_root"])
    output = resolve_repo_path(config["output_root"])
    interim.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (interim / "step4_holdout_dossiers.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in built)
    )
    checks.to_csv(interim / "step4_holdout_competence_checks.csv", index=False)
    outcome = candidates[[
        "dossier_id", "theta_ex_ante", "next_sue", "formation_date", "next_announcement_date"
    ]].copy()
    outcome.to_parquet(interim / "step4_holdout_outcome_key.parquet", index=False)
    mapping.to_csv(interim / "step4_holdout_placebo_mapping_private.csv", index=False)
    public_mapping = mapping[["dossier_id", "same_industry_bucket", "same_size_bucket"]].copy()
    public_mapping["time_permutation_rule"] = "seeded_abnormal_return_permutation_within_weekly_and_daily_segments"
    public_mapping.to_csv(output / "step4_holdout_placebo_audit.csv", index=False)
    summary = {
        "dossier_count": len(built),
        "competence_check_count": len(checks),
        "information_conditions": [
            "no_price",
            "informative_price_path",
            "matched_placebo_path",
            "time_permuted_price_path",
        ],
        "all_donors_pre_cutoff": bool(mapping.donor_formation_date.le(mapping.target_formation_date).all()),
        "prohibited_key_check": "passed",
        "api_calls_made": 0,
    }
    (output / "step4_holdout_treatments_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build(read_config(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
