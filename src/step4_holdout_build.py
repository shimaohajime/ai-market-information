"""Build a locked, time-safe real-data sample for the E2 replication.

This command makes no model or network call. It selects 50 anonymous
firm-quarter dossiers from years disjoint from the 2012--2016 development
sample, using a fixed pre-outcome hash. Realized outcomes and identifiers stay
in the ignored interim key; public outputs contain only anonymous aggregates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from step4_phase0 import (
    REPO_ROOT,
    attach_sic,
    dossier_payload,
    eligible_rows,
    read_config,
    resolve_repo_path,
)


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_holdout_v1.json"
PROHIBITED_FIELDS = {
    "permno",
    "cusip",
    "ticker",
    "ibes_ticker",
    "formation_date",
    "next_sue",
    "theta_ex_ante",
    "next_announcement_date",
    "stock_price",
}


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def selection_plan(config: dict[str, Any]) -> dict[str, Any]:
    selection = config["selection"]
    years = [int(value) for value in selection["formation_years"]]
    per_year = int(selection["candidates_per_year"])
    expected = len(years) * per_year
    if expected != int(selection["candidate_count"]):
        raise ValueError("candidate_count must equal years times candidates_per_year.")
    if len(set(years)) != len(years) or years != sorted(years):
        raise ValueError("formation_years must be unique and sorted.")
    if set(years).intersection(range(2012, 2017)):
        raise ValueError("The holdout years must be disjoint from the 2012--2016 development window.")
    return {
        "dossier_count": expected,
        "formation_years": years,
        "candidates_per_year": per_year,
        "selection_seed": str(selection["selection_seed"]),
        "api_calls_made": 0,
    }


def select_holdout(eligible: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = config["selection"]
    years = [int(value) for value in selection["formation_years"]]
    per_year = int(selection["candidates_per_year"])
    frame = eligible.copy()
    frame["_hash"] = [
        stable_hash(f"{selection['selection_seed']}|{int(permno)}|{date.date().isoformat()}")
        for permno, date in zip(frame.permno, frame.formation_date, strict=True)
    ]
    selected: list[pd.Series] = []
    used_permnos: set[int] = set()
    global_sector_counts: Counter[str] = Counter()
    for year in years:
        pool = frame.loc[frame.formation_year.eq(year)].copy()
        if pool.empty:
            raise ValueError(f"No eligible candidates in holdout formation year {year}.")
        year_sector_counts: Counter[str] = Counter()
        for _ in range(per_year):
            candidates = pool.loc[~pool.permno.isin(used_permnos)].copy()
            if candidates.empty:
                raise ValueError(f"Could not select {per_year} distinct firms in holdout year {year}.")
            candidates["_year_sector_seen"] = candidates.industry_bucket.map(year_sector_counts).fillna(0).gt(0)
            candidates["_global_sector_count"] = candidates.industry_bucket.map(global_sector_counts).fillna(0)
            row = candidates.sort_values(
                ["_year_sector_seen", "_global_sector_count", "_hash"],
                ascending=[True, True, True],
            ).iloc[0]
            selected.append(row)
            used_permnos.add(int(row.permno))
            year_sector_counts[str(row.industry_bucket)] += 1
            global_sector_counts[str(row.industry_bucket)] += 1
    result = pd.DataFrame(selected).reset_index(drop=True)
    prefix = str(config["holdout"]["dossier_id_prefix"])
    result["dossier_id"] = [f"{prefix}{number:03d}" for number in range(1, len(result) + 1)]
    if len(result) != int(selection["candidate_count"]) or result.permno.duplicated().any():
        raise AssertionError("Holdout selection did not produce the locked one-firm-per-dossier design.")
    return result.drop(columns="_hash")


def verify_payload(payload: dict[str, Any], label: str) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    leaked = sorted(field for field in PROHIBITED_FIELDS if f'"{field}"' in rendered)
    if leaked:
        raise AssertionError(f"{label} leaked prohibited fields: {leaked}")


def public_manifest(selected: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    output = pd.DataFrame({
        "dossier_id": selected.dossier_id.astype(str),
        "formation_year": selected.formation_year.astype(int),
        "industry_bucket": selected.industry_bucket.astype(str),
        "size_bucket": selected.size_bucket.astype(str),
        "financial_statement_age_months": (selected.statement_age_days.astype(float) / 30.0).round().astype(int),
        "days_to_next_announcement": selected.days_to_next_announcement.astype(int),
    })
    requested = list(config["holdout"]["public_manifest_fields"])
    if set(requested) != set(output.columns):
        raise ValueError("public_manifest_fields must match the allowed anonymous manifest columns.")
    return output.loc[:, requested].sort_values("dossier_id").reset_index(drop=True)


def build(config: dict[str, Any]) -> dict[str, Any]:
    plan = selection_plan(config)
    panel = pd.read_parquet(resolve_repo_path(config["panel_path"]))
    eligible, diagnostics = eligible_rows(panel, config)
    eligible = attach_sic(eligible, resolve_repo_path(config["ccm_link_path"]))
    selected = select_holdout(eligible, config)
    dossiers = [dossier_payload(row, config) for _, row in selected.iterrows()]
    for dossier in dossiers:
        verify_payload(dossier, f"holdout dossier {dossier['dossier_id']}")
    interim = resolve_repo_path(config["interim_root"])
    output = resolve_repo_path(config["output_root"])
    interim.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    selected.to_parquet(interim / "step4_candidate_key.parquet", index=False)
    (interim / "step4_dossiers.jsonl").write_text(
        "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in dossiers)
    )
    manifest = public_manifest(selected, config)
    manifest.to_csv(output / "step4_holdout_public_source_manifest.csv", index=False)
    config_hash = hashlib.sha256(
        json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    summary = {
        **plan,
        "eligible_diagnostics": diagnostics,
        "selected_by_formation_year": {
            str(key): int(value) for key, value in selected.formation_year.value_counts().sort_index().items()
        },
        "selected_by_industry_bucket": {
            str(key): int(value) for key, value in selected.industry_bucket.value_counts().sort_index().items()
        },
        "candidate_key_path": str(interim / "step4_candidate_key.parquet"),
        "anonymous_dossiers_path": str(interim / "step4_dossiers.jsonl"),
        "public_manifest_path": str(output / "step4_holdout_public_source_manifest.csv"),
        "config_sha256": config_hash,
        "prohibited_key_check": "passed",
    }
    (output / "step4_holdout_selection_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["plan", "build"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.command == "plan":
        print(json.dumps(selection_plan(config), indent=2))
    else:
        print(json.dumps(build(config), indent=2))


if __name__ == "__main__":
    main()
