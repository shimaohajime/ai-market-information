"""Build time-safe inputs for the E2 controlled behavior experiment.

This script makes no model or network call.  It creates a matched-placebo
anonymous price path for each of the existing twenty development dossiers,
local answer keys for deterministic competence checks, and a local held-out
outcome key.  The latter two files are in ``data/interim`` and ignored by Git.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from step4_phase0 import REPO_ROOT, read_config, resolve_repo_path


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_v3_task12.json"
PROHIBITED_FIELDS = {"permno", "cusip", "ticker", "formation_date", "next_sue", "theta_ex_ante", "stock_price"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def scalar(value: float, digits: int = 4) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("E2 construction encountered a non-finite price value.")
    return round(numeric, digits)


def compound(values: list[float] | pd.Series) -> float:
    return float(np.prod(1.0 + np.asarray(values, dtype=float)) - 1.0)


def price_without_summary(price: dict[str, Any]) -> dict[str, Any]:
    result = dict(price)
    result.pop("price_summary", None)
    return result


def verify_no_prohibited(payload: dict[str, Any], label: str) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    leaked = sorted(field for field in PROHIBITED_FIELDS if f'"{field}"' in rendered)
    if leaked:
        raise AssertionError(f"{label} leaked prohibited fields: {leaked}")


def price_stats(dossier: dict[str, Any]) -> tuple[float, float]:
    summary = dossier["rich_price_information"]["price_summary"]
    return float(summary["stock_volatility_260d"]), float(summary["stock_return_260d"])


def donor_score(target: pd.Series, donor: pd.Series, dossiers: dict[str, dict[str, Any]], donor_use: Counter[str]) -> tuple[float, int, int]:
    target_vol, target_return = price_stats(dossiers[str(target.dossier_id)])
    donor_vol, donor_return = price_stats(dossiers[str(donor.dossier_id)])
    volatility_distance = abs(math.log(max(donor_vol, 1e-9) / max(target_vol, 1e-9)))
    return_distance = abs(donor_return - target_return)
    same_industry_penalty = 0.75 if str(target.industry_bucket) == str(donor.industry_bucket) else 0.0
    same_size_penalty = 0.10 if str(target.size_bucket) == str(donor.size_bucket) else 0.0
    score = volatility_distance + 0.5 * return_distance + same_industry_penalty + same_size_penalty
    tie = stable_hash(f"step4-v3-placebo-v1|{target.dossier_id}|{donor.dossier_id}")
    return score, int(donor_use[str(donor.dossier_id)]), tie


def choose_donors(candidates: pd.DataFrame, dossiers: dict[str, dict[str, Any]], config: dict[str, Any]) -> pd.DataFrame:
    """Select a pre-cutoff donor for each target without using outcomes.

    The target market path is retained later.  Donors supply only an
    idiosyncratic abnormal-return sequence and therefore do not identify a
    contemporaneous issuer to the model.
    """
    candidates = candidates.copy()
    candidates["formation_date"] = pd.to_datetime(candidates.formation_date)
    candidates = candidates.sort_values(["formation_date", "dossier_id"]).reset_index(drop=True)
    donor_use: Counter[str] = Counter()
    rows: list[dict[str, Any]] = []
    for target in candidates.itertuples(index=False):
        pool = candidates.loc[
            candidates.formation_date.le(target.formation_date)
            & candidates.dossier_id.ne(target.dossier_id)
        ].copy()
        if pool.empty:
            raise ValueError(f"No pre-cutoff placebo donor available for {target.dossier_id}.")
        choice = min(
            pool.itertuples(index=False),
            key=lambda donor: donor_score(pd.Series(target._asdict()), pd.Series(donor._asdict()), dossiers, donor_use),
        )
        donor_use[str(choice.dossier_id)] += 1
        rows.append({
            "dossier_id": str(target.dossier_id),
            "donor_dossier_id": str(choice.dossier_id),
            "target_formation_date": target.formation_date,
            "donor_formation_date": choice.formation_date,
            "same_industry_bucket": bool(str(target.industry_bucket) == str(choice.industry_bucket)),
            "same_size_bucket": bool(str(target.size_bucket) == str(choice.size_bucket)),
        })
    mapping = pd.DataFrame(rows)
    if mapping.dossier_id.duplicated().any():
        raise AssertionError("Each target must have exactly one matched-placebo donor.")
    if not mapping.donor_formation_date.le(mapping.target_formation_date).all():
        raise AssertionError("A placebo donor uses a post-target price history.")
    return mapping


def placebo_price_information(target: dict[str, Any], donor: dict[str, Any]) -> dict[str, Any]:
    """Combine target market returns with donor abnormal returns.

    The first forty weekly observations use weekly returns.  The final twelve
    weeks are regenerated from the target/donor daily rows so the supplied
    daily and weekly histories agree exactly over their overlap.
    """
    target_price = target["rich_price_information"]
    donor_price = donor["rich_price_information"]
    target_weekly = target_price["weekly_history"]
    donor_weekly = donor_price["weekly_history"]
    target_daily = target_price["recent_daily_history"]
    donor_daily = donor_price["recent_daily_history"]
    if not (len(target_weekly) == len(donor_weekly) == 52 and len(target_daily) == len(donor_daily) == 60):
        raise ValueError("Expected 52 weekly and 60 daily rows for both placebo inputs.")
    if [value["period_index"] for value in target_weekly] != [value["period_index"] for value in donor_weekly]:
        raise ValueError("Target and donor weekly histories have incompatible indexes.")
    if [value["trading_day_index"] for value in target_daily] != [value["trading_day_index"] for value in donor_daily]:
        raise ValueError("Target and donor daily histories have incompatible indexes.")

    weekly_returns: list[tuple[float, float]] = []
    for index in range(40):
        market = float(target_weekly[index]["market_return"])
        stock = market + float(donor_weekly[index]["abnormal_return"])
        if stock <= -1.0:
            raise ValueError("Matched placebo weekly stock return is invalid.")
        weekly_returns.append((stock, market))

    daily_returns: list[tuple[float, float]] = []
    for target_row, donor_row in zip(target_daily, donor_daily, strict=True):
        market = float(target_row["market_return"])
        stock = market + float(donor_row["abnormal_return"])
        if stock <= -1.0:
            raise ValueError("Matched placebo daily stock return is invalid.")
        daily_returns.append((stock, market))
    for index in range(12):
        block = daily_returns[index * 5:(index + 1) * 5]
        weekly_returns.append((compound([row[0] for row in block]), compound([row[1] for row in block])))

    weekly: list[dict[str, float | int]] = []
    gross = 1.0
    for index, ((stock, market), target_row) in enumerate(zip(weekly_returns, target_weekly, strict=True)):
        gross *= 1.0 + stock
        weekly.append({
            "period_index": int(target_row["period_index"]),
            "stock_return": scalar(stock),
            "market_return": scalar(market),
            "abnormal_return": scalar(stock - market),
            "rebased_price_index": scalar(100.0 * gross),
        })

    gross_before_daily = float(np.prod([1.0 + value[0] for value in weekly_returns[:40]]))
    daily: list[dict[str, float | int]] = []
    gross = gross_before_daily
    for (stock, market), target_row in zip(daily_returns, target_daily, strict=True):
        gross *= 1.0 + stock
        daily.append({
            "trading_day_index": int(target_row["trading_day_index"]),
            "stock_return": scalar(stock),
            "market_return": scalar(market),
            "abnormal_return": scalar(stock - market),
            "rebased_price_index": scalar(100.0 * gross),
        })
    if abs(float(daily[-1]["rebased_price_index"]) - float(weekly[-1]["rebased_price_index"])) > 0.001:
        raise AssertionError("Placebo weekly and daily price histories are inconsistent at the endpoint.")
    return {
        "price_information_version": "v2_daily_path_anonymous",
        "timing": "All price-derived values end on the last CRSP trading day strictly before the information cutoff.",
        "price_index_definition": "A split-adjusted return-implied index rebased to 100 at the start of the 260-trading-day history. It is not a dollar price.",
        "weekly_history": weekly,
        "recent_daily_history": daily,
    }


def build_checks(dossiers: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dossier_id, dossier in sorted(dossiers.items()):
        accounting = dossier["public_information"]["accounting_ratios"]
        price = price_without_summary(dossier["informative_price_information"])
        daily = price["recent_daily_history"]
        weekly = price["weekly_history"]
        checks = [
            (
                "financial_arithmetic",
                "Using only accounting_ratios, compute leverage minus cash_holdings, rounded to four decimals.",
                scalar(float(accounting["leverage"]) - float(accounting["cash_holdings"])),
                "no_price",
            ),
            (
                "financial_arithmetic",
                "Using only accounting_ratios, compute operating_profitability plus roa, rounded to four decimals.",
                scalar(float(accounting["operating_profitability"]) + float(accounting["roa"])),
                "no_price",
            ),
            (
                "price_path_arithmetic",
                "Using only recent_daily_history, compound stock_return over the final 20 rows and return the result rounded to four decimals.",
                scalar(compound([float(row["stock_return"]) for row in daily[-20:]])),
                "informative_price_path",
            ),
            (
                "price_path_arithmetic",
                "Using only weekly_history, compound stock_return and market_return over the final 12 rows, then return stock cumulative return minus market cumulative return rounded to four decimals.",
                scalar(
                    compound([float(row["stock_return"]) for row in weekly[-12:]])
                    - compound([float(row["market_return"]) for row in weekly[-12:]])
                ),
                "informative_price_path",
            ),
        ]
        for number, (check_type, question, answer, information_condition) in enumerate(checks, start=1):
            rows.append({
                "dossier_id": dossier_id,
                "check_id": f"{dossier_id}_task1_{number}",
                "check_type": check_type,
                "information_condition": information_condition,
                "question": question,
                "expected_numeric_answer": answer,
            })
    return pd.DataFrame(rows)


def build(config: dict[str, Any]) -> dict[str, Any]:
    source_root = resolve_repo_path(config["source_interim_root"])
    source_dossiers = {item["dossier_id"]: item for item in read_jsonl(source_root / "step4_price_v2_dossiers.jsonl")}
    candidates = pd.read_parquet(resolve_repo_path(config["source_candidate_key"]))
    if len(source_dossiers) != len(candidates) or set(source_dossiers) != set(candidates.dossier_id.astype(str)):
        raise ValueError("E2 source dossiers and candidate key are not aligned.")
    candidates = candidates.copy()
    candidates["dossier_id"] = candidates.dossier_id.astype(str)
    mapping = choose_donors(candidates, source_dossiers, config)
    placebo_by_target = {
        row.dossier_id: placebo_price_information(source_dossiers[row.dossier_id], source_dossiers[row.donor_dossier_id])
        for row in mapping.itertuples(index=False)
    }

    built: list[dict[str, Any]] = []
    for dossier_id in sorted(source_dossiers):
        source = source_dossiers[dossier_id]
        payload = {
            "dossier_id": dossier_id,
            "dossier_version": "v3_task12_anonymous",
            "public_information": source["public_information_without_price"],
            "informative_price_information": price_without_summary(source["rich_price_information"]),
            "matched_placebo_price_information": placebo_by_target[dossier_id],
        }
        verify_no_prohibited(payload, f"E2 dossier {dossier_id}")
        built.append(payload)
    checks = build_checks({item["dossier_id"]: item for item in built})
    outcome_key = candidates[["dossier_id", "theta_ex_ante", "next_sue", "formation_date", "next_announcement_date"]].copy()
    if outcome_key.theta_ex_ante.isna().any():
        raise ValueError("E2 public-only evaluation key has a missing held-out target.")

    interim = resolve_repo_path(config["interim_root"])
    output = resolve_repo_path(config["output_root"])
    interim.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    (interim / "step4_v3_task12_dossiers.jsonl").write_text(
        "".join(json.dumps(item, sort_keys=True, allow_nan=False) + "\n" for item in built)
    )
    checks.to_csv(interim / "step4_v3_competence_checks.csv", index=False)
    outcome_key.to_parquet(interim / "step4_v3_outcome_key.parquet", index=False)
    mapping.to_csv(interim / "step4_v3_placebo_mapping_private.csv", index=False)
    public_mapping = mapping[["dossier_id", "same_industry_bucket", "same_size_bucket"]].copy()
    public_mapping.to_csv(output / "step4_v3_placebo_matching_audit.csv", index=False)
    result = {
        "dossier_count": len(built),
        "competence_check_count": len(checks),
        "placebo_donor_count": int(mapping.donor_dossier_id.nunique()),
        "all_donors_pre_cutoff": bool(mapping.donor_formation_date.le(mapping.target_formation_date).all()),
        "prohibited_key_check": "passed",
        "api_calls_made": 0,
    }
    (output / "step4_v3_task12_build_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    print(json.dumps(build(read_config(args.config)), sort_keys=True))


if __name__ == "__main__":
    main()
