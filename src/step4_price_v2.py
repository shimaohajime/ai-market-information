"""Build anonymous, time-safe rich-price dossiers for the Step 4 v2 pilot.

This leaves the completed v1 dossier and its results untouched.  The output
contains only rebased return paths and derived statistics, never price levels,
dates, tickers, PERMNOs, or held-out earnings outcomes.
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


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_price_v2.json"
DAILY_COLUMNS = ["PERMNO", "DlyCalDt", "DlyRet", "DlyPrc", "vwretd"]
PROHIBITED_KEYS = {"permno", "cusip", "ticker", "formation_date", "next_sue", "theta_ex_ante", "stock_price"}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def scalar(value: float, digits: int) -> float:
    if not math.isfinite(float(value)):
        raise ValueError("Price-history construction encountered a non-finite value.")
    return round(float(value), digits)


def compound(values: pd.Series) -> float:
    return float((1.0 + values.astype(float)).prod() - 1.0)


def drawdown(price_index: pd.Series) -> float:
    return float((price_index / price_index.cummax() - 1.0).min())


def extract_daily_history(candidates: pd.DataFrame, source: Path) -> tuple[dict[int, pd.DataFrame], dict[int, int]]:
    """Read selected pre-cutoff observations, dropping only exact duplicate rows.

    CRSP can contain a repeated identical same-PERMNO/day record.  Such rows
    add no information and are collapsed with an audit count.  Any same-day
    duplicate whose displayed price or returns differ remains a hard error.
    """
    cutoffs = candidates.set_index("permno").formation_date.to_dict()
    selected_permnos = set(cutoffs)
    histories: dict[int, list[pd.DataFrame]] = {int(value): [] for value in selected_permnos}
    for chunk in pd.read_csv(
        source,
        usecols=DAILY_COLUMNS,
        parse_dates=["DlyCalDt"],
        chunksize=500_000,
        low_memory=False,
    ):
        chunk = chunk.rename(columns={"PERMNO": "permno", "DlyCalDt": "date", "DlyRet": "stock_return", "vwretd": "market_return"})
        chunk = chunk.loc[chunk.permno.isin(selected_permnos)].copy()
        if chunk.empty:
            continue
        chunk["cutoff"] = chunk.permno.map(cutoffs)
        chunk = chunk.loc[
            chunk.date.lt(chunk.cutoff)
            & chunk.stock_return.notna()
            & chunk.market_return.notna()
            & chunk.DlyPrc.notna()
        ]
        for permno, group in chunk.groupby("permno", sort=False):
            histories[int(permno)].append(group.drop(columns="cutoff"))
    result: dict[int, pd.DataFrame] = {}
    exact_duplicate_rows_removed: dict[int, int] = {}
    for permno, pieces in histories.items():
        if not pieces:
            raise ValueError(f"No eligible CRSP daily history for selected PERMNO {permno}.")
        history = pd.concat(pieces, ignore_index=True).sort_values("date")
        duplicate = history.loc[history.duplicated(["permno", "date"], keep=False)].copy()
        if not duplicate.empty:
            value_columns = ["stock_return", "market_return", "DlyPrc"]
            inconsistent = duplicate.groupby(["permno", "date"])[value_columns].nunique(dropna=False).gt(1).any(axis=1)
            if inconsistent.any():
                raise ValueError(f"Non-identical CRSP daily duplicates for selected PERMNO {permno}.")
            exact_duplicate_rows_removed[permno] = int(len(duplicate) - duplicate.drop_duplicates(["permno", "date"]).shape[0])
            history = history.drop_duplicates(["permno", "date"], keep="first")
        else:
            exact_duplicate_rows_removed[permno] = 0
        result[permno] = history.reset_index(drop=True)
    return result, exact_duplicate_rows_removed


def price_payload(history: pd.DataFrame, settings: dict[str, Any]) -> tuple[dict[str, Any], dict[str, float]]:
    digits = int(settings["round_digits"])
    recent_days = int(settings["daily_recent_days"])
    weeks = int(settings["weekly_history_weeks"])
    days_per_week = int(settings["trading_days_per_week"])
    required = max(int(settings["minimum_daily_observations"]), weeks * days_per_week)
    if len(history) < required:
        raise ValueError(f"Expected at least {required} daily observations; found {len(history)}.")
    window = history.tail(required).copy().reset_index(drop=True)
    gross = (1.0 + window.stock_return.astype(float)).cumprod()
    window["price_index"] = 100.0 * gross / gross.iloc[0]
    window["abnormal_return"] = window.stock_return.astype(float) - window.market_return.astype(float)

    weekly: list[dict[str, float | int]] = []
    for number in range(weeks):
        week = window.iloc[number * days_per_week:(number + 1) * days_per_week]
        weekly.append({
            "period_index": number - weeks,
            "stock_return": scalar(compound(week.stock_return), digits),
            "market_return": scalar(compound(week.market_return), digits),
            "abnormal_return": scalar(compound(week.stock_return) - compound(week.market_return), digits),
            "rebased_price_index": scalar(float(week.price_index.iloc[-1]), digits),
        })

    recent = window.tail(recent_days).reset_index(drop=True)
    daily_recent = [{
        "trading_day_index": number - recent_days,
        "stock_return": scalar(float(row.stock_return), digits),
        "market_return": scalar(float(row.market_return), digits),
        "abnormal_return": scalar(float(row.abnormal_return), digits),
        "rebased_price_index": scalar(float(row.price_index), digits),
    } for number, row in enumerate(recent.itertuples(index=False))]

    summary: dict[str, float] = {}
    for length in (20, 60, 120, required):
        sample = window.tail(length)
        summary[f"stock_return_{length}d"] = scalar(compound(sample.stock_return), digits)
        summary[f"market_return_{length}d"] = scalar(compound(sample.market_return), digits)
        summary[f"abnormal_return_{length}d"] = scalar(compound(sample.stock_return) - compound(sample.market_return), digits)
        summary[f"stock_volatility_{length}d"] = scalar(float(sample.stock_return.std(ddof=1)), digits)
    summary[f"maximum_drawdown_{required}d"] = scalar(drawdown(window.price_index), digits)
    payload = {
        "price_information_version": "v2_daily_path_anonymous",
        "timing": "All price-derived values end on the last CRSP trading day strictly before the information cutoff.",
        "price_index_definition": "A split-adjusted return-implied index rebased to 100 at the start of the 260-trading-day history. It is not a dollar price.",
        "price_summary": summary,
        "weekly_history": weekly,
        "recent_daily_history": daily_recent,
    }
    checks = {
        # Compute checks from the already rounded rows supplied to the model,
        # rather than the higher-precision CRSP inputs used to construct them.
        "last_20d_stock_return": scalar(
            compound(pd.Series([value["stock_return"] for value in daily_recent[-20:]])), digits
        ),
        "last_12w_abnormal_return": scalar(
            compound(pd.Series([value["stock_return"] for value in weekly[-12:]]))
            - compound(pd.Series([value["market_return"] for value in weekly[-12:]])),
            digits,
        ),
    }
    return payload, checks


def build(config: dict[str, Any]) -> dict[str, Any]:
    base_root = resolve_repo_path(config["base_interim_root"])
    output_root = resolve_repo_path(config["interim_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    candidates = pd.read_parquet(base_root / "step4_candidate_key.parquet")
    selection_mode = config.get("selection", {}).get("mode", "phase1_selected")
    if selection_mode == "phase1_selected":
        selected = candidates.loc[candidates.phase1_selected].copy()
    elif selection_mode == "all_phase0_candidates":
        # These are the full candidate set constructed before any LLM outcome
        # is observed.  This enables a cross-sectional expansion without
        # selecting on the pilot's realised model responses.
        selected = candidates.copy()
    else:
        raise ValueError(
            "selection.mode must be 'phase1_selected' or "
            f"'all_phase0_candidates', not {selection_mode!r}."
        )
    if selected.empty:
        raise ValueError("The selected rich-price-dossier sample is empty.")
    if selected.dossier_id.duplicated().any() or selected.permno.duplicated().any():
        raise ValueError("The selected rich-price-dossier sample is not one-to-one by dossier and PERMNO.")
    selected["formation_date"] = pd.to_datetime(selected.formation_date)
    base_dossiers = {value["dossier_id"]: value for value in read_jsonl(base_root / "step4_dossiers.jsonl")}
    histories, exact_duplicate_rows_removed = extract_daily_history(selected, Path(config["daily_crsp_path"]))

    dossiers: list[dict[str, Any]] = []
    checks: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        base = base_dossiers[row.dossier_id]
        public_information = dict(base["public_information"])
        summary_price_information = public_information.pop("strictly_lagged_market_history")
        rich_price_information, check_values = price_payload(histories[int(row.permno)], config["history"])
        dossier = {
            "dossier_id": str(row.dossier_id),
            "dossier_version": "v2_price_information",
            "public_information_without_price": public_information,
            "summary_price_information": summary_price_information,
            "rich_price_information": rich_price_information,
        }
        rendered = json.dumps(dossier, sort_keys=True)
        leaked = sorted(key for key in PROHIBITED_KEYS if f'"{key}"' in rendered)
        if leaked:
            raise AssertionError(f"Rich-price dossier leaked prohibited keys: {leaked}")
        dossiers.append(dossier)
        checks.extend([
            {
                "dossier_id": str(row.dossier_id),
                "check_id": f"{row.dossier_id}_last_20d_stock_return",
                "check_type": "daily_path_compounding",
                "question": "Using only recent_daily_history, compound the stock_return values for the final 20 trading-day rows and return the result rounded to four decimals.",
                "expected_numeric_answer": check_values["last_20d_stock_return"],
            },
            {
                "dossier_id": str(row.dossier_id),
                "check_id": f"{row.dossier_id}_last_12w_abnormal_return",
                "check_type": "weekly_path_compounding",
                "question": "Using only weekly_history, compound stock_return and market_return over the final 12 rows, then return stock cumulative return minus market cumulative return rounded to four decimals.",
                "expected_numeric_answer": check_values["last_12w_abnormal_return"],
            },
        ])
        history = histories[int(row.permno)]
        audit_rows.append({
            "dossier_id": str(row.dossier_id),
            "daily_observations_before_cutoff": int(len(history)),
            "last_daily_observation_strictly_before_cutoff": str(history.date.max().date()),
            "last_260_daily_observations_complete": True,
            "exact_duplicate_rows_removed": int(exact_duplicate_rows_removed[int(row.permno)]),
        })

    (output_root / "step4_price_v2_dossiers.jsonl").write_text(
        "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in dossiers)
    )
    pd.DataFrame(checks).to_csv(output_root / "step4_price_v2_checks.csv", index=False)
    pd.DataFrame(audit_rows).to_csv(output_root / "step4_price_v2_build_audit.csv", index=False)
    result = {
        "dossier_count": len(dossiers),
        "selection_mode": selection_mode,
        "check_count": len(checks),
        "exact_duplicate_rows_removed_total": int(sum(exact_duplicate_rows_removed.values())),
        "affected_permnos_with_exact_duplicates": int(sum(value > 0 for value in exact_duplicate_rows_removed.values())),
        "prohibited_key_check": "passed",
        "cutoff_rule": config["history"]["cutoff_rule"],
    }
    (output_root / "step4_price_v2_build_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_config(args.config)
    if args.command == "build":
        print(json.dumps(build(config)))


if __name__ == "__main__":
    main()
