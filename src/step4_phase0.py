"""Prepare the non-API assets and audit gates for the E2 LLM experiment.

This command deliberately makes no network or API call.  It selects anonymous
historical dossiers from the point-in-time E1 panel, keeps the actual outcome
only in an ignored local key, and creates the balanced scenario manifest,
prompt contract, cost estimates, and audit artifacts.
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


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "config" / "step4.json"


def read_config(path: Path) -> dict[str, Any]:
    with path.open() as handle:
        return json.load(handle)


def resolve_repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def require_columns(frame: pd.DataFrame, columns: list[str], source: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def sic_bucket(value: object) -> str:
    sic = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(sic):
        return "unclassified"
    code = int(sic)
    if 100 <= code <= 999:
        return "agriculture"
    if 1000 <= code <= 1499:
        return "mining"
    if 1500 <= code <= 1799:
        return "construction"
    if 2000 <= code <= 3999:
        return "manufacturing"
    if 4000 <= code <= 4999:
        return "transport_utilities"
    if 5000 <= code <= 5199:
        return "wholesale"
    if 5200 <= code <= 5999:
        return "retail"
    if 6000 <= code <= 6799:
        return "finance_real_estate"
    if 7000 <= code <= 8999:
        return "services"
    if 9100 <= code <= 9999:
        return "public_other"
    return "unclassified"


def attach_sic(eligible: pd.DataFrame, ccm_path: Path) -> pd.DataFrame:
    """Attach only a broad industry stratum; no company identity is exported."""
    if not ccm_path.exists():
        raise FileNotFoundError(f"Missing CCM link file required for industry strata: {ccm_path}")
    links = pd.read_csv(
        ccm_path,
        usecols=lambda name: str(name).lower() in {"lpermno", "linkdt", "linkenddt", "sic"},
        low_memory=False,
    )
    links.columns = [str(column).lower() for column in links.columns]
    require_columns(links, ["lpermno", "linkdt", "linkenddt", "sic"], "CCM link file")
    links["lpermno"] = pd.to_numeric(links.lpermno, errors="coerce")
    links["linkdt"] = pd.to_datetime(links.linkdt, errors="coerce")
    links["linkenddt"] = pd.to_datetime(links.linkenddt, errors="coerce").fillna(pd.Timestamp("2262-04-11"))
    links = links.dropna(subset=["lpermno", "linkdt"]).copy()
    links["lpermno"] = links.lpermno.astype("int64")

    probe = eligible.reset_index(drop=True).copy()
    probe["_candidate_row"] = np.arange(len(probe), dtype=np.int64)
    matched = probe.merge(links, left_on="permno", right_on="lpermno", how="left", validate="many_to_many")
    matched = matched.loc[
        matched.linkdt.le(matched.formation_date) & matched.formation_date.le(matched.linkenddt)
    ].copy()
    matched = matched.sort_values(["_candidate_row", "linkdt"], ascending=[True, False]).drop_duplicates(
        "_candidate_row", keep="first"
    )
    sic = matched.set_index("_candidate_row").sic.reindex(probe._candidate_row)
    probe["industry_bucket"] = sic.map(sic_bucket).to_numpy()
    return probe.drop(columns="_candidate_row")


def assign_size_bucket(frame: pd.DataFrame) -> pd.Series:
    labels = ["small", "lower_mid", "upper_mid", "large"]

    def bucket(values: pd.Series) -> pd.Series:
        result = pd.Series("unclassified", index=values.index, dtype="string")
        observed = values.dropna()
        if len(observed) >= len(labels):
            result.loc[observed.index] = pd.qcut(
                observed.rank(method="first"), q=len(labels), labels=labels
            ).astype("string")
        return result

    return frame.groupby("formation_date", group_keys=False).log_market_equity_lag1.apply(bucket)


def add_ex_ante_target(panel: pd.DataFrame, minimum_history: int) -> pd.DataFrame:
    """Create a robust outcome scale using only announcements before formation.

    The original I/B/E/S SUE is a valid held-out outcome, but its event-level
    scale is not directly observable at the dossier cutoff.  The experiment
    therefore uses a robust, expanding historical normalization.  Selection
    never uses the realized value of this transformed target.
    """
    required = ["permno", "formation_date", "next_announcement_date", "next_sue"]
    require_columns(panel, required, "E1 forecast panel")
    output = panel.copy()
    output["formation_date"] = pd.to_datetime(output.formation_date, errors="coerce")
    output["next_announcement_date"] = pd.to_datetime(output.next_announcement_date, errors="coerce")
    events = output.loc[
        output.next_announcement_date.notna() & output.next_sue.notna(),
        ["permno", "next_announcement_date", "next_sue"],
    ].copy()
    events["next_sue"] = pd.to_numeric(events.next_sue, errors="coerce")
    events = events.dropna().sort_values(["permno", "next_announcement_date"])
    events = events.drop_duplicates(["permno", "next_announcement_date"], keep="first")
    dates = pd.Series(output.formation_date.dropna().unique()).sort_values()
    moments: list[dict[str, object]] = []
    for date in dates:
        history = events.loc[events.next_announcement_date.lt(date), "next_sue"].to_numpy(dtype=float)
        if len(history) < minimum_history:
            moments.append({
                "formation_date": date,
                "preformation_target_history_n": len(history),
                "theta_center": np.nan,
                "theta_scale": np.nan,
            })
            continue
        center = float(np.median(history))
        scale = float(np.median(np.abs(history - center)) * 1.4826)
        if not math.isfinite(scale) or scale <= 0:
            scale = float(np.std(history, ddof=1))
        moments.append({
            "formation_date": date,
            "preformation_target_history_n": len(history),
            "theta_center": center,
            "theta_scale": scale if math.isfinite(scale) and scale > 0 else np.nan,
        })
    output = output.merge(pd.DataFrame(moments), on="formation_date", how="left", validate="many_to_one")
    clipped = output.next_sue.clip(
        lower=output.theta_center - 8.0 * output.theta_scale,
        upper=output.theta_center + 8.0 * output.theta_scale,
    )
    output["theta_ex_ante"] = (clipped - output.theta_center) / output.theta_scale
    return output


def eligible_rows(panel: pd.DataFrame, config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, int]]:
    selection = config["selection"]
    required = [
        "permno", "cusip", "formation_date", "accounting_available_date", "next_announcement_date",
        "next_sue", "days_to_next_announcement", *selection["required_features"],
    ]
    require_columns(panel, required, "E1 forecast panel")
    frame = add_ex_ante_target(panel, int(config["outcome"]["minimum_preformation_history"]))
    for column in ["formation_date", "accounting_available_date", "next_announcement_date"]:
        frame[column] = pd.to_datetime(frame[column], errors="coerce")
    frame["formation_year"] = frame.formation_date.dt.year
    in_years = frame.formation_year.isin(selection["formation_years"])
    timing_ok = (
        frame.accounting_available_date.le(frame.formation_date)
        & frame.days_to_next_announcement.between(
            selection["minimum_days_to_announcement"], selection["maximum_days_to_announcement"]
        )
        & frame.statement_age_days.le(selection["maximum_statement_age_days"])
    )
    observed_target = frame.theta_ex_ante.notna() & frame.next_announcement_date.notna()
    complete_features = frame[selection["required_features"]].notna().all(axis=1)
    finite_features = np.isfinite(frame[selection["required_features"]].to_numpy(dtype=float)).all(axis=1)
    eligible = frame.loc[in_years & timing_ok & observed_target & complete_features & finite_features].copy()
    diagnostics = {
        "panel_rows": int(len(frame)),
        "rows_in_vintage_window": int(in_years.sum()),
        "rows_with_observed_evaluation_target": int((in_years & observed_target).sum()),
        "rows_passing_timing_rules": int((in_years & timing_ok).sum()),
        "rows_with_complete_public_features": int((in_years & complete_features & finite_features).sum()),
        "rows_with_ex_ante_target_scale": int((in_years & observed_target).sum()),
        "eligible_rows": int(len(eligible)),
    }
    if eligible.empty:
        raise ValueError("No rows satisfy the pre-specified E2 eligibility rules.")
    eligible["size_bucket"] = assign_size_bucket(eligible)
    return eligible, diagnostics


def select_dossiers(eligible: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    selection = config["selection"]
    years = list(selection["formation_years"])
    per_year = int(selection["candidates_per_year"])
    if len(years) * per_year != int(selection["candidate_count"]):
        raise ValueError("candidate_count must equal len(formation_years) × candidates_per_year.")
    frame = eligible.copy()
    frame["_hash"] = [
        stable_hash(f"{selection['selection_seed']}|{permno}|{date.date().isoformat()}")
        for permno, date in zip(frame.permno, frame.formation_date, strict=True)
    ]
    selected: list[pd.Series] = []
    chosen_permnos: set[int] = set()
    global_sector_counts: Counter[str] = Counter()
    for year in years:
        pool = frame.loc[frame.formation_year.eq(year)].copy()
        year_sector_counts: Counter[str] = Counter()
        for _ in range(per_year):
            candidates = pool.loc[~pool.permno.isin(chosen_permnos)].copy()
            if candidates.empty:
                raise ValueError(f"Could not select {per_year} distinct-firm dossiers in {year}.")
            candidates["_year_sector_seen"] = candidates.industry_bucket.map(year_sector_counts).fillna(0).gt(0)
            candidates["_global_sector_count"] = candidates.industry_bucket.map(global_sector_counts).fillna(0)
            row = candidates.sort_values(
                ["_year_sector_seen", "_global_sector_count", "_hash"],
                ascending=[True, True, True],
            ).iloc[0]
            selected.append(row)
            chosen_permnos.add(int(row.permno))
            year_sector_counts[str(row.industry_bucket)] += 1
            global_sector_counts[str(row.industry_bucket)] += 1
    result = pd.DataFrame(selected).reset_index(drop=True)
    result["dossier_id"] = [f"D{number:03d}" for number in range(1, len(result) + 1)]
    result["phase1_selected"] = False
    phase1_years = [years[0], years[len(years) // 2], years[-1]]
    phase1_ids: list[int] = []
    selected_sectors: set[str] = set()
    for year in phase1_years[: int(selection["phase1_dossier_count"])]:
        options = result.loc[result.formation_year.eq(year) & ~result.industry_bucket.isin(selected_sectors)]
        if options.empty:
            options = result.loc[result.formation_year.eq(year)]
        choice = options.sort_values("_hash").index[0]
        phase1_ids.append(int(choice))
        selected_sectors.add(str(result.loc[choice, "industry_bucket"]))
    result.loc[phase1_ids, "phase1_selected"] = True
    if result.phase1_selected.sum() != int(selection["phase1_dossier_count"]):
        raise AssertionError("Phase 1 selection did not contain the requested number of dossiers.")
    return result.drop(columns=["_hash"])


def scalar(value: object, digits: int) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError("Dossier construction encountered a non-finite public value.")
    return round(numeric, digits)


def analyst_bucket(value: object) -> str:
    count = int(float(value))
    if count <= 3:
        return "1_to_3"
    if count <= 7:
        return "4_to_7"
    return "8_or_more"


def dossier_payload(row: pd.Series, config: dict[str, Any]) -> dict[str, Any]:
    dossier = config["dossier"]
    digits = int(dossier["round_digits"])
    return {
        "dossier_id": str(row.dossier_id),
        "dossier_version": dossier["version"],
        "public_information": {
            "industry_bucket": str(row.industry_bucket),
            "company_scale_bucket": str(row.size_bucket),
            "analyst_coverage_bucket": analyst_bucket(row.analyst_coverage),
            "financial_statement_age_months": int(round(float(row.statement_age_days) / 30.0)),
            "accounting_ratios": {
                name: scalar(row[name], digits) for name in dossier["accounting_features"]
            },
            "strictly_lagged_market_history": {
                name: scalar(row[name], digits) for name in dossier["return_features"]
            },
        },
    }


def build_scenarios(dossiers: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    clue = config["private_clue"]
    execution = config["execution"]
    selected = dossiers.loc[dossiers.phase1_selected].copy()
    records: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        for effort in execution["phase1_efforts"]:
            for replicate in range(1, int(execution["phase1_replicates"]) + 1):
                base = {
                    "dossier_id": row.dossier_id,
                    "model": execution["model"],
                    "reasoning_mode": execution["reasoning_mode"],
                    "reasoning_effort": effort,
                    "replicate": replicate,
                    "fresh_session": bool(execution["fresh_session_per_call"]),
                    "allow_tools": bool(execution["allow_tools"]),
                    "store": bool(execution["store"]),
                }
                records.append({
                    **base,
                    "scenario_type": "public_only",
                    "signal_value": np.nan,
                    "signal_noise_sd": np.nan,
                })
                for noise_sd in clue["noise_sds"]:
                    for signal_value in clue["signal_values"]:
                        records.append({
                            **base,
                            "scenario_type": "private_clue",
                            "signal_value": float(signal_value),
                            "signal_noise_sd": float(noise_sd),
                        })
    scenarios = pd.DataFrame(records)
    scenarios["scenario_id"] = [
        "S4_" + hashlib.sha256(
            f"{row.dossier_id}|{row.reasoning_effort}|{row.replicate}|"
            f"{row.scenario_type}|{row.signal_value}|{row.signal_noise_sd}".encode("utf-8")
        ).hexdigest()[:16]
        for row in scenarios.itertuples(index=False)
    ]
    scenarios["run_order"] = scenarios.scenario_id.map(lambda value: stable_hash(f"run-order|{value}"))
    scenarios = scenarios.sort_values("run_order").reset_index(drop=True)
    scenarios["run_order"] = np.arange(1, len(scenarios) + 1, dtype=np.int64)
    expected = (
        int(selected.shape[0])
        * len(execution["phase1_efforts"])
        * int(execution["phase1_replicates"])
        * (1 + len(clue["noise_sds"]) * len(clue["signal_values"]))
    )
    if len(scenarios) != expected or scenarios.scenario_id.duplicated().any():
        raise AssertionError("Phase 1 scenario construction failed its balanced-design check.")
    return scenarios[
        ["scenario_id", "run_order", "dossier_id", "model", "reasoning_mode", "reasoning_effort", "replicate",
         "scenario_type", "signal_value", "signal_noise_sd", "fresh_session", "allow_tools", "store"]
    ]


def build_public_checks(dossiers: pd.DataFrame) -> pd.DataFrame:
    """Known-answer public-information checks for the Phase 1 manipulation audit."""
    rows: list[dict[str, Any]] = []
    for row in dossiers.loc[dossiers.phase1_selected].itertuples(index=False):
        net_debt_less_cash = round(float(row.leverage) - float(row.cash_holdings), 4)
        return_spread = round(float(row.return_12m_lag1) - float(row.return_3m_lag1), 4)
        rows.extend([
            {
                "dossier_id": row.dossier_id,
                "check_id": f"{row.dossier_id}_net_debt_less_cash",
                "check_type": "arithmetic",
                "question": "Using the dossier values, compute leverage minus cash_holdings, rounded to four decimals.",
                "expected_numeric_answer": net_debt_less_cash,
            },
            {
                "dossier_id": row.dossier_id,
                "check_id": f"{row.dossier_id}_return_spread",
                "check_type": "arithmetic",
                "question": "Using the dossier values, compute return_12m_lag1 minus return_3m_lag1, rounded to four decimals.",
                "expected_numeric_answer": return_spread,
            },
        ])
    return pd.DataFrame(rows)


def build_check_scenarios(checks: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    execution = config["execution"]
    rows: list[dict[str, Any]] = []
    for check in checks.itertuples(index=False):
        for effort in execution["phase1_efforts"]:
            for replicate in range(1, int(execution["phase1_replicates"]) + 1):
                rows.append({
                    "check_scenario_id": "C4_" + hashlib.sha256(
                        f"{check.check_id}|{effort}|{replicate}".encode("utf-8")
                    ).hexdigest()[:16],
                    "dossier_id": check.dossier_id,
                    "check_id": check.check_id,
                    "model": execution["model"],
                    "reasoning_mode": execution["reasoning_mode"],
                    "reasoning_effort": effort,
                    "replicate": replicate,
                    "fresh_session": bool(execution["fresh_session_per_call"]),
                    "allow_tools": bool(execution["allow_tools"]),
                    "store": bool(execution["store"]),
                })
    scenarios = pd.DataFrame(rows)
    if scenarios.check_scenario_id.duplicated().any():
        raise AssertionError("Public-information check scenarios are not unique.")
    return scenarios.sort_values("check_scenario_id").reset_index(drop=True)


def cost_table(config: dict[str, Any]) -> pd.DataFrame:
    clue = config["private_clue"]
    execution = config["execution"]
    cost = config["cost_assumptions"]
    conditions_per_dossier = 1 + len(clue["signal_values"]) * len(clue["noise_sds"])
    plans = [
        ("phase1_forecasts", int(config["selection"]["phase1_dossier_count"]), execution["phase1_efforts"], int(execution["phase1_replicates"]), conditions_per_dossier),
        ("phase1_public_checks", int(config["selection"]["phase1_dossier_count"]), execution["phase1_efforts"], int(execution["phase1_replicates"]), 2),
        ("phase1_total", int(config["selection"]["phase1_dossier_count"]), execution["phase1_efforts"], int(execution["phase1_replicates"]), conditions_per_dossier + 2),
        ("phase2_20_dossiers_1_replicate", int(config["selection"]["candidate_count"]), execution["phase2_efforts"], 1, conditions_per_dossier),
        ("phase2_20_dossiers_3_replicates", int(config["selection"]["candidate_count"]), execution["phase2_efforts"], 3, conditions_per_dossier),
    ]
    rows: list[dict[str, Any]] = []
    for plan, n_dossiers, efforts, replicates, calls_per_dossier_effort in plans:
        for effort in efforts:
            calls = n_dossiers * replicates * calls_per_dossier_effort
            input_tokens = calls * int(cost["input_tokens_per_call"])
            output_tokens = calls * int(cost["expected_output_tokens_by_effort"][effort])
            max_output_tokens = calls * int(execution["max_output_tokens"])
            rows.append({
                "plan": plan,
                "effort": effort,
                "dossiers": n_dossiers,
                "replicates": replicates,
                "conditions_per_dossier_effort": calls_per_dossier_effort,
                "calls": calls,
                "estimated_input_tokens": input_tokens,
                "estimated_output_tokens": output_tokens,
                "estimated_cost_usd": round(
                    input_tokens / 1_000_000 * float(cost["input_usd_per_million_tokens"])
                    + output_tokens / 1_000_000 * float(cost["output_usd_per_million_tokens"]), 4
                ),
                "max_output_token_cost_usd": round(
                    input_tokens / 1_000_000 * float(cost["input_usd_per_million_tokens"])
                    + max_output_tokens / 1_000_000 * float(cost["output_usd_per_million_tokens"]), 4
                ),
            })
    detail = pd.DataFrame(rows)
    totals = detail.groupby("plan", as_index=False).agg(
        effort=("effort", lambda value: "all"),
        dossiers=("dossiers", "first"),
        replicates=("replicates", "first"),
        conditions_per_dossier_effort=("conditions_per_dossier_effort", "first"),
        calls=("calls", "sum"),
        estimated_input_tokens=("estimated_input_tokens", "sum"),
        estimated_output_tokens=("estimated_output_tokens", "sum"),
        estimated_cost_usd=("estimated_cost_usd", "sum"),
        max_output_token_cost_usd=("max_output_token_cost_usd", "sum"),
    )
    return pd.concat([detail, totals], ignore_index=True).sort_values(["plan", "effort"]).reset_index(drop=True)


def write_markdown_report(
    path: Path, diagnostics: dict[str, int], dossiers: pd.DataFrame, scenarios: pd.DataFrame, costs: pd.DataFrame,
    config: dict[str, Any],
) -> None:
    total = costs.loc[costs.effort.eq("all")].set_index("plan")
    phase1 = total.loc["phase1_total"]
    lines = [
        "# E2 dossier-construction audit",
        "",
        "This audit created no API request and retrieved no SEC filing. It prepares anonymous tabular dossiers from the point-in-time E1 panel.",
        "",
        "## Data and timing gates",
        "",
        f"- Eligible rows under the pre-specified rules: {diagnostics['eligible_rows']:,}.",
        f"- Selected dossiers: {len(dossiers)}; Phase 1 dossiers: {int(dossiers.phase1_selected.sum())}.",
        f"- All selected accounting availability dates precede or equal formation: {bool(dossiers.accounting_available_date.le(dossiers.formation_date).all())}.",
        f"- All selected announcements are {config['selection']['minimum_days_to_announcement']}–{config['selection']['maximum_days_to_announcement']} days after formation.",
        "- The scored target is an expanding, robust normalization of I/B/E/S SUE using only announcements before formation; its realized value is excluded from the dossier.",
        "- Dossiers exclude issuer identifiers, dates, current-price fields, and realized target fields. The local candidate key retains those fields only for later evaluation.",
        "",
        "## Phase 1 design",
        "",
        f"- {int(dossiers.phase1_selected.sum())} dossiers × {len(config['execution']['phase1_efforts'])} effort conditions × {config['execution']['phase1_replicates']} independent repetitions.",
        f"- Conditions per dossier-effort: 1 public-only control + {len(config['private_clue']['signal_values'])} signal values × {len(config['private_clue']['noise_sds'])} noise SDs = 7.",
        f"- Forecast manifest: {len(scenarios)} calls; public-information checks add 36 calls. All calls use fresh sessions and no tools.",
        "",
        "## Cost gate",
        "",
        f"- Phase 1 expected cost: ${phase1.estimated_cost_usd:.2f}; planning cap at `max_output_tokens`: ${phase1.max_output_token_cost_usd:.2f}.",
        f"- Cost assumptions use {config['execution']['model']} standard token rates checked on {config['cost_assumptions']['pricing_checked_on']}; actual reasoning-token usage is unknown until the first approved calls.",
        "",
        "## Go / no-go before Phase 2",
        "",
        "Proceed only if every Phase 1 response validates against the JSON schema, forecast SD is finite and positive, all calls record usage, and the cost remains below the approved cap.",
        "",
        "Interpret a lower private-clue slope as informational substitution only if the high-effort condition also passes the public-information arithmetic checks, responds monotonically to clue sign, and gives a weaker update for the higher-noise clue. The experiment's primary outcome is the finite-difference forecast-mean slope; the CARA-derived demand index is secondary.",
    ]
    path.write_text("\n".join(lines) + "\n")


def validate_artifacts(interim_root: Path, config: dict[str, Any]) -> dict[str, object]:
    candidate_key = interim_root / "step4_candidate_key.parquet"
    dossiers_path = interim_root / "step4_dossiers.jsonl"
    scenarios_path = interim_root / "scenario_manifest.csv"
    checks_path = interim_root / "public_information_checks.csv"
    check_scenarios_path = interim_root / "public_information_check_manifest.csv"
    for path in [candidate_key, dossiers_path, scenarios_path, checks_path, check_scenarios_path]:
        if not path.exists():
            raise FileNotFoundError(f"Missing E2 preparation artifact: {path}")
    candidates = pd.read_parquet(candidate_key)
    scenarios = pd.read_csv(scenarios_path)
    check_scenarios = pd.read_csv(check_scenarios_path)
    dossiers = [json.loads(line) for line in dossiers_path.read_text().splitlines() if line.strip()]
    selection = config["selection"]
    clue = config["private_clue"]
    execution = config["execution"]
    expected = (
        int(selection["phase1_dossier_count"])
        * len(execution["phase1_efforts"])
        * int(execution["phase1_replicates"])
        * (1 + len(clue["signal_values"]) * len(clue["noise_sds"]))
    )
    if len(candidates) != int(selection["candidate_count"]):
        raise AssertionError("Unexpected candidate count.")
    if candidates.permno.duplicated().any():
        raise AssertionError("A firm appears in more than one selected dossier.")
    if len(dossiers) != len(candidates):
        raise AssertionError("Dossier count does not equal candidate count.")
    if len(scenarios) != expected or scenarios.scenario_id.duplicated().any():
        raise AssertionError("Scenario manifest is not the intended balanced design.")
    expected_checks = (
        int(selection["phase1_dossier_count"])
        * 2
        * len(execution["phase1_efforts"])
        * int(execution["phase1_replicates"])
    )
    if len(check_scenarios) != expected_checks or check_scenarios.check_scenario_id.duplicated().any():
        raise AssertionError("Public-information check manifest is not the intended balanced design.")
    rendered = "\n".join(json.dumps(value, sort_keys=True) for value in dossiers)
    prohibited = ["permno", "cusip", "formation_date", "next_sue", "next_actual_eps", "stock_price"]
    leaks = [name for name in prohibited if f'"{name}"' in rendered]
    if leaks:
        raise AssertionError(f"Dossier payload leaked prohibited fields: {leaks}")
    return {
        "candidate_count": len(candidates),
        "phase1_forecast_scenarios": len(scenarios),
        "phase1_public_check_scenarios": len(check_scenarios),
        "phase1_total_api_calls": len(scenarios) + len(check_scenarios),
        "dossier_payload_leak_check": "passed",
        "api_calls_made": 0,
    }


def build(config: dict[str, Any]) -> dict[str, object]:
    panel_path = resolve_repo_path(config["panel_path"])
    if not panel_path.exists():
        raise FileNotFoundError(f"Missing E1 forecast panel: {panel_path}")
    panel = pd.read_parquet(panel_path)
    eligible, diagnostics = eligible_rows(panel, config)
    eligible = attach_sic(eligible, Path(config["ccm_link_path"]))
    dossiers = select_dossiers(eligible, config)
    payloads = [dossier_payload(row, config) for _, row in dossiers.iterrows()]

    interim_root = resolve_repo_path(config["interim_root"])
    output_root = resolve_repo_path(config["output_root"])
    interim_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    candidate_key_columns = [
        "dossier_id", "phase1_selected", "permno", "cusip", "formation_date", "accounting_available_date",
        "next_announcement_date", "next_sue", "theta_ex_ante", "theta_center", "theta_scale",
        "preformation_target_history_n", "days_to_next_announcement", "formation_year", "industry_bucket",
        "size_bucket", *config["selection"]["required_features"],
    ]
    dossiers[candidate_key_columns].to_parquet(interim_root / "step4_candidate_key.parquet", index=False)
    (interim_root / "step4_dossiers.jsonl").write_text(
        "".join(json.dumps(payload, allow_nan=False, sort_keys=True) + "\n" for payload in payloads)
    )
    scenarios = build_scenarios(dossiers, config)
    scenarios.to_csv(interim_root / "scenario_manifest.csv", index=False)
    checks = build_public_checks(dossiers)
    checks.to_csv(interim_root / "public_information_checks.csv", index=False)
    check_scenarios = build_check_scenarios(checks, config)
    check_scenarios.to_csv(interim_root / "public_information_check_manifest.csv", index=False)
    costs = cost_table(config)
    costs.to_csv(output_root / "phase0_cost_estimate.csv", index=False)

    public_manifest = dossiers[
        ["dossier_id", "phase1_selected", "formation_year", "industry_bucket", "size_bucket", "days_to_next_announcement"]
    ].copy()
    public_manifest.to_csv(output_root / "phase0_dossier_manifest.csv", index=False)
    pd.DataFrame([{"metric": name, "value": value} for name, value in diagnostics.items()]).to_csv(
        output_root / "phase0_data_audit.csv", index=False
    )
    pd.DataFrame([
        {"feature": name, "included_in_dossier": True, "group": "accounting_ratio"}
        for name in config["dossier"]["accounting_features"]
    ] + [
        {"feature": name, "included_in_dossier": True, "group": "strictly_lagged_market_history"}
        for name in config["dossier"]["return_features"]
    ] + [
        {"feature": name, "included_in_dossier": False, "group": "excluded_or_evaluation_only"}
        for name in config["dossier"]["excluded_columns"]
    ]).to_csv(output_root / "phase0_dossier_feature_manifest.csv", index=False)
    write_markdown_report(output_root / "phase0_gate_report.md", diagnostics, dossiers, scenarios, costs, config)
    result = validate_artifacts(interim_root, config)
    pd.DataFrame([result]).to_csv(output_root / "phase0_validation.csv", index=False)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "check"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    config = read_config(args.config)
    interim_root = resolve_repo_path(config["interim_root"])
    if args.command == "build":
        result = build(config)
    else:
        result = validate_artifacts(interim_root, config)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
