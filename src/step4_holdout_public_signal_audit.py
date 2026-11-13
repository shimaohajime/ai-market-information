"""Pre-API researcher-side audit for the E2 real-data price treatment.

The audit uses no LLM or external API. It asks a deliberately narrow question:
do target-firm pre-cutoff price features improve a fixed, non-LLM next-SUE
forecast relative to an accounting-only forecast and to two mechanically
constructed price placebos? Its result is an expenditure gate, not an
after-the-fact model-performance claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from step4_phase0 import (
    REPO_ROOT,
    analyst_bucket,
    attach_sic,
    dossier_payload,
    eligible_rows,
    read_config,
    resolve_repo_path,
)
from step4_price_v2 import extract_daily_history, price_payload


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_holdout_public_signal_audit_v1.json"
ACCOUNTING_FIELDS = [
    "operating_profitability",
    "roa",
    "asset_growth",
    "leverage",
    "cash_holdings",
    "accruals",
    "sales_growth",
    "tangibility",
]
PUBLIC_NUMERIC_FIELDS = ["financial_statement_age_months", *ACCOUNTING_FIELDS]
PUBLIC_CATEGORICAL_FIELDS = [
    "industry_bucket",
    "company_scale_bucket",
    "analyst_coverage_bucket",
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def stable_hash(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest()[:16], 16)


def training_config(base: dict[str, Any], audit: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    years = [int(value) for value in audit["training"]["formation_years"]]
    result["selection"]["formation_years"] = years
    result["selection"]["candidate_count"] = len(years) * int(audit["training"]["candidates_per_year"])
    result["selection"]["candidates_per_year"] = int(audit["training"]["candidates_per_year"])
    result["selection"]["selection_seed"] = str(audit["training"]["selection_seed"])
    return result


def select_training(eligible: pd.DataFrame, audit: dict[str, Any]) -> pd.DataFrame:
    training = audit["training"]
    years = [int(value) for value in training["formation_years"]]
    per_year = int(training["candidates_per_year"])
    frame = eligible.copy()
    frame["_hash"] = [
        stable_hash(f"{training['selection_seed']}|{int(permno)}|{date.date().isoformat()}")
        for permno, date in zip(frame.permno, frame.formation_date, strict=True)
    ]
    selected: list[pd.Series] = []
    used_permnos: set[int] = set()
    for year in years:
        pool = frame.loc[frame.formation_year.eq(year)].copy()
        candidates = (
            pool.loc[~pool.permno.isin(used_permnos)]
            .sort_values("_hash")
            .drop_duplicates("permno", keep="first")
        )
        if len(candidates) < per_year:
            raise ValueError(f"Insufficient distinct training firms in {year}.")
        chosen = candidates.head(per_year)
        selected.extend(chosen.iloc[index] for index in range(len(chosen)))
        used_permnos.update(int(value) for value in chosen.permno)
    result = pd.DataFrame(selected).reset_index(drop=True).drop(columns="_hash")
    if len(result) != len(years) * per_year or result.permno.duplicated().any():
        raise AssertionError("Training selection did not satisfy fixed one-observation-per-firm design.")
    return result


def public_features(public_information: dict[str, Any]) -> dict[str, Any]:
    accounting = public_information["accounting_ratios"]
    return {
        "financial_statement_age_months": float(public_information["financial_statement_age_months"]),
        **{field: float(accounting[field]) for field in ACCOUNTING_FIELDS},
        "industry_bucket": str(public_information["industry_bucket"]),
        "company_scale_bucket": str(public_information["company_scale_bucket"]),
        "analyst_coverage_bucket": str(public_information["analyst_coverage_bucket"]),
    }


def compound(values: list[float]) -> float:
    return float(np.prod(1.0 + np.asarray(values, dtype=float)) - 1.0)


def volatility(values: list[float]) -> float:
    if len(values) < 2:
        raise ValueError("Volatility needs at least two observations.")
    return float(np.std(np.asarray(values, dtype=float), ddof=1))


def price_features(price: dict[str, Any], settings: dict[str, Any]) -> dict[str, float]:
    """Use the identical anonymous path format shown to a model.

    Windows up to 60 days use the displayed daily rows. Longer windows use
    displayed weekly rows. This keeps the gate tied to information that is
    actually present in the real-data treatment.
    """
    daily = price["recent_daily_history"]
    weekly = price["weekly_history"]
    if len(daily) != 60 or len(weekly) != 52:
        raise ValueError("Price-feature audit expects 60 daily and 52 weekly observations.")
    windows = [int(value) for value in settings["return_windows_days"]]
    result: dict[str, float] = {}
    for window in windows:
        if window <= len(daily):
            rows = daily[-window:]
        else:
            weeks = window // 5
            if weeks * 5 != window or weeks > len(weekly):
                raise ValueError(f"Unsupported price window {window}.")
            rows = weekly[-weeks:]
        stock = [float(row["stock_return"]) for row in rows]
        market = [float(row["market_return"]) for row in rows]
        abnormal = [float(row["abnormal_return"]) for row in rows]
        result[f"stock_return_{window}d"] = compound(stock)
        result[f"market_return_{window}d"] = compound(market)
        result[f"abnormal_return_{window}d"] = compound(stock) - compound(market)
        result[f"stock_volatility_{window}d"] = volatility(stock)
        result[f"abnormal_volatility_{window}d"] = volatility(abnormal)
    recent_abnormal = [float(row["abnormal_return"]) for row in daily[-20:]]
    earlier_abnormal = [float(row["abnormal_return"]) for row in daily[-40:-20]]
    all_abnormal = np.asarray([float(row["abnormal_return"]) for row in daily], dtype=float)
    result["abnormal_momentum_20d_minus_prior20d"] = compound(recent_abnormal) - compound(earlier_abnormal)
    result["abnormal_return_autocorrelation_60d"] = (
        float(np.corrcoef(all_abnormal[:-1], all_abnormal[1:])[0, 1])
        if np.std(all_abnormal[:-1]) > 0 and np.std(all_abnormal[1:]) > 0 else 0.0
    )
    rebased = np.asarray([float(row["rebased_price_index"]) for row in weekly], dtype=float)
    result["maximum_drawdown_260d"] = float(np.min(rebased / np.maximum.accumulate(rebased) - 1.0))
    if not all(math.isfinite(value) for value in result.values()):
        raise ValueError("Price-feature extraction produced a non-finite value.")
    return result


def training_public_payload(row: pd.Series, base_config: dict[str, Any]) -> dict[str, Any]:
    work = row.copy()
    work["dossier_id"] = f"TRAIN{int(work.name):05d}"
    return dossier_payload(work, base_config)["public_information"]


def build_training_frame(
    selected: pd.DataFrame, base_config: dict[str, Any], audit: dict[str, Any],
) -> pd.DataFrame:
    histories, _ = extract_daily_history(selected[["permno", "formation_date"]], Path(audit["daily_crsp_path"]))
    rows: list[dict[str, Any]] = []
    for index, row in selected.iterrows():
        price, _ = price_payload(histories[int(row.permno)], audit["history"])
        values = public_features(training_public_payload(row, base_config))
        values.update(price_features(price, audit["features"]))
        values.update({
            "sample_id": f"T{index + 1:04d}",
            "formation_year": int(row.formation_year),
            "theta_ex_ante": float(row.theta_ex_ante),
        })
        rows.append(values)
    return pd.DataFrame(rows)


def build_holdout_frames(config: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    root = resolve_repo_path(config["holdout_final_root"])
    dossiers = {str(item["dossier_id"]): item for item in read_jsonl(root / config["holdout_dossier_filename"])}
    outcome = pd.read_parquet(root / config["holdout_outcome_filename"])
    outcome["dossier_id"] = outcome.dossier_id.astype(str)
    if set(dossiers) != set(outcome.dossier_id):
        raise ValueError("Holdout dossiers and outcomes are not aligned.")
    public_rows: list[dict[str, Any]] = []
    by_condition: dict[str, list[dict[str, Any]]] = {
        "informative_price_path": [],
        "matched_placebo_path": [],
        "time_permuted_price_path": [],
    }
    field_map = {
        "informative_price_path": "informative_price_information",
        "matched_placebo_path": "matched_placebo_price_information",
        "time_permuted_price_path": "time_permuted_price_information",
    }
    for dossier_id, dossier in sorted(dossiers.items()):
        public = public_features(dossier["public_information"])
        public["dossier_id"] = dossier_id
        public_rows.append(public)
        for condition, field in field_map.items():
            values = dict(public)
            values.update(price_features(dossier[field], config["features"]))
            by_condition[condition].append(values)
    public_frame = pd.DataFrame(public_rows).merge(
        outcome[["dossier_id", "theta_ex_ante"]], on="dossier_id", validate="one_to_one"
    )
    condition_frames = {
        condition: pd.DataFrame(rows).merge(
            outcome[["dossier_id", "theta_ex_ante"]], on="dossier_id", validate="one_to_one"
        )
        for condition, rows in by_condition.items()
    }
    return public_frame, condition_frames


def model_pipeline(numeric: list[str], categorical: list[str], alpha: float) -> Pipeline:
    preprocessor = ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    return Pipeline([("preprocessor", preprocessor), ("ridge", Ridge(alpha=float(alpha)))])


def choose_alpha(training: pd.DataFrame, numeric: list[str], categorical: list[str], alphas: list[float]) -> float:
    years = sorted(int(value) for value in training.formation_year.unique())
    fold_losses: dict[float, list[float]] = {float(alpha): [] for alpha in alphas}
    for validation_year in years[1:]:
        fit = training.loc[training.formation_year.lt(validation_year)]
        validate = training.loc[training.formation_year.eq(validation_year)]
        if fit.empty or validate.empty:
            continue
        for alpha in alphas:
            predictor = model_pipeline(numeric, categorical, float(alpha))
            predictor.fit(fit[numeric + categorical], fit.theta_ex_ante)
            error = predictor.predict(validate[numeric + categorical]) - validate.theta_ex_ante.to_numpy(dtype=float)
            fold_losses[float(alpha)].append(float(np.mean(error**2)))
    average = {
        alpha: float(np.mean(losses)) for alpha, losses in fold_losses.items() if losses
    }
    if not average:
        raise ValueError("No rolling training folds were available for ridge tuning.")
    return min(average, key=lambda alpha: (average[alpha], alpha))


def bootstrap_contrast(values: pd.Series, repetitions: int, seed: int) -> dict[str, Any]:
    sample = values.to_numpy(dtype=float)
    if not np.isfinite(sample).all() or len(sample) == 0:
        raise ValueError("Contrast contains missing or non-finite values.")
    rng = np.random.default_rng(seed)
    draws = sample[rng.integers(0, len(sample), size=(repetitions, len(sample)))].mean(axis=1)
    return {
        "dossiers": int(len(sample)),
        "mean": float(sample.mean()),
        "ci_95_lower": float(np.quantile(draws, 0.025)),
        "ci_95_upper": float(np.quantile(draws, 0.975)),
        "negative_dossiers": int((sample < 0).sum()),
    }


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    output = frame.copy()
    for column in output.columns:
        if pd.api.types.is_float_dtype(output[column]):
            output[column] = output[column].map(lambda value: f"{value:.4f}")
    lines = [
        "| " + " | ".join(str(column) for column in output.columns) + " |",
        "| " + " | ".join("---" for _ in output.columns) + " |",
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in output.itertuples(index=False, name=None))
    return "\n".join(lines)


def plan(config: dict[str, Any]) -> dict[str, Any]:
    years = [int(value) for value in config["training"]["formation_years"]]
    return {
        "api_calls_made": 0,
        "training_years": years,
        "training_dossiers": len(years) * int(config["training"]["candidates_per_year"]),
        "holdout_dossiers_expected": 50,
        "ridge_alpha_grid": list(config["features"]["ridge_alpha_grid"]),
        "gate_contrasts": [
            "genuine_minus_no_price_squared_error",
            "genuine_minus_matched_placebo_squared_error",
            "genuine_minus_time_permuted_placebo_squared_error",
        ],
    }


def run(config: dict[str, Any]) -> dict[str, Any]:
    base_config = read_config(resolve_repo_path(config["holdout_selection_config"]))
    train_config = training_config(base_config, config)
    panel = pd.read_parquet(resolve_repo_path(base_config["panel_path"]))
    eligible, _ = eligible_rows(panel, train_config)
    eligible = attach_sic(eligible, resolve_repo_path(base_config["ccm_link_path"]))
    selected = select_training(eligible, config)
    training = build_training_frame(selected, base_config, config)
    holdout_public, holdout_conditions = build_holdout_frames(config)
    price_numeric = sorted(column for column in training.columns if column.startswith((
        "stock_return_", "market_return_", "abnormal_return_", "stock_volatility_", "abnormal_volatility_",
    )) or column in {
        "abnormal_momentum_20d_minus_prior20d",
        "abnormal_return_autocorrelation_60d",
        "maximum_drawdown_260d",
    })
    base_numeric = list(PUBLIC_NUMERIC_FIELDS)
    categorical = list(PUBLIC_CATEGORICAL_FIELDS)
    missing = set(base_numeric + categorical + price_numeric).difference(training.columns)
    if missing:
        raise ValueError(f"Training feature frame is missing fields: {sorted(missing)}")
    alphas = [float(value) for value in config["features"]["ridge_alpha_grid"]]
    base_alpha = choose_alpha(training, base_numeric, categorical, alphas)
    price_alpha = choose_alpha(training, [*base_numeric, *price_numeric], categorical, alphas)
    base_model = model_pipeline(base_numeric, categorical, base_alpha)
    price_model = model_pipeline([*base_numeric, *price_numeric], categorical, price_alpha)
    base_model.fit(training[base_numeric + categorical], training.theta_ex_ante)
    price_model.fit(training[base_numeric + price_numeric + categorical], training.theta_ex_ante)
    target = holdout_public.theta_ex_ante.to_numpy(dtype=float)
    predictions: dict[str, np.ndarray] = {
        "no_price": base_model.predict(holdout_public[base_numeric + categorical]),
    }
    for condition, frame in holdout_conditions.items():
        predictions[condition] = price_model.predict(frame[base_numeric + price_numeric + categorical])
    cells = pd.DataFrame({
        "dossier_id": holdout_public.dossier_id,
        "theta_ex_ante": target,
        **{f"forecast_{condition}": values for condition, values in predictions.items()},
    })
    for condition, values in predictions.items():
        cells[f"squared_error_{condition}"] = (values - target)**2
    comparisons = [
        ("genuine_minus_no_price_squared_error", "informative_price_path", "no_price"),
        ("genuine_minus_matched_placebo_squared_error", "informative_price_path", "matched_placebo_path"),
        ("genuine_minus_time_permuted_placebo_squared_error", "informative_price_path", "time_permuted_price_path"),
    ]
    contrast_rows: list[dict[str, Any]] = []
    for number, (label, left, right) in enumerate(comparisons):
        stats = bootstrap_contrast(
            cells[f"squared_error_{left}"] - cells[f"squared_error_{right}"],
            int(config["analysis"]["bootstrap_repetitions"]),
            int(config["analysis"]["bootstrap_seed"]) + number,
        )
        contrast_rows.append({"contrast": label, **stats})
    contrasts = pd.DataFrame(contrast_rows)
    condition_summary = pd.DataFrame([
        {
            "condition": condition,
            "dossiers": int(len(cells)),
            "mean_squared_error": float(cells[f"squared_error_{condition}"].mean()),
        }
        for condition in ["no_price", "informative_price_path", "matched_placebo_path", "time_permuted_price_path"]
    ])
    gate = bool((contrasts.ci_95_upper < 0).all())
    interim = resolve_repo_path(config["interim_root"])
    output = resolve_repo_path(config["output_root"])
    interim.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    training.to_parquet(interim / "public_signal_training_private.parquet", index=False)
    cells.to_parquet(interim / "public_signal_holdout_cells_private.parquet", index=False)
    condition_summary.to_csv(output / "step4_holdout_public_signal_condition_summary.csv", index=False)
    contrasts.to_csv(output / "step4_holdout_public_signal_gate_contrasts.csv", index=False)
    summary = {
        **plan(config),
        "selected_ridge_alpha_accounting_only": base_alpha,
        "selected_ridge_alpha_accounting_plus_price": price_alpha,
        "researcher_side_public_signal_gate_passed": gate,
        "gate_rule": "Every genuine-minus-comparator squared-error contrast has bootstrap upper CI below zero.",
        "contrast_results": contrasts.to_dict(orient="records"),
    }
    (output / "step4_holdout_public_signal_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    report = "\n".join([
        "# E2 public-signal audit",
        "",
        "This is a non-LLM pre-API gate. The model output is not used in its construction.",
        "",
        "## Condition-level holdout squared error",
        "",
        markdown_table(condition_summary),
        "",
        "## Genuine-path contrasts",
        "",
        "Negative values favor the genuine target-firm path.",
        "",
        markdown_table(contrasts),
        "",
        f"Researcher-side gate passed: {gate}.",
    ])
    (output / "step4_holdout_public_signal_audit_report.md").write_text(report)
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
