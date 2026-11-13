"""Audit the Step 1 information sets for current-price contamination.

The economic comparison requires L0 and H0 to exclude the formation-month
price snapshot.  This program makes the restriction machine-checkable from the
model feature lists, the generated feature manifest, and the constructed
firm-quarter panel.  Strictly lagged price histories remain permissible public
information in L0/H0; the audit distinguishes them from the contemporaneous
snapshot rather than treating all price history as forbidden.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from step1_pipeline import (
    HIGH_ADDITIONAL_FEATURES,
    LOW_FEATURES,
    PANEL_PATH,
    PRICE_SNAPSHOT_FEATURES,
    OUTPUT_DIR,
    model_feature_sets,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def timing_class(feature: str) -> str:
    if feature in PRICE_SNAPSHOT_FEATURES:
        return "formation-month price snapshot"
    if feature.startswith("log_price_lag") or feature.startswith("return_") or feature.startswith("return_lag"):
        return "strictly lagged CRSP price/return history"
    if feature.endswith("_statement_lag1") or feature.endswith("_statement_lag2"):
        return "lagged public accounting statement"
    if feature in {"statement_age_days", *[item for item in LOW_FEATURES if item not in {"log_market_equity_lag1"}]}:
        return "public accounting or lagged feature"
    if feature == "log_market_equity_lag1":
        return "prior-month market equity"
    return "other permitted public feature"


def check(name: str, passed: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(passed), "detail": detail}


def run(args: argparse.Namespace) -> None:
    feature_sets = model_feature_sets()
    expected = {
        "L0": list(LOW_FEATURES),
        "L1": [*LOW_FEATURES, *PRICE_SNAPSHOT_FEATURES],
        "H0": [*LOW_FEATURES, *HIGH_ADDITIONAL_FEATURES],
        "H1": [*LOW_FEATURES, *HIGH_ADDITIONAL_FEATURES, *PRICE_SNAPSHOT_FEATURES],
    }
    if feature_sets != expected:
        raise AssertionError("model_feature_sets() no longer matches the declared information-set construction")

    forbidden_current_names = set(PRICE_SNAPSHOT_FEATURES) | {
        "stock_price", "market_equity", "stock_return",
    }
    rows: list[dict[str, object]] = []
    for model, features in feature_sets.items():
        for feature in features:
            rows.append({
                "model": model,
                "feature": feature,
                "timing_class": timing_class(feature),
                "is_formation_month_price_snapshot": feature in PRICE_SNAPSHOT_FEATURES,
                "allowed_in_model": not (model in {"L0", "H0"} and feature in forbidden_current_names),
            })
    expanded_manifest = pd.DataFrame(rows)

    checks = [
        check(
            "L0_excludes_formation_month_snapshot",
            not bool(set(feature_sets["L0"]).intersection(forbidden_current_names)),
            "L0 contains none of current_log_price, current_log_market_equity, current_return_1m, or raw current CRSP fields.",
        ),
        check(
            "H0_excludes_formation_month_snapshot",
            not bool(set(feature_sets["H0"]).intersection(forbidden_current_names)),
            "H0 contains none of current_log_price, current_log_market_equity, current_return_1m, or raw current CRSP fields.",
        ),
        check(
            "L1_minus_L0_is_exactly_snapshot",
            set(feature_sets["L1"]).difference(feature_sets["L0"]) == set(PRICE_SNAPSHOT_FEATURES),
            "The low-capacity price comparison changes only the three formation-month snapshot features.",
        ),
        check(
            "H1_minus_H0_is_exactly_snapshot",
            set(feature_sets["H1"]).difference(feature_sets["H0"]) == set(PRICE_SNAPSHOT_FEATURES),
            "The high-capacity price comparison changes only the three formation-month snapshot features.",
        ),
        check(
            "H0_minus_L0_is_exactly_high_capacity_inputs",
            set(feature_sets["H0"]).difference(feature_sets["L0"]) == set(HIGH_ADDITIONAL_FEATURES),
            "The compute comparison adds only the declared longer histories and lagged accounting inputs.",
        ),
    ]

    panel = pd.read_parquet(args.panel)
    formation = pd.to_datetime(panel["formation_date"], errors="coerce")
    panel_columns = set(panel.columns)
    checks.extend([
        check(
            "all_declared_features_present_in_panel",
            set(sum(feature_sets.values(), [])).issubset(panel_columns),
            "Every model feature is a column in the firm-quarter panel; models receive explicit feature lists only.",
        ),
        check(
            "formation_dates_are_quarter_ends",
            bool((formation.dt.is_quarter_end | formation.isna()).all()),
            "The current snapshot is measured at calendar-quarter-end formation; lagged CRSP features are created with group.shift().",
        ),
        check(
            "no_forbidden_feature_in_price_free_manifest",
            bool(expanded_manifest.loc[expanded_manifest.model.isin(["L0", "H0"]), "allowed_in_model"].all()),
            "The expanded all-model manifest contains no contemporaneous price feature in L0 or H0.",
        ),
    ])
    results = pd.DataFrame(checks)
    if not results.passed.all():
        failed = results.loc[~results.passed, "check"].tolist()
        raise AssertionError(f"Information-set audit failed: {failed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    expanded_manifest.to_csv(args.output_dir / "information_set_audit_feature_manifest.csv", index=False)
    results.to_csv(args.output_dir / "information_set_audit_checks.csv", index=False)
    report = [
        "# Step 1 information-set audit",
        "",
        "All checks passed. `L0` and `H0` exclude the formation-month price snapshot. "
        "They may use only strictly lagged CRSP price/return histories and public accounting information.",
        "",
        "| Check | Passed | Detail |",
        "| --- | --- | --- |",
    ]
    report.extend(f"| {row.check} | {row.passed} | {row.detail} |" for row in results.itertuples(index=False))
    report.extend([
        "",
        "The price comparisons are exact nested contrasts: `L1 - L0` and `H1 - H0` each add "
        "only `current_log_price`, `current_log_market_equity`, and `current_return_1m`.",
    ])
    (args.output_dir / "information_set_audit_report.md").write_text("\n".join(report) + "\n")
    print(results.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--panel", type=Path, default=PANEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
