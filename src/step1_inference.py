"""Audit and infer the Step 1 price--compute loss moments.

The Step 1 runner evaluates all four models with a common Gaussian scale in
each OOS year.  Consequently, differences in its Gaussian pseudo-log-loss are
exactly half of the corresponding squared-error differences divided by that
year's common scale squared.  This script documents that identity, reports
MSEs alongside losses, and supplies two-way clustered and quarter-block
bootstrap uncertainty for the SI moments.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCORES = REPO_ROOT / "output/step1/oos_scores.parquet"
DEFAULT_FOLDS = REPO_ROOT / "output/step1/oos_folds.csv"
DEFAULT_OUTPUT = REPO_ROOT / "output/step1"
MODELS = ["L0", "L1", "H0", "H1"]


def normal_two_sided_pvalue(statistic: float) -> float:
    """Normal-approximation two-sided p-value without an extra dependency."""
    return math.erfc(abs(statistic) / math.sqrt(2.0)) if np.isfinite(statistic) else np.nan


def cluster_score_meat(score: np.ndarray, labels: pd.Series) -> tuple[float, int]:
    groups = pd.Series(labels.to_numpy()).astype("category")
    codes = groups.cat.codes.to_numpy()
    n_groups = int(len(groups.cat.categories))
    sums = np.zeros(n_groups)
    np.add.at(sums, codes, score)
    return float(np.dot(sums, sums)), n_groups


def two_way_cluster_mean(values: pd.Series, firm: pd.Series, quarter: pd.Series) -> tuple[float, float, int, int]:
    """Mean and Cameron-Gelbach-Miller two-way clustered standard error."""
    y = values.to_numpy(dtype=float)
    n_obs = len(y)
    mean = float(np.mean(y))
    score = y - mean
    firm_meat, n_firm = cluster_score_meat(score, firm)
    quarter_meat, n_quarter = cluster_score_meat(score, quarter)
    pair = firm.astype(str) + "|" + quarter.astype(str)
    pair_meat, n_pair = cluster_score_meat(score, pair)
    # The variance of an intercept estimate divides the score meat by N^2.
    base = (n_obs - 1.0) / max(n_obs - 1.0, 1.0)
    firm_correction = base * n_firm / max(n_firm - 1.0, 1.0)
    quarter_correction = base * n_quarter / max(n_quarter - 1.0, 1.0)
    pair_correction = base * n_pair / max(n_pair - 1.0, 1.0)
    variance = (firm_correction * firm_meat + quarter_correction * quarter_meat - pair_correction * pair_meat) / n_obs**2
    return mean, math.sqrt(max(variance, 0.0)), n_firm, n_quarter


def quarter_block_bootstrap(values: pd.Series, quarter: pd.Series, draws: int, seed: int) -> tuple[float, float]:
    """Percentile CI from resampling entire formation quarters with replacement."""
    frame = pd.DataFrame({"value": values.to_numpy(dtype=float), "quarter": quarter.to_numpy()})
    blocks = frame.groupby("quarter", observed=True).agg(total=("value", "sum"), count=("value", "size"))
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(blocks), size=(draws, len(blocks)))
    totals = blocks.total.to_numpy()[sampled].sum(axis=1)
    counts = blocks["count"].to_numpy()[sampled].sum(axis=1)
    means = totals / counts
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def load_scores(scores_path: Path, folds_path: Path) -> pd.DataFrame:
    if not scores_path.exists() or not folds_path.exists():
        raise FileNotFoundError("Run `python3 src/step1_pipeline.py run` before Step 1 inference.")
    scores = pd.read_parquet(scores_path).copy()
    folds = pd.read_csv(folds_path)
    required = {"permno", "formation_date", "formation_year", "target_for_model"}
    required |= {f"prediction_{model}" for model in MODELS}
    required |= {f"loss_{model}" for model in MODELS}
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"Missing Step 1 OOS columns: {', '.join(missing)}")
    sigma = folds.groupby("test_year", as_index=False).agg(common_sigma=("common_sigma", "first"))
    if folds.groupby("test_year").common_sigma.nunique().gt(1).any():
        raise ValueError("The OOS fold file does not have a common sigma across models within every year.")
    scores = scores.merge(sigma, left_on="formation_year", right_on="test_year", how="left", validate="many_to_one")
    if scores.common_sigma.isna().any():
        raise ValueError("Some OOS rows could not be assigned a common sigma.")
    scores["formation_date"] = pd.to_datetime(scores["formation_date"])
    scores["formation_quarter"] = scores.formation_date.dt.to_period("Q").astype(str)
    scores["permno"] = pd.to_numeric(scores.permno, errors="coerce").astype("int64")
    for model in MODELS:
        scores[f"squared_error_{model}"] = (scores.target_for_model - scores[f"prediction_{model}"]) ** 2
    scores["S_loss"] = scores.loss_L0 - scores.loss_L1
    scores["F_loss"] = scores.loss_H0 - scores.loss_H1
    scores["V_loss"] = scores.loss_L1 - scores.loss_H1
    scores["E_loss"] = scores.loss_L0 - scores.loss_H0
    scores["sharing_gap_loss"] = scores.S_loss - scores.F_loss
    scores["S_mse"] = scores.squared_error_L0 - scores.squared_error_L1
    scores["F_mse"] = scores.squared_error_H0 - scores.squared_error_H1
    scores["V_mse"] = scores.squared_error_L1 - scores.squared_error_H1
    scores["E_mse"] = scores.squared_error_L0 - scores.squared_error_H0
    scores["sharing_gap_mse"] = scores.S_mse - scores.F_mse
    return scores


def loss_mse_identity(scores: pd.DataFrame) -> pd.DataFrame:
    """Verify the exact common-scale loss versus squared-error identity."""
    pairs = {
        "S": ("L0", "L1"),
        "F": ("H0", "H1"),
        "V": ("L1", "H1"),
        "E": ("L0", "H0"),
    }
    rows: list[dict[str, object]] = []
    for name, (first, second) in pairs.items():
        loss_diff = scores[f"loss_{first}"] - scores[f"loss_{second}"]
        scaled_mse_diff = (scores[f"squared_error_{first}"] - scores[f"squared_error_{second}"]) / (
            2.0 * scores.common_sigma**2
        )
        error = loss_diff - scaled_mse_diff
        rows.append({
            "moment": name,
            "max_abs_identity_error": float(error.abs().max()),
            "mean_abs_identity_error": float(error.abs().mean()),
        })
    accounting = scores.E_loss - (scores.V_loss + scores.S_loss - scores.F_loss)
    rows.append({
        "moment": "E_equals_V_plus_S_minus_F",
        "max_abs_identity_error": float(accounting.abs().max()),
        "mean_abs_identity_error": float(accounting.abs().mean()),
    })
    return pd.DataFrame(rows)


def inference_table(scores: pd.DataFrame, draws: int, seed: int) -> pd.DataFrame:
    labels = {
        "S_loss": "S = L0 - L1 (limited-observer price value)",
        "F_loss": "F = H0 - H1 (advanced-observer price value)",
        "V_loss": "V = L1 - H1 (compute value after price)",
        "E_loss": "E = L0 - H0 (compute value before price)",
        "sharing_gap_loss": "S - F (capacity-dependent price value)",
        "S_mse": "MSE analogue of S",
        "F_mse": "MSE analogue of F",
        "V_mse": "MSE analogue of V",
        "E_mse": "MSE analogue of E",
        "sharing_gap_mse": "MSE analogue of S - F",
    }
    rows: list[dict[str, object]] = []
    for offset, (column, label) in enumerate(labels.items()):
        mean, se, n_firm, n_quarter = two_way_cluster_mean(
            scores[column], scores.permno, scores.formation_quarter
        )
        lower, upper = quarter_block_bootstrap(
            scores[column], scores.formation_quarter, draws=draws, seed=seed + offset
        )
        statistic = mean / se if se > 0 else np.nan
        rows.append({
            "moment": column,
            "label": label,
            "mean": mean,
            "two_way_cluster_se": se,
            "two_way_ci_low": mean - 1.96 * se,
            "two_way_ci_high": mean + 1.96 * se,
            "two_way_p_value": normal_two_sided_pvalue(statistic),
            "quarter_block_ci_low": lower,
            "quarter_block_ci_high": upper,
            "n_obs": len(scores),
            "n_firms": n_firm,
            "n_formation_quarters": n_quarter,
        })
    return pd.DataFrame(rows)


def model_loss_mse_table(scores: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    annual: list[pd.DataFrame] = []
    for model in MODELS:
        rows.append({
            "model": model,
            "mean_gaussian_pseudo_log_loss": scores[f"loss_{model}"].mean(),
            "mean_squared_error": scores[f"squared_error_{model}"].mean(),
            "root_mean_squared_error": math.sqrt(scores[f"squared_error_{model}"].mean()),
        })
        by_year = scores.groupby("formation_year", as_index=False).agg(
            mean_loss=(f"loss_{model}", "mean"), mean_mse=(f"squared_error_{model}", "mean"),
            common_sigma=("common_sigma", "first"), n_obs=("permno", "size"),
        )
        by_year["model"] = model
        annual.append(by_year)
    return pd.DataFrame(rows), pd.concat(annual, ignore_index=True)


def write_report(summary: pd.DataFrame, identity: pd.DataFrame, output_dir: Path) -> None:
    by_moment = summary.set_index("moment")
    s = by_moment.loc["S_loss"]
    f = by_moment.loc["F_loss"]
    v = by_moment.loc["V_loss"]
    e = by_moment.loc["E_loss"]
    gap = by_moment.loc["sharing_gap_loss"]
    report = f"""# Step 1 inference and loss audit

All four models share a single Gaussian scale within an OOS year. Thus the
reported pseudo-log-loss differences are common-scale squared-error differences,
not model-specific predictive-variance improvements. The maximum numerical
error in the loss--MSE identity is reported in `loss_mse_identity_check.csv`.

| Moment | Mean | Two-way clustered SE | 95% CI | Quarter-block 95% CI |
| --- | ---: | ---: | ---: | ---: |
| S | {s['mean']:.6f} | {s['two_way_cluster_se']:.6f} | [{s['two_way_ci_low']:.6f}, {s['two_way_ci_high']:.6f}] | [{s['quarter_block_ci_low']:.6f}, {s['quarter_block_ci_high']:.6f}] |
| F | {f['mean']:.6f} | {f['two_way_cluster_se']:.6f} | [{f['two_way_ci_low']:.6f}, {f['two_way_ci_high']:.6f}] | [{f['quarter_block_ci_low']:.6f}, {f['quarter_block_ci_high']:.6f}] |
| V | {v['mean']:.6f} | {v['two_way_cluster_se']:.6f} | [{v['two_way_ci_low']:.6f}, {v['two_way_ci_high']:.6f}] | [{v['quarter_block_ci_low']:.6f}, {v['quarter_block_ci_high']:.6f}] |
| E | {e['mean']:.6f} | {e['two_way_cluster_se']:.6f} | [{e['two_way_ci_low']:.6f}, {e['two_way_ci_high']:.6f}] | [{e['quarter_block_ci_low']:.6f}, {e['quarter_block_ci_high']:.6f}] |
| S − F | {gap['mean']:.6f} | {gap['two_way_cluster_se']:.6f} | [{gap['two_way_ci_low']:.6f}, {gap['two_way_ci_high']:.6f}] | [{gap['quarter_block_ci_low']:.6f}, {gap['quarter_block_ci_high']:.6f}] |

The accounting identity is `E = V + S − F`, observation by observation up to
floating-point precision. Inference clusters jointly by PERMNO and formation
quarter; the block interval resamples entire formation quarters and is reported
as a complementary time-cluster robustness check.
"""
    (output_dir / "step1_inference_report.md").write_text(report)


def run(args: argparse.Namespace) -> None:
    scores = load_scores(args.scores, args.folds)
    identity = loss_mse_identity(scores)
    summary = inference_table(scores, draws=args.bootstrap_draws, seed=args.seed)
    model_table, annual_table = model_loss_mse_table(scores)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    identity.to_csv(args.output_dir / "loss_mse_identity_check.csv", index=False)
    summary.to_csv(args.output_dir / "step1_inference_summary.csv", index=False)
    model_table.to_csv(args.output_dir / "step1_model_loss_mse.csv", index=False)
    annual_table.to_csv(args.output_dir / "step1_model_loss_mse_by_year.csv", index=False)
    write_report(summary, identity, args.output_dir)
    print(summary.loc[summary.moment.str.endswith("loss"), ["moment", "mean", "two_way_cluster_se", "two_way_p_value"]].to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--scores", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--folds", type=Path, default=DEFAULT_FOLDS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-draws", type=int, default=1999)
    parser.add_argument("--seed", type=int, default=20261030)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
