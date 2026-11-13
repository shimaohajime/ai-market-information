"""Create SI-ready Step 1--2 tables, figures, and a results-section draft."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
STEP1 = REPO_ROOT / "output/step1"
STEP2 = REPO_ROOT / "output/step2"
DEFAULT_OUTPUT = REPO_ROOT / "output/si"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9,
                     "axes.titlesize": 10, "legend.fontsize": 8, "pdf.fonttype": 42, "ps.fonttype": 42})


def require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Missing required results file: {path}")
    return path


def save(figure: plt.Figure, path: Path) -> None:
    figure.tight_layout()
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    figure.savefig(path.with_suffix(".png"), dpi=240, bbox_inches="tight")
    plt.close(figure)


def plot_step1(inference: pd.DataFrame, models: pd.DataFrame, output: Path) -> None:
    order = ["L0", "L1", "H0", "H1"]
    models = models.set_index("model").loc[order].reset_index()
    moments = ["S_loss", "F_loss", "V_loss", "E_loss", "sharing_gap_loss"]
    labels = ["S = L0 − L1", "F = H0 − H1", "V = L1 − H1", "E = L0 − H0", "S − F"]
    forest = inference.set_index("moment").loc[moments].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    x = np.arange(len(models))
    ax.plot(x, models.mean_gaussian_pseudo_log_loss, color="#1f4e79", marker="o", lw=1.8)
    for row, xpos in zip(models.itertuples(index=False), x):
        ax.annotate(f"{row.mean_gaussian_pseudo_log_loss:.4f}", (xpos, row.mean_gaussian_pseudo_log_loss),
                    xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
    ax.set_xticks(x, ["L0\nlow, no price", "L1\nlow, price", "H0\nhigh, no price", "H1\nhigh, price"])
    ax.set_ylabel("OOS Gaussian pseudo-log-loss")
    ax.set_title("A. Predictive loss across simulated technologies")
    ax.grid(axis="y", color="0.88", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    ax = axes[1]
    y = np.arange(len(forest))[::-1]
    ax.errorbar(forest["mean"], y, xerr=1.96 * forest["two_way_cluster_se"], fmt="o", color="#1f4e79",
                ecolor="#1f4e79", capsize=3, lw=1.3)
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Loss difference (positive favors price or compute)")
    ax.set_title("B. OOS price and compute moments (95% clustered CI)")
    ax.grid(axis="x", color="0.88", lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    save(fig, output / "figure_si_step1_price_compute")


def plot_event(full: pd.DataFrame, rdq: pd.DataFrame, output: Path) -> None:
    labels = {
        "low_compute_price_signal": "Low price\nL1 − L0",
        "high_compute_price_signal": "High price\nH1 − H0",
        "high_compute_public_signal": "High public\nH0 − L0",
        "high_compute_post_price_signal": "High post-price\nH1 − L1",
    }
    order = list(labels)
    full = full.loc[full.outcome.eq("car_market_adjusted_0_p1_w")].set_index("specification").loc[order].reset_index()
    rdq = rdq.loc[rdq.outcome.eq("car_market_adjusted_0_p1_w")].set_index("specification").loc[order].reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.9), gridspec_kw={"width_ratios": [1.12, 1]})
    ax = axes[0]
    y = np.arange(len(order))[::-1]
    ax.errorbar(full.beta_residual, y + 0.12, xerr=1.96 * full.se_beta_residual, fmt="o", color="#1f4e79",
                capsize=3, label="Residual-SUE coefficient β")
    ax.errorbar(full.delta_increment, y - 0.12, xerr=1.96 * full.se_delta_increment, fmt="s", color="#b45f06",
                capsize=3, label="Incremental-signal coefficient δ")
    ax.axvline(0, color="0.35", lw=0.8)
    ax.set_yticks(y, [labels[name] for name in order])
    ax.set_xlabel("CAR[0,+1] coefficient")
    ax.set_title("A. Event-day response: full sample")
    # Keep the legend clear of the fourth (post-price) estimate.
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="x", color="0.88", lw=0.7)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax = axes[1]
    x = np.arange(len(order))
    ax.errorbar(x - 0.12, full.delta_increment, yerr=1.96 * full.se_delta_increment, fmt="o", color="#b45f06",
                capsize=3, label="Full event sample")
    ax.errorbar(x + 0.12, rdq.delta_increment, yerr=1.96 * rdq.se_delta_increment, fmt="s", color="#38761d",
                capsize=3, label="RDQ agreement ≤ 1 day")
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, [labels[name] for name in order])
    ax.set_ylabel("Incremental-signal coefficient δ")
    ax.set_title("B. Timing-validation robustness")
    ax.legend(loc="upper right", frameon=False)
    ax.grid(axis="y", color="0.88", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output / "figure_si_step2_event_response")


def plot_preannouncement(pre: pd.DataFrame, output: Path) -> None:
    outcome_labels = {
        "pre_car_market_adjusted_per_day_w": "All pre-event\n(daily average)",
        "pre_car_market_adjusted_early20_w": "First 20\ntrading days",
        "pre_car_market_adjusted_late20_w": "Last 20\ntrading days",
    }
    pre = pre.set_index(["outcome", "signal"])
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(outcome_labels))
    styles = [
        (-0.12, "q_H_public", "o", "#1f4e79", "High public signal H0 − L0"),
        (0.12, "q_H_post_price", "s", "#b45f06", "High post-price signal H1 − L1"),
    ]
    for displacement, signal, marker, color, label in styles:
        rows = [pre.loc[(outcome, signal)] for outcome in outcome_labels]
        ax.errorbar(x + displacement, [row.coefficient for row in rows],
                    yerr=[1.96 * row.std_error_two_way_cluster for row in rows],
                    fmt=marker, color=color, capsize=3, label=label)
    ax.axhline(0, color="0.35", lw=0.8)
    ax.set_xticks(x, list(outcome_labels.values()))
    ax.set_ylabel("Pre-announcement return coefficient")
    ax.set_title("High-compute signals and returns before earnings announcement")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", color="0.88", lw=0.7)
    ax.spines[["top", "right"]].set_visible(False)
    save(fig, output / "figure_si_step2_preannouncement")


def plot_persistence(persistence: pd.DataFrame, output: Path) -> None:
    """Display the future-SUE association of high-compute increments."""
    full = persistence.loc[persistence["sample"].eq("full event sample")].copy()
    labels = {
        "high public compute": "High public signal H0 − L0",
        "high post-price compute": "High post-price signal H1 − L1",
    }
    styles = {
        "high public compute": ("o", "#1f4e79"),
        "high post-price compute": ("s", "#b45f06"),
    }
    titles = {
        "formation_date": "A. Formation-date fixed effects",
        "firm_and_formation_date": "B. Firm + formation-date fixed effects",
    }
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.8), sharey=True)
    for ax, fixed_effects in zip(axes, titles, strict=True):
        for display, label in labels.items():
            subset = full.loc[
                full.display.eq(display) & full.fixed_effects.eq(fixed_effects)
            ].sort_values("horizon")
            marker, color = styles[display]
            ax.errorbar(
                subset.horizon, subset.delta_increment,
                yerr=1.96 * subset.se_delta_increment,
                fmt=f"-{marker}", color=color, capsize=3, label=label,
            )
        ax.axhline(0, color="0.35", lw=0.8)
        ax.set_xticks(range(5), ["t", "t+1", "t+2", "t+3", "t+4"])
        ax.set_xlabel("Earnings-surprise horizon")
        ax.set_title(titles[fixed_effects])
        ax.grid(axis="y", color="0.88", lw=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Future-SUE coefficient")
    axes[0].legend(frameon=False, loc="lower left")
    save(fig, output / "figure_si_step2_earnings_persistence")


def table_lines(frame: pd.DataFrame, columns: list[str], headings: list[str], precision: dict[str, int]) -> str:
    lines = ["| " + " | ".join(headings) + " |", "| " + " | ".join(["---"] * len(headings)) + " |"]
    for values in frame[columns].itertuples(index=False, name=None):
        cells = []
        for value, column in zip(values, columns):
            if isinstance(value, (float, np.floating)):
                digits = precision[column]
                if column.startswith("p") and value < 10 ** (-digits):
                    cells.append(f"<{10 ** (-digits):.{digits}f}")
                else:
                    cells.append(f"{value:.{digits}f}")
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def write_draft(inference: pd.DataFrame, event_full: pd.DataFrame, event_rdq: pd.DataFrame,
                rdq_diag: pd.DataFrame, pre: pd.DataFrame, integrated: pd.DataFrame,
                persistence: pd.DataFrame, output: Path) -> None:
    names = {"S_loss": "S = L0 − L1", "F_loss": "F = H0 − H1", "V_loss": "V = L1 − H1",
             "E_loss": "E = L0 − H0", "sharing_gap_loss": "S − F"}
    moments = inference.loc[inference.moment.isin(names)].copy()
    moments["name"] = moments.moment.map(names)
    event = event_full.loc[event_full.outcome.eq("car_market_adjusted_0_p1_w")].copy()
    event["name"] = event.specification.map({
        "low_compute_price_signal": "Low price: L1 − L0", "high_compute_price_signal": "High price: H1 − H0",
        "high_compute_public_signal": "High public compute: H0 − L0",
        "high_compute_post_price_signal": "High post-price compute: H1 − L1",
    })
    rdq = event_rdq.loc[event_rdq.outcome.eq("car_market_adjusted_0_p1_w"),
                        ["specification", "delta_increment", "se_delta_increment", "p_delta_equals_beta"]].copy()
    rdq = rdq.rename(columns={"delta_increment": "delta_rdq", "se_delta_increment": "se_delta_rdq",
                              "p_delta_equals_beta": "p_eq_beta_rdq"})
    event = event.merge(rdq, on="specification", how="left")
    rdq_rate = rdq_diag.loc[rdq_diag.metric.eq("rdq_match_within_1_calendar_day_rate"), "value"].iloc[0]
    moment_table = table_lines(
        moments, ["name", "mean", "two_way_cluster_se", "two_way_ci_low", "two_way_ci_high"],
        ["Moment", "Mean", "Two-way SE", "95% CI low", "95% CI high"],
        {"mean": 6, "two_way_cluster_se": 6, "two_way_ci_low": 6, "two_way_ci_high": 6},
    )
    event_table = table_lines(
        event, ["name", "beta_residual", "se_beta_residual", "delta_increment", "se_delta_increment", "p_delta_equals_beta", "delta_rdq", "se_delta_rdq", "p_eq_beta_rdq"],
        ["Signal", "β", "SE(β)", "δ", "SE(δ)", "p: δ=β", "δ RDQ", "SE RDQ", "p: δ=β RDQ"],
        {"beta_residual": 6, "se_beta_residual": 6, "delta_increment": 6, "se_delta_increment": 6,
         "p_delta_equals_beta": 4, "delta_rdq": 6, "se_delta_rdq": 6, "p_eq_beta_rdq": 4},
    )
    pre_table = table_lines(
        pre, ["outcome", "signal", "coefficient", "std_error_two_way_cluster", "ci_low", "ci_high", "p_value", "n_obs"],
        ["Pre-event outcome", "Signal", "Coefficient", "Two-way SE", "95% CI low", "95% CI high", "p", "N"],
        {"coefficient": 6, "std_error_two_way_cluster": 6, "ci_low": 6, "ci_high": 6, "p_value": 4},
    )
    integrated_full = integrated.loc[
        integrated["sample"].eq("full event sample") & integrated.display.eq("high public compute")
    ].iloc[0]
    persistence_full = persistence.loc[
        persistence["sample"].eq("full event sample") & persistence.fixed_effects.eq("formation_date")
    ].copy()
    persistence_firm = persistence.loc[
        persistence["sample"].eq("full event sample") & persistence.fixed_effects.eq("firm_and_formation_date")
    ].copy()
    persistence_rows = []
    for horizon in range(5):
        public = persistence_full.loc[
            persistence_full.display.eq("high public compute") & persistence_full.horizon.eq(horizon)
        ].iloc[0]
        post = persistence_full.loc[
            persistence_full.display.eq("high post-price compute") & persistence_full.horizon.eq(horizon)
        ].iloc[0]
        persistence_rows.append({
            "horizon": horizon,
            "public_delta": public.delta_increment,
            "public_se": public.se_delta_increment,
            "post_delta": post.delta_increment,
            "post_se": post.se_delta_increment,
        })
    persistence_table = table_lines(
        pd.DataFrame(persistence_rows), ["horizon", "public_delta", "public_se", "post_delta", "post_se"],
        ["Horizon", "H0 − L0", "SE", "H1 − L1", "SE"],
        {"public_delta": 6, "public_se": 6, "post_delta": 6, "post_se": 6},
    )
    firm_h4_public = persistence_firm.loc[
        persistence_firm.display.eq("high public compute") & persistence_firm.horizon.eq(4)
    ].iloc[0]
    firm_h4_post = persistence_firm.loc[
        persistence_firm.display.eq("high post-price compute") & persistence_firm.horizon.eq(4)
    ].iloc[0]
    s = moments.loc[moments.name.eq("S = L0 − L1")].iloc[0]
    f = moments.loc[moments.name.eq("F = H0 − H1")].iloc[0]
    v = moments.loc[moments.name.eq("V = L1 − H1")].iloc[0]
    gap = moments.loc[moments.name.eq("S − F")].iloc[0]
    draft = f"""# Supplementary empirical results: draft for Science SI

## S1. Out-of-sample simulated processing technologies

We evaluate four pre-announcement technologies on 58,807 firm-quarter observations using expanding annual training windows and strictly out-of-sample predictions. Within an OOS year, all models share one Gaussian scale. Loss differences therefore equal common-scale squared-error differences and do not combine mean prediction with model-specific variance prediction.

{moment_table}

The low-capacity price value is positive (S = {s['mean']:.6f}, clustered SE = {s['two_way_cluster_se']:.6f}). The high-capacity price value is smaller and its interval includes zero (F = {f['mean']:.6f}, clustered SE = {f['two_way_cluster_se']:.6f}). The capacity-dependent difference remains positive (S − F = {gap['mean']:.6f}, clustered SE = {gap['two_way_cluster_se']:.6f}), while the compute value after price observation is positive (V = {v['mean']:.6f}). See Figure SI.1.

## S2. Earnings-announcement event study

We link these out-of-sample observations to I/B/E/S quarterly EPS announcements and CRSP CIZ Daily returns. The primary outcome is market-adjusted CAR[0,+1], which accommodates unknown before-market versus after-market release time. The complete-window main sample has 55,740 observations. Compustat RDQ matches I/B/E/S within one calendar day for {rdq_rate:.1%} of events; the corresponding robustness sample has 54,929 observations.

{event_table}

Both high-compute components have event-day coefficients much smaller than their residual-SUE coefficients; equality is rejected in the full and RDQ-validated samples. Thus they do not resemble earnings-day surprise components. This result does not prove incorporation at the original formation date, because information could be incorporated between formation and earnings or a model signal could have limited valuation content. See Figure SI.2.

The four displayed increments are not independent because `H1 − L1 = (H0 − L0) + (H1 − H0) − (L1 − L0)`. In a joint regression on the linearly independent basis `(L1 − L0, H1 − H0, H0 − L0)`, the high-public-compute coefficient is {integrated_full.coefficient:.6f} (two-way SE {integrated_full.std_error_two_way_cluster:.6f}; p for zero = {integrated_full.p_zero:.3f}) versus a high-model residual coefficient of 0.004339. Equality is rejected (p < 0.0001). Replacing `H0 − L0` with `H1 − L1` gives fitted values identical to machine precision, so the post-price display is a reparameterization, not an additional independent result.

## S3. Returns between formation and earnings announcement

We test for post-formation drift in market-adjusted returns. The first outcome is average daily market-adjusted return from the formation trading date through the day before earnings; the next two isolate the first and last twenty trading days. The regressions absorb formation-date fixed effects and use two-way PERMNO/formation-date clustered standard errors.

{pre_table}

None of these post-formation estimates is distinguishable from zero. This is consistent with incorporation no later than formation but is not proof: null returns can also reflect model error, offsetting valuation news, or small effects. See Figure SI.3.

## S4. Earnings persistence diagnostic

To examine whether small event-day coefficients merely reflect prediction of a transitory earnings component, we relate the pre-announcement high-compute signals to SUE at the focal announcement and at up to four subsequent quarterly announcements. Successive announcements must be 45--180 days apart; the median gaps at horizons one through four are 91, 182, 273, and 364 days. We report both formation-date fixed effects and the stricter specification that additionally absorbs firm fixed effects, with two-way PERMNO/formation-date clustered standard errors.

{persistence_table}

With formation-date fixed effects, both high-compute increments remain positively associated with SUE four quarters later: {persistence_rows[4]['public_delta']:.6f} (SE {persistence_rows[4]['public_se']:.6f}) for `H0 − L0`, and {persistence_rows[4]['post_delta']:.6f} (SE {persistence_rows[4]['post_se']:.6f}) for `H1 − L1`. This captures persistent cross-sectional firm differences. Once firm fixed effects are also absorbed, the corresponding four-quarter coefficients are {firm_h4_public.delta_increment:.6f} (SE {firm_h4_public.se_delta_increment:.6f}) and {firm_h4_post.delta_increment:.6f} (SE {firm_h4_post.se_delta_increment:.6f}), respectively; the RDQ-restricted sample has the same negative signs. Thus within-firm high-compute increments revert rather than persist over this horizon. This diagnostic does not eliminate the alternative that the event-day result partly reflects differing valuation relevance; it should be reported as a limitation, not used to claim price incorporation. See Figure SI.4.

## S5. Scope

These analyses compare counterfactual information-processing technologies in the same market data. They neither measure adviser AI adoption nor identify a causal AI-adoption effect. An adviser-level AI-talent pilot can later test whether observed adoption changes S, F, and V; it should not be required for the Step 1--2 external-validity evidence reported here.
"""
    (output / "SI_EMPIRICAL_RESULTS_DRAFT.md").write_text(draft)


def write_stability_note(
    oos_year: pd.DataFrame,
    event_stability: pd.DataFrame,
    event_stability_rdq: pd.DataFrame,
    analyst_tercile: pd.DataFrame,
    size_tercile: pd.DataFrame,
    output: Path,
) -> None:
    """Write descriptive time- and cross-section-stability tables for the SI.

    These cells intentionally carry no new standard errors or significance
    labels: the point is to display all pre-existing splits without treating
    them as additional independent tests.
    """
    annual = oos_year.copy()
    annual["S_minus_F"] = (
        annual["S_limited_price_value"] - annual["F_high_compute_price_value"]
    )
    annual_table = table_lines(
        annual,
        [
            "formation_year",
            "n_obs",
            "S_limited_price_value",
            "F_high_compute_price_value",
            "S_minus_F",
            "V_compute_value_with_price",
        ],
        ["Year", "N", "S", "F", "S - F", "V"],
        {
            "S_limited_price_value": 6,
            "F_high_compute_price_value": 6,
            "S_minus_F": 6,
            "V_compute_value_with_price": 6,
        },
    )

    def descriptive_terciles(frame: pd.DataFrame, label: str) -> pd.DataFrame:
        selected = frame.copy()
        selected["S_minus_F"] = (
            selected["S_limited_price_value"]
            - selected["F_high_compute_price_value"]
        )
        selected["split"] = label + ": " + selected.iloc[:, 0].astype(str)
        return selected[
            [
                "split",
                "n_obs",
                "S_limited_price_value",
                "F_high_compute_price_value",
                "S_minus_F",
                "V_compute_value_with_price",
            ]
        ]

    terciles = pd.concat(
        [
            descriptive_terciles(analyst_tercile, "Analyst coverage"),
            descriptive_terciles(size_tercile, "Firm size"),
        ],
        ignore_index=True,
    )
    tercile_table = table_lines(
        terciles,
        [
            "split",
            "n_obs",
            "S_limited_price_value",
            "F_high_compute_price_value",
            "S_minus_F",
            "V_compute_value_with_price",
        ],
        ["Split", "N", "S", "F", "S - F", "V"],
        {
            "S_limited_price_value": 6,
            "F_high_compute_price_value": 6,
            "S_minus_F": 6,
            "V_compute_value_with_price": 6,
        },
    )

    selected_specs = [
        "high_compute_public_signal",
        "high_compute_post_price_signal",
    ]
    labels = {
        "high_compute_public_signal": "High public compute: H0 - L0",
        "high_compute_post_price_signal": "High post-price compute: H1 - L1",
    }

    def event_rows(frame: pd.DataFrame, source: str) -> pd.DataFrame:
        selected = frame.loc[frame.specification.isin(selected_specs)].copy()
        selected["signal"] = selected.specification.map(labels)
        selected["sample"] = source + ", " + selected["period"].str.replace("_", "-")
        return selected[
            [
                "sample",
                "signal",
                "n_obs",
                "delta_increment",
                "se_delta_increment",
                "p_delta_equals_beta",
            ]
        ]

    event = pd.concat(
        [
            event_rows(event_stability, "Full event sample"),
            event_rows(event_stability_rdq, "RDQ agreement <= 1 day"),
        ],
        ignore_index=True,
    )
    event_table = table_lines(
        event,
        [
            "sample",
            "signal",
            "n_obs",
            "delta_increment",
            "se_delta_increment",
            "p_delta_equals_beta",
        ],
        ["Sample and period", "Signal", "N", "Delta", "SE(Delta)", "p: Delta = beta"],
        {
            "delta_increment": 6,
            "se_delta_increment": 6,
            "p_delta_equals_beta": 4,
        },
    )
    n_positive_years = int((annual["S_minus_F"] > 0).sum())
    note = f"""# Supplementary empirical stability results

## S6. Descriptive stability across time and firm characteristics

This note displays pre-existing sample splits. It is descriptive: annual and
tercile rows are not additional independent hypothesis tests, and no claim is
made that their point estimates are statistically different from one another.

The capacity-dependent price-value gap, S - F, is positive in
{n_positive_years} of {len(annual)} formation years. It is positive in every
analyst-coverage and firm-size tercile below. The low-capacity price value S
and high-capacity post-price compute value V are also generally positive.

{annual_table}

{tercile_table}

## S7. Calendar-subsample event-time robustness

The table splits the earnings-event sample into 2018-2021 and 2022-2025 and
repeats the high-capacity public and post-price signal comparisons. In both
periods and in the RDQ-validated subsample, the point estimate on each
incremental signal is well below the residual-SUE coefficient; every equality
test rejects at conventional levels. This is a stability check for the
event-time association, not evidence that the effects are identical across
subperiods.

{event_table}
"""
    (output / "SI_EMPIRICAL_STABILITY_NOTE.md").write_text(note)


def run(args: argparse.Namespace) -> None:
    inference = pd.read_csv(require(STEP1 / "step1_inference_summary.csv"))
    models = pd.read_csv(require(STEP1 / "step1_model_loss_mse.csv"))
    event_full = pd.read_csv(require(STEP2 / "market_expectations_summary.csv"))
    event_rdq = pd.read_csv(require(STEP2 / "market_expectations_rdq_within_1_summary.csv"))
    rdq_diag = pd.read_csv(require(STEP2 / "rdq_ibes_timing_diagnostics.csv"))
    pre = pd.read_csv(require(STEP2 / "preannouncement_signal_returns.csv"))
    integrated = pd.read_csv(require(STEP2 / "step2_integrated_decomposition.csv"))
    persistence = pd.read_csv(require(STEP2 / "step2_earnings_persistence.csv"))
    oos_year = pd.read_csv(require(STEP1 / "oos_summary_by_year.csv"))
    event_stability = pd.read_csv(require(STEP2 / "market_expectations_stability.csv"))
    event_stability_rdq = pd.read_csv(
        require(STEP2 / "market_expectations_rdq_within_1_stability.csv")
    )
    analyst_tercile = pd.read_csv(
        require(STEP1 / "oos_summary_by_analyst_coverage_tercile.csv")
    )
    size_tercile = pd.read_csv(require(STEP1 / "oos_summary_by_size_tercile.csv"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_step1(inference, models, args.output_dir)
    plot_event(event_full, event_rdq, args.output_dir)
    plot_preannouncement(pre, args.output_dir)
    plot_persistence(persistence, args.output_dir)
    write_draft(inference, event_full, event_rdq, rdq_diag, pre, integrated, persistence, args.output_dir)
    write_stability_note(
        oos_year,
        event_stability,
        event_stability_rdq,
        analyst_tercile,
        size_tercile,
        args.output_dir,
    )
    print(f"Wrote SI drafts and four figures under {args.output_dir}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
