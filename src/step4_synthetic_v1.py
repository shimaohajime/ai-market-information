"""Controlled synthetic public-price-information benchmark for E2.

This implements a known Gaussian information structure, with a public price
tape that has a fixed length but a controlled number of payoff-linked signals.
The exact posterior and private-clue slope remain local and are never sent to
a model.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import datetime as dt
import json
import math
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from step4_phase0 import REPO_ROOT, read_config, resolve_repo_path


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_synthetic_v1_pilot.json"
RAW_NAME = "step4_synthetic_v1_raw_responses.jsonl"
SCHEMA = {
    "name": "step4_synthetic_posterior",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "posterior_mean": {"type": "number"},
            "posterior_sd": {"type": "number"},
        },
        "required": ["posterior_mean", "posterior_sd"],
    },
}
CONDITION_LINKED_COUNT = {
    "low_public_precision": "linked_observation_count_low",
    "high_public_precision": "price_observation_count",
    "placebo_public_tape": "zero",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value, allow_nan=False) + "\n" for value in values))


def render(template: str, values: dict[str, str]) -> str:
    output = template
    for key, value in values.items():
        output = output.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", output):
        raise ValueError("Unresolved prompt placeholder.")
    return output


def response_format() -> dict[str, Any]:
    return {"type": "json_schema", "name": SCHEMA["name"], "strict": True, "schema": SCHEMA["schema"]}


def reasoning_configuration(model: str, effort: str) -> dict[str, str]:
    if model.startswith("gpt-5.2"):
        return {"effort": effort}
    return {"mode": "standard", "effort": effort}


def clue_key(value: float | None) -> str:
    return "public_only" if value is None else f"{value:+.1f}"


def linked_count(config: dict[str, Any], condition: str) -> int:
    source = CONDITION_LINKED_COUNT[condition]
    return 0 if source == "zero" else int(config[source])


def posterior(
    linked_terms: list[tuple[float, float, float]], prior_sd: float, clue: float | None, clue_noise_sd: float,
) -> tuple[float, float, float]:
    """Exact posterior from (theta loading, residualized price, noise SD) terms."""
    precision = 1.0 / prior_sd**2 + sum(loading**2 / noise_sd**2 for loading, _, noise_sd in linked_terms)
    weighted_sum = sum(loading * residual / noise_sd**2 for loading, residual, noise_sd in linked_terms)
    public_variance = 1.0 / precision
    public_mean = public_variance * weighted_sum
    if clue is None:
        return public_mean, math.sqrt(public_variance), public_variance
    posterior_precision = precision + 1.0 / clue_noise_sd**2
    variance = 1.0 / posterior_precision
    mean = variance * (weighted_sum + clue / clue_noise_sd**2)
    return mean, math.sqrt(variance), public_variance


def build_assets(config: dict[str, Any]) -> dict[str, Any]:
    """Create public stimuli and a separate local exact-posterior key."""
    conditions = list(config["information_conditions"])
    if set(conditions) != set(CONDITION_LINKED_COUNT):
        raise ValueError("Synthetic benchmark requires its three named information conditions.")
    total = int(config["price_observation_count"])
    low = int(config["linked_observation_count_low"])
    if not 0 < low < total:
        raise ValueError("linked_observation_count_low must be between zero and total observations.")
    prior_sd = float(config["prior_sd"])
    price_noise_sd = float(config["price_noise_sd"])
    clue_noise_sd = float(config["private_clue_noise_sd"])
    design = str(config.get("price_tape_design", "homoskedastic_simple"))
    if design not in {"homoskedastic_simple", "heteroskedastic_factor"}:
        raise ValueError("Unsupported price_tape_design.")
    rng = np.random.default_rng(int(config["rng_seed"]))
    public_cases: list[dict[str, Any]] = []
    expected_rows: list[dict[str, Any]] = []
    for number in range(1, int(config["case_count"]) + 1):
        case_id = f"SYN{number:03d}"
        latent_theta = float(rng.normal(0.0, prior_sd))
        linked = np.round(rng.normal(latent_theta, price_noise_sd, size=total), 4)
        # Same unconditional scale as theta + eta, but independent of theta.
        controls = np.round(rng.normal(0.0, math.sqrt(prior_sd**2 + price_noise_sd**2), size=total), 4)
        order = rng.permutation(total)
        low_indexes = set(order[:low].tolist())
        if design == "heteroskedastic_factor":
            loading_grid = np.asarray(config["theta_loading_grid"], dtype=float)
            market_loading_grid = np.asarray(config["market_loading_grid"], dtype=float)
            noise_grid = np.asarray(config["idiosyncratic_noise_sd_grid"], dtype=float)
            theta_loadings = np.round(rng.choice(loading_grid, size=total), 3)
            market_loadings = np.round(rng.choice(market_loading_grid, size=total), 3)
            idiosyncratic_sds = np.round(rng.choice(noise_grid, size=total), 3)
            market_innovations = np.round(rng.normal(0.0, 1.0, size=total), 4)
            linked_errors = rng.normal(0.0, idiosyncratic_sds, size=total)
            control_errors = rng.normal(0.0, np.sqrt(theta_loadings**2 * prior_sd**2 + idiosyncratic_sds**2), size=total)
        for condition in conditions:
            n_linked = linked_count(config, condition)
            rows: list[dict[str, Any]] = []
            local_terms: list[tuple[float, float, float]] = []
            for index in range(total):
                is_linked = condition == "high_public_precision" or (
                    condition == "low_public_precision" and index in low_indexes
                )
                if design == "homoskedastic_simple":
                    value = float(linked[index] if is_linked else controls[index])
                    if is_linked:
                        local_terms.append((1.0, value, price_noise_sd))
                    rows.append({
                        "observation_index": index + 1,
                        "observation_type": "payoff_linked_price_innovation" if is_linked else "market_only_control",
                        "standardized_price_innovation": value,
                    })
                else:
                    theta_loading = float(theta_loadings[index])
                    market_loading = float(market_loadings[index])
                    noise_sd = float(idiosyncratic_sds[index])
                    market_innovation = float(market_innovations[index])
                    raw_price = (
                        theta_loading * latent_theta + market_loading * market_innovation + linked_errors[index]
                        if is_linked else market_loading * market_innovation + control_errors[index]
                    )
                    price_innovation = round(float(raw_price), 4)
                    if is_linked:
                        local_terms.append((theta_loading, price_innovation - market_loading * market_innovation, noise_sd))
                    rows.append({
                        "observation_index": index + 1,
                        "observation_type": "payoff_linked_price_innovation" if is_linked else "market_only_control",
                        "theta_loading": theta_loading,
                        "market_loading": market_loading,
                        "market_innovation": market_innovation,
                        "idiosyncratic_noise_sd": noise_sd,
                        "standardized_price_innovation": price_innovation,
                    })
            if len(local_terms) != n_linked:
                raise AssertionError("Stimulus linked-observation count does not match its condition.")
            public_case: dict[str, Any] = {
                "case_id": case_id,
                "prior": {"theta_mean": 0.0, "theta_sd": prior_sd},
                "price_innovation_model": {
                    "payoff_linked_equation": "z = theta + eta",
                    "payoff_linked_noise_sd": price_noise_sd,
                    "market_only_control": "Independent of theta; exclude from theta likelihood.",
                },
                "anonymous_price_tape": rows,
            }
            if design == "heteroskedastic_factor":
                public_case["price_innovation_model"] = {
                    "payoff_linked_equation": "r_t = a_t * theta + b_t * m_t + eta_t",
                    "payoff_linked_rule": "For a payoff-linked row, use its displayed theta_loading a_t, market_loading b_t, market_innovation m_t, and idiosyncratic_noise_sd. Residualize r_t by b_t*m_t before updating theta.",
                    "market_only_control": "Independent of theta; exclude from theta likelihood even though its numeric columns are displayed.",
                }
            public_cases.append({"case_id": case_id, "information_condition": condition, "public_case": public_case})
            for clue in [None, *[float(value) for value in config["private_clue_values"]]]:
                mean, sd, public_variance = posterior(local_terms, prior_sd, clue, clue_noise_sd)
                expected_rows.append({
                    "case_id": case_id,
                    "information_condition": condition,
                    "clue_key": clue_key(clue),
                    "expected_posterior_mean": mean,
                    "expected_posterior_sd": sd,
                    "public_posterior_variance": public_variance,
                    "theoretical_private_clue_slope": public_variance / (public_variance + clue_noise_sd**2),
                    "hidden_latent_theta": latent_theta,
                })
    interim = resolve_repo_path(config["interim_root"])
    write_jsonl(interim / "synthetic_public_cases.jsonl", public_cases)
    pd.DataFrame(expected_rows).to_csv(interim / "synthetic_expected_key_private.csv", index=False)
    summary = {
        "case_count": int(config["case_count"]),
        "public_case_rows": len(public_cases),
        "expected_key_rows": len(expected_rows),
        "linked_observations": {condition: linked_count(config, condition) for condition in conditions},
        "price_tape_design": design,
        "api_calls_made": 0,
    }
    (interim / "synthetic_generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    # The public prompt assets must contain no simulated payoff realization.
    rendered = json.dumps(public_cases)
    if "hidden_latent_theta" in rendered:
        raise AssertionError("Synthetic public asset leaked hidden latent theta.")
    return summary


def load_assets(config: dict[str, Any]) -> tuple[dict[tuple[str, str], dict[str, Any]], pd.DataFrame]:
    interim = resolve_repo_path(config["interim_root"])
    public_cases = {
        (str(value["case_id"]), str(value["information_condition"])): value["public_case"]
        for value in read_jsonl(interim / "synthetic_public_cases.jsonl")
    }
    expected = pd.read_csv(interim / "synthetic_expected_key_private.csv")
    return public_cases, expected


def clue_block(clue: float | None, noise_sd: float) -> str:
    if clue is None:
        return "No independent private clue is supplied. Use the public tape only."
    return (
        "An independent synthetic private research clue follows s = theta + xi, with xi independent of "
        f"the public tape and xi standard deviation {noise_sd:.1f}. The observed clue is s = {clue:+.1f}."
    )


def request_item(
    config: dict[str, Any], model_key: str, case_id: str, condition: str, public_case: dict[str, Any],
    clue: float | None, replicate_id: str,
) -> dict[str, Any]:
    model = str(config["models"][model_key]["model"])
    clue_label = clue_key(clue).replace("+", "plus").replace("-", "minus").replace(".", "p")
    scenario_id = f"SYNV1_{case_id}_{condition}_{model_key}_{replicate_id}_{clue_label}"
    user = render(
        (REPO_ROOT / "prompt_templates" / "step4_synthetic_v1_user.md").read_text().strip(),
        {
            "SCENARIO_JSON": json.dumps(public_case, sort_keys=True, separators=(",", ":")),
            "CLUE_BLOCK": clue_block(clue, float(config["private_clue_noise_sd"])),
        },
    )
    payload = {
        "model": model,
        "reasoning": reasoning_configuration(model, str(config["models"][model_key].get("reasoning_effort", config["execution"]["reasoning_effort"]))),
        "max_output_tokens": int(config["execution"]["max_output_tokens"]),
        "store": bool(config["execution"]["store"]),
        "metadata": {"experiment": str(config["experiment_id"]), "scenario_id": scenario_id, "replicate_id": replicate_id},
        "text": {"format": response_format()},
        "input": [
            {"role": "system", "content": (REPO_ROOT / "prompt_templates" / "step4_synthetic_v1_system.md").read_text().strip()},
            {"role": "user", "content": user},
        ],
    }
    return {
        "scenario_id": scenario_id,
        "case_id": case_id,
        "information_condition": condition,
        "model_key": model_key,
        "requested_model": model,
        "replicate_id": replicate_id,
        "clue_key": clue_key(clue),
        "response_kind": "posterior",
        "connection_retry_attempts": int(config["execution"].get("connection_retry_attempts", 1)),
        "request": payload,
    }


def build_requests(config: dict[str, Any], model_key: str) -> list[dict[str, Any]]:
    public_cases, _ = load_assets(config)
    result: list[dict[str, Any]] = []
    for replicate in range(1, int(config["replicates"]) + 1):
        replicate_id = f"r{replicate}"
        for (case_id, condition), public_case in sorted(public_cases.items()):
            for clue in [None, *[float(value) for value in config["private_clue_values"]]]:
                result.append(request_item(config, model_key, case_id, condition, public_case, clue, replicate_id))
    ids = [str(value["scenario_id"]) for value in result]
    if len(ids) != len(set(ids)):
        raise AssertionError("Synthetic design produced duplicate scenario IDs.")
    return result


def existing_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    """Return one usable record per scenario, retaining failed attempts in raw JSONL.

    Transient connection failures are not experimental observations.  A later
    completed, schema-valid retry replaces an earlier failure for resume and
    analysis purposes; repeated valid responses never create a new draw.
    """
    by_id: dict[str, dict[str, Any]] = {}
    for value in read_jsonl(path):
        scenario_id = str(value["scenario_id"])
        current = by_id.get(scenario_id)
        valid = value.get("status") == "completed" and value.get("parse_status") == "valid"
        current_valid = current is not None and current.get("status") == "completed" and current.get("parse_status") == "valid"
        if current is None or (valid and not current_valid):
            by_id[scenario_id] = value
    return list(by_id.values())


def preflight(config: dict[str, Any], model_key: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
    model = config["models"][model_key]
    input_chars = sum(len(json.dumps(item["request"]["input"], separators=(",", ":"))) for item in requests)
    input_tokens = math.ceil(input_chars / 4)
    output_tokens = len(requests) * int(config["execution"]["max_output_tokens"])
    input_cost = input_tokens * float(model["input_usd_per_million_tokens"]) / 1_000_000
    output_cost = output_tokens * float(model["output_usd_per_million_tokens"]) / 1_000_000
    return {
        "experiment_id": config["experiment_id"],
        "model_key": model_key,
        "model": model["model"],
        "case_count": int(config["case_count"]),
        "request_count": len(requests),
        "request_count_by_condition": {condition: sum(value["information_condition"] == condition for value in requests) for condition in config["information_conditions"]},
        "estimated_input_cost_usd": input_cost,
        "maximum_possible_cost_usd": input_cost + output_cost,
        "configured_cost_cap_usd": float(config["maximum_allowed_cost_usd"][model_key]),
    }


def parse_response(response: Any, item: dict[str, Any]) -> dict[str, Any]:
    raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    result = {
        **{key: item[key] for key in ["scenario_id", "case_id", "information_condition", "model_key", "requested_model", "replicate_id", "clue_key", "response_kind"]},
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": raw.get("status"),
        "model": raw.get("model"),
        "usage": raw.get("usage"),
        "raw_response": raw,
    }
    try:
        parsed = json.loads(getattr(response, "output_text", raw.get("output_text", "")))
        mean = float(parsed["posterior_mean"])
        sd = float(parsed["posterior_sd"])
        if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
            raise ValueError("non-finite posterior or non-positive posterior SD")
        result.update({"posterior_mean": mean, "posterior_sd": sd, "parse_status": "valid"})
    except Exception as exc:
        result.update({"parse_status": "invalid", "parse_error": str(exc)})
    return result


def invoke(item: dict[str, Any]) -> dict[str, Any]:
    from openai import APIConnectionError, OpenAI

    attempts = max(1, int(item.get("connection_retry_attempts", 1)))
    client = OpenAI()
    for attempt in range(1, attempts + 1):
        try:
            result = parse_response(client.responses.create(**item["request"]), item)
            result["connection_attempts"] = attempt
            return result
        except APIConnectionError as exc:
            if attempt < attempts:
                time.sleep(float(2 ** (attempt - 1)))
                continue
            error = exc
        except Exception as exc:
            error = exc
            break
    return {
        **{key: item[key] for key in ["scenario_id", "case_id", "information_condition", "model_key", "requested_model", "replicate_id", "clue_key", "response_kind"]},
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "api_error",
        "parse_status": "not_parsed",
        "error_type": type(error).__name__,
        "error_message": str(error),
        "connection_attempts": attempts,
    }


def execute(requests: list[dict[str, Any]], raw_path: Path, max_workers: int, quiet: bool) -> dict[str, int]:
    prior = existing_records(raw_path)
    completed = {
        str(value["scenario_id"])
        for value in prior
        if value.get("status") == "completed" and value.get("parse_status") == "valid"
    }
    pending = [value for value in requests if str(value["scenario_id"]) not in completed]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a") as handle, concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(invoke, value): value for value in pending}
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            result = future.result()
            handle.write(json.dumps(result, allow_nan=False) + "\n")
            handle.flush()
            if not quiet:
                print(json.dumps({"completed_this_run": number, "remaining": len(pending) - number, "scenario_id": result["scenario_id"]}), flush=True)
    records = existing_records(raw_path)
    return {
        "records_total": len(records),
        "records_written_this_run": len(pending),
        "completed": int(sum(value.get("status") == "completed" for value in records)),
        "valid": int(sum(value.get("parse_status") == "valid" for value in records)),
    }


def bootstrap(values: pd.Series, repetitions: int, seed: int) -> tuple[float, float, float]:
    sample = values.dropna().to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    draws = sample[rng.integers(0, len(sample), size=(repetitions, len(sample)))].mean(axis=1)
    return float(sample.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    printable = frame.copy()
    for column in printable.columns:
        if pd.api.types.is_float_dtype(printable[column]):
            printable[column] = printable[column].map(lambda value: f"{value:.4f}" if pd.notna(value) else "")
    cols = [str(column) for column in printable.columns]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    lines.extend("| " + " | ".join(str(value).replace("|", "\\|") for value in row) + " |" for row in printable.itertuples(index=False, name=None))
    return "\n".join(lines)


def analyze(config: dict[str, Any], output: Path, model_keys: list[str] | None = None) -> dict[str, Any]:
    raw_path = output / RAW_NAME
    records = pd.DataFrame(existing_records(raw_path))
    if records.empty:
        raise FileNotFoundError(f"No synthetic raw output found at {raw_path}")
    selected_models = list(config["models"]) if model_keys is None else model_keys
    unknown_models = sorted(set(selected_models) - set(config["models"]))
    if unknown_models:
        raise ValueError(f"Unknown model key(s) in analysis: {unknown_models}")
    records = records.loc[records.model_key.isin(selected_models)].copy()
    expected_count = int(config["case_count"]) * len(config["information_conditions"]) * 3 * len(selected_models) * int(config["replicates"])
    if len(records) != expected_count or not records.parse_status.eq("valid").all():
        raise ValueError("Synthetic run is incomplete or has invalid structured outputs.")
    _, expected = load_assets(config)
    merged = records.merge(expected.drop(columns=["hidden_latent_theta"]), on=["case_id", "information_condition", "clue_key"], validate="many_to_one")
    merged["posterior_mean_abs_error"] = (merged.posterior_mean - merged.expected_posterior_mean).abs()
    merged["posterior_sd_abs_error"] = (merged.posterior_sd - merged.expected_posterior_sd).abs()
    public = merged.loc[merged.clue_key.eq("public_only")].copy()
    public_summary = (
        public.groupby("model_key", as_index=False)
        .agg(cases=("case_id", "size"), mean_posterior_mean_abs_error=("posterior_mean_abs_error", "mean"), mean_posterior_sd_abs_error=("posterior_sd_abs_error", "mean"))
    )
    private = merged.loc[merged.clue_key.ne("public_only")].copy()
    means = private.pivot_table(index=["case_id", "information_condition", "model_key", "replicate_id"], columns="clue_key", values="posterior_mean")
    if not {"-0.8", "+0.8"}.issubset(means.columns):
        raise ValueError("Synthetic private-clue grid is incomplete.")
    beta = ((means["+0.8"] - means["-0.8"]) / 1.6).rename("model_private_clue_slope").reset_index()
    beta = beta.merge(expected[["case_id", "information_condition", "theoretical_private_clue_slope"]].drop_duplicates(), on=["case_id", "information_condition"], validate="many_to_one")
    beta = beta.groupby(["case_id", "information_condition", "model_key", "theoretical_private_clue_slope"], as_index=False).agg(model_private_clue_slope=("model_private_clue_slope", "mean"))
    beta["slope_error"] = beta.model_private_clue_slope - beta.theoretical_private_clue_slope
    slope_summary = (
        beta.groupby(["model_key", "information_condition"], as_index=False)
        .agg(cases=("case_id", "size"), theoretical_slope=("theoretical_private_clue_slope", "mean"), model_slope=("model_private_clue_slope", "mean"), mean_slope_error=("slope_error", "mean"), mean_abs_slope_error=("slope_error", lambda values: float(values.abs().mean())))
    )
    contrasts: list[dict[str, Any]] = []
    for model_key, group in beta.groupby("model_key"):
        wide = group.pivot(index="case_id", columns="information_condition", values="model_private_clue_slope")
        for label, left, right in [
            ("high_minus_low", "high_public_precision", "low_public_precision"),
            ("high_minus_placebo", "high_public_precision", "placebo_public_tape"),
        ]:
            values = wide[left] - wide[right]
            mean, lower, upper = bootstrap(values, int(config["analysis"]["bootstrap_repetitions"]), int(config["analysis"]["bootstrap_seed"]) + len(contrasts))
            contrasts.append({"model_key": model_key, "contrast": label, "cases": int(values.notna().sum()), "mean": mean, "ci_95_lower": lower, "ci_95_upper": upper, "negative_cases": int((values < 0).sum())})
    costs: dict[str, float] = {}
    for model_key, group in records.groupby("model_key"):
        input_tokens = sum(float((value or {}).get("input_tokens", 0)) for value in group.usage)
        output_tokens = sum(float((value or {}).get("output_tokens", 0)) for value in group.usage)
        prices = config["models"][model_key]
        costs[model_key] = (input_tokens * float(prices["input_usd_per_million_tokens"]) + output_tokens * float(prices["output_usd_per_million_tokens"])) / 1_000_000
    output.mkdir(parents=True, exist_ok=True)
    public_summary.to_csv(output / "synthetic_public_posterior_accuracy.csv", index=False)
    slope_summary.to_csv(output / "synthetic_private_clue_slopes.csv", index=False)
    pd.DataFrame(contrasts).to_csv(output / "synthetic_slope_contrasts.csv", index=False)
    result = {"analysis_model_keys": selected_models, "actual_api_cost_usd": costs, "public_accuracy": public_summary.to_dict(orient="records"), "slopes": slope_summary.to_dict(orient="records"), "contrasts": contrasts}
    (output / "synthetic_summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    report = "\n".join([
        f"# E2 synthetic benchmark: {config['experiment_id']}",
        "",
        "This is a synthetic mechanism benchmark. The unit of inference is the synthetic case.",
        f"Analysed model conditions: {', '.join(selected_models)}.",
        "",
        "## Public posterior accuracy",
        "",
        markdown_table(public_summary),
        "",
        "## Private-clue slopes versus exact Gaussian benchmark",
        "",
        markdown_table(slope_summary),
        "",
        "## Within-model slope contrasts",
        "",
        markdown_table(pd.DataFrame(contrasts)),
        "",
        f"Recorded API cost: ${sum(costs.values()):.4f}.",
    ])
    (output / "synthetic_report.md").write_text(report)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build", "dry-run", "execute", "analyze"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--model-key")
    parser.add_argument("--output-subdir", default="run_v1")
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--model-keys", nargs="+", help="For analysis, restrict to complete named model conditions.")
    parser.add_argument("--acknowledge-paid-api-calls", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    if args.command == "build":
        print(json.dumps(build_assets(config), allow_nan=False))
        return
    build_assets(config)
    output = resolve_repo_path(config["output_root"]) / args.output_subdir
    if args.command == "analyze":
        print(json.dumps(analyze(config, output, args.model_keys), allow_nan=False))
        return
    if args.model_key is None:
        raise ValueError("--model-key is required for dry-run and execute.")
    if args.model_key not in config["models"]:
        raise ValueError("--model-key is not defined in the selected config.")
    requests = build_requests(config, args.model_key)
    preview = preflight(config, args.model_key, requests)
    output.mkdir(parents=True, exist_ok=True)
    (output / f"synthetic_preflight_{args.model_key}.json").write_text(json.dumps(preview, indent=2) + "\n")
    if args.command == "dry-run":
        print(json.dumps({"network_calls_made": 0, **preview}, allow_nan=False))
        return
    if not args.acknowledge_paid_api_calls:
        raise PermissionError("execute requires --acknowledge-paid-api-calls")
    if preview["maximum_possible_cost_usd"] > preview["configured_cost_cap_usd"]:
        raise ValueError("Maximum possible cost exceeds the configured cap.")
    workers = int(args.max_workers or config["execution"]["max_workers"])
    if workers < 1 or workers > 16:
        raise ValueError("--max-workers must be between 1 and 16.")
    result = execute(requests, output / RAW_NAME, workers, args.quiet)
    (output / "synthetic_execution_summary.json").write_text(json.dumps({**preview, **result}, indent=2) + "\n")
    print(json.dumps({**preview, **result}, allow_nan=False))


if __name__ == "__main__":
    main()
