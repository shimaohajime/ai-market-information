"""Run the locked E2 development experiment.

All calls are fresh Responses API requests.  The runner supports only the
pre-specified competence, public-context, and behavioral conditions,
records explicit replicate IDs, and safely resumes an interrupted run without
duplicating a scenario.
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

from step4_phase0 import REPO_ROOT, read_config, resolve_repo_path


DEFAULT_CONFIG = REPO_ROOT / "config" / "step4_v3_task12.json"
RAW_NAME = "step4_v3_raw_responses.jsonl"
FORECAST_SCHEMA = {
    "name": "step4_v3_forecast",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "forecast_mean": {"type": "number"},
            "forecast_sd": {"type": "number"},
        },
        "required": ["forecast_mean", "forecast_sd"],
    },
}
ANSWER_SCHEMA = {
    "name": "step4_v3_numeric_answer",
    "strict": True,
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "properties": {"answer": {"type": "number"}},
        "required": ["answer"],
    },
}
PROHIBITED_FIELDS = ["permno", "cusip", "ticker", "formation_date", "next_sue", "theta_ex_ante", "stock_price"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def safe_subdir(value: str) -> Path:
    result = Path(value)
    if result.name != value or value in {"", ".", ".."}:
        raise ValueError("--output-subdir must be a single safe directory name.")
    return result


def render(template: str, replacements: dict[str, str]) -> str:
    result = template
    for key, value in replacements.items():
        result = result.replace("{{" + key + "}}", value)
    if re.search(r"\{\{[A-Z_]+\}\}", result):
        raise ValueError("Prompt template contains an unresolved placeholder.")
    return result


def response_format(schema: dict[str, Any]) -> dict[str, Any]:
    return {"type": "json_schema", "name": schema["name"], "strict": schema["strict"], "schema": schema["schema"]}


def reasoning_configuration(model: str, effort: str) -> dict[str, str]:
    if model.startswith("gpt-5.2"):
        return {"effort": effort}
    return {"mode": "standard", "effort": effort}


def price_for_condition(dossier: dict[str, Any], condition: str) -> dict[str, Any] | None:
    if condition == "no_price":
        return None
    source = {
        "informative_price_path": "informative_price_information",
        "matched_placebo_path": "matched_placebo_price_information",
        "time_permuted_price_path": "time_permuted_price_information",
    }.get(condition)
    if source is None:
        raise ValueError(f"Unsupported information condition {condition!r}.")
    price = dict(dossier[source])
    price.pop("price_summary", None)
    return price


def price_block(price: dict[str, Any] | None) -> str:
    if price is None:
        return "No additional historical price path is supplied."
    return "Additional anonymous pre-cutoff historical price path:\n\n" + json.dumps(
        {"historical_price_path": price}, sort_keys=True, separators=(",", ":")
    )


def clue_block(signal: float | None, noise_sd: float | None) -> str:
    if signal is None:
        return "No separate private research clue is supplied."
    return (
        "Synthetic private clue (research intervention only): an independent research source reports "
        f"a next-surprise nowcast of {signal:+.1f}. Its zero-mean measurement-error standard deviation is "
        f"{noise_sd:.1f} in the same standardized unit. The clue is not a re-summary of the dossier and is "
        "not real nonpublic information."
    )


def no_leak(payload: dict[str, Any], label: str) -> None:
    rendered = json.dumps(payload, sort_keys=True)
    leaked = [field for field in PROHIBITED_FIELDS if f'"{field}"' in rendered]
    if leaked:
        raise AssertionError(f"{label} leaks prohibited fields: {leaked}")


def load_assets(config: dict[str, Any]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    interim = resolve_repo_path(config["interim_root"])
    dossier_name = str(config.get("dossier_filename", "step4_v3_task12_dossiers.jsonl"))
    check_name = str(config.get("checks_filename", "step4_v3_competence_checks.csv"))
    dossiers = {item["dossier_id"]: item for item in read_jsonl(interim / dossier_name)}
    checks = pd_read_csv_records(interim / check_name)
    return dossiers, checks


def pd_read_csv_records(path: Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_csv(path).to_dict(orient="records")


def make_forecast_item(
    config: dict[str, Any], model_key: str, task: str, dossier: dict[str, Any], condition: str,
    replicate_id: str, signal: float | None, noise_sd: float | None,
) -> dict[str, Any]:
    model = str(config["models"][model_key]["model"])
    public_json = json.dumps(
        {"dossier_id": dossier["dossier_id"], "public_information": dossier["public_information"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    user = render(
        (REPO_ROOT / "prompt_templates" / "step4_v3_forecast_user.md").read_text().strip(),
        {
            "PUBLIC_JSON": public_json,
            "PRICE_BLOCK": price_block(price_for_condition(dossier, condition)),
            "CLUE_BLOCK": clue_block(signal, noise_sd),
        },
    )
    scenario_id = f"V3_{task}_{dossier['dossier_id']}_{model_key}_{condition}_{replicate_id}"
    if signal is not None:
        scenario_id += f"_{signal:+.1f}_{noise_sd:.1f}"
    payload = {
        "model": model,
        "reasoning": reasoning_configuration(model, str(config["execution"]["reasoning_effort"])),
        "max_output_tokens": int(config["execution"]["max_output_tokens"]),
        "store": bool(config["execution"]["store"]),
        "metadata": {"experiment": str(config.get("experiment_id", "step4_v3_task12")), "task": task, "scenario_id": scenario_id, "replicate_id": replicate_id},
        "text": {"format": response_format(FORECAST_SCHEMA)},
        "input": [
            {"role": "system", "content": (REPO_ROOT / "prompt_templates" / "step4_v3_forecast_system.md").read_text().strip()},
            {"role": "user", "content": user},
        ],
    }
    no_leak(payload["input"], scenario_id)
    return {
        "scenario_id": scenario_id,
        "task": task,
        "dossier_id": dossier["dossier_id"],
        "requested_model": model,
        "model_key": model_key,
        "replicate_id": replicate_id,
        "information_condition": condition,
        "signal_value": signal,
        "signal_noise_sd": noise_sd,
        "response_kind": "forecast",
        "connection_retry_attempts": int(config["execution"].get("connection_retry_attempts", 1)),
        "request": payload,
    }


def make_check_item(
    config: dict[str, Any], model_key: str, dossier: dict[str, Any], check: dict[str, Any], replicate_id: str,
) -> dict[str, Any]:
    model = str(config["models"][model_key]["model"])
    condition = str(check["information_condition"])
    public_json = json.dumps(
        {"dossier_id": dossier["dossier_id"], "public_information": dossier["public_information"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    user = render(
        (REPO_ROOT / "prompt_templates" / "step4_v3_check_user.md").read_text().strip(),
        {
            "PUBLIC_JSON": public_json,
            "PRICE_BLOCK": price_block(price_for_condition(dossier, condition)),
            "QUESTION": str(check["question"]),
        },
    )
    scenario_id = f"V3_competence_{dossier['dossier_id']}_{check['check_id']}_{model_key}_{replicate_id}"
    payload = {
        "model": model,
        "reasoning": reasoning_configuration(model, str(config["execution"]["reasoning_effort"])),
        "max_output_tokens": int(config["execution"]["max_output_tokens"]),
        "store": bool(config["execution"]["store"]),
        "metadata": {"experiment": str(config.get("experiment_id", "step4_v3_task12")), "task": "competence", "scenario_id": scenario_id, "replicate_id": replicate_id},
        "text": {"format": response_format(ANSWER_SCHEMA)},
        "input": [
            {"role": "system", "content": (REPO_ROOT / "prompt_templates" / "step4_v3_check_system.md").read_text().strip()},
            {"role": "user", "content": user},
        ],
    }
    no_leak(payload["input"], scenario_id)
    return {
        "scenario_id": scenario_id,
        "task": "competence",
        "dossier_id": dossier["dossier_id"],
        "check_id": str(check["check_id"]),
        "check_type": str(check["check_type"]),
        "expected_numeric_answer": float(check["expected_numeric_answer"]),
        "requested_model": model,
        "model_key": model_key,
        "replicate_id": replicate_id,
        "information_condition": condition,
        "signal_value": None,
        "signal_noise_sd": None,
        "response_kind": "answer",
        "connection_retry_attempts": int(config["execution"].get("connection_retry_attempts", 1)),
        "request": payload,
    }


def build_requests(config: dict[str, Any], task: str, model_key: str) -> list[dict[str, Any]]:
    if task not in config["tasks"]:
        raise ValueError(f"Unknown task {task!r}.")
    if model_key not in config["models"]:
        raise ValueError(f"Unknown model key {model_key!r}.")
    dossiers, checks = load_assets(config)
    task_config = config["tasks"][task]
    replicate_ids = [f"r{value}" for value in range(1, int(task_config["replicates"]) + 1)]
    requests: list[dict[str, Any]] = []
    if task == "competence":
        for replicate_id in replicate_ids:
            for check in checks:
                requests.append(make_check_item(config, model_key, dossiers[str(check["dossier_id"])], check, replicate_id))
    elif task == "public_forecast":
        for replicate_id in replicate_ids:
            for dossier in dossiers.values():
                for condition in task_config["price_conditions"]:
                    requests.append(make_forecast_item(config, model_key, task, dossier, condition, replicate_id, None, None))
    elif task == "mechanism":
        for replicate_id in replicate_ids:
            for dossier in dossiers.values():
                for condition in task_config["price_conditions"]:
                    for signal in task_config["signal_values"]:
                        requests.append(
                            make_forecast_item(
                                config, model_key, task, dossier, condition, replicate_id,
                                float(signal), float(task_config["signal_noise_sd"]),
                            )
                        )
    ids = [str(item["scenario_id"]) for item in requests]
    if len(ids) != len(set(ids)):
        raise AssertionError("E2 design has duplicate scenario IDs.")
    return requests


def existing_records(raw_path: Path) -> list[dict[str, Any]]:
    if not raw_path.exists():
        return []
    # A UI timeout can leave an in-flight shell process alive.  If a user then
    # starts a second process, raw output can contain repeated scenario IDs.
    # Preserve the complete raw audit trail, but retain only the first completed
    # pre-specified draw for all resume and analysis decisions.
    records_by_id: dict[str, dict[str, Any]] = {}
    for item in read_jsonl(raw_path):
        scenario_id = str(item["scenario_id"])
        current = records_by_id.get(scenario_id)
        valid = item.get("status") == "completed" and item.get("parse_status") == "valid"
        current_valid = current is not None and current.get("status") == "completed" and current.get("parse_status") == "valid"
        if current is None or (valid and not current_valid):
            records_by_id[scenario_id] = item
    return list(records_by_id.values())


def cost_preview(config: dict[str, Any], task: str, model_key: str, requests: list[dict[str, Any]]) -> dict[str, Any]:
    model_config = config["models"][model_key]
    char_count = sum(len(json.dumps(item["request"]["input"], separators=(",", ":"))) for item in requests)
    approximate_input_tokens = math.ceil(char_count / 4)
    max_output_tokens = len(requests) * int(config["execution"]["max_output_tokens"])
    input_cost = approximate_input_tokens * float(model_config["input_usd_per_million_tokens"]) / 1_000_000
    max_output_cost = max_output_tokens * float(model_config["output_usd_per_million_tokens"]) / 1_000_000
    return {
        "task": task,
        "model_key": model_key,
        "model": model_config["model"],
        "dossier_count": len({str(item["dossier_id"]) for item in requests}),
        "request_count": len(requests),
        "request_count_by_information_condition": {
            condition: sum(item["information_condition"] == condition for item in requests)
            for condition in sorted({str(item["information_condition"]) for item in requests})
        },
        "input_characters": char_count,
        "approximate_input_tokens": approximate_input_tokens,
        "max_output_tokens": max_output_tokens,
        "estimated_input_cost_usd": input_cost,
        "maximum_possible_cost_usd": input_cost + max_output_cost,
        "configured_cost_cap_usd": float(config["tasks"][task]["maximum_allowed_cost_usd"][model_key]),
    }


def parse_response(response: Any, item: dict[str, Any]) -> dict[str, Any]:
    raw = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    record = {
        **{key: item.get(key) for key in [
            "scenario_id", "task", "dossier_id", "check_id", "check_type", "expected_numeric_answer",
            "requested_model", "model_key", "replicate_id", "information_condition", "signal_value", "signal_noise_sd",
            "response_kind",
        ] if key in item},
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": raw.get("status"),
        "model": raw.get("model"),
        "usage": raw.get("usage"),
        "raw_response": raw,
    }
    try:
        parsed = json.loads(getattr(response, "output_text", raw.get("output_text", "")))
        if item["response_kind"] == "forecast":
            mean, sd = float(parsed["forecast_mean"]), float(parsed["forecast_sd"])
            if not math.isfinite(mean) or not math.isfinite(sd) or sd <= 0:
                raise ValueError("non-finite forecast or non-positive forecast SD")
            record.update({"forecast_mean": mean, "forecast_sd": sd, "parse_status": "valid"})
        else:
            answer = float(parsed["answer"])
            if not math.isfinite(answer):
                raise ValueError("non-finite numeric answer")
            record.update({"answer": answer, "parse_status": "valid"})
    except Exception as exc:
        record.update({"parse_status": "invalid", "parse_error": str(exc)})
    return record


def failed_record(item: dict[str, Any], exc: Exception) -> dict[str, Any]:
    return {
        **{key: item.get(key) for key in [
            "scenario_id", "task", "dossier_id", "check_id", "check_type", "expected_numeric_answer",
            "requested_model", "model_key", "replicate_id", "information_condition", "signal_value", "signal_noise_sd",
            "response_kind",
        ] if key in item},
        "completed_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "api_error",
        "parse_status": "not_parsed",
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def invoke(item: dict[str, Any]) -> dict[str, Any]:
    from openai import APIConnectionError, OpenAI

    attempts = max(1, int(item.get("connection_retry_attempts", 1)))
    client = OpenAI()
    for attempt in range(1, attempts + 1):
        try:
            record = parse_response(client.responses.create(**item["request"]), item)
            record["connection_attempts"] = attempt
            return record
        except APIConnectionError as exc:
            if attempt < attempts:
                time.sleep(float(2 ** (attempt - 1)))
                continue
            error = exc
        except Exception as exc:
            error = exc
            break
    record = failed_record(item, error)
    record["connection_attempts"] = attempts
    return record


def execute(
    requests: list[dict[str, Any]], raw_path: Path, max_workers: int, quiet: bool, max_new_requests: int | None,
) -> dict[str, int]:
    prior = existing_records(raw_path)
    completed_ids = {
        str(item["scenario_id"])
        for item in prior
        if item.get("status") == "completed" and item.get("parse_status") == "valid"
    }
    pending = [item for item in requests if str(item["scenario_id"]) not in completed_ids]
    if max_new_requests is not None:
        pending = pending[:max_new_requests]
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with raw_path.open("a") as handle, concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(invoke, item): item for item in pending}
        for number, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            record = future.result()
            handle.write(json.dumps(record, allow_nan=False) + "\n")
            handle.flush()
            if not quiet:
                print(json.dumps({"completed_this_run": number, "remaining": len(pending) - number, "scenario_id": record["scenario_id"]}), flush=True)
    records = existing_records(raw_path)
    return {
        "records_total": len(records),
        "records_written_this_run": len(pending),
        "completed": int(sum(item.get("status") == "completed" for item in records)),
        "valid": int(sum(item.get("parse_status") == "valid" for item in records)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["dry-run", "execute"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--task", choices=["competence", "public_forecast", "mechanism"], required=True)
    parser.add_argument("--model-key", choices=["gpt6", "gpt52"], required=True)
    parser.add_argument("--output-subdir", required=True)
    parser.add_argument("--max-workers", type=int)
    parser.add_argument("--max-new-requests", type=int, help="Run at most this many pending requests, then exit safely.")
    parser.add_argument("--quiet", action="store_true", help="Suppress per-request progress lines.")
    parser.add_argument("--acknowledge-paid-api-calls", action="store_true")
    args = parser.parse_args()
    config = read_config(args.config)
    output = resolve_repo_path(config["output_root"]) / safe_subdir(args.output_subdir)
    requests = build_requests(config, args.task, args.model_key)
    preview = cost_preview(config, args.task, args.model_key, requests)
    output.mkdir(parents=True, exist_ok=True)
    (output / "step4_v3_preflight.json").write_text(json.dumps(preview, indent=2) + "\n")
    if args.command == "dry-run":
        print(json.dumps({"network_calls_made": 0, **preview}, allow_nan=False))
        return
    if not args.acknowledge_paid_api_calls:
        raise PermissionError("execute requires --acknowledge-paid-api-calls")
    if preview["maximum_possible_cost_usd"] > preview["configured_cost_cap_usd"]:
        raise ValueError("The maximum possible API cost exceeds the configured per-model task cap.")
    workers = int(args.max_workers or config["execution"]["max_workers"])
    if workers < 1 or workers > 16:
        raise ValueError("--max-workers must be between 1 and 16.")
    if args.max_new_requests is not None and args.max_new_requests < 1:
        raise ValueError("--max-new-requests must be positive.")
    result = execute(requests, output / RAW_NAME, workers, args.quiet, args.max_new_requests)
    (output / "step4_v3_execution_summary.json").write_text(json.dumps({**preview, **result}, indent=2) + "\n")
    print(json.dumps({**preview, **result}, allow_nan=False))


if __name__ == "__main__":
    main()
