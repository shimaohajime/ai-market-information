"""Cross-check I/B/E/S earnings-announcement dates against Compustat RDQ.

This is a timing audit, not a replacement source for the primary event date.
It appends the closest same-PERMNO Compustat RDQ and flags agreement within
0, 1, 3, 7, and 30 calendar days.  The one-day matched sample can then be
passed to the Step 2 market-expectations runner as a high-confidence timing
robustness sample.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel.parquet"
DEFAULT_DATA_ROOT = Path(os.environ.get("AI_MARKET_DATA_DIR", REPO_ROOT / "data" / "raw")).expanduser()
DEFAULT_CRSP_ROOT = Path(os.environ.get("AI_MARKET_CRSP_ROOT", DEFAULT_DATA_ROOT / "crsp")).expanduser()
DEFAULT_COMP_Q = DEFAULT_CRSP_ROOT / "compustat_quaterly.csv.gz"
DEFAULT_OUTPUT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel_rdq.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "output/step2"


def lower_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


def read_compustat_quarterly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing Compustat Quarterly source: {path}")
    usecols = [
        "GVKEY", "LPERMNO", "LINKPRIM", "LINKTYPE", "LINKDT", "LINKENDDT", "datadate", "rdq",
        "indfmt", "consol", "popsrc", "datafmt", "curcdq", "fic",
    ]
    raw = lower_columns(pd.read_csv(path, usecols=usecols, low_memory=False))
    for column in ["datadate", "rdq", "linkdt", "linkenddt"]:
        raw[column] = pd.to_datetime(raw[column].replace({"E": None, "B": None}), errors="coerce")
    raw["lpermno"] = pd.to_numeric(raw.lpermno, errors="coerce")
    raw["linkenddt"] = raw.linkenddt.fillna(pd.Timestamp("2262-04-11"))
    selected = raw.loc[
        raw.indfmt.eq("INDL")
        & raw.consol.eq("C")
        & raw.popsrc.eq("D")
        & raw.datafmt.eq("STD")
        & raw.curcdq.eq("USD")
        & raw.fic.eq("USA")
        & raw.linkprim.isin(["P", "C"])
        & raw.linktype.isin(["LC", "LU"])
        & raw.lpermno.notna()
        & raw.rdq.notna()
    ].copy()
    selected["permno"] = selected.lpermno.astype("int64")
    selected = selected.loc[selected.linkdt.le(selected.rdq) & selected.rdq.le(selected.linkenddt)].copy()
    selected = selected.sort_values(["permno", "rdq", "datadate", "gvkey"]).drop_duplicates(
        ["permno", "rdq", "datadate"], keep="last"
    )
    return selected[["permno", "gvkey", "datadate", "rdq"]]


def append_rdq(panel: pd.DataFrame, comp: pd.DataFrame, max_gap_days: int) -> pd.DataFrame:
    panel = panel.copy()
    panel["announcement_date"] = pd.to_datetime(panel.announcement_date, errors="coerce")
    panel["permno"] = pd.to_numeric(panel.permno, errors="coerce").astype("Int64")
    events = panel[["event_id", "permno", "announcement_date"]].dropna().copy()
    candidates = events.merge(comp, on="permno", how="left", validate="many_to_many")
    candidates["rdq_ibes_calendar_day_gap"] = (candidates.rdq - candidates.announcement_date).dt.days
    candidates["abs_rdq_ibes_calendar_day_gap"] = candidates.rdq_ibes_calendar_day_gap.abs()
    candidates = candidates.loc[candidates.abs_rdq_ibes_calendar_day_gap.le(max_gap_days)].copy()
    candidates = candidates.sort_values(
        ["event_id", "abs_rdq_ibes_calendar_day_gap", "datadate", "gvkey"],
        ascending=[True, True, False, True],
    ).drop_duplicates("event_id", keep="first")
    keep = ["event_id", "gvkey", "datadate", "rdq", "rdq_ibes_calendar_day_gap", "abs_rdq_ibes_calendar_day_gap"]
    output = panel.merge(candidates[keep], on="event_id", how="left", validate="one_to_one")
    output = output.rename(columns={"gvkey": "rdq_gvkey", "datadate": "rdq_datadate", "rdq": "rdq_date"})
    for gap in [0, 1, 3, 7, 30]:
        output[f"rdq_match_within_{gap}_calendar_day"] = output.abs_rdq_ibes_calendar_day_gap.le(gap).fillna(False)
    return output


def diagnostics(panel: pd.DataFrame) -> pd.DataFrame:
    rows = [
        {"metric": "event_panel_rows", "value": len(panel)},
        {"metric": "any_rdq_match_within_30_calendar_days", "value": int(panel.rdq_date.notna().sum())},
    ]
    for gap in [0, 1, 3, 7, 30]:
        column = f"rdq_match_within_{gap}_calendar_day"
        rows.append({"metric": column, "value": int(panel[column].sum())})
        rows.append({"metric": f"{column}_rate", "value": float(panel[column].mean())})
    matched = panel.loc[panel.rdq_date.notna(), "rdq_ibes_calendar_day_gap"]
    rows.extend([
        {"metric": "rdq_ibes_gap_median_calendar_days", "value": float(matched.median()) if len(matched) else np.nan},
        {"metric": "rdq_ibes_gap_mean_abs_calendar_days", "value": float(matched.abs().mean()) if len(matched) else np.nan},
    ])
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> None:
    if not args.panel.exists():
        raise FileNotFoundError(f"Missing Step 2 event panel: {args.panel}")
    panel = pd.read_parquet(args.panel)
    comp = read_compustat_quarterly(args.compustat_quarterly)
    output = append_rdq(panel, comp, args.max_gap_days)
    args.output_panel.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_panel, index=False, compression="zstd")
    result = diagnostics(output)
    result.to_csv(args.output_dir / "rdq_ibes_timing_diagnostics.csv", index=False)
    print(result.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--compustat-quarterly", type=Path, default=DEFAULT_COMP_Q)
    parser.add_argument("--output-panel", type=Path, default=DEFAULT_OUTPUT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-gap-days", type=int, default=30)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
