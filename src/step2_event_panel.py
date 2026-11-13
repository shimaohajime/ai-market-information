"""Prepare the E1 earnings-announcement event panel.

The public-data forecast comparison measures the predictive contribution of a
contemporaneous price under simulated limited and richer processors. This
event-time analysis asks how much earnings news is reflected in returns once
that price has formed. This runner joins each out-of-sample forecast observation to
the first CRSP trading date on or after its I/B/E/S announcement date.

I/B/E/S Summary contains dates but no release timestamps.  Consequently, the
primary return is market-adjusted CAR[0,+1], which captures a same-day or
after-market reaction without pretending to know the announcement time.
The output supports a counterfactual processing comparison, not an AI-adoption
or causal test.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("AI_MARKET_DATA_DIR", REPO_ROOT / "data" / "raw")).expanduser()
DEFAULT_CRSP_ROOT = Path(os.environ.get("AI_MARKET_CRSP_ROOT", DEFAULT_DATA_ROOT / "crsp")).expanduser()
DEFAULT_DAILY = DEFAULT_CRSP_ROOT / "crsp_daily_stock_2010_2025.csv.gz"
DEFAULT_SCORES = REPO_ROOT / "output/step1/oos_scores.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "output/step2"
DEFAULT_EVENT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel.parquet"


def sql_path(path: Path) -> str:
    """Quote a filesystem path for a DuckDB SQL string literal."""
    return path.as_posix().replace("'", "''")


def prepare_events(scores_path: Path, event_start: pd.Timestamp, event_end: pd.Timestamp) -> pd.DataFrame:
    """Keep one time-safe E1 forecast row for every earnings event."""
    if not scores_path.exists():
        raise FileNotFoundError(
            f"Missing E1 out-of-sample scores at {scores_path}. Run `python3 src/step1_pipeline.py run` first."
        )
    scores = pd.read_parquet(scores_path).copy()
    required = {
        "permno", "formation_date", "next_announcement_date", "next_sue",
        "S_limited_price_value", "F_high_compute_price_value", "V_compute_value_with_price",
    }
    missing = sorted(required.difference(scores.columns))
    if missing:
        raise ValueError(f"E1 score file is missing required columns: {', '.join(missing)}")
    scores["formation_date"] = pd.to_datetime(scores["formation_date"], errors="coerce")
    scores["next_announcement_date"] = pd.to_datetime(scores["next_announcement_date"], errors="coerce")
    scores["permno"] = pd.to_numeric(scores["permno"], errors="coerce").astype("Int64")
    events = scores.loc[
        scores.permno.notna()
        & scores.formation_date.notna()
        & scores.next_announcement_date.notna()
        & scores.next_announcement_date.between(event_start, event_end)
        & scores.formation_date.lt(scores.next_announcement_date)
    ].copy()
    # The Step 1 construction normally gives a unique key.  If an edge case
    # repeats an event, retain the latest information set that preceded it.
    events = events.sort_values(
        ["permno", "next_announcement_date", "formation_date"],
        ascending=[True, True, False],
    ).drop_duplicates(["permno", "next_announcement_date"], keep="first")
    events["permno"] = events.permno.astype("int64")
    events = events.rename(columns={"next_announcement_date": "announcement_date"})
    events = events.reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype=np.int64)
    return events


def query_event_days(events: pd.DataFrame, daily_path: Path) -> pd.DataFrame:
    """Attach CRSP [-1,+1] trading-day return rows without loading all CRSP into pandas."""
    if not daily_path.exists():
        raise FileNotFoundError(
            f"Missing CRSP daily file at {daily_path}. Download the CIZ Daily Stock File from WRDS "
            "and save it with this name, or pass --daily-path."
        )
    con = duckdb.connect()
    con.register("events", events)
    source = sql_path(daily_path)
    query = f"""
        WITH raw_daily AS (
            SELECT
                try_cast(trim(PERMNO) AS BIGINT) AS permno,
                try_cast(trim(DlyCalDt) AS DATE) AS trade_date,
                try_cast(trim(DlyRet) AS DOUBLE) AS dlyret,
                try_cast(trim(VWRETD) AS DOUBLE) AS vwretd,
                try_cast(trim(VWRETX) AS DOUBLE) AS vwretx,
                try_cast(trim(SPRTRN) AS DOUBLE) AS sprtrn,
                try_cast(trim(DlyCap) AS DOUBLE) AS dlycap,
                trim(PrimaryExch) AS primaryexch,
                trim(USIncFlg) AS usincflg,
                trim(IssuerType) AS issuertype,
                trim(SecurityType) AS securitytype,
                trim(SecuritySubType) AS securitysubtype,
                trim(ShareType) AS sharetype
            FROM read_csv_auto(
                '{source}', header = true, all_varchar = true, quote = '"',
                strict_mode = false, null_padding = true
            )
        ), daily AS (
            SELECT
                permno, trade_date, dlyret, vwretd, vwretx, sprtrn, dlycap,
                row_number() OVER (PARTITION BY permno ORDER BY trade_date) AS trading_index
            FROM raw_daily
            WHERE primaryexch IN ('N', 'A', 'Q')
              AND usincflg = 'Y'
              AND securitytype = 'EQTY'
              AND securitysubtype = 'COM'
              AND sharetype = 'NS'
              AND permno IS NOT NULL
              AND trade_date IS NOT NULL
        ), anchor_dates AS (
            SELECT e.event_id, min(d.trade_date) AS event_trade_date
            FROM events AS e
            LEFT JOIN daily AS d
              ON d.permno = e.permno
             AND d.trade_date >= CAST(e.announcement_date AS DATE)
             AND d.trade_date <= CAST(e.announcement_date AS DATE) + INTERVAL 7 DAYS
            GROUP BY e.event_id
        ), anchored AS (
            SELECT e.*, a.event_trade_date, d.trading_index AS event_trading_index
            FROM events AS e
            LEFT JOIN anchor_dates AS a USING (event_id)
            LEFT JOIN daily AS d
              ON d.permno = e.permno
             AND d.trade_date = a.event_trade_date
        )
        SELECT a.*, d.trade_date, d.dlyret, d.vwretd, d.vwretx, d.sprtrn, d.dlycap,
               d.trading_index - a.event_trading_index AS relative_trading_day
        FROM anchored AS a
        LEFT JOIN daily AS d
          ON d.permno = a.permno
         AND d.trading_index BETWEEN a.event_trading_index - 1 AND a.event_trading_index + 1
        ORDER BY a.event_id, relative_trading_day
    """
    days = con.execute(query).fetchdf()
    con.close()
    for column in ["formation_date", "announcement_date", "event_trade_date", "trade_date"]:
        days[column] = pd.to_datetime(days[column], errors="coerce")
    return days


def _window_metrics(group: pd.DataFrame, lower: int, upper: int, prefix: str) -> dict[str, object]:
    """Calculate returns only when every intended trading-day return is observed."""
    window = group.loc[group.relative_trading_day.between(lower, upper)].copy()
    expected_days = upper - lower + 1
    complete = (
        len(window) == expected_days
        and window.relative_trading_day.nunique() == expected_days
        and window[["dlyret", "vwretd"]].notna().all().all()
    )
    output: dict[str, object] = {f"complete_{prefix}": bool(complete), f"n_days_{prefix}": int(len(window))}
    if not complete:
        output.update({
            f"stock_return_{prefix}": np.nan,
            f"market_return_{prefix}": np.nan,
            f"car_market_adjusted_{prefix}": np.nan,
            f"bhar_market_adjusted_{prefix}": np.nan,
        })
        return output
    stock = float((1.0 + window.dlyret).prod() - 1.0)
    market = float((1.0 + window.vwretd).prod() - 1.0)
    output.update({
        f"stock_return_{prefix}": stock,
        f"market_return_{prefix}": market,
        f"car_market_adjusted_{prefix}": float((window.dlyret - window.vwretd).sum()),
        f"bhar_market_adjusted_{prefix}": float((1.0 + stock) / (1.0 + market) - 1.0),
    })
    return output


def summarize_event_days(days: pd.DataFrame) -> pd.DataFrame:
    """Collapse trading-day rows to the E1 event-level panel."""
    static = [
        "event_id", "permno", "formation_date", "formation_year", "formation_quarter", "announcement_date",
        "next_sue", "target_for_model", "log_market_equity_lag1", "analyst_coverage", "size_tercile",
        "analyst_coverage_tercile", "prediction_L0", "prediction_L1", "prediction_H0", "prediction_H1",
        "loss_L0", "loss_L1", "loss_H0", "loss_H1", "S_limited_price_value",
        "F_high_compute_price_value", "V_compute_value_with_price", "event_trade_date", "event_trading_index",
    ]
    available = [column for column in static if column in days]
    rows: list[dict[str, object]] = []
    for event_id, group in days.groupby("event_id", sort=False, dropna=False):
        row = group.iloc[0][available].to_dict()
        row["event_id"] = int(event_id)
        row["trading_date_anchor_found"] = bool(pd.notna(row.get("event_trade_date")))
        row.update(_window_metrics(group, 0, 1, "0_p1"))
        row.update(_window_metrics(group, -1, 1, "m1_p1"))
        rows.append(row)
    panel = pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)
    return panel


def write_diagnostics(panel: pd.DataFrame, events: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    diagnostics = {
        "step1_oos_events_in_requested_date_range": len(events),
        "event_panel_rows": len(panel),
        "event_panel_duplicate_event_ids": int(panel.event_id.duplicated().sum()),
        "event_panel_duplicate_permno_announcement_keys": int(panel.duplicated(["permno", "announcement_date"]).sum()),
        "formation_strictly_before_announcement": bool((panel.formation_date < panel.announcement_date).all()),
        "crsp_trading_date_anchor_rate": float(panel.trading_date_anchor_found.mean()),
        "complete_car_0_p1_rate": float(panel.complete_0_p1.mean()),
        "complete_car_m1_p1_rate": float(panel.complete_m1_p1.mean()),
        "primary_outcome": "car_market_adjusted_0_p1",
    }
    pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics.items()]).to_csv(
        output_dir / "event_panel_diagnostics.csv", index=False
    )
    (output_dir / "step2_data_note.md").write_text(
        "# E1 earnings-announcement event panel\n\n"
        "Each row is one out-of-sample forecast observation linked to its next I/B/E/S quarterly EPS announcement. "
        "The CRSP event date is the first trading date on or after the I/B/E/S date. "
        "Because I/B/E/S Summary has no intraday release time, `car_market_adjusted_0_p1` is the primary event-return measure; "
        "it is the sum of `DlyRet - VWRETD` on the anchor date and next trading date.\n\n"
        "This file prepares the information-content test. It does not identify actual adviser AI adoption or establish a causal adoption effect.\n"
    )


def build(args: argparse.Namespace) -> Path:
    event_start = pd.Timestamp(args.event_start)
    event_end = pd.Timestamp(args.event_end)
    if event_end < event_start:
        raise ValueError("--event-end must be on or after --event-start")
    events = prepare_events(args.scores_path, event_start, event_end)
    if events.empty:
        raise ValueError("No E1 out-of-sample events fall inside the requested event range.")
    print(f"Linking {len(events):,} E1 out-of-sample events to CRSP daily returns...", flush=True)
    days = query_event_days(events, args.daily_path)
    panel = summarize_event_days(days)
    args.event_panel.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(args.event_panel, index=False, compression="zstd")
    write_diagnostics(panel, events, args.output_dir)
    print(f"Wrote {len(panel):,} event rows to {args.event_panel}", flush=True)
    return args.event_panel


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["build"], nargs="?", default="build")
    parser.add_argument("--daily-path", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--scores-path", type=Path, default=DEFAULT_SCORES)
    parser.add_argument("--event-panel", type=Path, default=DEFAULT_EVENT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--event-start", default="2018-01-01")
    # The Annual Update daily extract ends 2025-12-31.  Ending on 12/30
    # guarantees that the primary [0,+1] window stays in the extract.
    parser.add_argument("--event-end", default="2025-12-30")
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
