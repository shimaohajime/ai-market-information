"""Add formation-to-announcement CRSP return measures to the Step 2 panel.

For each event, the formation anchor is the first CRSP trading date on or
after the Step 1 formation date.  Pre-announcement returns exclude both that
anchor and the announcement trading date.  The resulting measures distinguish
an early 20-trading-day response, a final 20-trading-day response, and the
entire variable-length interval before earnings are announced.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import duckdb
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel_rdq.parquet"
DEFAULT_DATA_ROOT = Path(os.environ.get("AI_MARKET_DATA_DIR", REPO_ROOT / "data" / "raw")).expanduser()
DEFAULT_CRSP_ROOT = Path(os.environ.get("AI_MARKET_CRSP_ROOT", DEFAULT_DATA_ROOT / "crsp")).expanduser()
DEFAULT_DAILY = DEFAULT_CRSP_ROOT / "crsp_daily_stock_2010_2025.csv.gz"
DEFAULT_OUTPUT_PANEL = REPO_ROOT / "data/interim/step2_earnings_event_panel_pre_returns.parquet"
DEFAULT_OUTPUT = REPO_ROOT / "output/step2"


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def calculate_preannouncement_returns(panel: pd.DataFrame, daily_path: Path) -> pd.DataFrame:
    if not daily_path.exists():
        raise FileNotFoundError(f"Missing CRSP daily file: {daily_path}")
    events = panel[["event_id", "permno", "formation_date", "event_trade_date"]].copy()
    events["formation_date"] = pd.to_datetime(events.formation_date, errors="coerce")
    events["event_trade_date"] = pd.to_datetime(events.event_trade_date, errors="coerce")
    events["permno"] = pd.to_numeric(events.permno, errors="coerce").astype("Int64")
    events = events.dropna().copy()
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
                trim(PrimaryExch) AS primaryexch,
                trim(USIncFlg) AS usincflg,
                trim(SecurityType) AS securitytype,
                trim(SecuritySubType) AS securitysubtype,
                trim(ShareType) AS sharetype
            FROM read_csv_auto(
                '{source}', header = true, all_varchar = true, quote = '"',
                strict_mode = false, null_padding = true
            )
        ), daily AS (
            SELECT permno, trade_date, dlyret, vwretd,
                   row_number() OVER (PARTITION BY permno ORDER BY trade_date) AS trading_index
            FROM raw_daily
            WHERE primaryexch IN ('N', 'A', 'Q')
              AND usincflg = 'Y'
              AND securitytype = 'EQTY'
              AND securitysubtype = 'COM'
              AND sharetype = 'NS'
              AND permno IS NOT NULL
              AND trade_date IS NOT NULL
        ), formation_dates AS (
            SELECT e.event_id, min(d.trade_date) AS formation_trade_date
            FROM events AS e
            LEFT JOIN daily AS d
              ON d.permno = e.permno
             AND d.trade_date >= CAST(e.formation_date AS DATE)
             AND d.trade_date <= CAST(e.event_trade_date AS DATE)
            GROUP BY e.event_id
        ), anchors AS (
            SELECT e.event_id, e.permno, f.formation_trade_date, e.event_trade_date,
                   fd.trading_index AS formation_trading_index,
                   ed.trading_index AS event_trading_index
            FROM events AS e
            LEFT JOIN formation_dates AS f USING (event_id)
            LEFT JOIN daily AS fd
              ON fd.permno = e.permno AND fd.trade_date = f.formation_trade_date
            LEFT JOIN daily AS ed
              ON ed.permno = e.permno AND ed.trade_date = CAST(e.event_trade_date AS DATE)
        ), pre_days AS (
            SELECT a.event_id, a.formation_trade_date, a.event_trade_date,
                   d.trading_index - a.formation_trading_index AS days_after_formation,
                   a.event_trading_index - d.trading_index AS days_before_event,
                   d.dlyret, d.vwretd
            FROM anchors AS a
            LEFT JOIN daily AS d
              ON d.permno = a.permno
             AND d.trading_index BETWEEN a.formation_trading_index + 1 AND a.event_trading_index - 1
        )
        SELECT
            event_id,
            min(formation_trade_date) AS formation_trade_date,
            min(event_trade_date) AS event_trade_date,
            count(dlyret) FILTER (WHERE dlyret IS NOT NULL AND vwretd IS NOT NULL) AS pre_trading_days,
            sum(dlyret - vwretd) FILTER (WHERE dlyret IS NOT NULL AND vwretd IS NOT NULL) AS pre_car_market_adjusted,
            count(dlyret) FILTER (
                WHERE days_after_formation BETWEEN 1 AND 20 AND dlyret IS NOT NULL AND vwretd IS NOT NULL
            ) AS pre_early20_days,
            sum(dlyret - vwretd) FILTER (
                WHERE days_after_formation BETWEEN 1 AND 20 AND dlyret IS NOT NULL AND vwretd IS NOT NULL
            ) AS pre_car_market_adjusted_early20,
            count(dlyret) FILTER (
                WHERE days_before_event BETWEEN 1 AND 20 AND dlyret IS NOT NULL AND vwretd IS NOT NULL
            ) AS pre_late20_days,
            sum(dlyret - vwretd) FILTER (
                WHERE days_before_event BETWEEN 1 AND 20 AND dlyret IS NOT NULL AND vwretd IS NOT NULL
            ) AS pre_car_market_adjusted_late20
        FROM pre_days
        GROUP BY event_id
    """
    result = con.execute(query).fetchdf()
    con.close()
    for column in ["formation_trade_date", "event_trade_date"]:
        result[column] = pd.to_datetime(result[column], errors="coerce")
    return result


def run(args: argparse.Namespace) -> None:
    if not args.panel.exists():
        raise FileNotFoundError(f"Missing event panel: {args.panel}")
    panel = pd.read_parquet(args.panel)
    returns = calculate_preannouncement_returns(panel, args.daily_path)
    output = panel.merge(returns, on="event_id", how="left", suffixes=("", "_pre"), validate="one_to_one")
    output["pre_car_market_adjusted_per_day"] = output.pre_car_market_adjusted / output.pre_trading_days
    output["pre_complete_early20"] = output.pre_early20_days.eq(20)
    output["pre_complete_late20"] = output.pre_late20_days.eq(20)
    args.output_panel.parent.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_parquet(args.output_panel, index=False, compression="zstd")
    diagnostics = pd.DataFrame([
        {"metric": "event_rows", "value": len(output)},
        {"metric": "formation_anchor_rate", "value": float(output.formation_trade_date.notna().mean())},
        {"metric": "pre_return_rate", "value": float(output.pre_car_market_adjusted.notna().mean())},
        {"metric": "median_pre_trading_days", "value": float(output.pre_trading_days.median())},
        {"metric": "complete_early20_rate", "value": float(output.pre_complete_early20.mean())},
        {"metric": "complete_late20_rate", "value": float(output.pre_complete_late20.mean())},
    ])
    diagnostics.to_csv(args.output_dir / "preannouncement_return_diagnostics.csv", index=False)
    print(diagnostics.to_string(index=False), flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["run"], nargs="?", default="run")
    parser.add_argument("--panel", type=Path, default=DEFAULT_PANEL)
    parser.add_argument("--daily-path", type=Path, default=DEFAULT_DAILY)
    parser.add_argument("--output-panel", type=Path, default=DEFAULT_OUTPUT_PANEL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
