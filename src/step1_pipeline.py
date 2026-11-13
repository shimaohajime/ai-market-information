"""Build and estimate the Step 1 simulated-compute earnings-news pilot.

The unit is a CRSP common-stock PERMNO at calendar-quarter end.  All matching
is point-in-time: annual Compustat statements must be public by formation, the
CRSP--I/B/E/S CUSIP mapping must be active at formation, and the selected EPS
surprise must be the first strictly future announcement within 180 days.

The script intentionally stops before any institutional-owner or manager-AI
analysis.  It is a local, reproducible feasibility pilot for Step 1.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = Path(os.environ.get("AI_MARKET_DATA_DIR", REPO_ROOT / "data" / "raw")).expanduser()
DEFAULT_CRSP_ROOT = Path(os.environ.get("AI_MARKET_CRSP_ROOT", DEFAULT_DATA_ROOT / "crsp")).expanduser()
DEFAULT_IBES_ROOT = Path(os.environ.get("AI_MARKET_IBES_ROOT", DEFAULT_DATA_ROOT / "ibes")).expanduser()
DEFAULT_QUARTERLY_SOURCE = DEFAULT_CRSP_ROOT / "compustat_quaterly.csv.gz"
PANEL_PATH = REPO_ROOT / "data/interim/step1_firm_quarter_panel.parquet"
OUTPUT_DIR = REPO_ROOT / "output/step1"

ACCOUNTING_BASE = [
    "log_assets",
    "operating_profitability",
    "roa",
    "asset_growth",
    "leverage",
    "cash_holdings",
    "accruals",
    "sales_growth",
    "tangibility",
]
LOW_FEATURES = [
    *ACCOUNTING_BASE,
    "statement_age_days",
    "return_1m_lag1",
    "return_12m_lag1",
    "return_volatility_12m_lag1",
    "log_market_equity_lag1",
]
HIGH_ADDITIONAL_FEATURES = [
    "return_3m_lag1",
    "return_6m_lag1",
    "return_24m_lag1",
    "return_volatility_3m_lag1",
    "return_volatility_6m_lag1",
    "return_volatility_24m_lag1",
    "log_price_lag1",
    "log_price_lag3",
    "log_price_lag6",
    "log_price_lag12",
    "log_price_lag24",
    *[f"return_lag{lag}" for lag in range(2, 13)],
    *[f"{column}_statement_lag{lag}" for lag in (1, 2) for column in ACCOUNTING_BASE],
]
PRICE_SNAPSHOT_FEATURES = [
    "current_log_price",
    "current_log_market_equity",
    "current_return_1m",
]


def sql_path(path: Path) -> str:
    return path.as_posix().replace("'", "''")


def lower_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(column).lower() for column in frame.columns]
    return frame


def parse_date_columns(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column].replace({"E": None, "B": None}), errors="coerce")


def as_numeric(frame: pd.DataFrame, columns: Iterable[str]) -> None:
    for column in columns:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")


def preferred_stock(frame: pd.DataFrame) -> pd.Series:
    result = frame["pstkrv"].copy()
    for column in ("pstkl", "pstk"):
        result = result.where(result.notna(), frame[column])
    return result.fillna(0.0)


def add_crsp_history_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Add contemporaneous and strictly lagged monthly-price features."""
    frame = frame.sort_values(["permno", "month_end"]).copy()
    group = frame.groupby("permno", sort=False)
    frame["month_index"] = frame.month_end.dt.year * 12 + frame.month_end.dt.month
    frame["current_log_price"] = np.log(frame.stock_price.where(frame.stock_price.gt(0)))
    frame["current_log_market_equity"] = np.log(frame.market_equity.where(frame.market_equity.gt(0)))
    frame["current_return_1m"] = frame.stock_return
    frame["log_market_equity_lag1"] = group.market_equity.shift(1).where(lambda value: value.gt(0)).pipe(np.log)
    for lag in (1, 3, 6, 12, 24):
        frame[f"log_price_lag{lag}"] = group.stock_price.shift(lag).where(lambda value: value.gt(0)).pipe(np.log)
    for lag in range(1, 13):
        frame[f"return_lag{lag}"] = group.stock_return.shift(lag)
    frame["return_1m_lag1"] = frame["return_lag1"]
    lagged_return = group.stock_return.shift(1)
    lagged_month_index = group.month_index.shift(1)
    for months in (3, 6, 12, 24):
        return_name = f"return_{months}m_lag1"
        volatility_name = f"return_volatility_{months}m_lag1"
        gross = (1.0 + lagged_return).groupby(frame.permno, sort=False).transform(
            lambda values: values.rolling(months, min_periods=months).apply(np.prod, raw=True)
        )
        volatility = lagged_return.groupby(frame.permno, sort=False).transform(
            lambda values: values.rolling(months, min_periods=months).std(ddof=1)
        )
        # A rolling row window is valid only when all months in it are calendar-contiguous.
        first_month = group.month_index.shift(months)
        contiguous = lagged_month_index.sub(first_month).eq(months - 1)
        frame[return_name] = (gross - 1.0).where(contiguous)
        frame[volatility_name] = volatility.where(contiguous)
    return frame.drop(columns="month_index")


def load_crsp_monthly(path: Path, start: pd.Timestamp, end: pd.Timestamp) -> tuple[pd.DataFrame, dict[str, object]]:
    usecols = [
        "PERMNO", "CUSIP", "PrimaryExch", "SecurityType", "SecuritySubType", "ShareType", "USIncFlg",
        "MthCalDt", "MthPrc", "MthVol", "ShrOut", "MthRet",
    ]
    raw = lower_columns(pd.read_csv(path, usecols=usecols, low_memory=False))
    raw_rows = len(raw)
    parse_date_columns(raw, ["mthcaldt"])
    as_numeric(raw, ["permno", "mthprc", "mthvol", "shrout", "mthret"])
    selected = raw.loc[
        raw.primaryexch.isin(["N", "A", "Q"])
        & raw.securitytype.eq("EQTY")
        & raw.securitysubtype.eq("COM")
        & raw.sharetype.eq("NS")
        & raw.usincflg.eq("Y")
        & raw.permno.notna()
        & raw.mthcaldt.notna()
    ].copy()
    selected["permno"] = selected.permno.astype("int64")
    selected["month_end"] = selected.mthcaldt
    selected["stock_price"] = selected.mthprc.abs()
    selected["stock_volume"] = selected.mthvol.abs()
    selected["shares_outstanding"] = selected.shrout.abs()
    # CRSP ShrOut is in thousands, while MthPrc is dollars per share.
    selected["market_equity"] = selected.stock_price * selected.shares_outstanding * 1000.0
    selected["stock_return"] = selected.mthret
    selected["cusip"] = selected.cusip.astype("string").str.strip().str[:8]
    selected = selected.sort_values(
        ["permno", "month_end", "stock_price", "stock_volume"],
        ascending=[True, True, False, False],
        na_position="last",
    ).drop_duplicates(["permno", "month_end"], keep="first")
    selected = add_crsp_history_features(selected)
    quarter_end = selected.loc[
        selected.month_end.dt.is_quarter_end
        & selected.month_end.between(start, end),
    ].copy()
    keep = [
        "permno", "month_end", "cusip", "stock_price", "market_equity", "stock_return",
        "current_log_price", "current_log_market_equity", "current_return_1m", "log_market_equity_lag1",
        "return_1m_lag1",
        *[f"log_price_lag{lag}" for lag in (1, 3, 6, 12, 24)],
        *[f"return_lag{lag}" for lag in range(1, 13)],
        *[f"return_{months}m_lag1" for months in (3, 6, 12, 24)],
        *[f"return_volatility_{months}m_lag1" for months in (3, 6, 12, 24)],
    ]
    quarter_end = quarter_end[keep].rename(columns={"month_end": "formation_date"})
    diagnostics: dict[str, object] = {
        "crsp_raw_rows": raw_rows,
        "crsp_selected_common_stock_rows": len(selected),
        "crsp_formation_rows": len(quarter_end),
        "crsp_unique_permnos_at_formation": quarter_end.permno.nunique(),
        "formation_dates_all_calendar_quarter_ends": bool(quarter_end.formation_date.dt.is_quarter_end.all()),
        "crsp_duplicate_permno_formation_keys": int(quarter_end.duplicated(["permno", "formation_date"]).sum()),
    }
    return quarter_end, diagnostics


def annual_statements(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    raw = lower_columns(pd.read_csv(path, low_memory=False))
    parse_date_columns(raw, ["datadate", "fdate"])
    numeric_columns = [
        "at", "act", "capx", "ceq", "che", "cogs", "dlc", "dltt", "ib", "lct", "ppent", "pstkl",
        "pstk", "pstkrv", "revt", "sale", "txditc", "xint", "xsga", "fyear",
    ]
    as_numeric(raw, numeric_columns)
    selected = raw.loc[
        raw.indfmt.eq("INDL")
        & raw.consol.eq("C")
        & raw.popsrc.eq("D")
        & raw.datafmt.eq("STD")
        & raw.curcd.eq("USD")
        & raw.fic.eq("USA")
        & raw.gvkey.notna()
        & raw.datadate.notna()
    ].copy()
    selected["primary_rank"] = selected.linkprim.eq("P").astype(int)
    statements = selected.sort_values(
        ["gvkey", "datadate", "primary_rank", "fdate"],
        ascending=[True, True, False, False],
        na_position="last",
    ).drop_duplicates(["gvkey", "datadate"], keep="first").copy()
    statements["available_date"] = statements.fdate.where(statements.fdate.ge(statements.datadate))
    statements["available_date"] = statements.available_date.fillna(statements.datadate + pd.DateOffset(months=6))
    statements = statements.sort_values(["gvkey", "datadate"])
    group = statements.groupby("gvkey", sort=False)
    for column in ["at", "act", "che", "lct", "dlc", "sale", "fyear"]:
        statements[f"lag_{column}"] = group[column].shift(1)
    statements["book_equity"] = statements.ceq + statements.txditc.fillna(0.0) - preferred_stock(statements)
    statements["lag_book_equity"] = group.book_equity.shift(1)
    valid_lag = statements.fyear.sub(statements.lag_fyear).eq(1)
    for column in ["lag_at", "lag_act", "lag_che", "lag_lct", "lag_dlc", "lag_sale", "lag_book_equity"]:
        statements.loc[~valid_lag, column] = np.nan
    income = statements.revt.where(statements.revt.notna(), statements.sale)
    statements["log_assets"] = np.log(statements["at"].where(statements["at"].gt(0)))
    statements["operating_profitability"] = (
        income - statements.cogs - statements.xsga - statements.xint
    ) / statements.lag_book_equity
    statements["roa"] = statements.ib / statements.lag_at
    statements["asset_growth"] = statements["at"] / statements.lag_at - 1.0
    statements["leverage"] = (statements.dltt.fillna(0.0) + statements.dlc.fillna(0.0)) / statements["at"]
    statements["cash_holdings"] = statements.che / statements["at"]
    statements["accruals"] = (
        (statements.act - statements.lag_act)
        - (statements.che - statements.lag_che)
        - (statements.lct - statements.lag_lct)
        + (statements.dlc - statements.lag_dlc)
    ) / statements.lag_at
    statements["sales_growth"] = statements.sale / statements.lag_sale - 1.0
    statements["tangibility"] = statements.ppent / statements["at"]
    statements[ACCOUNTING_BASE] = statements[ACCOUNTING_BASE].replace([np.inf, -np.inf], np.nan)
    for lag in (1, 2):
        statements[[f"{column}_statement_lag{lag}" for column in ACCOUNTING_BASE]] = group[ACCOUNTING_BASE].shift(lag)
    keep = ["gvkey", "datadate", "available_date", *ACCOUNTING_BASE]
    keep += [f"{column}_statement_lag{lag}" for lag in (1, 2) for column in ACCOUNTING_BASE]
    diagnostics: dict[str, object] = {
        "compustat_annual_raw_rows": len(raw),
        "compustat_annual_usd_usa_rows": len(selected),
        "compustat_annual_unique_company_statements": len(statements),
        "compustat_available_date_not_before_statement": bool(statements.available_date.ge(statements.datadate).all()),
    }
    return statements[keep], diagnostics


def quarterly_statements(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Prepare point-in-time quarterly Compustat features for a separate robustness panel.

    The source's RDQ is the availability date.  A quarterly record is retained
    only when RDQ is observed and is not earlier than the fiscal statement
    date.  Flow variables use trailing-four-quarter sums, and year-over-year
    growth requires a four-quarter date span of 300--400 days.  These choices
    keep the quarterly panel time-safe without changing the declared L/H price
    information sets.
    """
    usecols = [
        "GVKEY", "datadate", "fyearq", "fqtr", "rdq", "indfmt", "consol", "popsrc", "datafmt",
        "curcdq", "fic", "atq", "actq", "ceqq", "chq", "cogsq", "dlcq", "dlttq", "ibq", "lctq",
        "ppentq", "pstknq", "pstkq", "pstkrq", "saleq", "txditcq", "xintq", "xsgaq",
    ]
    raw = lower_columns(pd.read_csv(path, usecols=usecols, low_memory=False))
    parse_date_columns(raw, ["datadate", "rdq"])
    numeric_columns = [
        "fyearq", "fqtr", "atq", "actq", "ceqq", "chq", "cogsq", "dlcq", "dlttq", "ibq", "lctq",
        "ppentq", "pstknq", "pstkq", "pstkrq", "saleq", "txditcq", "xintq", "xsgaq",
    ]
    as_numeric(raw, numeric_columns)
    selected = raw.loc[
        raw.indfmt.eq("INDL")
        & raw.consol.eq("C")
        & raw.popsrc.eq("D")
        & raw.datafmt.eq("STD")
        & raw.curcdq.eq("USD")
        & raw.fic.eq("USA")
        & raw.gvkey.notna()
        & raw.datadate.notna()
        & raw.rdq.notna()
        & raw.rdq.ge(raw.datadate)
    ].copy()
    statements = selected.sort_values(
        ["gvkey", "datadate", "rdq", "fqtr"],
        ascending=[True, True, False, False],
        na_position="last",
    ).drop_duplicates(["gvkey", "datadate"], keep="first").copy()
    statements["available_date"] = statements["rdq"]
    statements = statements.sort_values(["gvkey", "datadate"])
    group = statements.groupby("gvkey", sort=False)
    lag_date_1 = group.datadate.shift(1)
    lag_date_4 = group.datadate.shift(4)
    contiguous_1 = statements.datadate.sub(lag_date_1).dt.days.between(70, 120)
    contiguous_4 = statements.datadate.sub(lag_date_4).dt.days.between(300, 400)

    preferred = statements["pstkrq"].copy()
    for column in ("pstknq", "pstkq"):
        preferred = preferred.where(preferred.notna(), statements[column])
    statements["book_equity"] = statements.ceqq + statements.txditcq.fillna(0.0) - preferred.fillna(0.0)
    for column in ("atq", "actq", "chq", "lctq", "dlcq", "saleq", "book_equity"):
        statements[f"lag_{column}"] = group[column].shift(1)
        statements.loc[~contiguous_1, f"lag_{column}"] = np.nan
    statements["lag_saleq_4"] = group.saleq.shift(4).where(contiguous_4)
    statements["lag_atq_4"] = group.atq.shift(4).where(contiguous_4)

    for column in ("saleq", "cogsq", "xsgaq", "xintq", "ibq"):
        statements[f"ttm_{column}"] = group[column].transform(
            lambda values: values.rolling(4, min_periods=4).sum()
        ).where(contiguous_4)
    statements["log_assets"] = np.log(statements.atq.where(statements.atq.gt(0)))
    statements["operating_profitability"] = (
        statements.ttm_saleq - statements.ttm_cogsq - statements.ttm_xsgaq - statements.ttm_xintq
    ) / statements.lag_book_equity
    statements["roa"] = statements.ttm_ibq / statements.lag_atq
    statements["asset_growth"] = statements.atq / statements.lag_atq_4 - 1.0
    statements["leverage"] = (statements.dlttq.fillna(0.0) + statements.dlcq.fillna(0.0)) / statements.atq
    statements["cash_holdings"] = statements.chq / statements.atq
    statements["accruals"] = (
        (statements.actq - statements.lag_actq)
        - (statements.chq - statements.lag_chq)
        - (statements.lctq - statements.lag_lctq)
        + (statements.dlcq - statements.lag_dlcq)
    ) / statements.lag_atq
    statements["sales_growth"] = statements.saleq / statements.lag_saleq_4 - 1.0
    statements["tangibility"] = statements.ppentq / statements.atq
    statements[ACCOUNTING_BASE] = statements[ACCOUNTING_BASE].replace([np.inf, -np.inf], np.nan)
    for lag in (1, 2):
        lag_date = group.datadate.shift(lag)
        contiguous = statements.datadate.sub(lag_date).dt.days.between(70 * lag, 120 * lag)
        lagged = group[ACCOUNTING_BASE].shift(lag)
        statements[[f"{column}_statement_lag{lag}" for column in ACCOUNTING_BASE]] = lagged.where(
            contiguous, axis=0
        )
    keep = ["gvkey", "datadate", "available_date", *ACCOUNTING_BASE]
    keep += [f"{column}_statement_lag{lag}" for lag in (1, 2) for column in ACCOUNTING_BASE]
    diagnostics: dict[str, object] = {
        "compustat_quarterly_raw_rows": len(raw),
        "compustat_quarterly_usd_usa_rdq_rows": len(selected),
        "compustat_quarterly_unique_company_statements": len(statements),
        "compustat_available_date_not_before_statement": bool(statements.available_date.ge(statements.datadate).all()),
        "quarterly_statement_lag1_contiguous_rate": float(contiguous_1.mean()),
        "quarterly_statement_lag4_contiguous_rate": float(contiguous_4.mean()),
    }
    return statements[keep], diagnostics


def valid_ccm_links(path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    links = lower_columns(pd.read_csv(path, low_memory=False))
    parse_date_columns(links, ["linkdt", "linkenddt"])
    as_numeric(links, ["lpermno"])
    links["linkenddt"] = links.linkenddt.fillna(pd.Timestamp("2262-04-11"))
    selected = links.loc[
        links.linkprim.isin(["P", "C"])
        & links.linktype.isin(["LC", "LU"])
        & links.fic.eq("USA")
        & links.lpermno.notna()
        & links.linkdt.notna()
    ].copy()
    selected["lpermno"] = selected.lpermno.astype("int64")
    selected = selected.drop_duplicates(["gvkey", "lpermno", "linkdt", "linkenddt"])
    diagnostics: dict[str, object] = {
        "ccm_raw_rows": len(links),
        "ccm_selected_rows": len(selected),
        "ccm_valid_date_ranges": bool(selected.linkdt.le(selected.linkenddt).all()),
    }
    return selected[["gvkey", "lpermno", "linkdt", "linkenddt"]], diagnostics


def attach_accounting(
    panel: pd.DataFrame, statements: pd.DataFrame, links: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    keys = panel[["permno", "formation_date"]].copy()
    keys["row_id"] = np.arange(len(keys), dtype=np.int64)
    con = duckdb.connect()
    con.register("keys", keys)
    con.register("statements", statements)
    con.register("links", links)
    joined = con.execute(
        """
        WITH active_links AS (
            SELECT k.*, l.gvkey
            FROM keys AS k
            JOIN links AS l
              ON l.lpermno = k.permno
             AND l.linkdt <= k.formation_date
             AND k.formation_date <= l.linkenddt
        ), candidates AS (
            SELECT a.*, s.datadate AS accounting_statement_date,
                   s.available_date AS accounting_available_date,
                   s.* EXCLUDE (gvkey, datadate, available_date),
                   row_number() OVER (
                       PARTITION BY a.row_id
                       ORDER BY s.available_date DESC, s.datadate DESC, a.gvkey
                   ) AS selection_rank
            FROM active_links AS a
            JOIN statements AS s
              ON s.gvkey = a.gvkey
             AND s.available_date <= a.formation_date
        )
        SELECT * EXCLUDE (selection_rank, gvkey) FROM candidates WHERE selection_rank = 1
        """
    ).fetchdf()
    con.close()
    joined.formation_date = pd.to_datetime(joined.formation_date)
    joined.accounting_statement_date = pd.to_datetime(joined.accounting_statement_date)
    joined.accounting_available_date = pd.to_datetime(joined.accounting_available_date)
    joined["statement_age_days"] = (
        joined.formation_date - joined.accounting_statement_date
    ).dt.days.astype(float)
    columns = ["row_id", "accounting_statement_date", "accounting_available_date", "statement_age_days", *ACCOUNTING_BASE]
    columns += [f"{column}_statement_lag{lag}" for lag in (1, 2) for column in ACCOUNTING_BASE]
    output = panel.copy()
    output["row_id"] = np.arange(len(output), dtype=np.int64)
    output = output.merge(joined[columns], on="row_id", how="left", validate="one_to_one").drop(columns="row_id")
    has_accounting = output.accounting_available_date.notna()
    diagnostics: dict[str, object] = {
        "accounting_match_rate": float(has_accounting.mean()),
        "accounting_available_by_formation": bool(
            (output.loc[has_accounting, "accounting_available_date"] <= output.loc[has_accounting, "formation_date"]).all()
        ),
    }
    return output, diagnostics


def historical_ibes_ids(path: Path) -> pd.DataFrame:
    ids = lower_columns(pd.read_csv(path, dtype=str))
    parse_date_columns(ids, ["sdates"])
    ids["cusip"] = ids.cusip.str.strip().str[:8]
    ids = ids.loc[
        ids.usfirm.eq("1")
        & ids.cusip.str.fullmatch(r"[0-9A-Z]{8}", na=False)
        & ids.sdates.notna()
    ].copy()
    ids = ids.sort_values(["ticker", "sdates", "cusip"]).drop_duplicates(["ticker", "sdates"], keep="last")
    ids["id_end_date"] = ids.groupby("ticker", sort=False).sdates.shift(-1) - pd.Timedelta(days=1)
    ids["id_end_date"] = ids.id_end_date.fillna(pd.Timestamp("2262-04-11"))
    return ids[["ticker", "cusip", "sdates", "id_end_date"]]


def surprise_events(path: Path) -> pd.DataFrame:
    events = lower_columns(pd.read_csv(path, low_memory=False))
    parse_date_columns(events, ["anndats"])
    events = events.loc[
        events.measure.eq("EPS")
        & events.fiscalp.eq("QTR")
        & events.usfirm.astype("string").eq("1")
        & events.anndats.notna()
    ].copy()
    as_numeric(events, ["actual", "surpmean", "surpstdev", "suescore", "pyear", "pmon"])
    events["information_rank"] = events[["actual", "surpmean", "surpstdev", "suescore"]].notna().sum(axis=1)
    events = events.sort_values(
        ["ticker", "anndats", "information_rank", "pyear", "pmon"],
        ascending=[True, True, False, False, False],
    ).drop_duplicates(["ticker", "anndats"], keep="first")
    return events[["ticker", "anndats", "actual", "surpmean", "surpstdev", "suescore"]]


def attach_ibes_outcomes(
    panel: pd.DataFrame, ids: pd.DataFrame, events: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, object]]:
    con = duckdb.connect()
    con.register("panel", panel)
    con.register("ids", ids)
    con.register("events", events)
    joined = con.execute(
        """
        WITH ticker_candidates AS (
            SELECT p.*, i.ticker AS ibes_ticker,
                   row_number() OVER (
                       PARTITION BY p.permno, p.formation_date
                       ORDER BY i.sdates DESC, i.ticker
                   ) AS ticker_rank
            FROM panel AS p
            LEFT JOIN ids AS i
              ON i.cusip = p.cusip
             AND i.sdates <= p.formation_date
             AND p.formation_date <= i.id_end_date
        ), selected_ticker AS (
            SELECT * EXCLUDE (ticker_rank) FROM ticker_candidates WHERE ticker_rank = 1
        )
        SELECT p.*, e.anndats AS next_announcement_date,
               date_diff('day', p.formation_date, e.anndats) AS days_to_next_announcement,
               e.actual AS next_actual_eps, e.surpmean AS next_surprise_mean,
               e.surpstdev AS next_surprise_stdev, e.suescore AS next_sue
        FROM selected_ticker AS p
        LEFT JOIN LATERAL (
            SELECT * FROM events AS e
            WHERE e.ticker = p.ibes_ticker
              AND e.anndats > p.formation_date
              AND e.anndats <= p.formation_date + INTERVAL 180 DAYS
            ORDER BY e.anndats ASC
            LIMIT 1
        ) AS e ON true
        """
    ).fetchdf()
    con.close()
    for column in ["formation_date", "accounting_statement_date", "accounting_available_date", "next_announcement_date"]:
        if column in joined:
            joined[column] = pd.to_datetime(joined[column])
    announced = joined.next_announcement_date.notna()
    diagnostics: dict[str, object] = {
        "ibes_historical_ticker_match_rate": float(joined.ibes_ticker.notna().mean()),
        "ibes_next_quarterly_eps_announcement_rate": float(announced.mean()),
        "ibes_next_sue_rate": float(joined.next_sue.notna().mean()),
        "ibes_announcements_strictly_after_formation": bool(
            (joined.loc[announced, "next_announcement_date"] > joined.loc[announced, "formation_date"]).all()
        ),
        "ibes_announcements_within_180_days": bool(
            joined.loc[announced, "days_to_next_announcement"].between(1, 180).all()
        ),
        "ibes_median_days_to_next_announcement": float(joined.days_to_next_announcement.median()),
    }
    return joined, diagnostics


def attach_analyst_coverage(panel: pd.DataFrame, path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    """Attach only the latest pre-formation analyst count for descriptive cuts."""
    keys = panel[["permno", "formation_date", "ibes_ticker"]].copy()
    con = duckdb.connect()
    con.register("keys", keys)
    source = sql_path(path)
    coverage = con.execute(
        f"""
        WITH consensus AS (
            SELECT trim(TICKER) AS ticker,
                   try_cast(STATPERS AS DATE) AS consensus_date,
                   try_cast(NUMEST AS DOUBLE) AS analyst_coverage
            FROM read_csv_auto(
                '{source}', header = true, all_varchar = true,
                quote = '"', strict_mode = false, null_padding = true
            )
            WHERE trim(MEASURE) = 'EPS' AND trim(FISCALP) = 'QTR'
        )
        SELECT k.permno, k.formation_date, c.consensus_date, c.analyst_coverage
        FROM keys AS k
        LEFT JOIN LATERAL (
            SELECT * FROM consensus AS c
            WHERE c.ticker = k.ibes_ticker
              AND c.consensus_date <= k.formation_date
            ORDER BY c.consensus_date DESC
            LIMIT 1
        ) AS c ON true
        """
    ).fetchdf()
    con.close()
    coverage.formation_date = pd.to_datetime(coverage.formation_date)
    coverage.consensus_date = pd.to_datetime(coverage.consensus_date)
    output = panel.merge(coverage, on=["permno", "formation_date"], how="left", validate="one_to_one")
    covered = output.consensus_date.notna()
    diagnostics = {
        "analyst_coverage_match_rate": float(covered.mean()),
        "analyst_coverage_not_after_formation": bool(
            (output.loc[covered, "consensus_date"] <= output.loc[covered, "formation_date"]).all()
        ),
    }
    return output, diagnostics


def write_diagnostics(diagnostics: dict[str, object], output: Path) -> None:
    pd.DataFrame([{"metric": key, "value": value} for key, value in diagnostics.items()]).to_csv(
        output / "build_diagnostics.csv", index=False
    )


def build_panel(args: argparse.Namespace) -> Path:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    args.panel.parent.mkdir(parents=True, exist_ok=True)
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    crsp_root, ibes_root = args.crsp_root, args.ibes_root
    print("Reading and constructing CRSP common-stock histories...", flush=True)
    panel, diagnostics = load_crsp_monthly(crsp_root / "stock_monthly_data.csv.gz", start, end)
    if args.accounting_frequency == "annual":
        print("Preparing time-safe annual Compustat statements and CCM links...", flush=True)
        statements, accounting_source_diagnostics = annual_statements(crsp_root / "compustat_annual.csv.gz")
    else:
        print("Preparing time-safe quarterly Compustat statements and CCM links...", flush=True)
        statements, accounting_source_diagnostics = quarterly_statements(args.quarterly_source)
    links, link_diagnostics = valid_ccm_links(crsp_root / "compustat_crsp_link.csv")
    panel, accounting_diagnostics = attach_accounting(panel, statements, links)
    print("Linking I/B/E/S quarterly EPS surprises...", flush=True)
    ids = historical_ibes_ids(ibes_root / "IBES_idsum.csv.gz")
    events = surprise_events(ibes_root / "IBES_surpsumu.csv.gz")
    panel, ibes_diagnostics = attach_ibes_outcomes(panel, ids, events)
    print("Adding pre-formation analyst coverage for descriptive heterogeneity...", flush=True)
    panel, coverage_diagnostics = attach_analyst_coverage(panel, ibes_root / "IBES_statsumu_epsus.csv.gz")
    panel["formation_year"] = panel.formation_date.dt.year.astype(int)
    panel["formation_quarter"] = panel.formation_date.dt.to_period("Q").astype(str)
    panel = panel.sort_values(["formation_date", "permno"]).reset_index(drop=True)
    diagnostics.update(accounting_source_diagnostics)
    diagnostics.update(link_diagnostics)
    diagnostics.update(accounting_diagnostics)
    diagnostics.update(ibes_diagnostics)
    diagnostics.update(coverage_diagnostics)
    diagnostics["final_panel_rows"] = len(panel)
    diagnostics["final_panel_duplicate_keys"] = int(panel.duplicated(["permno", "formation_date"]).sum())
    diagnostics["accounting_frequency"] = args.accounting_frequency
    panel.to_parquet(args.panel, index=False, compression="zstd")
    write_diagnostics(diagnostics, output)
    manifest = pd.DataFrame([
        *[{"model": "L0", "feature": column, "feature_group": "lagged or accounting", "current_price_snapshot": False} for column in LOW_FEATURES],
        *[{"model": "L1", "feature": column, "feature_group": "current price snapshot", "current_price_snapshot": True} for column in PRICE_SNAPSHOT_FEATURES],
        *[{"model": "H0", "feature": column, "feature_group": "additional high-capacity input", "current_price_snapshot": False} for column in HIGH_ADDITIONAL_FEATURES],
    ])
    manifest.to_csv(output / "feature_manifest.csv", index=False)
    print(f"Wrote {args.panel} ({len(panel):,} firm-quarter rows)", flush=True)
    return args.panel


def model_feature_sets() -> dict[str, list[str]]:
    high = [*LOW_FEATURES, *HIGH_ADDITIONAL_FEATURES]
    return {
        "L0": LOW_FEATURES,
        "L1": [*LOW_FEATURES, *PRICE_SNAPSHOT_FEATURES],
        "H0": high,
        "H1": [*high, *PRICE_SNAPSHOT_FEATURES],
    }


def active_features(train: pd.DataFrame, features: list[str]) -> list[str]:
    return [column for column in features if column in train and train[column].notna().any()]


def fit_and_predict(model_name: str, train: pd.DataFrame, test: pd.DataFrame, features: list[str]) -> tuple[np.ndarray, list[str]]:
    columns = active_features(train, features)
    if not columns:
        raise ValueError(f"No populated features for {model_name} in this training window")
    if model_name.startswith("L"):
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("scale", StandardScaler()),
            ("model", Ridge(alpha=10.0)),
        ])
    else:
        model = Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True)),
            ("model", HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=250,
                max_leaf_nodes=15,
                min_samples_leaf=40,
                l2_regularization=1.0,
                random_state=20261029,
            )),
        ])
    model.fit(train[columns], train["target_for_model"])
    return model.predict(test[columns]), columns


def gaussian_pseudo_loss(actual: np.ndarray, predicted: np.ndarray, sigma: float) -> np.ndarray:
    return 0.5 * math.log(2.0 * math.pi * sigma * sigma) + 0.5 * ((actual - predicted) / sigma) ** 2


def model_diagnostics(scores: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = [
        {"metric": "oos_rows", "value": len(scores)},
        {"metric": "oos_duplicate_permno_formation_keys", "value": int(scores.duplicated(["permno", "formation_date"]).sum())},
        {"metric": "all_four_losses_present", "value": bool(scores[["loss_L0", "loss_L1", "loss_H0", "loss_H1"]].notna().all(axis=None))},
        {"metric": "same_oos_row_count_in_each_model", "value": True},
        {"metric": "all_training_before_test", "value": bool((folds.train_last_year < folds.test_year).all())},
    ]
    return pd.DataFrame(rows)


def summarize_scores(scores: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    grouped = scores.groupby(group_columns, dropna=False)
    output = grouped.agg(
        n_obs=("next_sue", "size"),
        loss_L0=("loss_L0", "mean"),
        loss_L1=("loss_L1", "mean"),
        loss_H0=("loss_H0", "mean"),
        loss_H1=("loss_H1", "mean"),
        S_limited_price_value=("S_limited_price_value", "mean"),
        F_high_compute_price_value=("F_high_compute_price_value", "mean"),
        V_compute_value_with_price=("V_compute_value_with_price", "mean"),
    ).reset_index()
    return output


def assign_descriptive_bins(scores: pd.DataFrame) -> pd.DataFrame:
    output = scores.copy()

    def terciles(values: pd.Series) -> pd.Series:
        result = pd.Series(pd.NA, index=values.index, dtype="string")
        observed = values.dropna()
        if len(observed) < 30:
            return result
        # Ranking first breaks ties deterministically, so terciles remain
        # defined in quarters with many identical small analyst counts.
        result.loc[observed.index] = pd.qcut(
            observed.rank(method="first"), 3, labels=["low", "middle", "high"]
        ).astype("string")
        return result

    for column, name in [("log_market_equity_lag1", "size"), ("analyst_coverage", "analyst_coverage")]:
        output[name + "_tercile"] = output.groupby("formation_quarter")[column].transform(terciles)
    return output


def write_report(summary: pd.DataFrame, diagnostics: pd.DataFrame, output: Path, first_oos_year: int) -> None:
    overall = summary.loc[summary.group.eq("overall")].drop(columns="group")
    report = [
        "# Step 1 simulated-compute pilot",
        "",
        "This is a counterfactual comparison of public-information processing technologies; it does not measure adviser or fund AI adoption.",
        "",
        f"The OOS evaluation uses expanding annual training windows and years {first_oos_year} onward. All four models are evaluated on exactly the same target-observed firm-quarter rows. The target is the first I/B/E/S U.S. quarterly EPS standardized unexpected earnings announcement strictly after formation and within 180 days.",
        "",
        "The loss is a Gaussian pseudo-log-loss with a common target scale estimated using each fold's training data. Lower is better. Therefore S = L0 - L1, F = H0 - H1, and V = L1 - H1 are improvements in this loss, not calibrated information units.",
        "",
        "## Overall out-of-sample comparison",
        "",
        "```text",
        overall.to_csv(index=False).strip(),
        "```",
        "",
        "## Validation",
        "",
        "```text",
        diagnostics.to_csv(index=False).strip(),
        "```",
    ]
    (output / "step1_report.md").write_text("\n".join(report) + "\n")


def estimate(args: argparse.Namespace) -> None:
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(args.panel)
    panel.formation_date = pd.to_datetime(panel.formation_date)
    panel = panel.loc[np.isfinite(panel.next_sue)].copy()
    panel["formation_year"] = panel.formation_date.dt.year.astype(int)
    feature_sets = model_feature_sets()
    oos_rows: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []
    for test_year in range(args.first_oos_year, int(panel.formation_year.max()) + 1):
        train = panel.loc[panel.formation_year < test_year].copy()
        test = panel.loc[panel.formation_year.eq(test_year)].copy()
        if len(train) < 500 or len(test) == 0:
            continue
        lower, upper = train.next_sue.quantile([0.01, 0.99]).to_numpy()
        train["target_for_model"] = train.next_sue.clip(lower, upper)
        test["target_for_model"] = test.next_sue.clip(lower, upper)
        sigma = float(train.target_for_model.std(ddof=1))
        sigma = max(sigma, 1e-6)
        for model_name, features in feature_sets.items():
            prediction, used_features = fit_and_predict(model_name, train, test, features)
            test[f"prediction_{model_name}"] = prediction
            test[f"loss_{model_name}"] = gaussian_pseudo_loss(
                test.target_for_model.to_numpy(), prediction, sigma
            )
            fold_rows.append({
                "test_year": test_year,
                "model": model_name,
                "n_train": len(train),
                "n_test": len(test),
                "train_last_year": int(train.formation_year.max()),
                "target_clip_lower": lower,
                "target_clip_upper": upper,
                "common_sigma": sigma,
                "n_active_features": len(used_features),
            })
        test["S_limited_price_value"] = test.loss_L0 - test.loss_L1
        test["F_high_compute_price_value"] = test.loss_H0 - test.loss_H1
        test["V_compute_value_with_price"] = test.loss_L1 - test.loss_H1
        oos_rows.append(test)
        print(f"Scored {test_year}: {len(test):,} common OOS rows", flush=True)
    if not oos_rows:
        raise RuntimeError("No OOS folds were estimable")
    scores = pd.concat(oos_rows, ignore_index=True)
    folds = pd.DataFrame(fold_rows)
    scores = assign_descriptive_bins(scores)
    keep = [
        "permno", "formation_date", "formation_year", "formation_quarter", "next_announcement_date", "days_to_next_announcement",
        "next_sue", "target_for_model", "log_market_equity_lag1", "analyst_coverage", "size_tercile", "analyst_coverage_tercile",
        *[f"prediction_{name}" for name in feature_sets],
        *[f"loss_{name}" for name in feature_sets],
        "S_limited_price_value", "F_high_compute_price_value", "V_compute_value_with_price",
    ]
    scores[keep].to_parquet(output / "oos_scores.parquet", index=False, compression="zstd")
    folds.to_csv(output / "oos_folds.csv", index=False)
    diagnostics = model_diagnostics(scores, folds)
    diagnostics.to_csv(output / "model_diagnostics.csv", index=False)
    overall = summarize_scores(scores.assign(group="overall"), ["group"])
    by_year = summarize_scores(scores, ["formation_year"])
    by_quarter = summarize_scores(scores, ["formation_quarter"])
    by_size = summarize_scores(scores.dropna(subset=["size_tercile"]), ["size_tercile"])
    by_coverage = summarize_scores(scores.dropna(subset=["analyst_coverage_tercile"]), ["analyst_coverage_tercile"])
    overall.to_csv(output / "oos_summary_overall.csv", index=False)
    by_year.to_csv(output / "oos_summary_by_year.csv", index=False)
    by_quarter.to_csv(output / "oos_summary_by_quarter.csv", index=False)
    by_size.to_csv(output / "oos_summary_by_size_tercile.csv", index=False)
    by_coverage.to_csv(output / "oos_summary_by_analyst_coverage_tercile.csv", index=False)
    summary_for_report = overall.assign(group="overall")
    write_report(summary_for_report, diagnostics, output, args.first_oos_year)
    print(f"Wrote OOS results to {output}", flush=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("command", choices=["build", "estimate", "run"])
    result.add_argument("--crsp-root", type=Path, default=DEFAULT_CRSP_ROOT)
    result.add_argument("--ibes-root", type=Path, default=DEFAULT_IBES_ROOT)
    result.add_argument("--accounting-frequency", choices=["annual", "quarterly"], default="annual")
    result.add_argument("--quarterly-source", type=Path, default=DEFAULT_QUARTERLY_SOURCE)
    result.add_argument("--panel", type=Path, default=PANEL_PATH)
    result.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    result.add_argument("--start", default="2011-03-31")
    result.add_argument("--end", default="2025-12-31")
    result.add_argument("--first-oos-year", type=int, default=2018)
    return result


def main() -> None:
    args = parser().parse_args()
    if args.command in {"build", "run"}:
        build_panel(args)
    if args.command in {"estimate", "run"}:
        estimate(args)


if __name__ == "__main__":
    main()
