"""M11 매수 이유 정기 점검.

위성 종목마다 '이 이야기가 깨지는 조건'을 기준으로 주기적으로(기본 90일, 분기 실적 주기) 확인한다.
결과는 유지 / 주의 / 깨짐. '깨짐'은 평단과 무관하게 정리를 검토한다.
기록은 투자일기와 같은 journal.db 의 reviews 테이블에 둔다.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .journal import DB

STATUSES = ["유지", "주의", "깨짐"]
MIN_NOTE = 10


def _conn(db: Path | str = DB) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE IF NOT EXISTS reviews (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        ticker TEXT, review_date TEXT, status TEXT, note TEXT, break_condition TEXT, created_at TEXT)""")
    return c


def save_review(ticker: str, status: str, note: str, break_condition: str, review_date: str,
                db: Path | str = DB) -> int:
    if status not in STATUSES:
        raise ValueError(f"상태는 {', '.join(STATUSES)} 중 하나여야 합니다.")
    if len(note.strip()) < MIN_NOTE:
        raise ValueError(f"점검 메모를 {MIN_NOTE}자 이상 적으세요 (무엇을 보고 판단했는지).")
    with _conn(db) as c:
        cur = c.execute(
            "INSERT INTO reviews (ticker, review_date, status, note, break_condition, created_at) VALUES (?,?,?,?,?,?)",
            (ticker.upper().strip(), str(review_date), status, note.strip(), break_condition.strip(),
             datetime.now().isoformat(timespec="seconds")))
        return int(cur.lastrowid)


def load_reviews(db: Path | str = DB) -> pd.DataFrame:
    with _conn(db) as c:
        return pd.read_sql_query("SELECT * FROM reviews ORDER BY review_date DESC, id DESC", c)


def _latest_text(df: pd.DataFrame, col: str) -> str:
    """최신순 정렬된 표에서 비어 있지 않은 첫 값."""
    vals = df[col].dropna().astype(str).str.strip() if col in df else pd.Series(dtype=str)
    vals = vals[vals != ""]
    return vals.iloc[0] if len(vals) else ""


def review_status(tickers: list[str], reviews: pd.DataFrame, trades: pd.DataFrame,
                  today: date, every_days: int) -> pd.DataFrame:
    """종목별 점검 현황. 기한 지남·미점검·'깨짐' 이 due=True, due 가 먼저 오도록 정렬."""
    rows = []
    for t in tickers:
        r = reviews[reviews["ticker"] == t] if not reviews.empty else reviews
        buys = trades[(trades["ticker"] == t) & (trades["side"] == "매수")] if not trades.empty else trades
        last = pd.to_datetime(r["review_date"].iloc[0]).date() if len(r) else None
        days = (today - last).days if last else None
        status = r["status"].iloc[0] if len(r) else None
        rows.append({
            "ticker": t, "last_review": last, "days_since": days, "last_status": status,
            "due": last is None or days >= every_days or status == "깨짐",
            "break_condition": _latest_text(r, "break_condition") or _latest_text(buys, "break_condition"),
        })
    out = pd.DataFrame(rows, columns=["ticker", "last_review", "days_since", "last_status", "due", "break_condition"])
    return out.sort_values("due", ascending=False, kind="stable").reset_index(drop=True)
