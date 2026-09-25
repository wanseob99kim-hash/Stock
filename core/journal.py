"""M4 투자일기 · 매수 체크리스트 (SQLite).

매수는 4대 자격(이효석 3탄)을 모두 채워야 저장된다.
금지 상품은 차단, 그 밖의 규칙 위반은 '위반 사유'를 적어야 저장되고 준수율 KPI에 반영된다.
"""
from __future__ import annotations

import sqlite3
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .ips import IPS

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "journal.db"

CHECKLIST = {
    "thesis": "① 왜 사는가 — 한 문장 내러티브",
    "fundamentals": "② 매출·이익은 얼마이고 늘고 있는가",
    "valuation": "③ 왜 싸다고 보는가 (PER 밴드·PEG 등 기준)",
    "plan": "④ 매수·매도 계획 (분할 횟수, 목표가, 손절/재검토 조건)",
}
MIN_LEN = 10


def _conn(db: Path | str = DB) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE IF NOT EXISTS trades (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        trade_date TEXT, account TEXT, ticker TEXT, side TEXT,
        quantity REAL, price REAL,
        thesis TEXT, fundamentals TEXT, valuation TEXT, plan TEXT, break_condition TEXT,
        violations TEXT, override_reason TEXT, compliant INTEGER,
        created_at TEXT)""")
    cols = {r[1] for r in c.execute("PRAGMA table_info(trades)")}
    for col, typ in (("realized_pnl", "REAL"), ("currency", "TEXT"), ("fx", "REAL"),
                     ("realized_krw", "REAL"), ("realized_approx", "INTEGER")):   # 이전 버전 DB 호환
        if col not in cols:
            c.execute(f"ALTER TABLE trades ADD COLUMN {col} {typ}")
    return c


def check_trade(ips: IPS, valued: pd.DataFrame, ticker: str, side: str, amount_usd: float,
                name: str = "", fields: dict | None = None) -> dict:
    """저장 전 검사. 반환: {'blocked': [...], 'violations': [...], 'missing': [...]}"""
    fields = fields or {}
    t = ticker.upper().strip()
    blocked, violations, missing = [], [], []

    if side == "매수":
        reason = ips.is_banned(t, name)
        if reason:
            blocked.append(reason)
        for k, label in CHECKLIST.items():
            if len(str(fields.get(k, "")).strip()) < MIN_LEN:
                missing.append(label)

        total = valued["value_usd"].sum() + amount_usd
        bucket = ips.bucket_of(t)
        if bucket == "satellite" and total > 0:
            cur = float(valued.loc[valued["ticker"] == t, "value_usd"].sum())
            w_after = (cur + amount_usd) / total
            cap = ips.rules.get("max_single_stock", 0.07)
            if w_after > cap:
                violations.append(f"매수 후 {t} 비중 {w_after:.1%} > 한도 {cap:.0%}")
            sat_names = set(valued.loc[(valued["bucket"] == "satellite") & (valued["ticker"] != "CASH"), "ticker"])
            if t not in sat_names and len(sat_names) >= ips.rules.get("max_satellite_count", 10):
                violations.append(f"위성 {len(sat_names)}종목 보유 중 — 신규 편입 전 기존 종목 정리 필요")
    return {"blocked": blocked, "violations": violations, "missing": missing}


def save_trade(record: dict, violations: list[str], override_reason: str = "", db: Path | str = DB) -> int:
    with _conn(db) as c:
        cur = c.execute(
            """INSERT INTO trades (trade_date, account, ticker, side, quantity, price, thesis, fundamentals,
               valuation, plan, break_condition, violations, override_reason, compliant, created_at,
               realized_pnl, currency, fx, realized_krw, realized_approx)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (str(record.get("trade_date", date.today())), record.get("account", "B"),
             record["ticker"].upper(), record["side"], float(record.get("quantity", 0)),
             float(record.get("price", 0)), record.get("thesis", ""), record.get("fundamentals", ""),
             record.get("valuation", ""), record.get("plan", ""), record.get("break_condition", ""),
             " | ".join(violations), override_reason, 0 if violations else 1,
             datetime.now().isoformat(timespec="seconds"),
             record.get("realized_pnl"), record.get("currency"), record.get("fx"),
             record.get("realized_krw"), record.get("realized_approx")))
        return int(cur.lastrowid)


def load_trades(db: Path | str = DB) -> pd.DataFrame:
    with _conn(db) as c:
        return pd.read_sql_query("SELECT * FROM trades ORDER BY trade_date DESC, id DESC", c)


def compliance_rate(trades: pd.DataFrame, month: str | None = None) -> float | None:
    """규칙 준수율 = 준수 거래 / 전체 거래. month='YYYY-MM' 이면 해당 월만."""
    df = trades
    if month:
        df = df[df["trade_date"].astype(str).str.startswith(month)]
    if df.empty:
        return None
    return float(df["compliant"].mean())


def realized_by_currency(trades: pd.DataFrame, year: str) -> dict:
    """해당 연도 매도 실현 손익 합계 {통화: 금액}. 세금 계산용 기초 자료."""
    if trades.empty or "realized_pnl" not in trades:
        return {}
    df = trades[trades["trade_date"].astype(str).str.startswith(year) & trades["realized_pnl"].notna()]
    return {k: float(v) for k, v in df.groupby("currency")["realized_pnl"].sum().items()}
