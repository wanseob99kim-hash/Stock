"""M10 성과 기록: 날짜별 스냅샷 저장 → 위성·코어·전체 수익률을 벤치마크(VOO)와 비교.

수익률은 '직전 스냅샷의 보유 수량'에 새 가격을 곱해 구간별로 계산하고 이어 붙인다(시간가중).
그래서 적립·추가매수로 평가액이 늘어도 수익률에는 섞이지 않는다.
한계: 매도한 종목은 매도가를 모르므로 그 구간 수익률을 0%로 본다.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "history.db"
BENCH = "benchmark"
GROUPS = ["satellite", "core", "total", BENCH]


def _conn(db: Path | str = DB) -> sqlite3.Connection:
    c = sqlite3.connect(db)
    c.execute("""CREATE TABLE IF NOT EXISTS snapshots (
        snap_date TEXT, ticker TEXT, bucket TEXT,
        quantity REAL, price_usd REAL, value_usd REAL, saved_at TEXT,
        PRIMARY KEY (snap_date, ticker))""")
    return c


def save_snapshot(valued: pd.DataFrame, snap_date: str, bench_price: float | None,
                  bench_ticker: str = "VOO", db: Path | str = DB) -> None:
    """그날의 보유 현황을 저장. 같은 날 다시 저장하면 덮어쓴다(마지막 값 유지)."""
    now = datetime.now().isoformat(timespec="seconds")
    rows = [(snap_date, r.ticker, r.bucket, float(r.quantity), float(r.price_usd), float(r.value_usd), now)
            for r in valued.itertuples()]
    if bench_price:
        rows.append((snap_date, f"^{bench_ticker}", BENCH, 1.0, float(bench_price), float(bench_price), now))
    with _conn(db) as c:
        c.execute("DELETE FROM snapshots WHERE snap_date = ?", (snap_date,))
        c.executemany("INSERT INTO snapshots VALUES (?,?,?,?,?,?,?)", rows)


def load_snapshots(db: Path | str = DB) -> pd.DataFrame:
    with _conn(db) as c:
        return pd.read_sql_query("SELECT * FROM snapshots ORDER BY snap_date", c)


def _group_mask(df: pd.DataFrame, group: str) -> pd.Series:
    if group == "satellite":
        return df["bucket"] == "satellite"
    if group == "core":
        return df["bucket"].str.startswith("core")
    if group == "total":
        return df["bucket"] != BENCH
    return df["bucket"] == BENCH


def chain_returns(snaps: pd.DataFrame) -> pd.DataFrame:
    """스냅샷 날짜별 누적 수익률 (index=snap_date, columns=GROUPS). 첫날은 0."""
    if snaps.empty:
        return pd.DataFrame(columns=GROUPS)
    dates = sorted(snaps["snap_date"].unique())
    growth = {g: [1.0] for g in GROUPS}
    for d0, d1 in zip(dates, dates[1:]):
        prev = snaps[snaps["snap_date"] == d0]
        new_px = snaps[snaps["snap_date"] == d1].set_index("ticker")["price_usd"]
        for g in GROUPS:
            p = prev[_group_mask(prev, g)]
            base = (p["quantity"] * p["price_usd"]).sum()
            px1 = p["ticker"].map(new_px).fillna(p["price_usd"])
            r = (p["quantity"] * px1).sum() / base if base else 1.0
            growth[g].append(growth[g][-1] * r)
    return pd.DataFrame({g: [x - 1 for x in v] for g, v in growth.items()}, index=pd.Index(dates, name="date"))


def hypothetical_growth(valued: pd.DataFrame, hist_usd: pd.DataFrame, bench: pd.Series) -> pd.DataFrame:
    """참고용: '지금 보유 수량을 기간 처음부터 그대로 들고 있었다면'의 누적 수익률.

    시세 히스토리가 없는 종목(현금 등)은 현재가로 고정한다.
    """
    idx = hist_usd.index
    px = hist_usd.reindex(idx).ffill().bfill()
    out = {}
    for g in ["satellite", "core", "total"]:
        v = valued[_group_mask(valued, g)]
        series = pd.Series(0.0, index=idx)
        for r in v.itertuples():
            series += r.quantity * (px[r.ticker] if r.ticker in px else r.price_usd)
        out[g] = series / series.iloc[0] - 1 if len(series) and series.iloc[0] else series * 0
    b = bench.reindex(idx).ffill().bfill()
    out[BENCH] = b / b.iloc[0] - 1
    return pd.DataFrame(out, index=idx)
