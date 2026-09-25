"""M10 성과 기록 테스트 (네트워크 없이)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.history import chain_returns, hypothetical_growth, load_snapshots, save_snapshot  # noqa: E402


def make_valued(rows):
    """rows: (ticker, bucket, quantity, price_usd)"""
    df = pd.DataFrame(rows, columns=["ticker", "bucket", "quantity", "price_usd"])
    df["value_usd"] = df["quantity"] * df["price_usd"]
    return df


@pytest.fixture
def db(tmp_path):
    return tmp_path / "history.db"


def test_snapshot_same_day_overwrites(db):
    save_snapshot(make_valued([("VOO", "core_sp500", 10, 100.0)]), "2026-01-02", 100.0, db=db)
    save_snapshot(make_valued([("VOO", "core_sp500", 12, 101.0)]), "2026-01-02", 101.0, db=db)
    snaps = load_snapshots(db)
    voo = snaps[snaps["ticker"] == "VOO"]
    assert len(voo) == 1 and voo["quantity"].iloc[0] == 12
    assert (snaps["bucket"] == "benchmark").sum() == 1


def test_chain_returns_excludes_contributions(db):
    # 1일차: NVDA 10주 @100
    save_snapshot(make_valued([("NVDA", "satellite", 10, 100.0)]), "2026-01-02", 500.0, db=db)
    # 2일차: 가격 110 (+10%), 같은 날 10주 추가매수 → 평가액은 2배지만 수익률은 +10%여야 함
    save_snapshot(make_valued([("NVDA", "satellite", 20, 110.0)]), "2026-01-03", 505.0, db=db)
    r = chain_returns(load_snapshots(db))
    assert r.loc["2026-01-03", "satellite"] == pytest.approx(0.10)
    assert r.loc["2026-01-03", "benchmark"] == pytest.approx(0.01)
    assert r.loc["2026-01-02", "satellite"] == pytest.approx(0.0)


def test_chain_returns_links_periods(db):
    save_snapshot(make_valued([("NVDA", "satellite", 10, 100.0)]), "2026-01-02", 100.0, db=db)
    save_snapshot(make_valued([("NVDA", "satellite", 10, 110.0)]), "2026-01-03", 100.0, db=db)
    save_snapshot(make_valued([("NVDA", "satellite", 10, 99.0)]), "2026-01-04", 100.0, db=db)
    r = chain_returns(load_snapshots(db))
    assert r.loc["2026-01-04", "satellite"] == pytest.approx(-0.01)  # 1.1 * 0.9 - 1


def test_chain_returns_sold_ticker_counts_as_flat(db):
    save_snapshot(make_valued([("NVDA", "satellite", 10, 100.0), ("AVGO", "satellite", 10, 100.0)]),
                  "2026-01-02", 100.0, db=db)
    # AVGO 매도(다음 스냅샷에 없음) → 매도가 모름 → 0% 로 취급, NVDA +20%
    save_snapshot(make_valued([("NVDA", "satellite", 10, 120.0)]), "2026-01-03", 100.0, db=db)
    r = chain_returns(load_snapshots(db))
    assert r.loc["2026-01-03", "satellite"] == pytest.approx(0.10)


def test_chain_returns_empty():
    assert chain_returns(pd.DataFrame()).empty


def test_hypothetical_growth():
    idx = pd.date_range("2026-01-01", periods=3)
    hist = pd.DataFrame({"NVDA": [100.0, 110.0, 120.0], "VOO": [50.0, 50.0, 55.0]}, index=idx)
    valued = make_valued([("NVDA", "satellite", 1, 120.0), ("VOO", "core_sp500", 2, 55.0),
                          ("CASH", "core_cash", 100, 1.0)])
    g = hypothetical_growth(valued, hist, hist["VOO"])
    assert g["satellite"].iloc[-1] == pytest.approx(0.20)
    assert g["benchmark"].iloc[-1] == pytest.approx(0.10)
    # 전체: 시작 100+100+100=300 → 끝 120+110+100=330
    assert g["total"].iloc[-1] == pytest.approx(0.10)
