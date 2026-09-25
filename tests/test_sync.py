"""2순위 일기→보유 종목 반영, 3순위 비중 초과 조치 테스트."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.balancer import overweight_actions  # noqa: E402
from core.health import health_check  # noqa: E402
from core.ips import load_ips  # noqa: E402
from core.journal import load_trades, realized_by_currency, save_trade  # noqa: E402
from core.portfolio import apply_trade, value_portfolio  # noqa: E402

IPS = load_ips()
COLS = ["account", "ticker", "market", "quantity", "avg_cost", "sector", "thesis"]


def holdings(rows):
    return pd.DataFrame(rows, columns=COLS)


BASE = holdings([
    ["B", "NVDA", "US", 10, 100.0, "Technology", "AI"],
    ["B", "CASH", "US", 5000, 1.0, "", ""],
])


# ── apply_trade ─────────────────────────────────────────────
def test_buy_existing_updates_weighted_avg_cost():
    h, realized = apply_trade(BASE, "B", "NVDA", "매수", 10, 200.0)
    row = h[h["ticker"] == "NVDA"].iloc[0]
    assert row["quantity"] == 20 and row["avg_cost"] == pytest.approx(150.0)
    assert realized is None
    assert BASE.loc[0, "quantity"] == 10  # 원본은 그대로 (불변)


def test_buy_new_ticker_adds_row_with_market_and_thesis():
    h, _ = apply_trade(BASE, "B", "005930", "매수", 5, 70000, sector="Technology", thesis="메모리 회복")
    row = h[h["ticker"] == "005930"].iloc[0]
    assert row["market"] == "KR" and row["quantity"] == 5 and row["avg_cost"] == 70000
    assert row["sector"] == "Technology" and row["thesis"] == "메모리 회복"


def test_buy_same_ticker_other_account_is_separate_row():
    h, _ = apply_trade(BASE, "A", "NVDA", "매수", 1, 120.0)
    assert len(h[h["ticker"] == "NVDA"]) == 2


def test_sell_partial_records_realized_and_keeps_avg():
    h, realized = apply_trade(BASE, "B", "NVDA", "매도", 4, 130.0)
    row = h[h["ticker"] == "NVDA"].iloc[0]
    assert row["quantity"] == 6 and row["avg_cost"] == 100.0
    assert realized == pytest.approx(120.0)  # (130-100)*4


def test_sell_all_removes_row():
    h, realized = apply_trade(BASE, "B", "NVDA", "매도", 10, 90.0)
    assert "NVDA" not in set(h["ticker"])
    assert realized == pytest.approx(-100.0)


def test_sell_more_than_held_raises():
    with pytest.raises(ValueError, match="보유 수량"):
        apply_trade(BASE, "B", "NVDA", "매도", 11, 100.0)
    with pytest.raises(ValueError, match="보유하지 않은"):
        apply_trade(BASE, "B", "AAPL", "매도", 1, 100.0)


def test_cash_adjust_for_us_trades():
    h, _ = apply_trade(BASE, "B", "NVDA", "매수", 10, 100.0, adjust_cash=True)
    assert h.loc[h["ticker"] == "CASH", "quantity"].iloc[0] == pytest.approx(4000)
    h, _ = apply_trade(BASE, "B", "NVDA", "매도", 10, 110.0, adjust_cash=True)
    assert h.loc[h["ticker"] == "CASH", "quantity"].iloc[0] == pytest.approx(6100)
    with pytest.raises(ValueError, match="현금"):
        apply_trade(BASE, "B", "NVDA", "매수", 100, 100.0, adjust_cash=True)


def test_invalid_quantity_or_price_raises():
    with pytest.raises(ValueError):
        apply_trade(BASE, "B", "NVDA", "매수", 0, 100.0)
    with pytest.raises(ValueError):
        apply_trade(BASE, "B", "NVDA", "매수", 1, 0)


# ── 실현 손익 기록 ──────────────────────────────────────────
def test_realized_pnl_saved_and_summed(tmp_path):
    db = tmp_path / "j.db"
    save_trade({"ticker": "NVDA", "side": "매도", "trade_date": "2026-03-01", "realized_pnl": 120.0,
                "currency": "USD"}, [], db=db)
    save_trade({"ticker": "005930", "side": "매도", "trade_date": "2026-04-01", "realized_pnl": 50000.0,
                "currency": "KRW"}, [], db=db)
    save_trade({"ticker": "NVDA", "side": "매도", "trade_date": "2025-12-01", "realized_pnl": 999.0,
                "currency": "USD"}, [], db=db)
    s = realized_by_currency(load_trades(db), "2026")
    assert s == {"USD": pytest.approx(120.0), "KRW": pytest.approx(50000.0)}


# ── 비중 초과 조치 ──────────────────────────────────────────
@pytest.fixture
def valued():
    h = holdings([
        ["A", "VOO", "US", 80, 100.0, "", ""],           # 평가 8,000 / 원가 8,000
        ["B", "NVDA", "US", 10, 50.0, "Technology", "x"],  # 평가 1,000 / 원가 500 → 상승으로 초과
        ["B", "MSFT", "US", 10, 100.0, "Technology", "x"],  # 평가 1,000 / 원가 1,000 → 매수로 초과
    ])
    return value_portfolio(h, {"VOO": 100.0, "NVDA": 100.0, "MSFT": 100.0}, IPS, 1400.0)


def test_overweight_actions_amounts_and_cause(valued):
    a = overweight_actions(valued, 0.07).set_index("ticker")
    # 총자산 10,000 · 한도 7% = 700 → 초과 300
    assert a.loc["NVDA", "trim_usd"] == pytest.approx(300)
    assert a.loc["NVDA", "trim_shares"] == 3
    assert a.loc["NVDA", "dilute_usd"] == pytest.approx(1000 / 0.07 - 10000)
    assert a.loc["NVDA", "gain_on_trim_usd"] == pytest.approx(150)  # 3주 × (100-50)
    assert a.loc["NVDA", "cause"] == "상승"   # 원가 기준 500/9,500 = 5.3% ≤ 7%
    assert a.loc["MSFT", "cause"] == "매수"   # 원가 기준 1,000/9,500 = 10.5% > 7%


def test_overweight_actions_empty_when_within_cap(valued):
    assert overweight_actions(valued, 0.5).empty


def test_health_penalty_softer_when_overweight_by_gain(valued):
    items = health_check(valued, pd.DataFrame(), IPS)["items"].set_index("항목")
    assert items.loc["단일 종목 비중", "감점"] == -15  # 상승 -5 + 매수 -10
    assert "상승" in items.loc["단일 종목 비중", "상세"]
