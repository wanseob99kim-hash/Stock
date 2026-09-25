"""5순위 원화 기준 손익 · 해외주식 양도세 테스트."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ips import load_ips  # noqa: E402
from core.journal import load_trades, save_trade  # noqa: E402
from core.portfolio import apply_trade, value_portfolio  # noqa: E402
from core.tax import harvest_plan, krw_breakdown, realized_krw, tax_summary  # noqa: E402

IPS = load_ips()
TAX = {"deduction_krw": 2_500_000, "rate": 0.22, "taxable_accounts": ["B"]}
COLS = ["account", "ticker", "market", "quantity", "avg_cost", "avg_fx", "sector", "thesis"]


def holdings(rows):
    return pd.DataFrame(rows, columns=COLS)


# ── 평균 매입 환율 ──────────────────────────────────────────
def test_buy_updates_avg_fx_weighted_by_usd_cost():
    h = holdings([["B", "NVDA", "US", 10, 100.0, 1300.0, "", ""]])
    out, _ = apply_trade(h, "B", "NVDA", "매수", 10, 300.0, fx=1400.0)
    # 원화 원가 = 1,000×1,300 + 3,000×1,400 = 5,500,000 / 달러 원가 4,000 → 1,375
    assert out.loc[0, "avg_fx"] == pytest.approx(1375.0)


def test_buy_new_us_row_gets_fx_kr_row_does_not():
    h = holdings([])
    out, _ = apply_trade(h, "B", "NVDA", "매수", 1, 100.0, fx=1350.0)
    out, _ = apply_trade(out, "B", "005930", "매수", 1, 70000, fx=1350.0)
    assert out.set_index("ticker").loc["NVDA", "avg_fx"] == 1350.0
    assert np.isnan(out.set_index("ticker").loc["005930", "avg_fx"])


def test_buy_with_unknown_existing_fx_stays_unknown():
    h = holdings([["B", "NVDA", "US", 10, 100.0, np.nan, "", ""]])
    out, _ = apply_trade(h, "B", "NVDA", "매수", 10, 100.0, fx=1400.0)
    assert np.isnan(out.loc[0, "avg_fx"])


def test_apply_trade_works_without_avg_fx_column():
    h = pd.DataFrame([["B", "NVDA", "US", 1, 100.0, "", ""]], columns=[c for c in COLS if c != "avg_fx"])
    out, _ = apply_trade(h, "B", "NVDA", "매수", 1, 100.0)
    assert out.loc[0, "quantity"] == 2


# ── 원화 실현 손익 ──────────────────────────────────────────
def test_realized_krw_uses_both_fx_rates():
    # 매입 $100 @1,300 → 매도 $120 @1,400, 10주: 10×(168,000−130,000)
    v, approx = realized_krw("US", 10, 120.0, 100.0, 1300.0, 1400.0)
    assert v == pytest.approx(380_000) and not approx


def test_realized_krw_unknown_avg_fx_is_approximate():
    v, approx = realized_krw("US", 10, 120.0, 100.0, np.nan, 1400.0)
    assert v == pytest.approx(280_000) and approx


def test_realized_krw_kr_market_is_local():
    v, approx = realized_krw("KR", 10, 80000, 70000, np.nan, 1400.0)
    assert v == pytest.approx(100_000) and not approx


# ── 원화 기준 손익 분해 ─────────────────────────────────────
def test_krw_breakdown_splits_stock_and_fx_effect():
    h = holdings([["B", "NVDA", "US", 10, 100.0, 1300.0, "", ""],
                  ["B", "005930", "KR", 10, 70000, np.nan, "", ""],
                  ["B", "MSFT", "US", 1, 100.0, np.nan, "", ""]])
    v = value_portfolio(h, {"NVDA": 120.0, "005930": 80000 / 1400, "MSFT": 110.0}, IPS, 1400.0)
    b = krw_breakdown(v, 1400.0).set_index("ticker")
    assert b.loc["NVDA", "stock_effect"] == pytest.approx(10 * 20 * 1300)     # 260,000
    assert b.loc["NVDA", "fx_effect"] == pytest.approx(10 * 120 * 100)        # 120,000
    assert b.loc["NVDA", "pnl_krw"] == pytest.approx(1_680_000 - 1_300_000)
    assert b.loc["005930", "fx_effect"] == 0 and b.loc["005930", "pnl_krw"] == pytest.approx(100_000)
    assert not b.loc["MSFT", "fx_known"] and b.loc["MSFT", "fx_effect"] == 0


# ── 올해 양도세 ─────────────────────────────────────────────
def _trade(db, date_, account, ticker, currency, rkrw):
    save_trade({"ticker": ticker, "side": "매도", "trade_date": date_, "account": account,
                "currency": currency, "realized_krw": rkrw}, [], db=db)


def test_tax_summary_nets_gains_losses_and_filters(tmp_path):
    db = tmp_path / "j.db"
    _trade(db, "2026-03-01", "B", "NVDA", "USD", 4_000_000)
    _trade(db, "2026-04-01", "B", "MSFT", "USD", -500_000)      # 손익 통산
    _trade(db, "2026-05-01", "B", "005930", "KRW", 9_000_000)   # 국내 주식 제외
    _trade(db, "2026-06-01", "A", "VOO", "USD", 9_000_000)      # 과세 계좌 아님
    _trade(db, "2025-06-01", "B", "NVDA", "USD", 9_000_000)     # 다른 연도
    s = tax_summary(load_trades(db), "2026", TAX)
    assert s["gain_krw"] == pytest.approx(3_500_000)
    assert s["taxable_krw"] == pytest.approx(1_000_000)
    assert s["tax_krw"] == pytest.approx(220_000)
    assert s["remaining_deduction_krw"] == 0


def test_tax_summary_empty():
    s = tax_summary(pd.DataFrame(), "2026", TAX)
    assert s["gain_krw"] == 0 and s["remaining_deduction_krw"] == 2_500_000


# ── 절세 매도 후보 ──────────────────────────────────────────
@pytest.fixture
def breakdown():
    h = holdings([["B", "NVDA", "US", 100, 100.0, 1300.0, "", ""],     # 주당 원화 이익 (200×1400 − 130,000)=150,000
                  ["B", "MSFT", "US", 10, 500.0, 1400.0, "", ""],      # 주당 원화 손실 (400×1400 − 700,000)=−140,000
                  ["A", "VOO", "US", 10, 100.0, 1000.0, "", ""],       # 비과세 계좌
                  ["B", "005930", "KR", 10, 1000, np.nan, "", ""]])    # 국내
    v = value_portfolio(h, {"NVDA": 200.0, "MSFT": 400.0, "VOO": 500.0, "005930": 2000 / 1400}, IPS, 1400.0)
    return krw_breakdown(v, 1400.0)


def test_harvest_gains_up_to_remaining_deduction(breakdown):
    p = harvest_plan(breakdown, remaining_krw=2_500_000, taxable_krw=0, taxable_accounts=["B"])
    g = p["gains"].set_index("ticker")
    assert list(g.index) == ["NVDA"]
    assert g.loc["NVDA", "shares"] == 16                       # floor(2,500,000 / 150,000)
    assert g.loc["NVDA", "realized_krw"] == pytest.approx(2_400_000)
    assert p["losses"].empty


def test_harvest_losses_when_over_deduction(breakdown):
    p = harvest_plan(breakdown, remaining_krw=0, taxable_krw=1_000_000, taxable_accounts=["B"])
    loss = p["losses"].set_index("ticker")
    assert list(loss.index) == ["MSFT"]
    assert loss.loc["MSFT", "shares"] == 8                     # ceil(1,000,000 / 140,000)
    assert loss.loc["MSFT", "realized_krw"] == pytest.approx(-1_120_000)
    assert p["gains"].empty
