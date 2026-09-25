"""엔진 단위 테스트 (네트워크 없이 가짜 시세로)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.balancer import contribution_plan, rebalance_orders, satellite_split  # noqa: E402
from core.health import health_check  # noqa: E402
from core.ips import load_ips  # noqa: E402
from core.journal import check_trade, compliance_rate, load_trades, save_trade  # noqa: E402
from core.portfolio import bucket_weights, value_portfolio  # noqa: E402

IPS = load_ips()


def make_holdings(rows):
    return pd.DataFrame(rows, columns=["account", "ticker", "market", "quantity", "avg_cost", "sector", "thesis"])


@pytest.fixture
def valued():
    h = make_holdings([
        ["A", "VOO", "US", 35, 100, "", ""],      # 3,500
        ["A", "QQQM", "US", 15, 100, "", ""],     # 1,500
        ["A", "SCHD", "US", 10, 100, "", ""],     # 1,000
        ["A", "SGOV", "US", 10, 100, "", ""],     # 1,000
        ["B", "NVDA", "US", 20, 100, "Technology", "AI 칩 병목"],  # 2,000
        ["B", "MSFT", "US", 10, 100, "Technology", ""],          # 1,000
    ])
    prices = {t: 100.0 for t in h["ticker"]}
    prices["CASH"] = 1.0
    return value_portfolio(h, prices, IPS, 1400.0)


def test_ips_targets_sum_to_one():
    assert abs(sum(b.target for b in IPS.buckets.values()) - 1) < 1e-9


def test_bucket_classification():
    assert IPS.bucket_of("voo") == "core_sp500"
    assert IPS.bucket_of("360750") == "core_sp500"
    assert IPS.bucket_of("NVDA") == "satellite"
    assert IPS.bucket_of("CASH") == "core_cash"


def test_banned():
    assert IPS.is_banned("TQQQ")
    assert IPS.is_banned("122630", "KODEX 레버리지")
    assert IPS.is_banned("VOO") is None


def test_on_target_portfolio_no_rebalance(valued):
    bw = bucket_weights(valued, IPS)
    assert bw["drift"].abs().max() < 1e-9
    assert (rebalance_orders(bw, IPS)["action"] == "유지").all()


def test_contribution_fills_underweight_first(valued):
    # VOO 가 절반으로 떨어진 상황 → 적립금은 S&P500 버킷부터
    v = valued.copy()
    v.loc[v["ticker"] == "VOO", "value_usd"] = 1750
    bw = bucket_weights(v, IPS)
    plan = contribution_plan(bw, 1000)
    assert plan["buy_usd"].sum() == pytest.approx(1000, abs=0.05)
    top = plan.sort_values("buy_usd", ascending=False).iloc[0]
    assert top["bucket"] == "core_sp500"
    assert (plan["buy_usd"] >= 0).all()


def test_contribution_surplus_goes_by_target(valued):
    bw = bucket_weights(valued, IPS)
    plan = contribution_plan(bw, 1000)
    assert plan.set_index("bucket")["buy_usd"]["core_sp500"] == pytest.approx(350, abs=0.05)


def test_satellite_split_respects_cap(valued):
    split = satellite_split(valued, 5000, 0.07, 0)
    # 총자산 10,000 기준 7% = 700 → NVDA(2,000), MSFT(1,000) 모두 한도 초과라 여유 0
    assert split["buy_usd"].sum() == 0


def test_health_penalties(valued):
    hist = pd.DataFrame()
    res = health_check(valued, hist, IPS)
    items = res["items"].set_index("항목")["감점"]
    assert items["단일 종목 비중"] == -20            # NVDA 20%, MSFT 10% > 7%
    assert items["섹터 집중"] == -10                 # Technology 100% of 위성
    assert items["보유 사유 미기록"] == -5           # MSFT
    assert res["score"] == 65 and not res["passed"]


def test_health_correlation_and_trailing(valued):
    idx = pd.bdate_range("2025-01-01", periods=260)
    base = np.linspace(100, 60, 260)                 # 고점 대비 -40%
    hist = pd.DataFrame({"NVDA": base, "MSFT": base * 1.01}, index=idx)
    res = health_check(valued, hist, IPS)
    assert res["avg_corr"] > 0.7
    assert {a["ticker"] for a in res["trailing_alerts"]} == {"NVDA", "MSFT"}
    assert res["items"].set_index("항목")["감점"]["추세 훼손(200일선 하향)"] == -10


def test_trade_checks(valued, tmp_path):
    empty = {k: "" for k in ["thesis", "fundamentals", "valuation", "plan"]}
    r = check_trade(IPS, valued, "SOXL", "매수", 100, "", empty)
    assert r["blocked"]
    full = {k: "충분히 긴 설명 문장입니다" for k in empty}
    r = check_trade(IPS, valued, "AAPL", "매수", 100, "", full)
    assert not r["blocked"] and not r["missing"] and not r["violations"]
    r = check_trade(IPS, valued, "AAPL", "매수", 2000, "", full)
    assert any("한도" in v for v in r["violations"])

    db = tmp_path / "j.db"
    save_trade({"ticker": "AAPL", "side": "매수", "trade_date": "2026-09-24", **full}, [], db=db)
    save_trade({"ticker": "AAPL", "side": "매수", "trade_date": "2026-09-25", **full}, ["한도 초과"], "확신", db=db)
    t = load_trades(db)
    assert len(t) == 2
    assert compliance_rate(t, "2026-09") == pytest.approx(0.5)
