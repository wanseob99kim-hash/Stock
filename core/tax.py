"""M12 원화 기준 손익 · 해외주식 양도소득세 (참고용 추정).

원화 손익 = 주가 효과 + 환율 효과
  주가 효과 = 수량 × (현재가 − 평단) × 평균 매입 환율
  환율 효과 = 수량 × 현재가 × (현재 환율 − 평균 매입 환율)
평균 매입 환율(avg_fx)을 모르면 현재 환율로 대신 계산하고 환율 효과는 0 으로 둔다(추정 표시).

양도세: 해외 상장 주식·ETF만 대상, 같은 해 이익·손실 통산, 연 250만 원 공제 후 22%(지방세 포함).
국내 상장 주식(소액주주)은 제외. 실제 신고 금액은 증권사 양도세 계산 결과를 따른다.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

BREAKDOWN_COLS = ["ticker", "account", "market", "quantity", "price_usd", "value_krw", "cost_krw", "pnl_krw",
                  "stock_effect", "fx_effect", "fx_known", "gain_per_share_krw"]
PLAN_COLS = ["ticker", "shares", "amount_usd", "realized_krw"]


def _known(x) -> bool:
    return x is not None and not (isinstance(x, float) and math.isnan(x)) and x > 0


def realized_krw(market: str, quantity: float, price: float, avg_cost: float, avg_fx: float | None,
                 fx: float) -> tuple[float, bool]:
    """매도 1건의 원화 실현 손익과 추정 여부. 국내 종목은 원화 그대로."""
    if market == "KR":
        return quantity * (price - avg_cost), False
    if _known(avg_fx):
        return quantity * (price * fx - avg_cost * avg_fx), False
    return quantity * (price - avg_cost) * fx, True


def krw_breakdown(valued: pd.DataFrame, usdkrw: float) -> pd.DataFrame:
    """종목별 원화 평가액·원가·손익과 주가/환율 효과."""
    rows = []
    for r in valued.itertuples():
        q, px, cost = float(r.quantity), float(r.price_usd), float(r.avg_cost)
        value_krw = float(r.value_usd) * usdkrw
        if r.market == "KR":
            cost_krw, stock, fx_eff, known = q * cost, value_krw - q * cost, 0.0, True
        else:
            f0 = getattr(r, "avg_fx", np.nan)
            known = _known(f0)
            base_fx = f0 if known else usdkrw
            cost_krw = q * cost * base_fx
            stock = q * (px - cost) * base_fx
            fx_eff = q * px * (usdkrw - f0) if known else 0.0
        pnl = value_krw - cost_krw
        rows.append([r.ticker, r.account, r.market, q, px, value_krw, cost_krw, pnl, stock, fx_eff, known,
                     pnl / q if q else 0.0])
    return pd.DataFrame(rows, columns=BREAKDOWN_COLS)


def tax_summary(trades: pd.DataFrame, year: str, tax: dict) -> dict:
    """해당 연도 해외주식 양도차익(원)·과세표준·예상 세액·남은 공제."""
    ded, rate = float(tax.get("deduction_krw", 2_500_000)), float(tax.get("rate", 0.22))
    gain, approx = 0.0, 0
    if not trades.empty and "realized_krw" in trades:
        df = trades[trades["trade_date"].astype(str).str.startswith(year)
                    & (trades["currency"] == "USD")
                    & trades["account"].isin(tax.get("taxable_accounts", ["B"]))
                    & trades["realized_krw"].notna()]
        gain = float(df["realized_krw"].sum())
        approx = int(df["realized_approx"].fillna(0).sum()) if "realized_approx" in df else 0
    taxable = max(0.0, gain - ded)
    return {"gain_krw": gain, "taxable_krw": taxable, "tax_krw": taxable * rate,
            "remaining_deduction_krw": max(0.0, ded - max(gain, 0.0)), "approx_count": approx,
            "deduction_krw": ded, "rate": rate}


def harvest_plan(breakdown: pd.DataFrame, remaining_krw: float, taxable_krw: float,
                 taxable_accounts: list[str]) -> dict:
    """절세 매도 후보 (각 행은 '이 종목 하나로 한다면'의 대안).

    gains : 남은 공제 한도만큼 이익 종목 일부 매도 → 비과세로 이익 확정, 재매수하면 취득가가 올라감
    losses: 공제를 넘은 이익이 있을 때 손실 종목 매도로 상계할 수 있는 수량
    """
    c = breakdown[(breakdown["market"] == "US") & breakdown["account"].isin(taxable_accounts)
                  & (breakdown["ticker"] != "CASH") & (breakdown["quantity"] > 0)]
    gains, losses = [], []
    for r in c.itertuples():
        gps = r.gain_per_share_krw
        if remaining_krw > 0 and gps > 0:
            n = min(int(r.quantity), math.floor(remaining_krw / gps))
            if n > 0:
                gains.append([r.ticker, n, n * r.price_usd, n * gps])
        if taxable_krw > 0 and gps < 0:
            n = min(int(r.quantity), math.ceil(taxable_krw / -gps))
            losses.append([r.ticker, n, n * r.price_usd, n * gps])
    return {"gains": pd.DataFrame(gains, columns=PLAN_COLS), "losses": pd.DataFrame(losses, columns=PLAN_COLS)}
