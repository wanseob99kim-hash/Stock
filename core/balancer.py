"""M2 코어-위성 밸런서.

1) 적립금 배분 (매도 없는 리밸런싱):
   매수금액_i = max(0, 목표비중_i × (V + C) − V_i), 합이 C를 넘으면 비례 축소
2) 전체 리밸런싱: 목표 비중으로 되돌리는 매수/매도 금액
"""
from __future__ import annotations

import pandas as pd

from .ips import IPS


def contribution_plan(bw: pd.DataFrame, contribution: float) -> pd.DataFrame:
    """이번 달 적립금 C를 가장 모자란 버킷부터 채운다."""
    V = bw["value_usd"].sum()
    need = (bw["target"] * (V + contribution) - bw["value_usd"]).clip(lower=0)
    total_need = need.sum()
    if contribution <= 0:
        alloc = need * 0
    elif total_need <= contribution or total_need == 0:
        alloc = need.copy()
        leftover = contribution - total_need
        alloc += leftover * bw["target"]          # 남는 돈은 목표 비중대로
    else:
        alloc = need * (contribution / total_need)
    out = bw[["bucket", "name", "value_usd", "weight", "target"]].copy()
    out["buy_usd"] = alloc.round(2)
    out["weight_after"] = (out["value_usd"] + out["buy_usd"]) / (V + contribution) if (V + contribution) else 0
    return out


def rebalance_orders(bw: pd.DataFrame, ips: IPS, force: bool = False) -> pd.DataFrame:
    """밴드 이탈(또는 force=True) 시 목표 비중으로 되돌리는 금액. +매수 / −매도."""
    V = bw["value_usd"].sum()
    thr = ips.rules.get("drift_threshold", 0.05)
    out = bw[["bucket", "name", "weight", "target", "drift"]].copy()
    trigger = force or (bw["drift"].abs() > thr).any() or bw["out_of_band"].any()
    out["trade_usd"] = ((bw["target"] - bw["weight"]) * V).round(2) if trigger else 0.0
    out["action"] = out["trade_usd"].map(lambda x: "매수" if x > 0.5 else ("매도" if x < -0.5 else "유지"))
    return out


def satellite_split(valued: pd.DataFrame, buy_usd: float, cap: float, contribution: float) -> pd.DataFrame:
    """위성 배정액을 기존 위성 종목에 나누는 참고안.

    종목별 7% 한도까지 남은 여유(headroom)에 비례해 배분한다.
    어느 종목을 살지는 매수 체크리스트(M4)를 거쳐 사용자가 최종 결정한다.
    """
    sat = valued[(valued["bucket"] == "satellite") & (valued["ticker"] != "CASH")].copy()
    if sat.empty or buy_usd <= 0:
        return pd.DataFrame(columns=["ticker", "weight", "headroom_usd", "buy_usd"])
    V_after = valued["value_usd"].sum() + contribution
    sat["headroom_usd"] = (cap * V_after - sat["value_usd"]).clip(lower=0)
    room = sat["headroom_usd"].sum()
    if room <= 0:
        sat["buy_usd"] = 0.0
    else:
        sat["buy_usd"] = (sat["headroom_usd"] / room * min(buy_usd, room)).round(2)
    return sat[["ticker", "weight", "headroom_usd", "buy_usd"]]
