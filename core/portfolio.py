"""M1 포트폴리오: 보유 종목 로드·평가·버킷 분류."""
from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .ips import IPS

ROOT = Path(__file__).resolve().parent.parent
HOLDINGS = ROOT / "data" / "holdings.csv"
EXAMPLE = ROOT / "data" / "holdings.example.csv"  # 처음 실행 시 복사해 쓰는 예시
COLUMNS = ["account", "ticker", "market", "quantity", "avg_cost", "sector", "thesis"]


def load_holdings(path: Path | str = HOLDINGS) -> pd.DataFrame:
    if not Path(path).exists() and Path(path) == HOLDINGS:
        shutil.copy(EXAMPLE, HOLDINGS)
    df = pd.read_csv(path, dtype={"ticker": str}).fillna({"sector": "", "thesis": "", "account": "B", "market": "US"})
    for c in COLUMNS:
        if c not in df:
            df[c] = ""
    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df["market"] = df["market"].astype(str).str.upper()
    df["account"] = df["account"].astype(str).str.upper()
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce").fillna(0.0)
    df["avg_cost"] = pd.to_numeric(df["avg_cost"], errors="coerce").fillna(0.0)
    df = df[df["ticker"] != ""]
    return df[COLUMNS].reset_index(drop=True)


def save_holdings(df: pd.DataFrame, path: Path | str = HOLDINGS) -> None:
    df[COLUMNS].to_csv(path, index=False, encoding="utf-8")


def value_portfolio(holdings: pd.DataFrame, prices_usd: dict, ips: IPS, usdkrw: float) -> pd.DataFrame:
    """종목별 평가액(USD)·비중·손익·버킷을 붙인 표."""
    df = holdings.copy()
    df["price_usd"] = df["ticker"].map(prices_usd).fillna(0.0)
    fx = df["market"].map(lambda m: usdkrw if m == "KR" else 1.0)
    df["cost_usd"] = df["quantity"] * df["avg_cost"] / fx
    df["value_usd"] = df["quantity"] * df["price_usd"]
    df["pnl_pct"] = (df["value_usd"] / df["cost_usd"] - 1).where(df["cost_usd"] > 0)
    df["bucket"] = df["ticker"].map(ips.bucket_of)
    total = df["value_usd"].sum()
    df["weight"] = df["value_usd"] / total if total else 0.0
    return df


def bucket_weights(valued: pd.DataFrame, ips: IPS) -> pd.DataFrame:
    """버킷별 현재 vs 목표 비중."""
    total = valued["value_usd"].sum()
    g = valued.groupby("bucket")["value_usd"].sum()
    rows = []
    for key, b in ips.buckets.items():
        v = float(g.get(key, 0.0))
        w = v / total if total else 0.0
        rows.append({
            "bucket": key, "name": b.name, "value_usd": v, "weight": w,
            "target": b.target, "drift": w - b.target,
            "band_low": b.band[0], "band_high": b.band[1],
            "out_of_band": not (b.band[0] <= w <= b.band[1]),
        })
    return pd.DataFrame(rows)
