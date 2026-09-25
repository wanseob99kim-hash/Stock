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


def _market_of(ticker: str) -> str:
    return "KR" if ticker.isdigit() else "US"


def _adjust_cash(df: pd.DataFrame, account: str, delta: float) -> pd.DataFrame:
    """같은 계좌 CASH(달러) 수량에 delta 를 더한 새 표. 부족하면 ValueError."""
    m = (df["account"] == account) & (df["ticker"] == "CASH")
    have = float(df.loc[m, "quantity"].sum())
    if have + delta < -1e-9:
        raise ValueError(f"{account}계좌 현금 부족: 보유 ${have:,.2f}, 필요 ${-delta:,.2f}")
    if m.any():
        return df.assign(quantity=df["quantity"].where(~m, have + delta))
    row = pd.DataFrame([[account, "CASH", "US", delta, 1.0, "", ""]], columns=COLUMNS)
    return pd.concat([df, row], ignore_index=True)


def apply_trade(holdings: pd.DataFrame, account: str, ticker: str, side: str, quantity: float, price: float,
                sector: str = "", thesis: str = "", adjust_cash: bool = False) -> tuple[pd.DataFrame, float | None]:
    """거래 1건을 반영한 새 보유 종목 표와 실현 손익(현지통화, 매수면 None). 원본은 바꾸지 않는다.

    매수: 수량 합산, 평단은 가중평균. 매도: 평단 유지, 수량 0이면 행 삭제.
    adjust_cash=True 이면 미국 종목 거래 금액을 같은 계좌 CASH 에서 빼거나 더한다.
    """
    t = str(ticker).upper().strip()
    if quantity <= 0 or price <= 0:
        raise ValueError("수량과 단가는 0보다 커야 합니다.")
    df = holdings.copy().reset_index(drop=True)
    m = (df["account"] == account) & (df["ticker"] == t)
    q0 = float(df.loc[m, "quantity"].sum())
    realized = None

    if side == "매수":
        if m.any():
            i = df.index[m][0]
            a0 = float(df.at[i, "avg_cost"])
            df.at[i, "quantity"] = q0 + quantity
            df.at[i, "avg_cost"] = (q0 * a0 + quantity * price) / (q0 + quantity)
            for col, val in (("sector", sector), ("thesis", thesis)):
                if val and not str(df.at[i, col]).strip():
                    df.at[i, col] = val
        else:
            row = pd.DataFrame([[account, t, _market_of(t), quantity, price, sector, thesis]], columns=COLUMNS)
            df = pd.concat([df, row], ignore_index=True)
    else:
        if not m.any():
            raise ValueError(f"{account}계좌에 보유하지 않은 종목입니다: {t}")
        if quantity > q0 + 1e-9:
            raise ValueError(f"매도 수량 {quantity:g} > 보유 수량 {q0:g} ({t})")
        i = df.index[m][0]
        realized = (price - float(df.at[i, "avg_cost"])) * quantity
        left = q0 - quantity
        df = df.drop(index=i) if left <= 1e-9 else df.assign(quantity=df["quantity"].where(df.index != i, left))

    if adjust_cash and _market_of(t) == "US":
        amount = quantity * price
        df = _adjust_cash(df, account, -amount if side == "매수" else amount)
    return df.reset_index(drop=True), realized
