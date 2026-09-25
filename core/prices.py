"""시세 수집 (yfinance). 실패하면 캐시 → 평단 순으로 대체한다.

- 미국: 티커 그대로 (VOO, NVDA)
- 한국: 6자리 코드 → '.KS'(코스피) 로 조회, 없으면 '.KQ'(코스닥) 재시도
- 한국 주식은 원/달러 환율(KRW=X)로 USD 환산
2차 단계에서 증권사 실시간 API(한국투자증권 KIS)로 교체할 자리다.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE = ROOT / "data" / "cache"
CACHE.mkdir(parents=True, exist_ok=True)
QUOTE_CACHE = CACHE / "quotes.json"


def yf_symbol(ticker: str, market: str) -> str:
    t = str(ticker).upper()
    if market.upper() == "KR" and not t.endswith((".KS", ".KQ")):
        return f"{t}.KS"
    return t


def _load_cache() -> dict:
    if QUOTE_CACHE.exists():
        try:
            return json.loads(QUOTE_CACHE.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _save_cache(c: dict) -> None:
    QUOTE_CACHE.write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")


def fetch_history(symbols: list[str], period: str = "1y") -> pd.DataFrame:
    """종가 히스토리 (열 = 심볼). 실패 시 빈 DataFrame."""
    if not symbols:
        return pd.DataFrame()
    try:
        import yfinance as yf
        df = yf.download(symbols, period=period, auto_adjust=True, progress=False, threads=True)
        if df is None or df.empty:
            return pd.DataFrame()
        close = df["Close"] if "Close" in df else df
        if isinstance(close, pd.Series):
            close = close.to_frame(symbols[0])
        return close.dropna(how="all")
    except Exception:
        return pd.DataFrame()


def _has_data(hist: pd.DataFrame, s: str) -> bool:
    return not hist.empty and s in hist and hist[s].dropna().size > 0


def _retry_missing(hist: pd.DataFrame, symbols: list[str], period: str) -> pd.DataFrame:
    """hist 에 데이터가 없는 심볼만 하나씩 다시 받아 합친다."""
    for s in symbols:
        if _has_data(hist, s):
            continue
        h2 = fetch_history([s], period)
        if not _has_data(h2, s):
            continue
        hist = h2[[s]] if hist.empty else hist.drop(columns=s, errors="ignore").join(h2[[s]], how="outer")
    return hist


def get_quotes(holdings: pd.DataFrame, period: str = "1y") -> tuple[dict, pd.DataFrame, float, dict]:
    """보유 종목 현재가(USD)와 히스토리.

    반환: (price_usd: {ticker: price}, history_usd: DataFrame, usdkrw, source: {ticker: 'live'|'cache'|'avg_cost'})
    """
    cache = _load_cache()
    rows = holdings[holdings["ticker"].str.upper() != "CASH"]
    sym_map = {r.ticker: yf_symbol(r.ticker, r.market) for r in rows.itertuples()}
    symbols = sorted(set(sym_map.values()) | {"KRW=X"})
    hist = fetch_history(symbols, period)

    # 코스닥 재시도: .KS 로 데이터가 없으면 .KQ
    for t, s in list(sym_map.items()):
        if s.endswith(".KS") and (hist.empty or s not in hist or hist[s].dropna().empty):
            alt = s.replace(".KS", ".KQ")
            h2 = fetch_history([alt], period)
            if not h2.empty and alt in h2:
                hist = hist.join(h2, how="outer") if not hist.empty else h2
                sym_map[t] = alt

    # 개별 재시도: 일괄 다운로드에서 일부 심볼만 빠지는 경우(요청 제한·타임아웃)가 있어 한 번 더 받는다
    hist = _retry_missing(hist, sorted(set(sym_map.values()) | {"KRW=X"}), period)

    usdkrw = None
    if not hist.empty and "KRW=X" in hist and hist["KRW=X"].dropna().size:
        usdkrw = float(hist["KRW=X"].dropna().iloc[-1])
        cache["KRW=X"] = {"price": usdkrw, "ts": time.time()}
    elif "KRW=X" in cache:
        usdkrw = cache["KRW=X"]["price"]
    usdkrw = usdkrw or 1400.0  # 최후 기본값

    prices, source = {}, {}
    hist_usd = pd.DataFrame(index=hist.index if not hist.empty else None)
    for r in rows.itertuples():
        s = sym_map[r.ticker]
        fx = usdkrw if r.market.upper() == "KR" else 1.0
        series = hist[s].dropna() if (not hist.empty and s in hist) else pd.Series(dtype=float)
        if series.size:
            px = float(series.iloc[-1])
            cache[s] = {"price": px, "ts": time.time()}
            source[r.ticker] = "live"
            hist_usd[r.ticker] = hist[s] / fx
        elif s in cache:
            px = cache[s]["price"]
            source[r.ticker] = "cache"
        else:
            px = float(r.avg_cost)
            source[r.ticker] = "avg_cost"
        prices[r.ticker] = px / fx
    prices["CASH"] = 1.0
    source["CASH"] = "live"
    _save_cache(cache)
    return prices, hist_usd, usdkrw, source
