"""M3 포트폴리오 건강 검진 (100점 만점, 감점 방식). 기획서 4.3 표 그대로."""
from __future__ import annotations

from itertools import combinations

import pandas as pd

from .balancer import overweight_actions
from .ips import IPS

OVER_BY_BUY = 10   # 많이 사서 한도 초과: 규칙 위반
OVER_BY_GAIN = 5   # 올라서 한도 초과: 잘된 결과라 감점 완화, 비중 조정만 권고


def _trend_broken(series: pd.Series) -> bool:
    """종가 < 200일선 이면서 200일선이 20거래일 전보다 낮음."""
    s = series.dropna()
    if s.size < 220:
        return False
    ma = s.rolling(200).mean()
    return bool(s.iloc[-1] < ma.iloc[-1] and ma.iloc[-1] < ma.iloc[-21])


def drawdown_from_high(series: pd.Series) -> float | None:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[-1] / s.max() - 1)


def health_check(valued: pd.DataFrame, hist_usd: pd.DataFrame, ips: IPS) -> dict:
    r = ips.rules
    items = []  # (항목, 감점, 상세)

    sat = valued[(valued["bucket"] == "satellite") & (valued["ticker"] != "CASH")]
    n = len(sat)

    # 1. 위성 종목 수
    over = max(0, n - r.get("max_satellite_count", 10))
    items.append(("위성 종목 수", -5 * over, f"{n}개 (한도 {r.get('max_satellite_count', 10)}개)"))

    # 2. 단일 종목 비중
    cap = r.get("max_single_stock", 0.07)
    heavy = overweight_actions(valued, cap)
    by_gain = heavy["cause"] == "상승"
    items.append(("단일 종목 비중", -OVER_BY_BUY * int((~by_gain).sum()) - OVER_BY_GAIN * int(by_gain.sum()),
                  ", ".join(f"{r.ticker} {r.weight:.1%}({r.cause})" for r in heavy.itertuples())
                  or f"모두 {cap:.0%} 이하"))

    # 3. 섹터 집중
    ded, detail = 0, []
    if n:
        sec = sat.assign(sector=sat["sector"].replace("", "미분류")).groupby("sector")
        share = sec["value_usd"].sum() / sat["value_usd"].sum()
        cnt = sec.size()
        for s_name in share.index:
            if s_name == "미분류":
                continue
            if share[s_name] > r.get("max_sector_share", 0.5):
                ded -= 10
                detail.append(f"{s_name} 위성의 {share[s_name]:.0%}")
            if cnt[s_name] > r.get("max_stocks_per_sector", 3):
                ded -= 5
                detail.append(f"{s_name} {cnt[s_name]}종목")
        if (sat["sector"] == "").any():
            detail.append(f"섹터 미입력 {int((sat['sector'] == '').sum())}종목")
    items.append(("섹터 집중", ded, ", ".join(detail) or "양호"))

    # 4. 가짜 분산 (평균 상관계수)
    corr_val = None
    cols = [t for t in sat["ticker"] if t in hist_usd.columns]
    if len(cols) >= 2:
        rets = hist_usd[cols].pct_change().dropna(how="all")
        c = rets.corr()
        pairs = [c.loc[a, b] for a, b in combinations(cols, 2) if pd.notna(c.loc[a, b])]
        if pairs:
            corr_val = float(sum(pairs) / len(pairs))
    thr = r.get("correlation_threshold", 0.7)
    items.append(("가짜 분산(평균 상관계수)", -10 if (corr_val is not None and corr_val > thr) else 0,
                  f"{corr_val:.2f} (경고선 {thr})" if corr_val is not None else "데이터 부족"))

    # 5. 보유 사유 미기록
    no_thesis = sat[sat["thesis"].astype(str).str.strip() == ""]
    items.append(("보유 사유 미기록", -5 * len(no_thesis), ", ".join(no_thesis["ticker"]) or "모두 기록됨"))

    # 6. 추세 훼손
    broken = [t for t in sat["ticker"] if t in hist_usd and _trend_broken(hist_usd[t])]
    items.append(("추세 훼손(200일선 하향)", -5 * len(broken), ", ".join(broken) or "없음"))

    # 7. 금지 상품 보유
    banned = [t for t in valued["ticker"] if ips.is_banned(t)]
    items.append(("금지 상품 보유", -15 * len(banned), ", ".join(banned) or "없음"))

    # (참고) 이익 추정 하향 항목은 컨센서스 데이터 확보 후 2차에서 추가

    score = max(0, 100 + sum(d for _, d, _ in items))

    # 트레일링 경보: 고점 대비 -20% 위성 종목 → 물타기 4단계 게이트 대상
    ts = r.get("trailing_stop", 0.2)
    alerts = []
    for t in sat["ticker"]:
        if t in hist_usd:
            dd = drawdown_from_high(hist_usd[t])
            if dd is not None and dd <= -ts:
                alerts.append({"ticker": t, "drawdown": dd})

    return {
        "score": score,
        "passed": score >= r.get("health_pass_score", 70),
        "items": pd.DataFrame(items, columns=["항목", "감점", "상세"]),
        "trailing_alerts": alerts,
        "avg_corr": corr_val,
    }
