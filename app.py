"""주식앱 MVP — 코어-위성 포트폴리오 운영 시스템.

실행:  streamlit run app.py
기획서: 주식앱 PDCA 기획서 (M1 보유종목 · M2 밸런서 · M3 계좌 검진 · M4 투자일기)
"""
from __future__ import annotations

from datetime import date
from io import StringIO

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.balancer import contribution_plan, overweight_actions, rebalance_orders, satellite_split
from core.health import health_check
from core.history import chain_returns, hypothetical_growth, load_snapshots, save_snapshot
from core.ips import load_ips
from core.journal import CHECKLIST, check_trade, compliance_rate, load_trades, realized_by_currency, save_trade
from core.portfolio import apply_trade, bucket_weights, load_holdings, save_holdings, value_portfolio
from core.prices import fetch_history, get_quotes
from core.review import STATUSES, load_reviews, review_status, save_review
from core.tax import harvest_plan, krw_breakdown, realized_krw, tax_summary

st.set_page_config(page_title="코어-위성 포트폴리오", layout="wide")

SERIES_1 = "#2a78d6"   # 현재 비중 (categorical slot 1)
TARGET_INK = "#52514e"  # 목표 표시 (text-secondary 계열, 시리즈 색 아님)
BAND_FILL = "#e8e6df"  # 허용 밴드 (recessive)
SERIES_2 = "#eb6834"   # 코어 (categorical slot 2)
PERF_LINES = [("satellite", "위성", SERIES_1, "solid"), ("core", "코어", SERIES_2, "solid")]


@st.cache_data(ttl=900, show_spinner="시세를 불러오는 중…")
def _quotes(holdings_json: str):
    h = pd.read_json(StringIO(holdings_json), dtype={"ticker": str})
    return get_quotes(h)


@st.cache_data(ttl=900, show_spinner=False)
def _benchmark(symbol: str) -> pd.Series:
    df = fetch_history([symbol], "1y")
    return df[symbol].dropna() if symbol in df else pd.Series(dtype=float)


def fmt_usd(x: float) -> str:
    return f"${x:,.0f}"


def fmt_krw(x: float) -> str:
    return f"{x / 10_000:+,.0f}만 원" if abs(x) >= 10_000 else f"{x:+,.0f}원"


ips = load_ips()
holdings = load_holdings()
prices, hist, usdkrw, source = _quotes(holdings.to_json())
valued = value_portfolio(holdings, prices, ips, usdkrw)
bw = bucket_weights(valued, ips)
health = health_check(valued, hist, ips)
trades = load_trades()
SAT_TICKERS = [t for t, b in zip(valued["ticker"], valued["bucket"]) if b == "satellite" and t != "CASH"]
reviews = review_status(SAT_TICKERS, load_reviews(), trades, date.today(), ips.rules.get("thesis_review_days", 90))
bench = hist[ips.benchmark].dropna() if ips.benchmark in hist else _benchmark(ips.benchmark)

# 날짜별 자산 기록: 평단으로 대체된 종목이 있으면 수익률이 왜곡되므로 저장하지 않는다
if "avg_cost" not in source.values():
    save_snapshot(valued, date.today().isoformat(), float(bench.iloc[-1]) if bench.size else None, ips.benchmark)

st.title("코어-위성 포트폴리오")
st.caption(f"투자정책서 {ips.version} · 원/달러 {usdkrw:,.1f} · 시세는 15분 지연될 수 있음(yfinance)")

stale = [t for t, s in source.items() if s != "live"]
if stale:
    st.warning("실시간 시세를 못 받은 종목: " + ", ".join(f"{t}({source[t]})" for t in stale)
               + " — 캐시 또는 평단으로 계산했습니다.")

tab_dash, tab_perf, tab_tax, tab_contrib, tab_health, tab_review, tab_journal, tab_hold, tab_ips = st.tabs(
    ["대시보드", "성과 비교", "원화·세금", "이번 달 적립", "계좌 검진", "매수 이유 점검", "투자일기", "보유 종목",
     "투자정책서"])

# ── 대시보드 ────────────────────────────────────────────────
with tab_dash:
    total = valued["value_usd"].sum()
    cost = valued["cost_usd"].sum()
    month = date.today().strftime("%Y-%m")
    cr = compliance_rate(trades, month)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("총자산", fmt_usd(total))
    c2.metric("평가손익", fmt_usd(total - cost), f"{(total / cost - 1):.1%}" if cost else None)
    c3.metric("포트폴리오 건강 점수", f"{health['score']}점")
    (c3.success if health["passed"] else c3.error)("합격" if health["passed"] else "점검 필요 — 계좌 검진 탭 확인")
    c4.metric("이번 달 규칙 준수율", f"{cr:.0%}" if cr is not None else "거래 없음")
    broken = reviews.loc[reviews["last_status"] == "깨짐", "ticker"].tolist()
    due = reviews.loc[reviews["due"] & (reviews["last_status"] != "깨짐"), "ticker"].tolist()
    if broken:
        st.error("매수 이유가 깨진 종목: " + ", ".join(broken) + " — 정리를 검토하세요 (매수 이유 점검 탭)")
    if due:
        st.warning("매수 이유 점검할 때가 된 종목: " + ", ".join(due) + " — 매수 이유 점검 탭")

    st.subheader("자산군 비중: 현재 vs 목표")
    order = bw.iloc[::-1]
    fig = go.Figure()
    fig.add_bar(y=order["name"], x=order["band_high"] - order["band_low"], base=order["band_low"],
                orientation="h", marker_color=BAND_FILL, name="허용 밴드", width=0.8,
                hovertemplate="%{y}<br>허용 %{base:.0%} ~ %{x:.0%}<extra></extra>")
    fig.add_bar(y=order["name"], x=order["weight"], orientation="h", marker_color=SERIES_1,
                name="현재 비중", width=0.4, text=[f"{w:.1%}" for w in order["weight"]],
                textposition="outside",
                hovertemplate="%{y}<br>현재 %{x:.1%}<extra></extra>")
    fig.add_scatter(y=order["name"], x=order["target"], mode="markers", name="목표",
                    marker=dict(symbol="line-ns", size=22, line=dict(width=3, color=TARGET_INK)),
                    hovertemplate="%{y}<br>목표 %{x:.0%}<extra></extra>")
    fig.update_layout(barmode="overlay", height=320, margin=dict(l=10, r=40, t=10, b=10),
                      xaxis=dict(tickformat=".0%", gridcolor="#eeeeee", rangemode="tozero"),
                      legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    show = bw.assign(**{
        "현재": bw["weight"].map("{:.1%}".format), "목표": bw["target"].map("{:.0%}".format),
        "이탈": bw["drift"].map("{:+.1%}".format), "평가액": bw["value_usd"].map(fmt_usd),
        "상태": bw["out_of_band"].map({True: "밴드 이탈", False: "정상"})})
    st.dataframe(show[["name", "평가액", "현재", "목표", "이탈", "상태"]].rename(columns={"name": "자산군"}),
                 hide_index=True, use_container_width=True)

    st.subheader("보유 종목")
    v = valued.sort_values("value_usd", ascending=False)
    st.dataframe(pd.DataFrame({
        "계좌": v["account"], "종목": v["ticker"],
        "자산군": v["bucket"].map(lambda k: ips.buckets[k].name),
        "평가액": v["value_usd"].map(fmt_usd), "비중": v["weight"].map("{:.1%}".format),
        "수익률": v["pnl_pct"].map(lambda x: f"{x:+.1%}" if pd.notna(x) else "-"),
        "보유 사유": v["thesis"]}), hide_index=True, use_container_width=True)

# ── 성과 비교 (M10) ─────────────────────────────────────────
def perf_chart(df: pd.DataFrame, bench_name: str) -> go.Figure:
    fig = go.Figure()
    for key, label, color, dash in PERF_LINES + [("benchmark", bench_name, TARGET_INK, "dash")]:
        fig.add_scatter(x=df.index, y=df[key], mode="lines", name=label,
                        line=dict(color=color, width=2, dash=dash),
                        hovertemplate=f"{label} %{{y:+.1%}}<extra></extra>")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                      yaxis=dict(tickformat="+.0%", gridcolor="#eeeeee", zeroline=True, zerolinecolor="#cccccc"),
                      legend=dict(orientation="h", y=-0.15), plot_bgcolor="rgba(0,0,0,0)")
    return fig


def perf_metrics(df: pd.DataFrame, bench_name: str) -> None:
    last = df.iloc[-1]
    m1, m2, m3 = st.columns(3)
    m1.metric("위성 누적 수익률", f"{last['satellite']:+.1%}")
    m2.metric(f"{bench_name} 누적 수익률", f"{last['benchmark']:+.1%}")
    gap = last["satellite"] - last["benchmark"]
    m3.metric(f"위성 − {bench_name}", f"{gap:+.1%}p", "위성이 앞섬" if gap >= 0 else "위성이 뒤처짐",
              delta_color="normal" if gap >= 0 else "inverse")


with tab_perf:
    bname = ips.benchmark
    st.markdown(f"위성(개별주)을 들고 있는 이유는 **{bname}보다 더 벌기 위해서**입니다. "
                f"위성이 3년 연속 {bname}보다 못하면 12월 점검에서 위성 비중 축소를 검토하세요.")

    st.subheader("실제 기록")
    snaps = load_snapshots()
    days = snaps["snap_date"].nunique() if not snaps.empty else 0
    if days < 2:
        st.info(f"기록 {days}일째입니다. 앱을 연 날마다 자동으로 저장되며, 이틀째부터 비교 그래프가 나옵니다.")
    else:
        real = chain_returns(snaps)
        st.caption(f"{real.index[0]} ~ {real.index[-1]} · {days}일 기록 · 적립·추가매수 금액은 수익률에서 제외(시간가중)")
        perf_metrics(real, bname)
        st.plotly_chart(perf_chart(real, bname), use_container_width=True)
        with st.expander("표로 보기"):
            st.dataframe(real.rename(columns={"satellite": "위성", "core": "코어", "total": "전체",
                                              "benchmark": bname}).map("{:+.2%}".format),
                         use_container_width=True)

    st.subheader("참고: 지금 구성을 1년 전부터 들고 있었다면")
    if hist.empty or not bench.size:
        st.info("시세 히스토리를 받지 못해 계산할 수 없습니다.")
    else:
        hypo = hypothetical_growth(valued, hist, bench)
        st.caption("현재 보유 수량 그대로 1년을 보유했다고 가정한 계산입니다. 실제 매매 시점과 다르며, "
                   "지금 살아남은 종목만 보므로 실제보다 좋게 나오기 쉽습니다(생존 편향).")
        perf_metrics(hypo, bname)
        st.plotly_chart(perf_chart(hypo, bname), use_container_width=True)

# ── 원화·세금 (M12) ──────────────────────────────────────────
PLAN_RENAME = {"ticker": "종목", "shares": "매도 주수", "amount_usd": "매도 금액", "realized_krw": "확정 손익(원)"}


def show_plan(df: pd.DataFrame) -> None:
    st.dataframe(df.assign(amount_usd=df["amount_usd"].map(fmt_usd), realized_krw=df["realized_krw"].map(fmt_krw))
                 .rename(columns=PLAN_RENAME), hide_index=True, use_container_width=True)


with tab_tax:
    bd = krw_breakdown(valued, usdkrw)
    st.subheader("원화 기준 손익")
    k1, k2, k3 = st.columns(3)
    k1.metric("평가손익 (원화)", fmt_krw(bd["pnl_krw"].sum()))
    k2.metric("주가 효과", fmt_krw(bd["stock_effect"].sum()), help="환율이 매입 때 그대로였다면의 손익")
    k3.metric("환율 효과", fmt_krw(bd["fx_effect"].sum()), help="매입 이후 원/달러 환율 변화로 생긴 손익")
    unknown = bd.loc[(bd["market"] == "US") & ~bd["fx_known"], "ticker"].tolist()
    if unknown:
        st.warning("평균 매입 환율 미입력: " + ", ".join(unknown) + " — 현재 환율로 추정해 환율 효과가 0으로 잡힙니다. "
                   "'보유 종목' 탭의 '평균 매입 환율' 칸을 채우세요(증권사 잔고의 매입환율).")
    st.dataframe(pd.DataFrame({
        "종목": bd["ticker"], "평가액": bd["value_krw"].map(lambda x: f"{x / 10_000:,.0f}만 원"),
        "손익": bd["pnl_krw"].map(fmt_krw), "주가 효과": bd["stock_effect"].map(fmt_krw),
        "환율 효과": bd["fx_effect"].map(fmt_krw),
        "환율": bd["fx_known"].map({True: "확정", False: "추정"})}), hide_index=True, use_container_width=True)

    year = str(date.today().year)
    tcfg = ips.tax
    ts = tax_summary(trades, year, tcfg)
    st.subheader(f"{year}년 해외주식 양도소득세 (추정)")
    t1, t2, t3 = st.columns(3)
    t1.metric("실현 양도차익", fmt_krw(ts["gain_krw"]))
    t2.metric("남은 기본공제", f"{ts['remaining_deduction_krw'] / 10_000:,.0f}만 원")
    t3.metric("예상 세액", f"{ts['tax_krw'] / 10_000:,.0f}만 원", help=f"(양도차익 − 공제) × {ts['rate']:.0%}")
    st.caption(f"대상: {', '.join(tcfg.get('taxable_accounts', ['B']))}계좌의 해외 상장 주식·ETF, 같은 해 이익·손실 통산. "
               "국내 상장 주식(소액주주)은 제외. 투자일기에 기록한 매도만 집계합니다. 신고·납부는 다음 해 5월.")
    if ts["approx_count"]:
        st.caption(f"매입 환율을 몰라 추정한 매도 {ts['approx_count']}건이 포함되어 있습니다.")

    plan = harvest_plan(bd, ts["remaining_deduction_krw"], ts["taxable_krw"], tcfg.get("taxable_accounts", ["B"]))
    if not plan["gains"].empty:
        st.markdown(f"**공제 한도 채우기** — 올해 남은 공제 {ts['remaining_deduction_krw'] / 10_000:,.0f}만 원까지 "
                    "이익을 세금 없이 확정할 수 있습니다. 팔고 다시 사면 취득가가 올라가 나중 세금이 줄어듭니다. "
                    "각 줄은 '이 종목 하나로 채운다면'의 대안입니다.")
        show_plan(plan["gains"])
    if not plan["losses"].empty:
        st.markdown(f"**손실로 상계하기** — 공제를 넘은 이익 {ts['taxable_krw'] / 10_000:,.0f}만 원을 "
                    "손실 종목 매도로 줄일 수 있습니다. 매수 이유가 깨진 종목부터 검토하세요.")
        show_plan(plan["losses"])
    st.info("12월에는 결제일 기준으로 연도가 정해집니다. 미국 주식은 거래 다음 날 결제되므로 마지막 거래일 "
            "며칠 전까지 매도하세요. 이 화면은 참고용 추정이며, 실제 신고는 증권사 양도세 계산 결과를 따르세요.")

# ── 이번 달 적립 (M2) ───────────────────────────────────────
with tab_contrib:
    st.markdown("적립금은 **가장 모자란 자산군부터** 채웁니다. 팔지 않고 비중을 맞추는 방법입니다.")
    contrib = st.number_input("이번 달 적립금 (USD)", min_value=0.0, value=2000.0, step=100.0)
    plan = contribution_plan(bw, contrib)
    st.dataframe(pd.DataFrame({
        "자산군": plan["name"], "매수 금액": plan["buy_usd"].map(fmt_usd),
        "현재": plan["weight"].map("{:.1%}".format), "매수 후": plan["weight_after"].map("{:.1%}".format),
        "목표": plan["target"].map("{:.0%}".format)}), hide_index=True, use_container_width=True)

    sat_buy = float(plan.loc[plan["bucket"] == "satellite", "buy_usd"].sum())
    if sat_buy > 0:
        st.markdown(f"**위성 배정 {fmt_usd(sat_buy)} 참고 배분** — 종목별 7% 한도까지 남은 여유에 비례. "
                    "매수 전 투자일기 체크리스트를 먼저 작성하세요.")
        split = satellite_split(valued, sat_buy, ips.rules["max_single_stock"], contrib)
        st.dataframe(split.assign(weight=split["weight"].map("{:.1%}".format),
                                  headroom_usd=split["headroom_usd"].map(fmt_usd),
                                  buy_usd=split["buy_usd"].map(fmt_usd))
                     .rename(columns={"ticker": "종목", "weight": "현재 비중", "headroom_usd": "한도까지 여유",
                                      "buy_usd": "참고 매수액"}), hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("리밸런싱 점검")
    orders = rebalance_orders(bw, ips)
    if (orders["action"] == "유지").all():
        st.success(f"모든 자산군이 목표 ±{ips.rules['drift_threshold']:.0%}p와 허용 밴드 안에 있습니다. 할 일 없음.")
    else:
        st.info("밴드 이탈이 있습니다. 적립금으로 먼저 보정하고, 부족할 때만 아래 매도를 검토하세요. "
                "전체 리밸런싱은 연 1회(12월)가 원칙입니다.")
        st.dataframe(pd.DataFrame({
            "자산군": orders["name"], "현재": orders["weight"].map("{:.1%}".format),
            "목표": orders["target"].map("{:.0%}".format), "조치": orders["action"],
            "금액": orders["trade_usd"].map(lambda x: fmt_usd(abs(x)))}), hide_index=True, use_container_width=True)

# ── 계좌 검진 (M3) ─────────────────────────────────────────
with tab_health:
    col1, col2 = st.columns([1, 3])
    col1.metric("포트폴리오 건강 점수", f"{health['score']} / 100")
    (col1.success if health["passed"] else col1.error)(
        "합격" if health["passed"] else f"{ips.rules['health_pass_score']}점 미만 — 감점 항목을 정리하세요")
    col2.dataframe(health["items"], hide_index=True, use_container_width=True)

    cap = ips.rules["max_single_stock"]
    over = overweight_actions(valued, cap)
    if not over.empty:
        st.subheader(f"비중 초과 종목 — 한도 {cap:.0%}로 맞추는 방법")
        st.markdown("**상승**: 매수 당시엔 한도 안이었는데 올라서 넘음(감점 5) · "
                    "**매수**: 원가 기준으로도 한도 초과(감점 10). 아래 셋 중 하나를 고르세요.")
        st.dataframe(pd.DataFrame({
            "종목": over["ticker"], "현재 비중": over["weight"].map("{:.1%}".format), "원인": over["cause"],
            "① 한도까지 매도": [f"{fmt_usd(u)} (약 {n}주)" for u, n in zip(over["trim_usd"], over["trim_shares"])],
            "② 절반만 매도": over["trim_usd"].map(lambda u: fmt_usd(u / 2)),
            "③ 안 팔고 다른 자산 적립": over["dilute_usd"].map(fmt_usd),
            "①의 예상 실현 이익": over["gain_on_trim_usd"].map(fmt_usd)}),
            hide_index=True, use_container_width=True)
        st.caption("① 매도 대금은 계좌에 남으니 '이번 달 적립'에서 모자란 자산군으로 옮기세요. "
                   "② 이익 일부만 확정하고 나머지는 추세를 따라갑니다. "
                   "③ 금액이 크면 현실적이지 않습니다 — 적립 몇 달 치인지 보고 판단하세요. "
                   "해외주식 양도차익은 연 250만 원 공제 후 22% 과세이니 연말 매도 시 ① 이익을 참고하세요.")

    st.subheader("트레일링 경보 → 물타기 4단계 게이트")
    if not health["trailing_alerts"]:
        st.success(f"고점 대비 -{ips.rules['trailing_stop']:.0%} 이하로 떨어진 위성 종목이 없습니다.")
    for a in health["trailing_alerts"]:
        t = a["ticker"]
        thesis = valued.loc[valued["ticker"] == t, "thesis"].iloc[0] or "(보유 사유 미기록)"
        with st.expander(f"{t} — 1년 고점 대비 {a['drawdown']:.0%}", expanded=True):
            st.markdown(f"**매수 당시 내러티브:** {thesis}")
            rv = reviews[reviews["ticker"] == t]
            if len(rv) and rv["last_review"].iloc[0]:
                st.markdown(f"**최근 매수 이유 점검:** {rv['last_review'].iloc[0]} · {rv['last_status'].iloc[0]}")
            g1 = st.checkbox("1. 기업가치: 내가 산 이유(매출·마진·경쟁력·가이던스)가 그대로인가?", key=f"g1{t}")
            g2 = st.checkbox("2. 가격: 지금 이익 기준으로 PER 밴드 하단 근처인가? (평단 대비가 아니라)", key=f"g2{t}")
            g3 = st.checkbox("3. 차트: 저점 갱신이 멈추고 올라가는 힘이 생겼는가?", key=f"g3{t}")
            w = float(valued.loc[valued["ticker"] == t, "weight"].iloc[0])
            g4 = w < ips.rules["max_single_stock"]
            st.markdown(f"4. 비중: 현재 {w:.1%} — {'추가 여유 있음' if g4 else '한도 도달, 추가매수 불가'}")
            if not g1:
                st.error("판정: 정리 검토 — 매수 근거가 깨졌다면 평단과 무관하게 판단합니다.")
            elif not (g2 and g3):
                st.warning("판정: 기다림 — 추가매수도 손절도 아닙니다. 기다리는 것도 투자입니다.")
            elif not g4:
                st.warning("판정: 추가매수 불가 — 비중 한도. 보유 유지.")
            else:
                st.success("판정: 분할 추가매수 허용 — 투자일기에 계획을 먼저 기록하세요.")

# ── 매수 이유 점검 (M11) ────────────────────────────────────
STATUS_HELP = {"유지": "산 이유 그대로 — 계속 보유", "주의": "흔들리는 신호 — 다음 실적에서 재확인",
               "깨짐": "깨지는 조건 충족 — 평단과 무관하게 정리 검토"}


def review_form(row) -> None:
    t = row.ticker
    thesis = valued.loc[valued["ticker"] == t, "thesis"].iloc[0] or "(보유 사유 미기록 — 보유 종목 탭에서 입력)"
    st.markdown(f"**산 이유:** {thesis}")
    if row.last_review:
        st.caption(f"최근 점검 {row.last_review} ({int(row.days_since)}일 전) · {row.last_status}")
    with st.form(f"review_{t}"):
        cond = st.text_input("이 이야기가 깨지는 조건", value=row.break_condition,
                             placeholder="예: Azure 성장률 20% 미만 2분기 연속")
        status = st.radio("판정", STATUSES, horizontal=True, format_func=lambda s: f"{s} — {STATUS_HELP[s]}")
        note = st.text_area("무엇을 보고 판단했나 (실적·가이던스·뉴스)", height=68)
        if not st.form_submit_button("점검 기록"):
            return
    if not cond.strip():
        st.error("깨지는 조건을 먼저 적으세요. 기준이 없으면 점검할 수 없습니다.")
        return
    try:
        save_review(t, status, note, cond, date.today().isoformat())
    except ValueError as e:
        st.error(str(e))
        return
    st.session_state["review_flash"] = f"{t} 점검을 기록했습니다: {status}"
    st.rerun()


with tab_review:
    every = ips.rules.get("thesis_review_days", 90)
    st.markdown(f"위성 종목마다 **{every}일에 한 번**(실적 발표 뒤가 좋음) '깨지는 조건'을 기준으로 산 이유가 "
                "아직 유효한지 확인합니다. 떨어지고 나서가 아니라 **떨어지기 전에** 판단 기준을 점검하는 것이 목적입니다.")
    if flash := st.session_state.pop("review_flash", None):
        st.success(flash)
    if reviews.empty:
        st.info("위성 종목이 없습니다.")
    for row in reviews.itertuples():
        mark = "🔴 깨짐" if row.last_status == "깨짐" else ("🟡 점검 필요" if row.due else "🟢 " + str(row.last_status))
        with st.expander(f"{row.ticker} — {mark}", expanded=bool(row.due)):
            review_form(row)
            if row.last_status == "깨짐":
                st.error("정리 검토: 투자일기에서 매도를 기록하면 보유 종목에 반영됩니다.")

    history = load_reviews()
    if not history.empty:
        st.subheader("점검 기록")
        st.dataframe(history[["review_date", "ticker", "status", "break_condition", "note"]]
                     .rename(columns={"review_date": "점검일", "ticker": "종목", "status": "판정",
                                      "break_condition": "깨지는 조건", "note": "메모"}),
                     hide_index=True, use_container_width=True)


# ── 투자일기 (M4) ─────────────────────────────────────────
def record_trade(rec: dict, violations: list, override: str, sync: bool, use_cash: bool, sector: str) -> None:
    """일기 저장 + (선택) 보유 종목 반영. 보유 반영이 불가능하면 아무것도 저장하지 않는다."""
    new_holdings = None
    if sync:
        pos = holdings[(holdings["account"] == rec["account"]) & (holdings["ticker"] == rec["ticker"])]
        if rec["side"] == "매도" and len(pos):
            rec["realized_krw"], rec["realized_approx"] = realized_krw(
                pos["market"].iloc[0], rec["quantity"], rec["price"], float(pos["avg_cost"].iloc[0]),
                float(pos["avg_fx"].iloc[0]), rec["fx"] or usdkrw)
        try:
            new_holdings, rec["realized_pnl"] = apply_trade(
                holdings, rec["account"], rec["ticker"], rec["side"], rec["quantity"], rec["price"],
                sector=sector, thesis=rec.get("thesis", ""), adjust_cash=use_cash, fx=rec["fx"])
        except ValueError as e:
            st.error(f"보유 종목에 반영할 수 없어 저장하지 않았습니다: {e}")
            return
    save_trade(rec, violations, override)
    msg = "저장했습니다." + (" (규칙 위반으로 기록됨)" if violations else "")
    if new_holdings is None:
        st.success(msg + " 보유 종목은 바꾸지 않았습니다.")
        return
    save_holdings(new_holdings)
    _quotes.clear()
    pnl = rec.get("realized_pnl")
    if pnl is not None:
        msg += f" 실현 손익 {pnl:+,.0f} {rec['currency']}"
        msg += f" (원화 {fmt_krw(rec['realized_krw'])})." if rec.get("realized_krw") is not None else "."
    st.session_state["journal_flash"] = msg + " 보유 종목에 반영했습니다."
    st.rerun()


# ── 투자일기 (M4) ──────────────────────────────────────────
with tab_journal:
    if flash := st.session_state.pop("journal_flash", None):
        st.success(flash)
    st.markdown("매수는 **네 칸을 모두 채워야** 저장됩니다. 금지 상품은 차단되고, 다른 규칙 위반은 사유를 남겨야 저장됩니다.")
    with st.form("trade"):
        a, b, c, d, e = st.columns(5)
        tdate = a.date_input("거래일", date.today())
        side = b.selectbox("구분", ["매수", "매도"])
        ticker = c.text_input("종목 (티커/코드)").upper().strip()
        qty = d.number_input("수량", min_value=0.0, step=1.0)
        price = e.number_input("단가(현지통화)", min_value=0.0, step=0.01)
        f1, f2, f3, f4 = st.columns(4)
        account = f1.selectbox("계좌", ["B", "A"], help="A=은퇴계좌(지수만), B=일반계좌")
        name = f2.text_input("상품명 (한국 ETF는 이름으로 레버리지/인버스/(H) 검사)")
        sector = f3.text_input("섹터 (위성 신규 매수 시)", help="예: Technology, Healthcare")
        trade_fx = f4.number_input("거래 환율 (원/달러, 미국 종목)", min_value=0.0, value=round(usdkrw, 1), step=0.1,
                                   help="증권사 체결 내역의 적용 환율. 원화 손익·양도세 계산에 씁니다.")
        fields = {k: st.text_area(label, height=68) for k, label in CHECKLIST.items()}
        brk = st.text_input("이 이야기가 깨지는 조건 (다모다란: 내러티브 무효 조건)")
        override = st.text_input("규칙 위반이 있을 때만: 그래도 하는 이유")
        s1, s2 = st.columns(2)
        sync = s1.checkbox("보유 종목에 자동 반영 (수량·평단)", value=True,
                           help="끄면 기록만 남깁니다. 과거 거래를 옮겨 적을 때 끄세요.")
        use_cash = s2.checkbox("달러 거래 금액을 CASH에서 빼기/더하기", value=False,
                               help="보유 종목의 CASH 행을 거래 금액만큼 조정합니다.")
        submitted = st.form_submit_button("검사 후 저장")

    if submitted:
        if not ticker:
            st.error("종목을 입력하세요.")
        else:
            is_kr = ticker.isdigit()
            amount = qty * price / (usdkrw if is_kr else 1.0)
            res = check_trade(ips, valued, ticker, side, amount, name, fields)
            if account == "A" and side == "매수" and ips.bucket_of(ticker) == "satellite":
                res["violations"].append("A계좌(은퇴계좌)에 개별주 매수 — 계좌 분리 원칙 위반")
            if res["blocked"]:
                st.error("차단: " + " / ".join(res["blocked"]))
            elif res["missing"]:
                st.error("체크리스트 미작성(각 10자 이상): " + ", ".join(res["missing"]))
            elif res["violations"] and not override.strip():
                st.warning("규칙 위반: " + " / ".join(res["violations"]) + " — 진행하려면 이유를 적으세요.")
            else:
                rec = {"trade_date": tdate, "account": account, "ticker": ticker, "side": side,
                       "quantity": qty, "price": price, "break_condition": brk,
                       "currency": "KRW" if is_kr else "USD", "fx": None if is_kr else trade_fx, **fields}
                record_trade(rec, res["violations"], override, sync, use_cash, sector)

    st.subheader("기록")
    year = str(date.today().year)
    realized = realized_by_currency(trades, year)
    if realized:
        cols = st.columns(len(realized) + 1)
        for col, (cur, amt) in zip(cols, sorted(realized.items(), reverse=True)):
            col.metric(f"{year}년 실현 손익 ({cur})", f"{amt:+,.0f}")
        cols[-1].caption("해외주식 양도차익은 연 250만 원까지 공제됩니다. 원화 환산은 거래일 환율 기준으로 따로 확인하세요.")
    if trades.empty:
        st.info("아직 기록이 없습니다.")
    else:
        st.dataframe(trades[["trade_date", "account", "ticker", "side", "quantity", "price", "realized_pnl",
                             "thesis", "break_condition", "violations", "compliant"]]
                     .rename(columns={"trade_date": "거래일", "account": "계좌", "ticker": "종목", "side": "구분",
                                      "quantity": "수량", "price": "단가", "realized_pnl": "실현손익", "thesis": "왜 샀나",
                                      "break_condition": "깨지는 조건", "violations": "위반", "compliant": "준수"}),
                     hide_index=True, use_container_width=True)

# ── 보유 종목 (M1) ─────────────────────────────────────────
with tab_hold:
    st.markdown("여기서 직접 고치거나 증권사 잔고를 CSV로 붙여넣으세요. "
                "한국 종목은 6자리 코드 + 시장 KR, 현금은 티커 CASH(수량=달러 금액, 평단 1).")
    edited = st.data_editor(holdings, num_rows="dynamic", use_container_width=True, column_config={
        "account": st.column_config.SelectboxColumn("계좌", options=["A", "B"]),
        "market": st.column_config.SelectboxColumn("시장", options=["US", "KR"]),
        "ticker": "종목", "quantity": "수량", "avg_cost": "평단(현지통화)",
        "avg_fx": st.column_config.NumberColumn("평균 매입 환율(미국)", format="%.1f",
                                                help="증권사 잔고의 매입환율. 원화 손익·양도세 계산용"),
        "sector": "섹터(위성만)", "thesis": st.column_config.TextColumn("보유 사유(한 문장)", width="large")})
    if st.button("저장하고 다시 계산"):
        save_holdings(edited)
        _quotes.clear()
        st.rerun()

# ── 투자정책서 ─────────────────────────────────────────────
with tab_ips:
    st.markdown("`config/ips.yaml` 을 편집해 바꿉니다. **규칙 변경은 연 1회(12월)** 가 원칙입니다.")
    st.dataframe(pd.DataFrame([{"자산군": b.name, "목표": f"{b.target:.0%}",
                                "허용 밴드": f"{b.band[0]:.0%} ~ {b.band[1]:.0%}",
                                "분류 종목": ", ".join(b.tickers) or "그 외 전부"} for b in ips.buckets.values()]),
                 hide_index=True, use_container_width=True)
    st.json(ips.rules)
    st.markdown("**금지 상품:** " + ", ".join(sorted(ips.banned_tickers)) +
                " · 상품명 키워드: " + ", ".join(ips.banned_keywords))
