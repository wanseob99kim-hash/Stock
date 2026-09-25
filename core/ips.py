"""투자정책서(IPS) 로드와 검증."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_IPS = ROOT / "config" / "ips.yaml"


@dataclass
class Bucket:
    key: str
    name: str
    target: float
    band: tuple[float, float]
    tickers: list[str] = field(default_factory=list)


@dataclass
class IPS:
    version: str
    base_currency: str
    buckets: dict[str, Bucket]
    rules: dict
    banned_tickers: set[str]
    banned_keywords: list[str]
    benchmark: str = "VOO"
    tax: dict = field(default_factory=dict)

    def bucket_of(self, ticker: str) -> str:
        """종목이 속한 버킷 키. 어디에도 없으면 satellite."""
        t = str(ticker).upper()
        for key, b in self.buckets.items():
            if t in b.tickers:
                return key
        return "satellite"

    def is_banned(self, ticker: str, name: str = "") -> str | None:
        """금지 상품이면 사유 문자열, 아니면 None."""
        t = str(ticker).upper()
        if t in self.banned_tickers:
            return f"{t}: 금지 목록(레버리지·인버스) 종목"
        for kw in self.banned_keywords:
            if kw and kw.lower() in (name or "").lower():
                return f"{name}: 상품명에 '{kw}' 포함 (레버리지·인버스·환헤지 금지)"
        return None


def load_ips(path: Path | str = DEFAULT_IPS) -> IPS:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    buckets = {}
    for key, b in raw["buckets"].items():
        buckets[key] = Bucket(
            key=key,
            name=b["name"],
            target=float(b["target"]),
            band=(float(b["band"][0]), float(b["band"][1])),
            tickers=[str(t).upper() for t in (b.get("tickers") or [])],
        )
    total = sum(b.target for b in buckets.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"IPS 목표 비중 합계가 100%가 아닙니다: {total:.1%}")
    banned = raw.get("banned", {})
    return IPS(
        version=str(raw.get("version", "")),
        base_currency=raw.get("base_currency", "USD"),
        buckets=buckets,
        rules=raw.get("rules", {}),
        banned_tickers={str(t).upper() for t in banned.get("tickers", [])},
        banned_keywords=list(banned.get("name_keywords", [])),
        benchmark=str(raw.get("benchmark", "VOO")).upper(),
        tax=raw.get("tax", {}) or {},
    )
