"""4순위 매수 이유 정기 점검 테스트."""
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.journal import load_trades, save_trade  # noqa: E402
from core.review import STATUSES, load_reviews, review_status, save_review  # noqa: E402

TODAY = date(2026, 9, 25)


@pytest.fixture
def db(tmp_path):
    return tmp_path / "j.db"


def test_never_reviewed_is_due(db):
    s = review_status(["NVDA"], load_reviews(db), load_trades(db), TODAY, 90).set_index("ticker")
    assert s.loc["NVDA", "due"] and pd.isna(s.loc["NVDA", "last_review"])
    assert s.loc["NVDA", "break_condition"] == ""


def test_recent_review_not_due_old_review_due(db):
    save_review("NVDA", "유지", "데이터센터 매출 계속 증가", "", "2026-08-01", db=db)
    save_review("MSFT", "유지", "Azure 성장 유지 확인함", "", "2026-05-01", db=db)
    s = review_status(["NVDA", "MSFT"], load_reviews(db), load_trades(db), TODAY, 90).set_index("ticker")
    assert not s.loc["NVDA", "due"] and s.loc["NVDA", "days_since"] == 55
    assert s.loc["MSFT", "due"]


def test_broken_status_is_always_flagged(db):
    save_review("NVDA", "깨짐", "경쟁사에 점유율 크게 뺏김", "", "2026-09-20", db=db)
    s = review_status(["NVDA"], load_reviews(db), load_trades(db), TODAY, 90).set_index("ticker")
    assert s.loc["NVDA", "last_status"] == "깨짐" and s.loc["NVDA", "due"]


def test_break_condition_prefers_latest_review_then_journal(db):
    save_trade({"ticker": "MSFT", "side": "매수", "trade_date": "2026-01-10",
                "break_condition": "Azure 성장률 20% 미만"}, [], db=db)
    s = review_status(["MSFT"], load_reviews(db), load_trades(db), TODAY, 90).set_index("ticker")
    assert s.loc["MSFT", "break_condition"] == "Azure 성장률 20% 미만"
    save_review("MSFT", "주의", "성장률 둔화 조짐 있음", "Azure 성장률 18% 미만", "2026-09-01", db=db)
    s = review_status(["MSFT"], load_reviews(db), load_trades(db), TODAY, 90).set_index("ticker")
    assert s.loc["MSFT", "break_condition"] == "Azure 성장률 18% 미만"


def test_due_sorted_first(db):
    save_review("NVDA", "유지", "데이터센터 매출 계속 증가", "", "2026-09-01", db=db)
    s = review_status(["NVDA", "AVGO"], load_reviews(db), load_trades(db), TODAY, 90)
    assert list(s["ticker"]) == ["AVGO", "NVDA"]


def test_save_review_validates(db):
    with pytest.raises(ValueError):
        save_review("NVDA", "몰라", "충분히 긴 메모입니다 정말로", "", "2026-09-01", db=db)
    with pytest.raises(ValueError):
        save_review("NVDA", STATUSES[0], "짧음", "", "2026-09-01", db=db)
