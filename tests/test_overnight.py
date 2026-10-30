"""Checks that the overnight window uses only data available before the next open.

    python -m pytest tests
"""
import numpy as np
import pandas as pd

from capstone.overnight import window_return

US = pd.Series([100.0, 101.0, 103.0, 102.0],
               index=pd.to_datetime(["2026-10-19", "2026-10-20", "2026-10-21", "2026-10-22"]))


def ret(start, end):
    r, n = window_return(US, pd.Series(pd.to_datetime([start])), pd.Series(pd.to_datetime([end])))
    return r[0], n[0]


def test_one_night_uses_same_date_us_session():
    # India closes Tue 20th, opens Wed 21st: the new info is the US session of the 20th.
    r, n = ret("2026-10-20", "2026-10-21")
    assert n == 1 and np.isclose(r, 101 / 100 - 1)


def test_never_uses_session_dated_on_open_day():
    # The US session of the 21st ends after India opens on the 21st, so it must be excluded.
    r, _ = ret("2026-10-20", "2026-10-21")
    assert not np.isclose(r, 103 / 100 - 1)


def test_multi_day_gap_compounds_every_session():
    # India closed 21st and 22nd: close on the 20th, next open on the 23rd covers three US sessions.
    r, n = ret("2026-10-20", "2026-10-23")
    assert n == 3 and np.isclose(r, 102 / 100 - 1)


def test_us_holiday_is_zero_not_missing():
    s = US.drop(pd.Timestamp("2026-10-20"))
    r, n = window_return(s, pd.Series(pd.to_datetime(["2026-10-20"])), pd.Series(pd.to_datetime(["2026-10-21"])))
    assert n[0] == 0 and r[0] == 0


def test_before_history_is_nan():
    r, _ = ret("2026-10-01", "2026-10-02")
    assert np.isnan(r)


def test_after_history_is_nan_not_zero():
    r, _ = ret("2026-10-26", "2026-10-27")
    assert np.isnan(r)
