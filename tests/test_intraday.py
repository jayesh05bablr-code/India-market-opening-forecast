import pandas as pd

from capstone.intraday import opening_range, parse_yahoo


def test_parse_yahoo_and_opening_range():
    # 2026-10-27 09:15 IST = 03:45 UTC
    start = int(pd.Timestamp("2026-10-27 03:45", tz="UTC").timestamp())
    stamps = [start + 60 * i for i in range(6)]
    body = {"chart": {"result": [{"timestamp": stamps, "indicators": {"quote": [{
        "open": [100, 101, 102, 101, 100, 99], "high": [101, 103, 102, 102, 101, 100],
        "low": [99, 100, 101, 100, 98, 97], "close": [101, 102, 101, 100, 99, 98],
        "volume": [10] * 6}]}}]}}
    df = parse_yahoo(body)
    assert df["datetime"].iloc[0] == pd.Timestamp("2026-10-27 09:15")
    orr = opening_range(df, "2026-10-27", 5)
    assert orr == {"open": 100.0, "high": 103.0, "low": 98.0}
    assert opening_range(df, "2026-10-27", 10) is None  # range not finished yet


def test_read_minutes_layouts():
    from capstone.scalping import read_minutes
    t = pd.date_range("2026-10-27 09:15", periods=3, freq="min")
    p = [100, 101, 102]
    files = [
        pd.DataFrame({"Date": t.strftime("%d/%m/%Y"), "Time": t.strftime("%H:%M"),
                      "Open": p, "High": p, "Low": p, "Close": p}).to_csv(index=False),
        "Export\n\n" + pd.DataFrame({"Timestamp (IST)": t, "Open Price": p, "High Price": p,
                                     "Low Price": p, "LTP": p}).to_csv(index=False),
    ]
    for f in files:
        d = read_minutes("x.csv", f.encode())
        assert d["datetime"].iloc[0] == pd.Timestamp("2026-10-27 09:15") and len(d) == 3


def test_read_preopen():
    from capstone.scalping import read_preopen
    data = ('SYMBOL,PREV. CLOSE,FINAL PRICE,FINAL VOLUME\n'
            'INFY,"1,166.00","1,151.95","87,719"\nTCS,"2,535.00","-","1"\n').encode()
    d = read_preopen("MW-Pre-Open-Market-NIFTY 50-27-Oct-2026.csv", data)
    assert list(d["symbol"]) == ["INFY"] and d["open"].iloc[0] == 1151.95 and d["date"].iloc[0] == "2026-10-27"
