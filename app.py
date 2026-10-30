"""Dashboard: streamlit run app.py"""
import json
from datetime import date

import altair as alt
import plotly.graph_objects as go
import pandas as pd
import streamlit as st

from capstone import charts, export, intraday, overnight, planner, refresh, scalping, tracxn, tracxn_excel, uploads
from capstone.explain import DRIVERS, OTHER, move
from capstone.paths import MODELS, PROCESSED, RAW, SECTOR_LINKS_FILE, SYNTHETIC_FLAG

UP, DOWN = "#2a78d6", "#e34948"  # diverging pair: blue = up, red = down

st.set_page_config(page_title="India Opening Forecast", layout="wide")
st.title("India market opening forecast")



def auto_update(why: str) -> None:
    """Retrain + re-predict in the background process, then reload the page with fresh numbers."""
    with st.spinner(f"{why}: retraining the model and updating predictions (this can take a minute)..."):
        ok, log = refresh.retrain_and_predict()
    st.cache_data.clear()
    st.session_state["update_result"] = (ok, log)
    st.rerun()


# Data saved since the last training (e.g. an upload whose retrain was interrupted): update now.
if refresh.stale() and not st.session_state.get("auto_tried"):
    st.session_state["auto_tried"] = True  # at most once per browser session if training keeps failing
    auto_update("Company data changed since the model was trained")

result = st.session_state.pop("update_result", None)
if result:
    ok, log = result
    if ok:
        st.success("Model retrained and predictions updated with the new company data.")
    else:
        st.error("Saved your data, but retraining failed. Details below (also in the terminal).")
    with st.expander("Training and prediction output"):
        st.code(log[-6000:] or "(no output)")
for msg in st.session_state.pop("saved_msgs", []):
    st.success(msg)

pred_path = PROCESSED / "predictions.csv"
story_path = PROCESSED / "overnight_story.json"
if not pred_path.exists():
    st.warning("No predictions yet. Run fetch_data, fetch_overnight, train and predict first.")
    st.stop()
if SYNTHETIC_FLAG.exists():
    st.error("SAMPLE DATA: these prices, US moves and rupee rates are synthetic, made to test the "
             "pipeline. They are not market data. Delete data/raw, data/us, data/fx and "
             "data/SYNTHETIC_DATA, then fetch real data.")

preds = pd.read_csv(pred_path)
story = json.loads(story_path.read_text()) if story_path.exists() else None
if "reason" not in preds.columns or story is None:
    st.info("These predictions come from the older model. Run fetch_overnight, train and predict "
            "to get overnight US and rupee reasons.")
    st.dataframe(preds, use_container_width=True, hide_index=True)
    st.stop()


def signed_bars(df: pd.DataFrame, label: str, value: str, title: str, height: int | None = None):
    """Horizontal bars from zero, blue for up and red for down, with the % written on each."""
    df = df.assign(pct=df[value] * 100, direction=(df[value] >= 0).map({True: "Up", False: "Down"}))
    base = alt.Chart(df).encode(
        y=alt.Y(f"{label}:N", sort=list(df.sort_values("pct", ascending=False)[label]), title=None,
                axis=alt.Axis(labelLimit=220, labelOverlap=False)),
        tooltip=[alt.Tooltip(f"{label}:N"), alt.Tooltip("pct:Q", title="Move %", format="+.2f")],
    )
    bars = base.mark_bar(cornerRadiusEnd=4, height=14).encode(
        x=alt.X("pct:Q", title=title),
        color=alt.Color("direction:N", scale=alt.Scale(domain=["Up", "Down"], range=[UP, DOWN]),
                        legend=None),
    )
    label = alt.Text("pct:Q", format="+.2f")
    right = base.mark_text(align="left", dx=4).encode(x="pct:Q", text=label).transform_filter("datum.pct >= 0")
    left = base.mark_text(align="right", dx=-4).encode(x="pct:Q", text=label).transform_filter("datum.pct < 0")
    chart = alt.layer(bars, right, left).properties(height=height or 34 * len(df) + 30)
    st.altair_chart(chart, use_container_width=True)


# --- Headline -------------------------------------------------------------------
st.caption(f"Forecast for the NSE open on **{story['next_open']}**, using closes from "
           f"{story['as_of_close']} and the US session of {story['us_session'] or 'n/a'}. "
           "Run predict after about 02:30 IST so the US close is included.")
if not story["overnight_complete"]:
    st.warning("The US session after India's last close is not in the data yet, so the overnight "
               "signal is missing. Run fetch_overnight after the US close, then predict again.")

moves = pd.DataFrame(story["moves"]).set_index("ticker")
avg_gap = preds["pred_gap_pct"].mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Expected opening", story["mood"], f"{avg_gap:+.2f}% average gap")
c2.metric("S&P 500 overnight", f"{moves.loc['SPY', 'ret'] * 100:+.2f}%")
c3.metric("Nasdaq 100 overnight", f"{moves.loc['QQQ', 'ret'] * 100:+.2f}%")
fx_chg = story.get("usdinr_chg")
if story.get("usdinr"):
    word = "weaker" if (fx_chg or 0) > 0 else "stronger"
    c4.metric("USD/INR", f"₹{story['usdinr']:.2f}",
              None if fx_chg is None or pd.isna(fx_chg) else f"Rupee {abs(fx_chg) * 100:.2f}% {word}",
              delta_color="off")

st.subheader("Why India should open this way")
st.markdown("\n".join(f"- {line}" for line in story["lines"]))

(tab_stocks, tab_sectors, tab_rupee, tab_invest, tab_scalp, tab_company, tab_profiles, tab_data,
 tab_model) = st.tabs(["Stocks", "US sectors and India", "Rupee", "Invest planner", "Intraday scalping",
                       "Company profile", "Upload company data", "Download data", "Model accuracy"])


@st.cache_data
def load_raw(sym: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(RAW / f"{sym}.csv", parse_dates=["date"])


def raw_prices(sym: str) -> pd.DataFrame | None:
    path = RAW / f"{sym}.csv"
    return load_raw(sym, path.stat().st_mtime) if path.exists() else None

# --- Stocks -----------------------------------------------------------------------
with tab_stocks:
    table = preds[["symbol", "india_sector", "close", "pred_open", "pred_gap_pct",
                   "pred_open_low", "pred_open_high"]].rename(columns={
        "symbol": "Stock", "india_sector": "Sector", "close": "Last close (₹)",
        "pred_open": "Expected open (₹)", "pred_gap_pct": "Gap %",
        "pred_open_low": "Low (₹)", "pred_open_high": "High (₹)"})
    st.dataframe(table.sort_values("Gap %", ascending=False), use_container_width=True, hide_index=True)
    st.caption("Low to high is the 10th to 90th percentile of recent out-of-sample errors.")

    symbol = st.selectbox("Explain one stock", preds["symbol"])
    row = preds.set_index("symbol").loc[symbol]
    st.info(row["reason"])
    contrib = pd.DataFrame({
        "driver": [*DRIVERS, OTHER],
        "effect": [row.get(f"contrib_{d}", 0.0) for d in [*DRIVERS, OTHER]],
    })
    left, right = st.columns(2)
    with left:
        st.markdown("**What moved the forecast** (percentage points of the gap)")
        signed_bars(contrib, "driver", "effect", "Effect on expected gap (%)")
        st.caption("What-if estimate: the forecast with that driver set to no move, compared with "
                   "the real forecast. It shows what the model reacted to, not proven cause.")
    with right:
        st.markdown("**The forecast in numbers**")
        st.dataframe(pd.DataFrame({
            "": ["Last close", "Expected open", "Likely low", "Likely high", "Expected gap"],
            symbol: [f"₹{row['close']:,.2f}", f"₹{row['pred_open']:,.2f}", f"₹{row['pred_open_low']:,.2f}",
                     f"₹{row['pred_open_high']:,.2f}", f"{row['pred_gap_pct']:+.2f}%"]}), hide_index=True)
    hist = raw_prices(symbol)
    if hist is not None:
        fc = {"date": pd.Timestamp(story["next_open"]), "open": row["pred_open"],
              "low": row["pred_open_low"], "high": row["pred_open_high"]}
        st.plotly_chart(charts.candles(hist, f"{symbol}: daily candles with the expected open (◆)",
                                       forecast=fc), use_container_width=True)
        st.caption("Green candle = closed above its open, red = below. Lines are 20- and 50-day "
                   "moving averages; bars underneath are volume. Use the 1M/3M/6M/1Y buttons or drag to zoom.")

# --- US sectors -----------------------------------------------------------------
with tab_sectors:
    links = pd.read_csv(SECTOR_LINKS_FILE)
    sectors = moves[moves["is_sector"]].reset_index()
    left, right = st.columns([2, 3])
    with left:
        st.markdown("**US sector moves overnight**")
        signed_bars(sectors, "name", "ret", "Move (%)")
    with right:
        st.markdown("**How each US sector maps to India**")
        held = preds.groupby("india_sector")["symbol"].apply(", ".join)
        m = links.assign(us_move=links["us_etf"].map(moves["ret"]))
        m["Your stocks"] = m["india_sector"].map(held).fillna("")
        m["What happened"] = m["us_move"].map(move)
        m = m.sort_values("Your stocks", ascending=False, key=lambda s: s.str.len())
        st.dataframe(m[["india_sector", "us_sector", "us_etf", "What happened", "Your stocks", "why_us"]]
                     .rename(columns={"india_sector": "Indian sector", "us_sector": "US sector",
                                      "us_etf": "ETF", "why_us": "Why it matters"}),
                     use_container_width=True, hide_index=True)
    adrs = preds.dropna(subset=["adr"])
    if len(adrs):
        st.markdown("**Indian companies listed in the US (ADRs)**")
        st.caption("An ADR trades in New York while India is closed, so it is often the clearest "
                   "overnight hint for that stock.")
        st.dataframe(adrs[["symbol", "adr", "adr_ret"]].assign(adr_ret=adrs["adr_ret"].map(move))
                     .rename(columns={"symbol": "NSE stock", "adr": "US ticker", "adr_ret": "Overnight"}),
                     hide_index=True)

# --- Rupee ------------------------------------------------------------------------
with tab_rupee:
    fx = overnight.load_fx().tail(180).rename("USD/INR").reset_index()
    if fx.empty:
        st.info("No USD/INR data yet. Run fetch_overnight.")
    else:
        fx_all = overnight.load_fx().rename("USD/INR").reset_index()
        st.plotly_chart(charts.line(fx_all, "date", {"USD/INR": "USD/INR"},
                                    "Rupees per US dollar (up means the rupee weakened)", "₹ per $",
                                    show_days=180), use_container_width=True)
    st.markdown(
        "**How the rupee affects the open.** When the rupee weakens, exporters that earn dollars "
        "(IT, pharma, metals) earn more in rupees, while importers (oil, autos, FMCG inputs) pay "
        "more. A falling rupee can also push foreign investors to sell Indian shares. A stronger "
        "rupee does the opposite. The model multiplies the rupee move by each sector's exposure "
        "below to get the rupee effect for each stock.")
    exp = pd.read_csv(SECTOR_LINKS_FILE)[["india_sector", "fx_exposure", "why_fx"]]
    exp["Weaker rupee is"] = exp["fx_exposure"].map(
        lambda e: "Good" if e > 0 else "Bad" if e < 0 else "Neutral")
    st.dataframe(exp.rename(columns={"india_sector": "Indian sector", "fx_exposure": "Exposure",
                                     "why_fx": "Why"}), use_container_width=True, hide_index=True)
    st.caption("Exposures are editable in data/sector_links.csv (+1 strong exporter, -1 strong importer).")

# --- Model accuracy ---------------------------------------------------------------
with tab_model:
    report = MODELS / "walk_forward_report.csv"
    if report.exists():
        summary = (pd.read_csv(report).groupby("method")[["mae_rupees", "mape_pct", "direction_acc_pct"]]
                   .mean().round(3).rename(columns={"mae_rupees": "MAE (₹)", "mape_pct": "MAPE %",
                                                    "direction_acc_pct": "Direction accuracy %"})
                   .rename(index={"model": "Model with overnight US + rupee",
                                  "model_without_overnight": "Same model without overnight data",
                                  "baseline_open_eq_close": "Baseline: open = last close",
                                  "baseline_mean_gap": "Baseline: average gap"}))
        st.markdown("**Walk-forward validation** (average over folds, lower error is better)")
        st.dataframe(summary)
        without = "Same model without overnight data"
        if without in summary.index:
            gain = summary.loc[without, "MAE (₹)"] - summary.loc["Model with overnight US + rupee", "MAE (₹)"]
            st.caption(f"Adding the overnight US and rupee features {'lowers' if gain > 0 else 'raises'} "
                       f"the average error by ₹{abs(gain):.2f} per share.")
    log_path = PROCESSED / "predictions_log.csv"
    raw = RAW / f"{symbol}.csv"
    if log_path.exists() and raw.exists():
        log = pd.read_csv(log_path)
        log = log[log["symbol"] == symbol].drop_duplicates("as_of_close", keep="last")
        hist = pd.read_csv(raw, parse_dates=["date"])
        hist["as_of_close"] = hist["date"].shift(1).dt.strftime("%Y-%m-%d")
        joined = log.merge(hist[["as_of_close", "open"]].rename(columns={"open": "actual_open"}),
                           on="as_of_close", how="inner")
        if len(joined):
            joined["as_of_close"] = pd.to_datetime(joined["as_of_close"])
            st.plotly_chart(charts.line(joined, "as_of_close", {"pred_open": "Predicted open",
                                                                "actual_open": "Actual open"},
                                        f"{symbol}: past predictions vs actual open", "₹"),
                            use_container_width=True)

# --- Invest planner ---------------------------------------------------------------
with tab_invest:
    st.warning("Educational tool for the capstone, not investment advice. The forecast is for the "
               "opening price, which is already priced in once the market opens, so this plan "
               "assumes you buy at the expected open and focuses on how much to put in each stock "
               "and where to cut losses. Past backtest results do not guarantee future returns.")
    c1, c2, c3 = st.columns(3)
    capital = c1.number_input("Amount to invest (₹)", min_value=1000, value=100000, step=5000)
    strategy = c2.selectbox("How to split", list(planner.STRATEGIES))
    hold = c3.slider("Holding period (trading days)", 1, 20, 5)
    c4, c5, c6, c7 = st.columns(4)
    stop_mult = c4.slider("Stop loss distance (× ATR)", 0.5, 4.0, 1.5, 0.25,
                          help="ATR is the stock's average daily range over 14 days.")
    reward = c5.slider("Target (× the stop distance)", 1.0, 4.0, 2.0, 0.5)
    risk_pct = c6.slider("Max loss per stock (% of amount)", 0.5, 5.0, 1.0, 0.5) / 100
    max_w = c7.slider("Max in one stock (%)", 20, 100, 40, 5) / 100
    st.caption(planner.STRATEGIES[strategy])

    prices = {}
    for sym in preds["symbol"]:
        path = RAW / f"{sym}.csv"
        if path.exists():
            prices[sym] = pd.read_csv(path, parse_dates=["date"])
    with st.spinner("Backtesting the stop/target plan on each stock's history..."):
        pl = planner.plan(preds, prices, capital, strategy, risk_pct, stop_mult, reward, hold, max_w)
    if pl.empty:
        st.info("No price history found for these stocks yet.")
    else:
        invested = pl["invest"].sum()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Invested", f"₹{invested:,.0f}", f"₹{capital - invested:,.0f} kept as cash", delta_color="off")
        m2.metric("Expected profit (backtest)", f"₹{pl['expected_profit'].sum():,.0f}",
                  f"{pl['expected_profit'].sum() / capital * 100:+.2f}% of amount", delta_color="off")
        m3.metric("If every target is hit", f"₹{pl['profit_at_target'].sum():,.0f}")
        m4.metric("If every stop is hit", f"₹{pl['loss_at_stop'].sum():,.0f}")
        table = pd.DataFrame({
            "Stock": pl["symbol"], "Sector": pl["sector"],
            "Split %": (pl["invest"] / capital * 100).round(1),
            "Shares": pl["shares"], "Amount (₹)": pl["invest"].round(0),
            "Buy near (₹)": pl["entry"].round(2), "Stop loss (₹)": pl["stop"].round(2),
            "Stop %": (pl["stop_pct"] * 100).round(2), "Target (₹)": pl["target"].round(2),
            "Max loss (₹)": pl["loss_at_stop"].round(0), "Profit at target (₹)": pl["profit_at_target"].round(0),
            "Backtest win rate %": (pl["win_rate"] * 100).round(1),
            "Avg return per trade %": (pl["avg_ret"] * 100).round(2),
            "Expected profit (₹)": pl["expected_profit"].round(0),
            "Forecast gap %": pl["pred_gap_pct"].round(2),
        })
        st.dataframe(table, use_container_width=True, hide_index=True)
        notes = []
        for r in pl.itertuples():
            if r.shares == 0:
                notes.append(f"**{r.symbol}**: nothing allocated (" + (
                    "its plan lost money in the backtest" if strategy == "Favour historical edge"
                    and not (r.avg_ret > 0) else "one share costs more than its allowed amount") + ").")
            elif r.limited_by_risk:
                notes.append(f"**{r.symbol}**: amount reduced so a stop-out loses at most "
                             f"{risk_pct * 100:.1f}% of your money.")
            if r.pred_gap_pct < -0.5:
                notes.append(f"**{r.symbol}**: the model expects a {r.pred_gap_pct:.2f}% gap down; "
                             "consider waiting for the open to settle before buying.")
        if notes:
            st.markdown("\n".join(f"- {n}" for n in notes))
        st.caption(f"Backtest: last ~500 trading days per stock, buying at each day's open, selling at "
                   f"the stop, the target or the close after {hold} days, whichever comes first "
                   "(a day that touches both counts as a stop). Expected profit = amount × average "
                   "return per trade. Brokerage and taxes are not included.")

# --- Company profiles (Tracxn) ---------------------------------------------------
with tab_profiles:
    st.subheader("1. Upload a CSV or Excel file (recommended)")
    st.markdown("The most accurate way to feed the model. A Tracxn company Excel export "
                "(TracxnExport-CompanyExcel-...xlsx) is read as it is. Your own file can hold fundamentals, "
                "shareholding (FII/DII), financials, investors and profile details, one sheet or table "
                "each. Column names are matched loosely (\"P/E\", \"PE Ratio\" and \"pe\" all work). "
                "Download the template to see the columns.")
    st.download_button("Download the Excel template", uploads.template_workbook(),
                       file_name="company_data_template.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
    with st.expander("What each sheet holds"):
        for kind, t in uploads.TEMPLATES.items():
            st.markdown(f"- **{kind}**: `{', '.join(t['columns'])}`. {t['about']}")
    up = st.file_uploader("CSV or Excel", type=["csv", "xlsx", "xls"], key="table_upload")
    if up is not None:
        try:
            tables = uploads.read_upload(up.name, up.getvalue())
        except Exception as exc:
            st.error(f"Could not read the file: {exc}")
            tables = {}
        ready = []
        if tracxn_excel.is_tracxn(tables):
            smap = overnight.stock_map()
            parsed, notes = tracxn_excel.parse(up.getvalue(), smap["symbol"],
                                               dict(zip(smap["symbol"], smap["india_sector"])))
            st.success("Tracxn company export recognised. Check the tables below, edit anything "
                       "that looks wrong, then save.")
            for note in notes:
                st.caption(note)
            for kind, df in parsed.items():
                st.markdown(f"**{kind.capitalize()}** ({len(df)} rows)")
                ready.append((kind, st.data_editor(df, num_rows="dynamic", key=f"tx_{kind}",
                                                   use_container_width=True)))
            tables = {}
        for name, raw_df in tables.items():
            if name == "README" or raw_df.empty:
                continue
            std = uploads.standardise(raw_df)
            kind = uploads.detect(std)
            kinds = list(uploads.TEMPLATES)
            st.markdown(f"**{name}**: {len(raw_df)} rows, recognised columns: "
                        f"{', '.join(std.columns) or 'none'}")
            if kind is None:
                st.warning("Needs a symbol column plus at least one data column from the template.")
                continue
            kind = st.selectbox("Save as", kinds, index=kinds.index(kind), key=f"kind_{name}")
            if kind in ("fundamentals", "shareholding") and "as_of" not in std:
                st.warning("No date column found; add an as_of column (date the numbers were public).")
                continue
            ready.append((kind, st.data_editor(std, num_rows="dynamic", key=f"ed_{name}",
                                               use_container_width=True)))
        if ready and st.button("Save uploaded data"):
            saved, failed = [], False
            for kind, df in ready:
                try:
                    n = uploads.merge_save(kind, df)
                    saved.append(f"Saved {n} {kind} rows.")
                except ValueError as exc:
                    st.error(f"{kind}: {exc}")
                    failed = True
            if saved and not failed:
                st.session_state["saved_msgs"] = saved
                auto_update("Saved " + ", ".join(k for k, _ in ready))
            for msg in saved:
                st.success(msg)

    st.subheader("2. Or upload a Tracxn profile PDF")
    st.markdown("Key numbers are pulled out for you to check and edit, then saved into "
                "`data/fundamentals.csv`. Saving retrains the model and updates the forecast automatically.")
    c1, c2 = st.columns([1, 2])
    symbol_up = c1.selectbox("Company", preds["symbol"], key="profile_symbol")
    as_of = c1.date_input("Data as of (report date)", value=date.today())
    pdf = c2.file_uploader("Tracxn profile (PDF)", type=["pdf"])
    if pdf is not None:
        data = pdf.getvalue()
        try:
            text = tracxn.pdf_text(data)
        except Exception as exc:
            st.error(f"Could not read this PDF ({type(exc).__name__}). Is it a scanned image?")
            text = ""
        if text:
            found = tracxn.extract(text, story.get("usdinr"))
            raw = {k[4:]: v for k, v in found.items() if k.startswith("raw_")}
            if raw:
                st.markdown("**Found in the PDF**")
                st.dataframe(pd.DataFrame(raw.items(), columns=["Field", "Value"]), hide_index=True)
            else:
                st.info("No standard fields recognised; fill the numbers in below from the PDF.")
            st.markdown("**Numbers to save for the model** (edit anything that looks wrong)")
            fields = {"market_cap_cr": "Market cap (₹ crore)", "pe": "P/E", "pb": "P/B",
                      "eps_ttm": "EPS, trailing 12 months (₹)", "roe_pct": "ROE %",
                      "debt_to_equity": "Debt to equity"}
            edit = st.data_editor(
                pd.DataFrame({"Field": list(fields.values()),
                              "Value": [found.get(k) for k in fields]}),
                disabled=["Field"], hide_index=True, key=f"edit_{pdf.name}")
            with st.expander("Text read from the PDF"):
                st.text(text[:5000])
            if st.button("Save to fundamentals"):
                values = dict(zip(fields, pd.to_numeric(edit["Value"], errors="coerce")))
                sector = preds.set_index("symbol").loc[symbol_up, "india_sector"]
                tracxn.save_fundamentals(symbol_up, as_of, values, sector)
                tracxn.save_profile(symbol_up, pdf.name, data, found)
                prof_row = tracxn.profile_row(symbol_up, found)
                if prof_row:
                    uploads.merge_save("profile", pd.DataFrame([prof_row]))
                st.session_state["saved_msgs"] = [f"Saved {symbol_up} fundamentals as of {as_of}."]
                auto_update(f"Saved {symbol_up} from the PDF")
    if tracxn.PROFILES_FILE.exists():
        st.markdown("**Profiles uploaded so far**")
        st.dataframe(pd.read_csv(tracxn.PROFILES_FILE), use_container_width=True, hide_index=True)

# --- Download data ------------------------------------------------------------------
with tab_data:
    st.markdown("Everything behind the model in one Excel workbook, one sheet per dataset, with a "
                "README sheet explaining each one. Use it for your own EDA.")
    parts = export.sheets()
    st.dataframe(pd.DataFrame([(n, len(d), len(d.columns)) for n, d in parts.items()],
                              columns=["Sheet", "Rows", "Columns"]), hide_index=True)
    if st.button("Prepare Excel file"):
        with st.spinner("Building the workbook..."):
            st.session_state["xlsx"] = export.workbook()
    if "xlsx" in st.session_state:
        st.download_button("Download capstone_data.xlsx", st.session_state["xlsx"],
                           file_name=f"capstone_data_{date.today():%Y%m%d}.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# --- Intraday scalping --------------------------------------------------------------
with tab_scalp:
    st.warning("Educational tool, not trading advice. Scalping the first minutes is the riskiest "
               "part of the day: spreads are wide and prices jump. SEBI's 2024 study found about 7 in "
               "10 individual intraday traders in the cash segment lost money in FY23. Paper-trade "
               "first and always use the stop.")
    st.markdown("**The rule (opening-range breakout).** Wait for the first few minutes after 09:15 to "
                "set a high and a low. Buy if price breaks above the high, or short if it breaks below "
                "the low, only in the direction of the bias. Stop at the other side of the range, "
                "target a multiple of the risk, and close whatever is open when the window ends.")
    c1, c2, c3, c4 = st.columns(4)
    s_capital = c1.number_input("Trading capital (₹)", min_value=1000, value=100000, step=5000, key="s_cap")
    s_risk = c2.slider("Risk per trade (% of capital)", 0.1, 2.0, 0.5, 0.1, key="s_risk") / 100
    s_reward = c3.slider("Target (× risk)", 1.0, 3.0, 1.5, 0.25, key="s_rew")
    s_lev = c4.slider("Leverage (MIS margin)", 1.0, 5.0, 1.0, 0.5, key="s_lev",
                      help="Exposure allowed per trade as a multiple of capital. 1 = no borrowing.")
    c5, c6, c7, c8 = st.columns(4)
    or_min = c5.selectbox("Opening range (minutes)", [5, 10, 15], index=0, key="s_or")
    window = c6.slider("Trade window after 09:15 (minutes)", 10, 60, 20, 5, key="s_win")
    buffer_pct = c7.number_input("Breakout buffer (%)", 0.0, 0.5, 0.05, 0.01, key="s_buf") / 100
    brokerage = c8.number_input("Brokerage per order (₹)", 0.0, 50.0, 20.0, 1.0, key="s_brk")
    bias_src = st.radio("Trade direction", ["Model's expected gap", "Actual opening gap", "Both directions"],
                        horizontal=True, key="s_bias")
    boost = 1.5  # first minutes are more volatile than the day average

    st.markdown("**Live 1-minute data.** After 09:15 + the opening range, click to pull today's "
                "1-minute candles for your stocks from the internet (Yahoo Finance, no key needed; "
                "Twelve Data as a backup if your plan covers NSE). The real opening range then "
                "replaces the estimate below, and the candles are kept for backtesting in Step 3.")
    if st.button("Fetch 1-minute data from the internet", key="fetch_min"):
        status = []
        with st.spinner("Fetching 1-minute candles..."):
            for sym in preds["symbol"]:
                src, n = intraday.update(sym)
                m = intraday.load(sym)
                status.append({"Stock": sym, "Source": src, "Minutes stored": n,
                               "Latest candle (IST)": m["datetime"].max() if len(m) else None})
        st.session_state["min_status"] = status
    if st.session_state.get("min_status"):
        st.dataframe(pd.DataFrame(st.session_state["min_status"]), hide_index=True, use_container_width=True)
        st.caption("Free 1-minute data can lag the market by a minute or two. Click again to refresh.")

    with st.expander("Or upload today's opening data yourself (CSV or Excel)"):
        st.markdown("Two kinds of file work here:\n"
                    "- **NSE pre-open file** (nseindia.com, Market Data, Pre-Open Market, Download CSV, "
                    "after 09:08): the real opening price of every stock. The stock picker is ignored.\n"
                    "- **1-minute candles** for one stock (`datetime, open, high, low, close`, or `date, time, ...`), "
                    "e.g. from your broker. The candles from 09:15 set the real opening range and are kept for Step 3.")
        u1, u2 = st.columns([1, 2])
        o_sym = u1.selectbox("Stock in this file", preds["symbol"], key="open_sym")
        o_up = u2.file_uploader("Opening minutes", type=["csv", "xlsx", "xls"], key="open_upload")
        if o_up is not None and st.session_state.get("open_saved") != (o_up.file_id, o_sym):
            pre = scalping.read_preopen(o_up.name, o_up.getvalue())
            if pre is not None:
                intraday.save_preopen(pre)
                st.session_state["open_saved"] = (o_up.file_id, o_sym)
                mine = pre[pre["symbol"].isin(preds["symbol"])]
                st.session_state["min_status"] = [{"Stock": r.symbol, "Source": f"NSE pre-open {r.date}",
                                                   "Minutes stored": 0, "Latest candle (IST)": None}
                                                  for r in mine.itertuples()]
                st.rerun()
            try:
                got = scalping.read_minutes(o_up.name, o_up.getvalue())
                kept = intraday.save(o_sym, got.assign(volume=got.get("volume")))
                st.session_state["open_saved"] = (o_up.file_id, o_sym)
                st.session_state["min_status"] = [{"Stock": o_sym, "Source": f"Upload: {o_up.name}",
                                                   "Minutes stored": len(kept),
                                                   "Latest candle (IST)": kept["datetime"].max()}]
                st.rerun()
            except (ValueError, KeyError) as exc:
                st.error(f"Could not use this file: {exc}")
    stored = [s for s in preds["symbol"] if len(intraday.load(s))]
    if stored:
        d1, d2 = st.columns([1, 2])
        dl_sym = d1.selectbox("Download stored 1-minute data", stored, key="dl_sym")
        d2.download_button(f"Download {dl_sym} 1-minute candles (CSV)",
                           intraday.load(dl_sym).to_csv(index=False).encode(),
                           file_name=f"{dl_sym}_1min.csv", mime="text/csv")

    today_ist = pd.Timestamp.now(tz="Asia/Kolkata").date()
    st.markdown(f"**Step 1. Opening range.** Before 09:{15 + or_min:02d} the table shows estimates from "
                f"each stock's volatility. After that, fetch live data (above) or type in the real open, "
                f"high and low of the first {or_min} minutes, and the plan updates.")
    est = []
    for r in preds.itertuples():
        h = raw_prices(r.symbol)
        if h is None or len(h) < 30:
            continue
        live = intraday.opening_range(intraday.load(r.symbol), today_ist, or_min)
        if live:
            est.append({"Stock": r.symbol, "Last close": round(r.close, 2), "Open": round(live["open"], 2),
                        "Range high": round(live["high"], 2), "Range low": round(live["low"], 2),
                        "Model gap %": round(r.pred_gap_pct, 2), "Source": f"Live, first {or_min} min today"})
            continue
        vol = float(h["close"].pct_change().tail(20).std())
        span = scalping.estimate_range(r.pred_open, vol, or_min, boost)
        est.append({"Stock": r.symbol, "Last close": round(r.close, 2), "Open": round(r.pred_open, 2),
                    "Range high": round(r.pred_open + span / 2, 2), "Range low": round(r.pred_open - span / 2, 2),
                    "Model gap %": round(r.pred_gap_pct, 2), "Source": "Estimate"})
    n_live = sum(e["Source"] != "Estimate" for e in est)
    pre_today = intraday.preopen(today_ist).set_index("symbol")
    if len(pre_today):
        for e in est:
            if e["Source"] == "Estimate" and e["Stock"] in pre_today.index:
                real = float(pre_today.loc[e["Stock"], "open"])
                half = (e["Range high"] - e["Range low"]) / 2
                e.update({"Open": round(real, 2), "Range high": round(real + half, 2),
                          "Range low": round(real - half, 2), "Source": "NSE pre-open open, estimated range"})
                n_live += 1
        chk = preds[preds["symbol"].isin(pre_today.index)][["symbol", "close", "pred_open", "pred_gap_pct"]].copy()
        if len(chk):
            chk["Actual open"] = chk["symbol"].map(pre_today["open"])
            chk["Actual gap %"] = (chk["Actual open"] / chk["close"] - 1) * 100
            chk["Direction right?"] = (chk["Actual gap %"] * chk["pred_gap_pct"] > 0).map({True: "Yes", False: "No"})
            chk.loc[chk["Actual gap %"].abs() < 0.05, "Direction right?"] = "Flat"
            st.markdown(f"**Today's forecast vs the real open** (NSE pre-open, {today_ist})")
            st.dataframe(chk.rename(columns={"symbol": "Stock", "close": "Last close", "pred_open": "Forecast open",
                                             "pred_gap_pct": "Forecast gap %"}).round(2),
                         hide_index=True, use_container_width=True)
            st.caption("Uses the last close the model saw. If NSE's previous close differs (e.g. predict was run "
                       "on older data), the gaps will be off by that difference.")
    ranges = st.data_editor(pd.DataFrame(est), disabled=["Stock", "Last close", "Model gap %", "Source"],
                            hide_index=True, use_container_width=True, key=f"or_{or_min}_{today_ist}_{n_live}")

    rows = []
    for r in ranges.itertuples():
        if bias_src == "Model's expected gap":
            bias = getattr(r, "_6")
        elif bias_src == "Actual opening gap":
            bias = r.Open - r._2
        else:
            bias = 0
        plan = scalping.plan_row(r.Stock, r.Open, r._4, r._5, bias, s_capital, s_risk, s_reward,
                                 s_lev, buffer_pct, brokerage)
        rows += list(plan.values())
    st.markdown("**Step 2. The plan**")
    if not rows:
        st.info("No setups: check the range high is above the range low.")
    else:
        sp = pd.DataFrame(rows)
        st.dataframe(pd.DataFrame({
            "Stock": sp["symbol"], "Side": sp["side"],
            "Enter if price crosses (₹)": sp["trigger"].round(2), "Stop (₹)": sp["stop"].round(2),
            "Target (₹)": sp["target"].round(2), "Qty": sp["qty"], "Exposure (₹)": sp["exposure"].round(0),
            "Net profit at target (₹)": sp["net_profit_at_target"].round(0),
            "Net loss at stop (₹)": sp["net_loss_at_stop"].round(0),
            "Charges (₹)": sp["charges"].round(0), "Move to break even (₹/share)": sp["breakeven_move"].round(2),
        }), use_container_width=True, hide_index=True)
        st.caption(f"Charges are approximate NSE intraday costs (brokerage, STT 0.025% on the sell side, "
                   f"exchange, SEBI, stamp duty, GST). Exit everything by "
                   f"09:{15 + window:02d}" if 15 + window < 60 else
                   f"Charges are approximate NSE intraday costs. Exit everything by "
                   f"{9 + (15 + window) // 60:02d}:{(15 + window) % 60:02d}.")

    st.markdown("**Step 3. Test the rule on real minute data (optional)**")
    st.markdown("Upload 1-minute candles for one stock (CSV or Excel with `datetime, open, high, low, close`, "
                "or `date, time, ...`), e.g. exported from your broker, or use the data fetched above. "
                "Free sources keep only about 7 days of 1-minute bars, so fetch daily to build history.")
    m_up = st.file_uploader("1-minute candles", type=["csv", "xlsx", "xls"], key="minute_upload")
    fetched = [s for s in preds["symbol"] if len(intraday.load(s))]
    bt_sym = None
    if m_up is None and fetched:
        bt_sym = st.selectbox("Or test on the 1-minute data fetched from the internet", fetched, key="bt_sym")
    if m_up is not None or bt_sym:
        try:
            mins = scalping.read_minutes(m_up.name, m_up.getvalue()) if m_up is not None \
                else intraday.load(bt_sym)[["datetime", "open", "high", "low", "close"]]
            mode = "Both directions" if bias_src == "Both directions" else "Gap direction only"
            bt = scalping.backtest(mins, or_min, window, s_reward, buffer_pct, mode,
                                   s_risk * s_capital, brokerage)
        except (ValueError, KeyError) as exc:
            st.error(f"Could not use this file: {exc}")
            bt = pd.DataFrame()
        trades = bt[bt["side"] != "No trade"] if len(bt) else bt
        if len(trades):
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Days tested", len(bt))
            k2.metric("Trades", len(trades))
            k3.metric("Win rate", f"{(trades['net_pnl'] > 0).mean() * 100:.1f}%")
            k4.metric("Net P&L after charges", f"₹{trades['net_pnl'].sum():,.0f}")
            curve = trades.assign(date=pd.to_datetime(trades["date"]), equity=trades["net_pnl"].cumsum())
            st.plotly_chart(charts.line(curve, "date", {"equity": "Cumulative net P&L"},
                                        "Backtest: cumulative profit after charges", "₹"),
                            use_container_width=True)
            st.dataframe(bt.round(2), use_container_width=True, hide_index=True)
            if "Model" in bias_src:
                st.caption("Past model forecasts aren't stored per day, so the backtest uses the actual "
                           "opening gap as the direction filter.")
        elif m_up is not None or bt_sym:
            st.info("No trades triggered in this data with these settings.")

# --- Company profile ----------------------------------------------------------------
with tab_company:
    comp = st.selectbox("Company", preds["symbol"], key="company_symbol")
    smap = overnight.stock_map().set_index("symbol")
    info = smap.loc[comp] if comp in smap.index else None
    prof = uploads.load("profile")
    prof = prof[prof["symbol"].astype(str).str.upper() == comp]
    sub = []
    if info is not None:
        sub.append(f"Sector: **{info['india_sector']}**")
        sub.append(f"US twin: **{info['us_sector']} ({info['us_etf']})**")
        if info["adr"]:
            sub.append(f"US-listed ADR: **{info['adr']}**")
    if len(prof):
        p0 = prof.iloc[-1]
        for col, label in [("founded", "Founded"), ("headquarters", "HQ"), ("employees", "Employees"),
                           ("ceo", "CEO"), ("website", "Website")]:
            if col in p0 and pd.notna(p0[col]):
                v = p0[col]
                if col in ("founded", "employees") and isinstance(v, (int, float)):
                    v = f"{int(v)}" if col == "founded" else f"{int(v):,}"
                sub.append(f"{label}: **{v}**")
    st.subheader(comp)
    st.markdown(" · ".join(sub))
    if len(prof) and "description" in prof and pd.notna(prof.iloc[-1].get("description")):
        st.write(prof.iloc[-1]["description"])

    fund = uploads.load("fundamentals")
    fund = fund[fund["symbol"].astype(str).str.upper() == comp].sort_values("as_of")
    st.markdown("#### Fundamentals")
    if fund.empty:
        st.info("No fundamentals yet. Add them on the Upload company data tab.")
    else:
        latest = fund.ffill().iloc[-1]
        cols = st.columns(6)
        mc = latest.get("market_cap_cr")
        if pd.notna(mc) and mc >= 1e5:
            latest["market_cap_cr"] = mc / 1e5  # show lakh crore so the tile fits
        mc_fmt = "₹{:,.2f}L Cr" if pd.notna(mc) and mc >= 1e5 else "₹{:,.0f} Cr"
        for c, (k, label, fmt) in zip(cols, [("market_cap_cr", "Market cap", mc_fmt),
                                             ("pe", "P/E", "{:.1f}"), ("pb", "P/B", "{:.1f}"),
                                             ("roe_pct", "ROE", "{:.1f}%"), ("debt_to_equity", "Debt/Equity", "{:.2f}"),
                                             ("eps_ttm", "EPS (TTM)", "₹{:.1f}")]):
            v = latest.get(k)
            c.metric(label, fmt.format(v) if pd.notna(v) else "–")
        st.caption(f"Latest values as of {latest['as_of']}.")
        hist_cols = [c for c in ["market_cap_cr", "pe", "pb", "eps_ttm", "roe_pct", "debt_to_equity"] if c in fund]
        shown = fund[["as_of", *hist_cols]].dropna(how="all", subset=hist_cols)
        st.dataframe(shown.iloc[::-1], use_container_width=True, hide_index=True)

    st.markdown("#### Financials")
    fin = uploads.load("financials")
    fin = fin[fin["symbol"].astype(str).str.upper() == comp].sort_values("period_end")
    if fin.empty:
        st.info("No results yet. Upload a financials sheet (period_end, revenue_cr, ebitda_cr, net_profit_cr...).")
    else:
        fig = go.Figure()
        for col, name, color in [("revenue_cr", "Revenue", "#2a78d6"), ("ebitda_cr", "EBITDA", "#1baf7a"),
                                 ("net_profit_cr", "Net profit", "#eb6834")]:
            if col in fin and fin[col].notna().any():
                fig.add_bar(x=fin["period_end"], y=fin[col], name=name, marker_color=color)
        fig.update_xaxes(type="category")
        fig.update_layout(barmode="group", height=380, title="Results by period (₹ crore)",
                          hovermode="x unified", margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
        st.plotly_chart(fig, use_container_width=True)
        if {"revenue_cr", "net_profit_cr"} <= set(fin.columns):
            fin = fin.assign(net_margin_pct=(fin["net_profit_cr"] / fin["revenue_cr"] * 100).round(1))
        st.dataframe(fin.drop(columns=["symbol"]).iloc[::-1], use_container_width=True, hide_index=True)

    st.markdown("#### Ownership: foreign (FII/FPI) vs domestic")
    sh_cols = ["promoter_pct", "fii_pct", "dii_pct", "public_pct", "govt_pct"]
    sh = fund[["as_of", *[c for c in sh_cols if c in fund]]].dropna(how="all", subset=[c for c in sh_cols if c in fund]) \
        if not fund.empty else pd.DataFrame()
    tx_own = fund[["as_of", "insiders_pct", "institutions_pct"]].dropna(
        subset=["insiders_pct", "institutions_pct"], how="all") \
        if {"insiders_pct", "institutions_pct"} <= set(fund.columns) else pd.DataFrame()
    if len(tx_own):
        o = tx_own.iloc[-1]
        e1, e2 = st.columns(2)
        e1.metric("Insiders (Tracxn)", f"{o['insiders_pct']:.2f}%" if pd.notna(o["insiders_pct"]) else "–")
        e2.metric("All institutions (Tracxn)",
                  f"{o['institutions_pct']:.2f}%" if pd.notna(o["institutions_pct"]) else "–")
        st.caption(f"From the Tracxn export as of {o['as_of']}. Tracxn does not split institutions into "
                   "foreign and domestic; the NSE shareholding pattern below does.")
    if sh.empty or len(sh.columns) <= 1:
        st.info("No shareholding pattern yet. Upload a shareholding sheet (as_of, promoter_pct, fii_pct, "
                "dii_pct, public_pct) from the company's quarterly filing on NSE/BSE.")
    else:
        last = sh.iloc[-1]
        prev = sh.iloc[-2] if len(sh) > 1 else None
        foreign = last.get("fii_pct", float("nan"))
        domestic = sum(last.get(c, 0) or 0 for c in ["promoter_pct", "dii_pct", "public_pct", "govt_pct"] if c in last)
        d1, d2, d3, d4 = st.columns(4)
        def chg(c):
            return None if prev is None or pd.isna(prev.get(c)) or pd.isna(last.get(c)) \
                else f"{last[c] - prev[c]:+.2f} pp vs previous filing"
        d1.metric("Foreign institutions (FII/FPI)", f"{foreign:.2f}%" if pd.notna(foreign) else "–", chg("fii_pct"))
        d2.metric("Domestic institutions (DII)", f"{last.get('dii_pct', float('nan')):.2f}%", chg("dii_pct"))
        d3.metric("Promoters", f"{last.get('promoter_pct', float('nan')):.2f}%", chg("promoter_pct"))
        d4.metric("All domestic holders", f"{domestic:.2f}%")
        fig = go.Figure()
        for col, name, color in [("promoter_pct", "Promoters", "#2a78d6"), ("fii_pct", "FII/FPI (foreign)", "#eb6834"),
                                 ("dii_pct", "DII (domestic institutions)", "#1baf7a"),
                                 ("public_pct", "Public / retail", "#eda100"), ("govt_pct", "Government", "#e87ba4")]:
            if col in sh and sh[col].notna().any():
                fig.add_bar(x=sh["as_of"], y=sh[col], name=name, marker_color=color,
                            hovertemplate=name + ": %{y:.2f}%<extra></extra>")
        fig.update_xaxes(type="category")
        fig.update_layout(barmode="stack", height=380, title="Shareholding pattern by filing (%)",
                          yaxis=dict(range=[0, 100], ticksuffix="%"), margin=dict(l=10, r=10, t=50, b=10),
                          legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Rising FII share usually means foreign money is buying in; rising DII share means "
                   "Indian mutual funds and insurers are. The model uses these percentages as features.")

    st.markdown("#### Major investors")
    inv = uploads.load("investors")
    inv = inv[inv["symbol"].astype(str).str.upper() == comp]
    if inv.empty:
        st.info("No investor list yet. Upload an investors sheet (investor, investor_type, country, stake_pct).")
    else:
        itype = inv.get("investor_type", pd.Series("", index=inv.index)).astype(str)
        country = inv.get("country", pd.Series("", index=inv.index)).astype(str).str.strip().str.lower()
        no_country = country.isin(["", "nan"])
        foreign_mask = itype.str.contains("FII|FPI|Foreign", case=False) | (~no_country & (country != "india"))
        domestic_mask = ~foreign_mask & (~no_country | itype.str.contains(
            "DII|Promoter|Mutual|Insurance|Government|Retail", case=False))
        unknown_mask = ~foreign_mask & ~domestic_mask

        def show(mask):
            part = inv[mask].drop(columns=["symbol"]).dropna(axis=1, how="all")
            if "stake_pct" in part:
                part = part.sort_values("stake_pct", ascending=False)
            st.dataframe(part, hide_index=True, use_container_width=True)

        cols = st.columns(3 if unknown_mask.any() else 2)
        for col, mask, title in zip(cols, [foreign_mask, domestic_mask, unknown_mask],
                                    ["Foreign", "Domestic", "Country not given (e.g. Tracxn lists)"]):
            with col:
                st.markdown(f"**{title}**")
                show(mask)

    tx = tracxn.PROFILES_FILE
    if tx.exists():
        t = pd.read_csv(tx)
        t = t[t["symbol"].astype(str).str.upper() == comp]
        if len(t):
            with st.expander("From uploaded Tracxn PDFs"):
                st.dataframe(t, hide_index=True, use_container_width=True)
