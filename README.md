# India market opening forecast

Predicts each NSE stock's next opening price from:

- the previous day's OHLCV (TejHQ API) and company fundamentals (CSV now; Bloomberg/Tracxn later), and
- **what happened overnight before India opens**: the US close (S&P 500, Nasdaq 100, the 11 US
  sector ETFs, and US-listed Indian ADRs, from Alpha Vantage) and the rupee (USD/INR, from
  Alpha Vantage history plus exchangerate.dev for the latest days).

The dashboard explains each forecast in plain words: which US sectors moved, how they map to
Indian sectors, how the rupee move helps or hurts each company, and how much of the forecast
comes from each factor.

## Keys

Copy `.env.example` to `.env` in the project folder and paste your keys after the `=` signs
(`run_all.bat` does the copy and opens Notepad for you). The `.env` file stays on your computer:
it is git-ignored, never printed, and must not be shared. Values in `.env` override
environment variables of the same name.

| Variable | Used for |
| --- | --- |
| `TWELVEDATA_API_KEY` | US ETFs, ADRs and USD/INR history (full history on a free key) |
| `EXCHANGERATE_API_KEY` | Latest USD/INR |
| `TEJ_API_KEY` | NSE prices (optional) |
| `ALPHAVANTAGE_API_KEY` | Optional fallback when no Twelve Data key is set |

Check them with `python -m capstone.check_keys` (shows only whether each is set and works).
If a network blocks outbound calls, allow `api.tejhq.dev`, `api.twelvedata.com` and `api.exchangerate.dev`.

## Quick start

One command: `./run_all.sh` (Mac/Linux) or double-click `run_all.bat` (Windows). Or step by step:

```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env                               # Mac: cp; then paste your keys into .env
python -m capstone.check_keys

python -m capstone.fetch_data --start 2015-01-01   # 1. NSE prices
python -m capstone.fetch_overnight                  # 2. US close + USD/INR
cp data/fundamentals_template.csv data/fundamentals.csv  # 3. fundamentals (optional)
python -m capstone.train                            # 4. walk-forward validation + save model
python -m capstone.predict                          # 5. next open + reasons
streamlit run app.py                                # 6. dashboard
```

**Daily routine:** after 4 pm IST run `fetch_data`. Next morning between about 02:30 IST (US
close) and 09:00 IST run `fetch_overnight`, then `predict`, then open the dashboard. Retrain weekly.

**Twelve Data free limits:** about 8 calls a minute and 800 a day; each call returns up to 5000
days, so the first run fetches full history (about 19 calls, 3 minutes).

**Alpha Vantage free limits** (only if you use it instead): about 25 calls a day, 5 a minute, and only the latest 100 days
per US series. `fetch_overnight` makes about 18 calls, waits between them, caches everything in
`data/us/` and `data/fx/`, and adds new days on each run, so history grows over time. For years of
US history at once, either use a premium key or drop CSVs with `date,close` columns into
`data/us/{TICKER}.csv`.

## Test without internet (synthetic sample)

```bash
python -m tests.make_synthetic_overnight   # fake prices, US moves and USD/INR
python -m capstone.train && python -m capstone.predict && streamlit run app.py
python -m pytest tests                     # checks the overnight window never peeks ahead
```

The sample has the Indian gaps built from the fake US and rupee moves, so the model will look
very accurate on it. That is by design and says nothing about real accuracy. The dashboard shows
a red banner while sample data is present. Before real data, delete `data/raw`, `data/us`,
`data/fx` and `data/SYNTHETIC_DATA`.

## How the overnight window works

NSE closes at 15:30 IST. The US session dated the same day closes at 01:30 or 02:30 IST, before
NSE opens at 09:15. For a close on day d and next open on day n, the model uses every US session
dated d to n-1 (several after a weekend or an Indian holiday, none on a US holiday). Moves are
measured from the last close before d to the last close before n, so nothing after the open leaks
in. `tests/test_overnight.py` checks this.

## Mappings you can edit

- `data/stock_map.csv`: NSE symbol, Indian sector, US ADR ticker (if any).
- `data/sector_links.csv`: Indian sector, matching US sector ETF, rupee exposure (+1 exporter,
  -1 importer) and the plain-language reasons shown on the dashboard.

## Layout

| Path | What it does |
| --- | --- |
| `capstone/tej_client.py` | NSE OHLCV from TejHQ; key from `TEJ_API_KEY` |
| `capstone/twelvedata_client.py` | US daily bars and USD/INR; key from `TWELVEDATA_API_KEY` (header, never in URLs) |
| `capstone/alphavantage_client.py` | US daily bars and FX_DAILY; key from `ALPHAVANTAGE_API_KEY` |
| `capstone/exchangerate_client.py` | Latest and recent USD/INR; key from `EXCHANGERATE_API_KEY` |
| `capstone/fetch_data.py` | Saves/tops up `data/raw/{SYMBOL}.csv` |
| `capstone/fetch_overnight.py` | Saves/tops up `data/us/*.csv` and `data/fx/USDINR.csv` |
| `capstone/overnight.py` | Lines up US and rupee moves with each India close-to-open window |
| `capstone/fundamentals.py` | CSV loader + point-in-time join |
| `capstone/features.py` | Price, overnight and fundamental features, and the target (overnight gap) |
| `capstone/train.py` | Walk-forward validation vs baselines and vs the same model without overnight data |
| `capstone/explain.py` | What-if attribution per driver and the plain-language reasons |
| `capstone/predict.py` | Next-open predictions, 10-90% band, reasons, market story |
| `capstone/planner.py` | Invest planner: split, ATR stop loss, target, backtested expected profit |
| `capstone/tracxn.py` | Reads Tracxn profile PDFs and saves checked numbers to `data/fundamentals.csv` |
| `capstone/uploads.py` | CSV/Excel upload of fundamentals, shareholding (FII/DII), financials, investors, profiles |
| `capstone/scalping.py` | Intraday opening-range planner, NSE charges, 1-minute backtest |
| `capstone/charts.py` | Candlestick and line charts (Plotly) |
| `capstone/export.py` | Builds the Excel workbook on the Download data tab |
| `app.py` | Streamlit dashboard |
| `tests/` | Synthetic data generators and the overnight-window tests |
