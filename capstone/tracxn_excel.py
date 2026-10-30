"""Read a Tracxn "Company Excel" export (TracxnExport-CompanyExcel-*.xlsx).

The export has a title block above each table, values like
"18,29,72,00,00,000 (Mar 31, 2026)" or "4.49x", and amounts in full rupees.
This turns it into the same tables the CSV/Excel upload uses:

- fundamentals: from "Public Market Data 8.1" (as_of = its "Data as on" date),
  with debt to equity from the "1.2" financial ratios sheet
- financials: annual revenue, EBITDA and net profit from "Companies Covered 1.1"
- profile: founded, HQ, employees, CEO, website, description
- investors: the institutional and angel investor lists

Tracxn's "Percent Insiders" and "Percent Institutions" are kept as
insiders_pct and institutions_pct. They are not the NSE promoter / FII / DII
split, so they are shown on the Company profile tab but not fed to the model.
"""
from __future__ import annotations

import io
import re

import pandas as pd

SHEETS = ("Companies Covered 1.1", "Public Market Data 8.1")
CRORE = 1e7
# Tracxn sometimes shows 2,147,483,647 (a 32-bit overflow) in place of a real amount.
_OVERFLOW = re.compile(r"^-?2147483647")


def is_tracxn(sheet_names) -> bool:
    return any(s in SHEETS for s in sheet_names)


def _table(raw: pd.DataFrame) -> pd.DataFrame:
    """Find the row starting with 'SNo.' and use it as the header (title rows above are dropped)."""
    raw = raw.dropna(axis=1, how="all")
    hits = raw.index[raw.apply(lambda r: r.astype(str).str.strip().eq("SNo.").any(), axis=1)]
    if not len(hits):
        return pd.DataFrame()
    h = hits[0]
    names = raw.loc[h].astype(str).str.strip()
    above = raw.shift(1).loc[h].ffill()  # group row such as "Revenue CAGR (%)" over "1 Year"
    names = [f"{g} {n}" if n.endswith("Year") and isinstance(g, str) else n for n, g in zip(names, above)]
    out = raw.loc[h + 1:].copy()
    out.columns = names
    out = out.dropna(how="all")
    return out[out["SNo."].astype(str).str.strip().str.match(r"^\d+(\.0)?$")]


def _date_of(v) -> str | None:
    m = re.search(r"\(([A-Za-z]{3} \d{1,2}, \d{4})\)", str(v))
    return pd.to_datetime(m.group(1)).strftime("%Y-%m-%d") if m else None


def _num(v, scale: float = 1.0) -> float | None:
    """'31.59% (Mar 31, 2026)' -> 31.59, '4.49x' -> 4.49, '3,28,594' -> 328594."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if isinstance(v, (int, float)):
        return float(v) / scale
    text = re.sub(r"\(.*?\)", "", str(v)).replace(",", "").replace("*", "").strip()
    if _OVERFLOW.match(text):
        return None
    text = re.sub(r"[%xX₹\s]+$", "", text).replace("₹", "")
    try:
        return float(text) / scale
    except ValueError:
        return None


def _symbol(row, known: set[str]) -> str | None:
    for col in ("Ticker", "Stock Ticker"):
        t = str(row.get(col, ""))
        m = re.search(r"([A-Z0-9&\-]+):NSE", t.upper())
        if m:
            return m.group(1)
    name = re.sub(r"[^A-Z0-9&]", "", str(row.get("Company Name", "")).upper())
    return name if name in known else None


def _ceo(text) -> str | None:
    """Key People Info lines look like 'Name; Role; email; background'."""
    for line in str(text).split("\n"):
        parts = [p.strip() for p in line.split(";")]
        if len(parts) > 1 and re.search(r"\bCEO\b|Managing Director", parts[1], re.I):
            return parts[0]
    return None


def _split(text) -> list[str]:
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return []
    return [p.strip() for p in re.split(r",?\s*\n|,\s*", str(text)) if p.strip() and p.strip() != "nan"]


def parse(data: bytes, known_symbols, sectors: dict | None = None) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Return ({kind: table}, notes). Tables use the column names of uploads.TEMPLATES."""
    known = {str(s).upper() for s in known_symbols}
    sectors = sectors or {}
    sheets = pd.read_excel(io.BytesIO(data), sheet_name=None, header=None)
    companies = _table(sheets["Companies Covered 1.1"]) if "Companies Covered 1.1" in sheets else pd.DataFrame()
    ratios = _table(sheets["1.2"]) if "1.2" in sheets else pd.DataFrame()
    market = _table(sheets["Public Market Data 8.1"]) if "Public Market Data 8.1" in sheets else pd.DataFrame()
    notes = []

    # Company Name -> NSE symbol, from whichever sheet has the ticker.
    names = {}
    for df in (companies, market):
        for _, r in df.iterrows():
            s = _symbol(r, known)
            if s:
                names[str(r["Company Name"]).strip()] = s
    def sym(r):
        return names.get(str(r.get("Company Name", "")).strip()) or _symbol(r, known)

    ratio_rows = {str(r["Company Name"]).strip(): r for _, r in ratios.iterrows()}
    fundamentals, financials, profiles, investors = [], [], [], []

    for _, r in market.iterrows():
        rr = ratio_rows.get(str(r["Company Name"]).strip(), {})
        as_of = pd.to_datetime(r.get("Data as on"), errors="coerce")
        roe = _num(r.get("Return on Equity TTM"))
        fundamentals.append({
            "symbol": sym(r), "as_of": None if pd.isna(as_of) else as_of.strftime("%Y-%m-%d"),
            "market_cap_cr": _num(r.get("Market Capitalization (INR)"), CRORE),
            "pe": _num(r.get("PE Ratio (TTM)")), "pb": _num(r.get("P/B MRQ")),
            "eps_ttm": _num(r.get("Diluted EPS TTM (INR)")),
            "roe_pct": roe if roe is not None else _num(rr.get("Latest Return on Equity (ROE)")),
            "debt_to_equity": _num(rr.get("Latest Debt-to-Equity Ratio")),
            "insiders_pct": _num(r.get("Percent Insiders")),
            "institutions_pct": _num(r.get("Percent Institutions")),
        })

    for _, r in companies.iterrows():
        s = sym(r)
        rev = r.get("Annual Revenue (INR)")
        period = _date_of(rev) or _date_of(r.get("Annual Net Profit (INR)"))
        if period:
            row = {"symbol": s, "period_end": period,
                   "revenue_cr": _num(rev, CRORE),
                   "ebitda_cr": _num(r.get("Annual EBITDA (INR)"), CRORE),
                   "net_profit_cr": _num(r.get("Annual Net Profit (INR)"), CRORE)}
            rr = ratio_rows.get(str(r["Company Name"]).strip())
            if rr is not None and _num(rr.get("Latest Net Debt (INR)")) is not None:
                row["net_debt_cr"] = _num(rr.get("Latest Net Debt (INR)"), CRORE)
            financials.append(row)
        hq = ", ".join(str(r[c]) for c in ("City", "State", "Country") if pd.notna(r.get(c)))
        emp = _num(r.get("Total Employee Count"))
        founded = _num(r.get("Founded Year"))
        profiles.append({
            "symbol": s, "founded": None if founded is None else int(founded), "headquarters": hq or None,
            "employees": None if emp is None else int(emp), "ceo": _ceo(r.get("Key People Info")),
            "website": r.get("Website") if pd.notna(r.get("Website")) else None,
            "description": r.get("Description") if pd.notna(r.get("Description")) else None,
        })
        for col, kind in (("Institutional Investors", "Institutional"), ("Angel Investors", "Angel")):
            for name in _split(r.get(col)):
                investors.append({"symbol": s, "investor": name, "investor_type": f"{kind} (Tracxn)"})

        # Prefer the 8.1 market cap; use 1.1 only if 8.1 had none.
        mc = _num(r.get("Latest Market Cap (INR)"), CRORE)
        if mc and not any(f["symbol"] == s and f["market_cap_cr"] for f in fundamentals):
            fundamentals.append({"symbol": s, "as_of": _date_of(r.get("Latest Market Cap (INR)")),
                                 "market_cap_cr": mc})

    tables = {}
    for kind, rows in (("fundamentals", fundamentals), ("financials", financials),
                       ("profile", profiles), ("investors", investors)):
        if rows:
            df = pd.DataFrame(rows)
            if kind == "fundamentals":
                df.insert(2, "sector", df["symbol"].map(sectors))
            tables[kind] = df

    all_syms = {s for df in tables.values() for s in df["symbol"]}
    if None in all_syms:
        notes.append("Some companies have no NSE ticker in the file. Type their NSE symbol in the "
                     "symbol column below, or those rows are skipped.")
    extra = sorted(s for s in all_syms if s and s not in known)
    if extra:
        notes.append(f"{', '.join(extra)} is not in data/stock_map.csv, so it is saved but not forecast.")
    if "fundamentals" in tables:
        notes.append("Tracxn's insider / institution percentages are saved for the Company profile tab. "
                     "They are not the NSE promoter, FII and DII split, so the model does not use them.")
    return tables, notes
