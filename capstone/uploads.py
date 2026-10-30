"""Upload company data as CSV or Excel: fundamentals, shareholding (FII/DII),
financials, investors and profile details.

Column names are matched loosely (e.g. "P/E", "PE Ratio" and "pe" all work),
the kind of table is detected from its columns, and every sheet of an Excel
file is read. Fundamentals and shareholding rows go into
data/fundamentals.csv, which the model joins point in time by as_of, so use
the date the numbers became public (results or filing date).
"""
from __future__ import annotations

import io
import re

import pandas as pd

from .fundamentals import NUMERIC
from .paths import DATA, FUNDAMENTALS_FILE

FINANCIALS_FILE = DATA / "financials.csv"
INVESTORS_FILE = DATA / "investors.csv"
PROFILES_FILE = DATA / "company_profiles.csv"

TEMPLATES = {
    "fundamentals": {
        "file": FUNDAMENTALS_FILE, "keys": ["symbol", "as_of"],
        "columns": ["symbol", "as_of", "market_cap_cr", "pe", "pb", "eps_ttm", "roe_pct", "debt_to_equity"],
        "about": "Valuation ratios. One row per company per date; as_of = date the numbers were public.",
    },
    "shareholding": {
        "file": FUNDAMENTALS_FILE, "keys": ["symbol", "as_of"],
        "columns": ["symbol", "as_of", "promoter_pct", "fii_pct", "dii_pct", "public_pct", "govt_pct"],
        "about": "Quarterly shareholding pattern in %. FII = foreign (FII/FPI), DII = domestic "
                 "institutions (mutual funds, insurers, banks). as_of = filing date (~21 days after quarter end).",
    },
    "financials": {
        "file": FINANCIALS_FILE, "keys": ["symbol", "period_end"],
        "columns": ["symbol", "period_end", "revenue_cr", "ebitda_cr", "net_profit_cr", "eps",
                    "total_debt_cr", "cash_cr"],
        "about": "Quarterly or annual results in ₹ crore. One row per company per period.",
    },
    "investors": {
        "file": INVESTORS_FILE, "keys": ["symbol", "investor"],
        "columns": ["symbol", "investor", "investor_type", "country", "stake_pct", "as_of"],
        "about": "Major holders. investor_type: Promoter, FII/FPI, DII, Mutual Fund, Insurance, "
                 "Government, PE/VC, Retail.",
    },
    "profile": {
        "file": PROFILES_FILE, "keys": ["symbol"],
        "columns": ["symbol", "founded", "headquarters", "employees", "ceo", "website", "description"],
        "about": "One row per company: basic details shown on the Company profile tab.",
    },
}

# Normalised header text -> our column name.
SYNONYMS = {
    "symbol": ["symbol", "ticker", "nse symbol", "nse code", "nse ticker", "stock", "scrip"],
    "as_of": ["as of", "as_of", "date", "report date", "filing date", "announcement date", "asof"],
    "period_end": ["period end", "period", "quarter", "quarter end", "year end", "fiscal period"],
    "market_cap_cr": ["market cap cr", "market cap", "mcap", "market capitalization", "market cap rs cr"],
    "pe": ["pe", "p e", "pe ratio", "price to earnings", "p e ratio", "stock p e"],
    "pb": ["pb", "p b", "price to book", "pb ratio"],
    "eps_ttm": ["eps ttm", "eps", "ttm eps", "earnings per share"],
    "roe_pct": ["roe pct", "roe", "return on equity", "roe %"],
    "debt_to_equity": ["debt to equity", "d e", "debt equity", "de ratio"],
    "promoter_pct": ["promoter pct", "promoters", "promoter", "promoter holding", "promoter %"],
    "fii_pct": ["fii pct", "fii", "fpi", "fiis", "fii fpi", "foreign institutions", "foreign portfolio investors"],
    "dii_pct": ["dii pct", "dii", "diis", "domestic institutions"],
    "public_pct": ["public pct", "public", "retail", "non institutions"],
    "govt_pct": ["govt pct", "government", "govt"],
    "revenue_cr": ["revenue cr", "revenue", "sales", "total income", "revenue from operations", "net sales"],
    "ebitda_cr": ["ebitda cr", "ebitda", "operating profit"],
    "net_profit_cr": ["net profit cr", "net profit", "pat", "profit after tax"],
    "total_debt_cr": ["total debt cr", "total debt", "borrowings", "debt"],
    "cash_cr": ["cash cr", "cash", "cash and equivalents"],
    "investor": ["investor", "investor name", "holder", "shareholder", "name"],
    "investor_type": ["investor type", "type", "category", "holder type"],
    "country": ["country", "origin"],
    "stake_pct": ["stake pct", "stake", "holding", "holding %", "stake %", "% holding"],
    "founded": ["founded", "founded year", "year founded", "incorporation year"],
    "headquarters": ["headquarters", "hq", "location", "city"],
    "employees": ["employees", "employee count", "headcount"],
    "ceo": ["ceo", "md ceo", "managing director"],
    "website": ["website", "url"],
    "description": ["description", "about", "overview", "business"],
}
_LOOKUP = {alias: col for col, aliases in SYNONYMS.items() for alias in aliases}
TEXT_COLS = {"symbol", "as_of", "period_end", "investor", "investor_type", "country", "headquarters",
             "ceo", "website", "description"}


def _norm(header: str) -> str:
    h = str(header).lower().replace("%", " pct ").replace("₹", " ")
    h = re.sub(r"\(.*?\)", " ", h)
    return re.sub(r"[^a-z0-9]+", " ", h).strip()


def _date(v) -> str | None:
    """Accept 2026-09-30, 30-09-2026, 30/09/2026, Sep 2026, Excel dates; day first for Indian style."""
    if pd.isna(v):
        return None
    if isinstance(v, (pd.Timestamp,)) or hasattr(v, "year"):
        return pd.Timestamp(v).strftime("%Y-%m-%d")
    text = str(v).strip()
    iso = re.match(r"^\d{4}-\d{1,2}-\d{1,2}", text)
    try:
        ts = pd.to_datetime(text, dayfirst=not iso)
    except (ValueError, TypeError):
        return None
    return ts.strftime("%Y-%m-%d")


def standardise(df: pd.DataFrame) -> pd.DataFrame:
    """Rename recognised columns to ours and drop the rest; clean numbers like '45.2%' or '1,234'."""
    rename = {}
    for c in df.columns:
        n = _norm(c)
        target = _LOOKUP.get(n) or _LOOKUP.get(n.replace(" pct", "").strip())
        if target and target not in rename.values():
            rename[c] = target
    out = df.rename(columns=rename)[list(rename.values())].copy()
    if "symbol" in out:
        out["symbol"] = out["symbol"].astype(str).str.upper().str.strip().str.replace(".NS", "", regex=False)
    for c in out.columns:
        if c in ("as_of", "period_end"):
            out[c] = out[c].map(_date)
        elif c not in TEXT_COLS:
            out[c] = pd.to_numeric(out[c].astype(str).str.replace(r"[,%₹\s]", "", regex=True)
                                   .replace({"": None, "nan": None, "-": None}), errors="coerce")
    return out.dropna(how="all")


def detect(df: pd.DataFrame) -> str | None:
    """Which template a standardised table matches best (needs symbol + one data column)."""
    if "symbol" not in df:
        return None
    best, score = None, 0
    for kind, t in TEMPLATES.items():
        data_cols = [c for c in t["columns"] if c not in ("symbol", "as_of", "period_end")]
        s = sum(c in df for c in data_cols)
        if s > score:
            best, score = kind, s
    return best


def read_upload(name: str, data: bytes) -> dict[str, pd.DataFrame]:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data), sheet_name=None)
    for enc in ("utf-8-sig", "latin-1"):
        try:
            return {name: pd.read_csv(io.BytesIO(data), encoding=enc)}
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read the CSV file")


def merge_save(kind: str, new: pd.DataFrame) -> int:
    """Add rows to the kind's CSV. Same keys: newer non-empty values win, others are kept."""
    t = TEMPLATES[kind]
    path, keys = t["file"], t["keys"]
    missing = [k for k in keys if k not in new]
    if missing:
        raise ValueError(f"missing column(s): {', '.join(missing)}")
    new = new.dropna(subset=keys)
    if kind == "financials" and "eps_ttm" in new and "eps" not in new:
        new = new.rename(columns={"eps_ttm": "eps"})  # an "EPS" header means quarterly EPS here
    old = pd.read_csv(path, dtype={"symbol": str}) if path.exists() else pd.DataFrame(columns=keys)
    both = pd.concat([old, new], ignore_index=True)
    merged = both.groupby(keys, as_index=False, sort=True).last()  # last non-null per column
    if path == FUNDAMENTALS_FILE:
        order = ["symbol", "as_of", *NUMERIC, "public_pct", "govt_pct", "sector"]
        merged = merged[[c for c in order if c in merged] + [c for c in merged if c not in order]]
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(path, index=False)
    return len(new)


def load(kind: str) -> pd.DataFrame:
    path = TEMPLATES[kind]["file"]
    return pd.read_csv(path) if path.exists() else pd.DataFrame(columns=TEMPLATES[kind]["columns"])


def template_workbook() -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as xl:
        pd.DataFrame([(k, t["about"]) for k, t in TEMPLATES.items()],
                     columns=["Sheet", "What to fill in"]).to_excel(xl, sheet_name="README", index=False)
        for kind, t in TEMPLATES.items():
            pd.DataFrame(columns=t["columns"]).to_excel(xl, sheet_name=kind, index=False)
    return buf.getvalue()
