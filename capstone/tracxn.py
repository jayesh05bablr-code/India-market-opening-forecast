"""Read a company profile PDF exported from Tracxn and pull out key numbers.

Extraction is best effort: Tracxn layouts vary, so every value is shown in an
editable table on the dashboard to check before it is saved into
data/fundamentals.csv (which the model uses after the next train).
"""
from __future__ import annotations

import io
import re
from datetime import date

import pandas as pd

from .fundamentals import NUMERIC
from .paths import DATA

PROFILE_DIR = DATA / "company_profiles"
PROFILES_FILE = DATA / "tracxn_profiles.csv"

# Label as it may appear in the PDF -> our field name.
LABELS = {
    "Market Cap": "market_cap", "Market Capitalization": "market_cap",
    "Annual Revenue": "revenue", "Revenue": "revenue",
    "Net Profit": "net_profit", "EBITDA": "ebitda",
    "Total Funding": "total_funding", "Latest Valuation": "valuation", "Valuation": "valuation",
    "Employee Count": "employees", "Employees": "employees",
    "Founded Year": "founded", "Founded": "founded",
    "Stage": "stage", "Location": "location",
    "P/E": "pe", "PE Ratio": "pe", "P/B": "pb", "EPS": "eps_ttm",
    "ROE": "roe_pct", "Debt to Equity": "debt_to_equity", "Debt/Equity": "debt_to_equity",
}
NUM = r"[-+]?\d[\d,]*\.?\d*"

# Earlier versions saved Tracxn extracts as company_profiles.csv; that name now holds
# the uploaded profile table, so move an old Tracxn file to its new name once.
_OLD = DATA / "company_profiles.csv"
if _OLD.exists() and not PROFILES_FILE.exists():
    try:
        if "file" in pd.read_csv(_OLD, nrows=1).columns:
            _OLD.rename(PROFILES_FILE)
    except Exception:
        pass


def pdf_text(data: bytes) -> str:
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(data))
    return "\n".join(page.extract_text() or "" for page in reader.pages)


def _value_after(text: str, label: str) -> str | None:
    # "Label: value", "Label value" on one line, or the value on the next line.
    m = re.search(rf"(?im)^\s*{re.escape(label)}\s*[:\-]?\s*(.+)$", text)
    if not m:
        return None
    value = m.group(1).strip()
    if not value:
        nxt = text[m.end():].lstrip("\n").split("\n", 1)[0].strip()
        value = nxt or None
    return value[:80] if value else None


def to_crore(value: str | None, usdinr: float | None) -> float | None:
    """'₹2.4L Cr' / '₹15,000 Cr' / '$29.1B' / '$450M' -> rupees crore."""
    if not value:
        return None
    m = re.search(NUM, value)
    if not m:
        return None
    x = float(m.group().replace(",", ""))
    v = value.upper().replace(" ", "")
    if "$" in v or "USD" in v:
        if not usdinr:
            return None
        mult = 1e9 if "B" in v else 1e6 if "M" in v else 1e3 if "K" in v else 1
        return x * mult * usdinr / 1e7
    if "LCR" in v or "LAKHCR" in v:
        return x * 1e5
    if "CR" in v:
        return x
    return None


def _number(value: str | None) -> float | None:
    m = re.search(NUM, value or "")
    return float(m.group().replace(",", "")) if m else None


def extract(text: str, usdinr: float | None) -> dict:
    found = {}
    for label, field in LABELS.items():
        if field not in found:
            v = _value_after(text, label)
            if v:
                found[field] = v
    out = {"raw_" + k: v for k, v in found.items()}
    out["market_cap_cr"] = to_crore(found.get("market_cap") or found.get("valuation"), usdinr)
    for f in ("pe", "pb", "eps_ttm", "roe_pct", "debt_to_equity"):
        out[f] = _number(found.get(f))
    return out


def save_fundamentals(symbol: str, as_of: date, values: dict, sector: str = "") -> None:
    """Add one dated row to data/fundamentals.csv (the point-in-time join uses as_of)."""
    from .uploads import merge_save
    row = {"symbol": symbol.upper(), "as_of": as_of.isoformat(), "sector": sector,
           **{c: values.get(c) for c in NUMERIC if values.get(c) is not None}}
    merge_save("fundamentals", pd.DataFrame([row]))


def save_profile(symbol: str, filename: str, data: bytes, extracted: dict) -> None:
    PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    (PROFILE_DIR / f"{symbol.upper()}_{date.today():%Y%m%d}_{safe}").write_bytes(data)
    row = {"symbol": symbol.upper(), "uploaded": date.today().isoformat(), "file": filename,
           **{k: v for k, v in extracted.items() if v is not None}}
    old = pd.read_csv(PROFILES_FILE) if PROFILES_FILE.exists() else pd.DataFrame()
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(PROFILES_FILE, index=False)


def profile_row(symbol: str, extracted: dict) -> dict | None:
    """Founded / employees / location from a PDF extract, for the Company profile tab."""
    row = {"symbol": symbol.upper()}
    founded = _number(extracted.get("raw_founded"))
    if founded and 1800 < founded < 2100:
        row["founded"] = int(founded)
    employees = _number(extracted.get("raw_employees"))
    if employees:
        row["employees"] = int(employees)
    if extracted.get("raw_location"):
        row["headquarters"] = extracted["raw_location"]
    return row if len(row) > 1 else None
