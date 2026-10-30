from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
MODELS = ROOT / "models"
SYMBOLS_FILE = DATA / "symbols.txt"
FUNDAMENTALS_FILE = DATA / "fundamentals.csv"


def load_symbols(path: Path = SYMBOLS_FILE) -> list[str]:
    lines = path.read_text().splitlines()
    return [s.strip().upper() for s in lines if s.strip() and not s.startswith("#")]

US_DIR = DATA / "us"
FX_DIR = DATA / "fx"
STOCK_MAP_FILE = DATA / "stock_map.csv"
SECTOR_LINKS_FILE = DATA / "sector_links.csv"
# Written by tests/make_synthetic_overnight.py so the dashboard can warn that numbers are fake.
SYNTHETIC_FLAG = DATA / "SYNTHETIC_DATA"
