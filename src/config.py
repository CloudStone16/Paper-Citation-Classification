"""Shared paths and constants for the pipeline."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"

RAW_CSV = RAW / "LMP2015.csv"
RAW_URL = "https://zenodo.org/records/4989552/files/LMP2015.csv?download=1"
RAW_MD5 = "f2efe15c4a2092443eaeb0f99ae43fd5"

# Citation bins used by the Stanford report: <60, 60-120, >120
BIN_EDGES = [60, 120]
BIN_LABELS = ["<60", "60-120", ">120"]

SEED = 42
TEST_SIZE = 0.15

for d in (RAW, INTERIM, PROCESSED):
    d.mkdir(parents=True, exist_ok=True)
