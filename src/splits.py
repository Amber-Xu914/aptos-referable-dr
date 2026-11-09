"""Locate and load the shared splits.csv.

The repo keeps it at the root (`splits.csv`); Part A's scripts expect
`data/part_c/splits.csv`. Both locations are accepted.
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CANDIDATES = (ROOT / "splits.csv", ROOT / "data" / "part_c" / "splits.csv")


def find_splits(path=None):
    if path:
        return Path(path)
    for p in CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(f"splits.csv not found in: {', '.join(map(str, CANDIDATES))}")


def load_splits(path=None):
    df = pd.read_csv(find_splits(path), dtype={"id_code": str})
    df["split"] = df["split"].str.strip().str.lower()
    if "label_conflict" not in df:
        df["label_conflict"] = 0
    return df
