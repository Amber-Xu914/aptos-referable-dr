"""Shared constants, folder paths and data loaders used by every Part A script.

Expected project layout (the project root is the folder that contains part_a/):

    project_root/
      data/
        raw/
          train.csv             APTOS labels: id_code, diagnosis (grade 0-4)
          train_images/         APTOS fundus photographs, {id_code}.png
        part_c/
          splits.csv            fixed train/val/test split from Part C
          cache_green_512/      green channel + CLAHE, 512 x 512 (input to Part A)
          cache_clahe_512/      colour + CLAHE, 512 x 512 (input to Part B, not used here)
        own_cache/green_512/    optional, written by make_green_cache.py
      part_a/                   Part A scripts
      outputs/part_a/           everything the scripts write (features, figures, predictions)

A different root can be given with --root or the APTOS_ROOT environment variable.
"""
import ast
import os
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SEED = 9027                      # same random seed as the group's data split
REFERABLE_MIN_GRADE = 2          # referable DR = grade 2 (moderate NPDR) or worse
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff")
# Feature name prefixes: global image statistics, bright lesions, dark lesions,
# vessels, and combined (dark + bright) lesions
FEATURE_PREFIXES = ("gl_", "br_", "dk_", "vs_", "ls_")
PROJECT_DEFAULT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Folder paths
# ---------------------------------------------------------------------------
def get_paths(root=None):
    """Return all input and output paths for a project root."""
    root = Path(root or os.environ.get("APTOS_ROOT") or PROJECT_DEFAULT).resolve()
    data = root / "data"
    return SimpleNamespace(
        root=root,
        raw_csv=data / "raw" / "train.csv",
        raw_images=data / "raw" / "train_images",
        splits=data / "part_c" / "splits.csv",
        green_c=data / "part_c" / "cache_green_512",
        green_own=data / "own_cache" / "green_512",
        out=root / "outputs" / "part_a",
    )


def green_dir(paths, which="auto"):
    """Choose the green-channel image folder.

    which = "c"    Part C's cache
            "own"  the cache written by make_green_cache.py
            "auto" Part C's cache if it contains images, otherwise the own cache
    """
    has = lambda d: d.is_dir() and any(p.suffix.lower() in IMAGE_EXTS for p in d.iterdir())
    if which == "c" or (which == "auto" and has(paths.green_c)):
        return paths.green_c
    if which == "own" or (which == "auto" and has(paths.green_own)):
        if which == "auto":
            print(f"NOTE: Part C's cache not found in {paths.green_c}; using {paths.green_own}")
        return paths.green_own
    raise FileNotFoundError(
        f"No green-channel images found.\n  Expected Part C's cache_green_512 folder in {paths.green_c.parent}\n"
        f"  or a cache made with:  python part_a/make_green_cache.py")


# ---------------------------------------------------------------------------
# Label and split tables
# ---------------------------------------------------------------------------
def _clean_ids(s):
    """Strip whitespace and any file extension from image ids."""
    s = s.astype(str).str.strip()
    return s.str.replace(r"(?i)\.(png|jpe?g|tiff?)$", "", regex=True)


def load_labels(path):
    """Read APTOS train.csv and add the binary referable label."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    df["id_code"] = _clean_ids(df["id_code"])
    df["diagnosis"] = df["diagnosis"].astype(int)
    df["referable"] = (df["diagnosis"] >= REFERABLE_MIN_GRADE).astype(int)
    return df


def load_splits(path):
    """Read splits.csv and normalise it (lower-case split names, ids without extension).

    Required columns: id_code, diagnosis, split. Optional: referable, label_conflict.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Expected Part C's splits.csv in {path.parent}")
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = {"id_code", "diagnosis", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}. Found: {list(df.columns)}")
    df["id_code"] = _clean_ids(df["id_code"])
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["diagnosis"] = df["diagnosis"].astype(int)
    if "referable" not in df.columns:
        df["referable"] = (df["diagnosis"] >= REFERABLE_MIN_GRADE).astype(int)
    return df


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------
def index_images(folder):
    """Map each image id (file name without extension) to its path."""
    folder = Path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder not found: {folder}")
    return {p.stem: p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS}


def load_green(path):
    """Load a cached green-channel image as a 2-D uint8 array."""
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise IOError(f"could not read {path}")
    if img.ndim == 3:                       # stored with 3 channels: OpenCV order is B, G, R, so take G
        img = img[..., 1] if img.shape[2] >= 2 else img[..., 0]
    if img.dtype == np.uint16:              # 16-bit image: rescale to 8-bit
        img = (img / 257).astype(np.uint8)
    elif img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(img)


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def stratified_subset(df, n_per_split, seed=SEED):
    """Random subset of about n_per_split images per split, keeping each split's grade mix."""
    parts = []
    for _, d in df.groupby("split"):
        frac = min(1.0, n_per_split / len(d))
        for _, x in d.groupby("diagnosis"):
            parts.append(x if frac >= 1 else x.sample(frac=frac, random_state=seed))
    return pd.concat(parts).reset_index(drop=True)


def parse_overrides(items):
    """Turn command-line overrides into a dictionary.

    ['dark_thr=15', 'fovea_dist=(130,240)'] -> {'dark_thr': 15, 'fovea_dist': (130, 240)}
    """
    out = {}
    for it in items or []:
        if "=" not in it:
            raise ValueError(f"--set expects name=value, got '{it}'")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out
