"""Compute the 38 classical features for every image and save them with the split labels.

Usage
  python part_a/extract_features.py                                  train + val (default)
  python part_a/extract_features.py --n-per-split 150                quick stratified subset
  python part_a/extract_features.py --set dark_thr=15 bright_thr=25  override detector parameters
  python part_a/extract_features.py --include-test                   also the test split (final run only)

Output: outputs/part_a/features_A.csv, one row per image (split columns + 38 features).

The test split is skipped by default. Feature extraction does not use labels, but skipping test
makes it easy to keep the rule that the test set is used only once, at the end.
"""
import argparse
import dataclasses
import multiprocessing as mp
import os
import sys
import time

import cv2
import pandas as pd

from common import green_dir, get_paths, index_images, load_green, load_splits, parse_overrides, stratified_subset, SEED
from detectors import Params, make_params
from features import compute_features

try:
    from tqdm import tqdm
except ImportError:                                  # progress bar is optional
    def tqdm(x, **kw):
        return x


# ---------------------------------------------------------------------------
# Per-image work (runs in a worker process)
# ---------------------------------------------------------------------------
def _work(job):
    """Features for one image; errors are returned instead of raised so the run continues."""
    id_code, path, pdict = job
    try:
        cv2.setNumThreads(1)
        f = compute_features(load_green(path), Params(**pdict))
        f["id_code"] = id_code
        return f
    except Exception as e:
        return {"id_code": id_code, "error": repr(e)}


def extract(df, green_index, params: Params, workers=1, progress=True):
    """Compute features for every id in df.

    Returns (df merged with the feature columns, list of (id, error) for images that failed).
    """
    pdict = dataclasses.asdict(params)
    jobs = [(r, str(green_index[r]), pdict) for r in df["id_code"] if r in green_index]
    missing = [r for r in df["id_code"] if r not in green_index]
    if missing:
        print(f"WARNING: {len(missing)} ids from splits.csv have no image file, e.g. {missing[:3]}")

    # run in parallel when more than one worker is requested
    if workers > 1:
        with mp.Pool(workers) as pool:
            it = pool.imap_unordered(_work, jobs, chunksize=8)
            rows = list(tqdm(it, total=len(jobs), disable=not progress))
    else:
        rows = [_work(j) for j in tqdm(jobs, disable=not progress)]

    errors = [(r["id_code"], r["error"]) for r in rows if "error" in r]
    ok = pd.DataFrame([r for r in rows if "error" not in r])
    return df.merge(ok, on="id_code", how="inner"), errors


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache", choices=["auto", "c", "own"], default="auto",
                    help="green images: Part C's cache (c), the own cache (own), or Part C's if present (auto)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--include-test", action="store_true", help="also process the test split (final run only)")
    ap.add_argument("--n-per-split", type=int, default=0, help="random stratified subset per split (0 = all)")
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="override detector parameters")
    ap.add_argument("--out", default=None, help="output CSV (default: outputs/part_a/features_A.csv)")
    a = ap.parse_args()

    # select images
    paths = get_paths(a.root)
    paths.out.mkdir(parents=True, exist_ok=True)
    params = make_params(parse_overrides(a.set))
    df = load_splits(paths.splits)
    keep = ["train", "val"] + (["test"] if a.include_test else [])
    df = df[df["split"].isin(keep)].reset_index(drop=True)
    if a.n_per_split:
        df = stratified_subset(df, a.n_per_split, SEED)
    print(f"{len(df)} images ({df['split'].value_counts().to_dict()}), {a.workers} worker(s)")
    print("parameters:", {k: v for k, v in dataclasses.asdict(params).items()})

    # extract and save
    t0 = time.time()
    out, errors = extract(df, index_images(green_dir(paths, a.cache)), params, a.workers)
    dest = a.out or (paths.out / "features_A.csv")
    out.to_csv(dest, index=False)
    print(f"done in {time.time() - t0:.0f}s -> {dest}  ({len(out)} rows, {out.shape[1]} columns)")
    if errors:
        print(f"{len(errors)} image(s) failed, e.g. {errors[:3]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
