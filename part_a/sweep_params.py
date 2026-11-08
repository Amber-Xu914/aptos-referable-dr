"""Measure the effect of one detector parameter on validation AUROC.

APTOS has no lesion outlines, so detector settings can only be judged by whether the downstream
classifier improves. For each value of the chosen parameter, this script:
  1. extracts features for a random subset of training images and all validation images,
  2. fits a quick logistic regression and random forest on the lesion features only
     (dk_, br_, ls_; the image statistics are left out so a dataset shortcut cannot hide the effect),
  3. reports validation AUROC.

Usage
  python part_a/sweep_params.py --param dark_thr --values 10 12 15
  python part_a/sweep_params.py --param bright_thr --values 20 25 30
  python part_a/sweep_params.py --param dark_line_len --values 11 15 21

Output: outputs/part_a/sweep_<param>.csv

Differences below about 0.01 are within noise for 549 validation images. Change one parameter at a
time and prefer the value whose contact sheets also look correct.
"""
import argparse
import ast
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import green_dir, SEED, get_paths, index_images, load_splits, parse_overrides
from detectors import make_params
from extract_features import extract
from train_models import build, scores, select_cols, xy


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache", choices=["auto", "c", "own"], default="auto",
                    help="green images: Part C's cache (c), the own cache (own), or Part C's if present (auto)")
    ap.add_argument("--param", required=True)
    ap.add_argument("--values", nargs="+", required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE", help="fixed overrides for other parameters")
    ap.add_argument("--n-train", type=int, default=600)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()

    # images: a random training subset (for speed) plus the full validation split
    paths = get_paths(a.root)
    paths.out.mkdir(parents=True, exist_ok=True)
    df = load_splits(paths.splits)
    tr = df[df["split"] == "train"].sample(min(a.n_train, (df["split"] == "train").sum()), random_state=SEED)
    sub = pd.concat([tr, df[df["split"] == "val"]]).reset_index(drop=True)
    index = index_images(green_dir(paths, a.cache))
    base = parse_overrides(a.set)

    rows = []
    for raw in a.values:
        try:
            val = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            val = raw

        # re-extract features with this parameter value
        params = make_params({**base, a.param: val})
        feats, errors = extract(sub, index, params, a.workers, progress=False)
        cols = [c for c in select_cols("bright_dark", list(feats.columns))]
        ftr, fva = feats[feats["split"] == "train"], feats[feats["split"] == "val"]
        res = {"param": a.param, "value": str(val), "n_errors": len(errors),
               "mean_dk_n": feats["dk_n"].mean(), "mean_br_n": feats["br_n"].mean()}

        # quick models with fixed settings, scored on validation
        for name in ("logreg", "rf"):
            hp = {"C": 1.0} if name == "logreg" else {"min_samples_leaf": 3}
            m = build(name, hp).fit(*xy(ftr, cols))
            res[f"val_auroc_{name}"] = roc_auc_score(fva["referable"], scores(m, xy(fva, cols)[0]))
        rows.append(res)
        print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()})

    out = pd.DataFrame(rows)
    dest = paths.out / f"sweep_{a.param}.csv"
    out.to_csv(dest, index=False)
    print("saved", dest)


if __name__ == "__main__":
    main()
