"""Validation metrics for comparing models in the ablation table (Part C).

Reads any prediction CSV in the group format
    image_id, split, true_grade, model_score, model_type
(model_score = probability of referable DR, grade >= 2), so it works for A's,
B's and C's files alike.

This is for choosing between model variants on the VALIDATION split. D's
evaluation script is the group's official evaluation (thresholds, calibration,
test set); these numbers should agree with it but do not replace it.

Metrics, following the group's agreed list:
    auroc               all validation images
    auroc_ci_lo/hi      95% bootstrap confidence interval
    auroc_g1_vs_ref     grade 1 vs grades 2-4 only. Removes grade 0, whose photos come
                        mostly from different cameras (Part A's "camera shortcut"), so
                        it tests the clinically hard boundary.
    auroc_no_conflict   excluding images flagged label_conflict in splits.csv
    spec_at_sens85      specificity at the threshold that gives >= 85% sensitivity
    sens_g2/g3/g4       sensitivity for each referable grade at that same threshold

Usage:
    python -m src.metrics results/part_a/preds_A_rf_all.csv results/part_c/preds_C_*.csv
"""

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.splits import load_splits

REFERABLE_MIN_GRADE = 2
TARGET_SENS = 0.85
SEED = 9027


def load_preds(path, split="val"):
    df = pd.read_csv(path, dtype={"image_id": str})
    missing = {"image_id", "split", "true_grade", "model_score"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s): {sorted(missing)}")
    df = df[df["split"].str.lower() == split].copy()
    if df.empty:
        raise ValueError(f"{path} has no rows with split == {split!r}")
    df["referable"] = (df["true_grade"] >= REFERABLE_MIN_GRADE).astype(int)
    return df


def auroc_ci(y, s, n_boot=1000, seed=SEED):
    """95% percentile bootstrap interval for AUROC."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if y[i].min() != y[i].max():
            vals.append(roc_auc_score(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def threshold_for_sensitivity(y, s, target=TARGET_SENS):
    """Highest threshold whose sensitivity (score >= t) is at least `target`."""
    pos = np.sort(np.asarray(s)[np.asarray(y) == 1])[::-1]
    k = int(np.ceil(target * len(pos)))          # need at least k positives at or above t
    return float(pos[k - 1])


def evaluate(df, conflict_ids=frozenset(), n_boot=1000):
    """All ablation metrics for one model's validation predictions."""
    y, s = df["referable"].to_numpy(), df["model_score"].to_numpy()
    lo, hi = auroc_ci(y, s, n_boot) if n_boot else (np.nan, np.nan)

    hard = df[df["true_grade"] >= 1]
    clean = df[~df["image_id"].isin(conflict_ids)]

    t = threshold_for_sensitivity(y, s)
    pred = s >= t
    out = {
        "n_val": len(df),
        "auroc": roc_auc_score(y, s),
        "auroc_ci_lo": lo,
        "auroc_ci_hi": hi,
        "auroc_g1_vs_ref": roc_auc_score(hard["referable"], hard["model_score"]),
        "auroc_no_conflict": roc_auc_score(clean["referable"], clean["model_score"]),
        "threshold_sens85": t,
        "sens_at_t": pred[y == 1].mean(),
        "spec_at_sens85": (~pred[y == 0]).mean(),
    }
    for g in (2, 3, 4):
        m = df["true_grade"].to_numpy() == g
        out[f"sens_g{g}"] = pred[m].mean() if m.any() else np.nan
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Validation metrics for prediction CSVs.")
    p.add_argument("preds", nargs="+")
    p.add_argument("--splits", default=None, help="path to splits.csv (default: auto-detect)")
    p.add_argument("--n-boot", type=int, default=1000)
    a = p.parse_args(argv)

    splits = load_splits(a.splits)
    conflict = frozenset(splits.loc[splits["label_conflict"] == 1, "id_code"])
    rows = []
    for path in a.preds:
        df = load_preds(path)
        name = df["model_type"].iloc[0] if "model_type" in df else path
        rows.append({"model": name, **evaluate(df, conflict, a.n_boot)})
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    sys.exit(main())
