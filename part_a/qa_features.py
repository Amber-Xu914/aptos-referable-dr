"""Quality assurance (QA) of the extracted features, using the training split only.

Checks
  1. Lesion counts and areas (dk_, br_, ls_) rise with grade from 0 to 4 (grade 1 is hard to see).
  2. Several lesion features have a single-feature AUROC for referable DR clearly above 0.5.
  3. Global image statistics (gl_) are weaker than lesion features. A strong gl_ feature suggests
     the data contain a shortcut (exposure, blur or camera differences between grades).
  4. No feature is constant, almost always zero, or missing.

Usage
  python part_a/qa_features.py

Output
  outputs/part_a/feature_trends.csv     one row per feature: AUROC, Spearman correlation, mean by grade
  outputs/part_a/feature_boxplots.png   distribution by grade for nine key features
"""
import argparse

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import FEATURE_PREFIXES, get_paths

# features shown in the box plots
KEY = ["ls_area_frac", "ls_n", "dk_n", "dk_area_frac", "br_n", "br_area_frac",
       "vs_skel_frac", "gl_sharp", "gl_mean"]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--features", default=None)
    a = ap.parse_args()
    paths = get_paths(a.root)
    df = pd.read_csv(a.features or paths.out / "features_A.csv")
    df = df[df["split"] == "train"]
    cols = [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]
    print(f"{len(df)} training images, {len(cols)} features")
    print("training grade counts:", df["diagnosis"].value_counts().sort_index().to_dict())

    # --- one row of statistics per feature ---
    rows = []
    for c in cols:
        x = df[c].astype(float)
        row = {"feature": c, "nan_frac": x.isna().mean(), "zero_frac": (x == 0).mean(), "n_unique": x.nunique()}
        xf = x.fillna(x.median())
        row["spearman_grade"] = spearmanr(xf, df["diagnosis"])[0] if xf.nunique() > 1 else np.nan
        row["auroc_referable"] = roc_auc_score(df["referable"], xf) if xf.nunique() > 1 else np.nan
        for g in sorted(df["diagnosis"].unique()):
            row[f"mean_g{g}"] = x[df["diagnosis"] == g].mean()
        rows.append(row)
    t = pd.DataFrame(rows)
    t["strength"] = (t["auroc_referable"] - 0.5).abs()        # distance from chance, either direction
    t = t.sort_values("strength", ascending=False).drop(columns="strength")
    out = paths.out / "feature_trends.csv"
    t.to_csv(out, index=False)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print("\nStrongest single features (AUROC for referable; 0.5 = no signal, below 0.5 = inverted):")
    print(t.head(12).round(3).to_string(index=False))

    # --- warnings ---
    print("\n--- warnings ---")
    bad = t[(t.n_unique <= 1) | (t.nan_frac > 0.05) | (t.zero_frac > 0.98)]
    print("constant / mostly-zero / NaN features:", bad["feature"].tolist() or "none")
    weak = t[(t.feature.str.startswith(("dk_", "br_"))) & (t.auroc_referable.sub(0.5).abs() < 0.05)]
    print("lesion features with almost no signal (|AUROC-0.5| < 0.05):", weak["feature"].tolist() or "none")
    gl = t[t.feature.str.startswith("gl_")]
    les = t[t.feature.str.startswith(("dk_", "br_"))]
    gl_best = (gl.auroc_referable - 0.5).abs().max() if len(gl) else 0
    les_best = (les.auroc_referable - 0.5).abs().max() if len(les) else 0
    print(f"best global feature |AUROC-0.5| = {gl_best:.3f}  vs best lesion feature = {les_best:.3f}",
          "  <-- possible shortcut: a global image statistic is as strong as any lesion feature"
          if gl_best >= les_best else "")
    corr = df[cols].corr(method="spearman").abs().values
    iu = np.triu_indices(len(cols), 1)
    print("feature pairs with |Spearman| > 0.95:", int((corr[iu] > 0.95).sum()),
          "(acceptable for random forests and regularised logistic regression)")
    means = df.groupby("diagnosis")[["dk_n", "br_n"]].mean().round(2)
    print("\nmean lesion counts by grade (should generally increase):\n", means.to_string())

    # --- box plots of key features by grade ---
    keys = [k for k in KEY if k in df.columns]
    fig, axes = plt.subplots(3, 3, figsize=(13, 10))
    for ax, k in zip(axes.ravel(), keys):
        data = [df.loc[df["diagnosis"] == g, k].dropna().values for g in sorted(df["diagnosis"].unique())]
        ax.boxplot(data, showfliers=False)
        ax.set_xticklabels(sorted(df["diagnosis"].unique()))
        ax.set_title(k)
        ax.set_xlabel("original grade")
    for ax in axes.ravel()[len(keys):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(paths.out / "feature_boxplots.png", dpi=110)
    print(f"\nsaved {out} and feature_boxplots.png")


if __name__ == "__main__":
    main()
