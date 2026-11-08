"""Train and compare classifiers on the classical features, and write prediction files.

Commands
  python part_a/train_models.py ablation
      Compare feature sets and classifiers on validation AUROC (the ablation ladder).
  python part_a/train_models.py final --model rf --featset all
      Fit one model on the training split and write validation predictions.
  python part_a/train_models.py final --model rf --featset all --final-test
      Also write test predictions. Intended to be run once, after all choices are fixed.

Protocol
  Models are fitted on the training split only.
  Feature set, classifier and hyperparameters are chosen on validation AUROC only.
  The test split is used only with --final-test.
  model_score is the predicted probability of referable DR (grade 2 or above).

Prediction file columns: image_id, split, true_grade, model_score, model_type
  model_type = A_<model>_<featset>, for example A_rf_all
"""
import argparse
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import QuantileTransformer
from sklearn.svm import SVC

from common import FEATURE_PREFIXES, SEED, get_paths

warnings.filterwarnings("ignore", message=".*n_quantiles.*")

# ---------------------------------------------------------------------------
# Ablation ladder: feature sets by name prefix.
# "prevalence" and "global" are controls: no features at all, and image statistics
# without any lesion detection (measures how much a dataset shortcut alone achieves).
# ---------------------------------------------------------------------------
FEATURE_SETS = {
    "prevalence": (),                              # predicts the base rate for everyone (AUROC 0.5)
    "global": ("gl_",),                            # image statistics only
    "bright": ("br_",),
    "dark": ("dk_",),
    "bright_dark": ("br_", "dk_", "ls_"),
    "lesions_vessels": ("br_", "dk_", "ls_", "vs_"),
    "all": ("gl_", "br_", "dk_", "ls_", "vs_"),
}

# Hyperparameter grids, kept small so that selection on 549 validation images does not overfit
GRIDS = {
    "logreg": [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)],
    "svm_rbf": [{"C": c, "gamma": g} for c in (0.3, 1.0, 3.0, 10.0) for g in ("scale", 0.01)],
    "rf": [{"min_samples_leaf": l, "max_features": m} for l in (1, 3, 5, 10) for m in ("sqrt", 0.5)],
}


# ---------------------------------------------------------------------------
# Model construction and scoring
# ---------------------------------------------------------------------------
def build(name, hp=None, pca=None, probability=False):
    """Build a scikit-learn pipeline for one classifier with the given hyperparameters."""
    hp = dict(hp or {})
    if name == "prevalence":
        return Pipeline([("clf", DummyClassifier(strategy="prior"))])
    imp = ("imp", SimpleImputer(strategy="median"))
    if name == "rf":
        # random forests need no feature scaling
        return Pipeline([imp, ("clf", RandomForestClassifier(
            n_estimators=500, class_weight="balanced_subsample", n_jobs=-1, random_state=SEED, **hp))])
    # Logistic regression and SVM need comparable feature scales. A rank-based transform to a normal
    # distribution handles heavy-tailed counts and areas. It is fitted inside the pipeline, on the
    # training data only, so no information leaks from validation or test.
    steps = [imp, ("qt", QuantileTransformer(n_quantiles=200, output_distribution="normal", random_state=SEED))]
    if pca:
        steps.append(("pca", PCA(n_components=pca, random_state=SEED)))
    if name == "logreg":
        clf = LogisticRegression(class_weight="balanced", max_iter=5000, **hp)
    elif name == "svm_rbf":
        clf = SVC(class_weight="balanced", probability=probability, random_state=SEED, **hp)
    else:
        raise ValueError(f"unknown model {name}")
    return Pipeline(steps + [("clf", clf)])


def scores(model, X):
    """Probability of the positive class where available, otherwise the decision function."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def auc_ci(y, s, n_boot=1000, seed=SEED):
    """95% bootstrap confidence interval for AUROC (images resampled with replacement)."""
    y, s = np.asarray(y), np.asarray(s)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if len(set(y[i])) == 2:
            vals.append(roc_auc_score(y[i], s[i]))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (np.nan, np.nan)


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load(features_path):
    """Features table and the list of feature columns."""
    df = pd.read_csv(features_path)
    return df, [c for c in df.columns if c.startswith(FEATURE_PREFIXES)]


def select_cols(featset, all_cols):
    """Feature columns belonging to a named feature set."""
    return [c for c in all_cols if c.startswith(FEATURE_SETS[featset])] if FEATURE_SETS[featset] else []


def xy(df, cols):
    """Feature matrix and referable labels."""
    X = df[cols].to_numpy(float) if cols else np.zeros((len(df), 1))
    return X, df["referable"].to_numpy(int)


def tune(model_name, cols, tr, va, pca=None):
    """Grid search: fit each setting on train, score on validation. Returns (best AUROC, best settings)."""
    Xtr, ytr = xy(tr, cols)
    Xva, yva = xy(va, cols)
    if model_name == "prevalence":
        return 0.5, {}
    best = (-1.0, None)
    for hp in GRIDS[model_name]:
        m = build(model_name, hp, pca).fit(Xtr, ytr)
        auc = roc_auc_score(yva, scores(m, Xva))
        if auc > best[0]:
            best = (auc, hp)
    return best


# ---------------------------------------------------------------------------
# "ablation" command
# ---------------------------------------------------------------------------
def cmd_ablation(a):
    """Every feature set x every classifier, scored on validation AUROC with a 95% CI."""
    paths = get_paths(a.root)
    df, all_cols = load(a.features or paths.out / "features_A.csv")
    tr, va = df[df["split"] == "train"], df[df["split"] == "val"]
    print(f"train {len(tr)} / val {len(va)} images; {len(all_cols)} features")
    rows = []
    for fs in FEATURE_SETS:
        cols = select_cols(fs, all_cols)
        for mn in (["prevalence"] if fs == "prevalence" else a.models):
            auc, hp = tune(mn, cols, tr, va, a.pca)
            if mn == "prevalence":
                lo, hi = 0.5, 0.5
            else:
                m = build(mn, hp, a.pca).fit(*xy(tr, cols))
                lo, hi = auc_ci(va["referable"], scores(m, xy(va, cols)[0]))
            rows.append({"featset": fs, "model": mn, "n_features": len(cols), "val_auroc": auc,
                         "ci_lo": lo, "ci_hi": hi, "best_hp": json.dumps(hp)})
            print(f"{fs:16s} {mn:10s} AUROC {auc:.3f}  [{lo:.3f}, {hi:.3f}]  {hp}")
    t = pd.DataFrame(rows)
    out = paths.out / "ablation_table.csv"
    t.to_csv(out, index=False)
    print(f"\nsaved {out}")
    print("Read it as a ladder: prevalence (0.5) < global (control) < bright / dark < bright_dark < all.")
    print("Differences smaller than the CI width are not reliable improvements.")


# ---------------------------------------------------------------------------
# "final" command
# ---------------------------------------------------------------------------
def diagnostics(va, s):
    """Validation summaries: mean score by grade, and AUROC overall and for sub-comparisons."""
    va = va.assign(score=s)
    print("\nmean predicted probability by original grade (should rise with grade):")
    print(va.groupby("diagnosis")["score"].agg(["mean", "count"]).round(3).to_string())
    print(f"AUROC, all val images:                      {roc_auc_score(va['referable'], va['score']):.3f}")
    if "label_conflict" in va.columns:
        k = va[va["label_conflict"].fillna(0) == 0]
        print(f"AUROC, excluding label_conflict images:     {roc_auc_score(k['referable'], k['score']):.3f}  (n={len(k)})")
    # grade 0 vs referable is affected by camera differences; grade 1 vs referable is not
    for name, sub in (("grade>=2 vs grade 0 only (easy)", va[va["diagnosis"] != 1]),
                      ("grade>=2 vs grade 1 only (hard boundary)", va[va["diagnosis"] != 0])):
        try:
            print(f"AUROC, {name}: {roc_auc_score(sub['referable'], sub['score']):.3f}")
        except ValueError:
            pass


def cmd_final(a):
    """Tune, fit on train, write predictions (validation, plus test with --final-test) and feature importance."""
    paths = get_paths(a.root)
    df, all_cols = load(a.features or paths.out / "features_A.csv")
    cols = select_cols(a.featset, all_cols)
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))

    # choose hyperparameters on validation, then fit on the training split only
    auc, hp = tune(a.model, cols, tr, va, a.pca)
    print(f"{a.model} on '{a.featset}' ({len(cols)} features): best val AUROC {auc:.3f} with {hp}")
    model = build(a.model, hp, a.pca, probability=True).fit(*xy(tr, cols))
    s_val = scores(model, xy(va, cols)[0])

    # predictions: validation always, test only when requested
    parts = [va.assign(model_score=s_val)]
    if a.final_test:
        if te.empty:
            raise SystemExit("No test rows in the features file. Re-run extract_features.py with --include-test.")
        parts.append(te.assign(model_score=scores(model, xy(te, cols)[0])))
        print(f"TEST predictions written for {len(te)} images.")
    else:
        print("Test split not used (add --final-test for the final run).")
    out = pd.concat(parts)

    # save predictions in the shared format, and the fitted model
    pred = pd.DataFrame({"image_id": out["id_code"], "split": out["split"], "true_grade": out["diagnosis"],
                         "model_score": out["model_score"].round(6), "model_type": f"A_{a.model}_{a.featset}"})
    pdir = paths.out / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    dest = pdir / f"preds_A_{a.model}_{a.featset}.csv"
    pred.to_csv(dest, index=False)
    joblib.dump({"model": model, "cols": cols, "hp": hp, "featset": a.featset}, pdir / f"model_A_{a.model}_{a.featset}.joblib")
    print("saved", dest)
    diagnostics(va, s_val)

    # permutation importance on validation: drop in AUROC when one feature is shuffled
    if cols:
        Xva, yva = xy(va, cols)
        pi = permutation_importance(model, Xva, yva, scoring="roc_auc", n_repeats=10, random_state=SEED)
        imp = pd.DataFrame({"feature": cols, "auroc_drop": pi.importances_mean, "sd": pi.importances_std})
        imp = imp.sort_values("auroc_drop", ascending=False)
        imp.to_csv(pdir / f"importance_A_{a.model}_{a.featset}.csv", index=False)
        print("\ntop features by permutation importance (val):")
        print(imp.head(8).round(4).to_string(index=False))


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("ablation", "final"):
        sp = sub.add_parser(name)
        sp.add_argument("--root", default=None)
        sp.add_argument("--features", default=None)
        sp.add_argument("--pca", type=float, default=None, help="PCA variance fraction for logreg/svm, e.g. 0.95")
        if name == "ablation":
            sp.add_argument("--models", nargs="+", default=["logreg", "svm_rbf", "rf"], choices=list(GRIDS))
            sp.set_defaults(fn=cmd_ablation)
        else:
            sp.add_argument("--model", required=True, choices=list(GRIDS))
            sp.add_argument("--featset", default="all", choices=[k for k in FEATURE_SETS if k != "prevalence"])
            sp.add_argument("--final-test", action="store_true", help="also write test predictions (final run only)")
            sp.set_defaults(fn=cmd_final)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
