"""
BIOM9027 - Shared evaluation script (Role D)
============================================

Task: referable diabetic retinopathy (APTOS 2019)
      referable = original grade >= 2 (moderate, severe, proliferative)

INPUT
-----
A CSV with one row per image (and per model) and these columns:
    image_id or id_code     : the APTOS image name (as in splits.csv)
    split                   : "val" or "test" (train rows are ignored)
    true_grade or diagnosis : the original 0-4 grade (the reference standard)
    model_score             : the model's probability that the image is referable (0 to 1)

Optional columns:
    model         : model/variant name, e.g. "A_classical", "B_resnet_finetune", "C_aug".
                    Several models can go in one CSV; each is evaluated separately and a
                    comparison table is written. The winner is picked on VALIDATION AUROC.
    image_quality : any label (e.g. "good", "blurry", "dark") for error analysis

Optional: --splits splits.csv adds dup_group / label_conflict to the error list.

HOW IT WORKS (and why)
----------------------
1. Thresholds are chosen on the VALIDATION set only.
2. The TEST set is then evaluated ONCE with those fixed thresholds.
   (Choosing the threshold on the test set would make the results look better than they are.)
3. Screening logic: missing a referable patient (false negative) is worse than an
   unnecessary referral (false positive), so the main threshold is chosen to reach
   a target sensitivity (default 90%) on validation, not the default 0.5.

USAGE
-----
Before all models are frozen (test rows are never read):
    python evaluate.py results.csv --splits splits.csv --val-only --target-sens 0.85

Final, once, after models and thresholds are frozen:
    python evaluate.py results.csv --out eval_output --splits splits.csv --quality image_quality.csv

In Colab:
    !python evaluate.py results.csv --out eval_output
"""

import argparse
import json
import os
import re

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, brier_score_loss

REFERABLE_MIN_GRADE = 2
GRADE_NAMES = {0: "0 No DR", 1: "1 Mild", 2: "2 Moderate", 3: "3 Severe", 4: "4 Proliferative"}


# ---------------------------------------------------------------------------
# Basic metrics
# ---------------------------------------------------------------------------
def metrics_at_threshold(y_true, scores, threshold):
    """All the confusion-matrix metrics for one threshold.  Positive = referable."""
    y_pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    def safe(a, b):
        return float(a / b) if b > 0 else float("nan")

    sens = safe(tp, tp + fn)          # of the truly referable, how many we caught
    spec = safe(tn, tn + fp)          # of the non-referable, how many we correctly cleared
    ppv = safe(tp, tp + fp)           # precision: of those we referred, how many needed it
    npv = safe(tn, tn + fn)           # of those we cleared, how many were truly fine
    f1 = safe(2 * ppv * sens, ppv + sens) if not np.isnan(ppv) and not np.isnan(sens) else float("nan")
    return {
        "threshold": float(threshold),
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "sensitivity": sens, "specificity": spec, "PPV_precision": ppv, "NPV": npv, "F1": f1,
        "referral_rate": safe(tp + fp, tp + fp + tn + fn),
    }


def threshold_for_sensitivity(y_true, scores, target_sens):
    """Highest threshold that still reaches the target sensitivity (keeps specificity as high as possible)."""
    pos_scores = np.sort(scores[y_true == 1])
    if len(pos_scores) == 0:
        raise ValueError("No referable cases in this set.")
    # we may miss at most floor((1 - target) * n_pos) positives
    n_miss_allowed = int(np.floor((1 - target_sens) * len(pos_scores) + 1e-9))
    return float(pos_scores[n_miss_allowed])


def threshold_for_npv_rule_out(y_true, scores, target_sens_low):
    """Low threshold for triage: below it we are confident the image is NOT referable."""
    return threshold_for_sensitivity(y_true, scores, target_sens_low)


def threshold_for_specificity(y_true, scores, target_spec):
    """Lowest threshold that reaches the target specificity (confident 'refer')."""
    neg_scores = np.sort(scores[y_true == 0])
    idx = int(np.ceil(target_spec * len(neg_scores))) - 1
    idx = min(max(idx, 0), len(neg_scores) - 1)
    return float(np.nextafter(neg_scores[idx], 1.0))


def auroc_grade1_vs_2to4(grades, scores):
    """AUROC using only grade 1 (negative) vs grades 2-4 (positive).
    Removes the easy grade-0 images, which in APTOS can be separated by camera/image
    statistics alone (a 'shortcut' found in Part A), so this shows lesion-based discrimination."""
    m = grades >= 1
    y = (grades[m] >= REFERABLE_MIN_GRADE).astype(int)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, scores[m]))


def evaluate_val_only(val, out, args, model_name):
    """Validation-only report. Use this BEFORE models are frozen: the test rows are never read."""
    os.makedirs(out, exist_ok=True)
    y, s, g = val["y_true"].values, val["model_score"].values, val["true_grade"].values
    t_main = threshold_for_sensitivity(y, s, args.target_sens)
    m = metrics_at_threshold(y, s, t_main)
    val = val.assign(pred_referable=(s >= t_main).astype(int))
    by_grade = (val.groupby("true_grade")["pred_referable"].agg(n="count", fraction_referred="mean")
                .reset_index().rename(columns={"true_grade": "grade"}))
    by_grade.to_csv(os.path.join(out, "val_referral_by_grade.csv"), index=False)
    row = {"model": model_name, "val_AUROC": float(roc_auc_score(y, s)),
           "val_AUROC_grade1_vs_2to4": auroc_grade1_vs_2to4(g, s),
           "threshold_from_val": t_main, "val_sensitivity": m["sensitivity"], "val_specificity": m["specificity"],
           **{f"val_referred_grade{int(r.grade)}": r.fraction_referred for r in by_grade.itertuples()}}
    print(f"\n=== {model_name} (VALIDATION ONLY, test not opened) ===")
    print(f"AUROC={row['val_AUROC']:.3f}, grade 1 vs 2-4 AUROC={row['val_AUROC_grade1_vs_2to4']:.3f}, "
          f"threshold={t_main:.3f} -> sens={m['sensitivity']:.3f}, spec={m['specificity']:.3f}")
    print(by_grade.round(3).to_string(index=False))
    return row


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------
def bootstrap_ci(y_true, scores, threshold, n_boot=1000, seed=0):
    """95% CI for AUROC, sensitivity, specificity, PPV, NPV by resampling images with replacement.
    Note: APTOS has no patient IDs, so we resample images, not patients."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    keys = ["AUROC", "sensitivity", "specificity", "PPV_precision", "NPV", "F1"]
    samples = {k: [] for k in keys}
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yt, sc = y_true[idx], scores[idx]
        if yt.min() == yt.max():
            continue  # need both classes
        m = metrics_at_threshold(yt, sc, threshold)
        samples["AUROC"].append(roc_auc_score(yt, sc))
        for k in keys[1:]:
            samples[k].append(m[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in samples.items()}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------
def calibration_table(y_true, scores, n_bins=10):
    """Does a score of 0.7 really mean ~70% of those images are referable?"""
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (scores >= lo) & (scores < hi) if hi < 1 else (scores >= lo) & (scores <= hi)
        if mask.sum() == 0:
            continue
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(mask.sum()),
                     "mean_predicted": float(scores[mask].mean()),
                     "observed_fraction_referable": float(y_true[mask].mean())})
    df = pd.DataFrame(rows)
    ece = float((df["n"] * (df["mean_predicted"] - df["observed_fraction_referable"]).abs()).sum() / len(scores))
    return df, ece


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_roc(y_val, s_val, y_test, s_test, op_point, out, model_name):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for y, s, label, style in [(y_val, s_val, "Validation", "--"), (y_test, s_test, "Test", "-")]:
        fpr, tpr, _ = roc_curve(y, s)
        ax.plot(fpr, tpr, style, label=f"{label} (AUROC = {roc_auc_score(y, s):.3f})")
    ax.plot([0, 1], [0, 1], ":", color="grey", label="Chance")
    ax.scatter([1 - op_point["specificity"]], [op_point["sensitivity"]], zorder=5, color="black",
               label=f"Chosen threshold on test ({op_point['threshold']:.2f})")
    ax.set_xlabel("1 - Specificity (false positive rate)")
    ax.set_ylabel("Sensitivity (true positive rate)")
    ax.set_title(f"ROC - referable DR - {model_name}")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "roc_curve.png"), dpi=200)
    plt.close(fig)


def plot_confusion(m, out, title):
    cm = np.array([[m["TN"], m["FP"]], [m["FN"], m["TP"]]])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = [["TN\n(correctly cleared)", "FP\n(unnecessary referral)"],
              ["FN\n(MISSED referable)", "TP\n(correctly referred)"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]}\n{labels[i][j]}", ha="center", va="center", fontsize=9,
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["Predicted: not referable", "Predicted: referable"], fontsize=8)
    ax.set_yticks([0, 1], ["True: not referable", "True: referable"], fontsize=8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "confusion_matrix_test.png"), dpi=200)
    plt.close(fig)


def plot_calibration(cal_df, ece, out):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], ":", color="grey", label="Perfect calibration")
    ax.plot(cal_df["mean_predicted"], cal_df["observed_fraction_referable"], "o-", label=f"Model (ECE = {ece:.3f})")
    ax.set_xlabel("Mean predicted score")
    ax.set_ylabel("Observed fraction referable")
    ax.set_title("Calibration (test set)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out, "calibration.png"), dpi=200)
    plt.close(fig)


def plot_sens_by_grade(grade_df, out):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(grade_df["grade"].astype(str), grade_df["fraction_referred"])
    for i, (v, n) in enumerate(zip(grade_df["fraction_referred"], grade_df["n"])):
        ax.text(i, v + 0.02, f"{v:.0%}\n(n={n})", ha="center", fontsize=8)
    ax.axvline(1.5, color="red", linestyle="--")
    ax.text(1.55, 1.05, "referable from here ->", color="red", fontsize=8)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("Fraction referred by the model")
    ax.set_xlabel("Original APTOS grade")
    ax.set_title("Referral rate by original grade (test set)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "referral_by_grade.png"), dpi=200)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def evaluate_one(df, out, args, model_name):
    """Full evaluation of ONE model. Writes everything into `out` and returns a summary row."""
    os.makedirs(out, exist_ok=True)
    args = argparse.Namespace(**{**vars(args), "out": out, "model_name": model_name})

    val = df[df["split"].isin(["val", "validation"])]
    test = df[df["split"] == "test"]
    if len(val) == 0 or len(test) == 0:
        raise SystemExit(f"[{model_name}] Need both 'val' and 'test' rows.")
    y_val, s_val = val["y_true"].values, val["model_score"].values
    y_test, s_test = test["y_true"].values, test["model_score"].values

    # ---- 1. thresholds chosen on VALIDATION only ----
    t_main = threshold_for_sensitivity(y_val, s_val, args.target_sens)
    t_low = threshold_for_npv_rule_out(y_val, s_val, args.triage_low_sens)
    t_high = max(threshold_for_specificity(y_val, s_val, args.triage_high_spec), t_low)

    # ---- 2. test set, evaluated once ----
    auroc_test = roc_auc_score(y_test, s_test)
    auroc_g1_test = auroc_grade1_vs_2to4(test["true_grade"].values, s_test)
    auroc_g1_val = auroc_grade1_vs_2to4(val["true_grade"].values, s_val)
    m_test_main = metrics_at_threshold(y_test, s_test, t_main)
    m_test_05 = metrics_at_threshold(y_test, s_test, 0.5)
    ci = bootstrap_ci(y_test, s_test, t_main, n_boot=args.n_boot)

    # sensitivity analysis: same threshold, test set WITHOUT label-conflict images (if splits.csv was given)
    clean = None
    if "label_conflict" in test.columns:
        keep = test["label_conflict"].fillna(0).astype(int) == 0
        yc, sc_ = test.loc[keep, "y_true"].values, test.loc[keep, "model_score"].values
        clean = {"n": int(keep.sum()), "n_excluded": int((~keep).sum()),
                 "AUROC": float(roc_auc_score(yc, sc_)), **metrics_at_threshold(yc, sc_, t_main)}

    # specificity at 90% sensitivity on test (a common summary number)
    fpr, tpr, _ = roc_curve(y_test, s_test)
    spec_at_90 = float(1 - fpr[np.argmax(tpr >= 0.90)])

    # ---- 3. referral rate by original grade ----
    test = test.copy()
    test["pred_referable"] = (test["model_score"] >= t_main).astype(int)
    grade_df = (test.groupby("true_grade")["pred_referable"]
                .agg(n="count", fraction_referred="mean").reset_index()
                .rename(columns={"true_grade": "grade"}))

    # ---- 4. calibration ----
    cal_df, ece = calibration_table(y_test, s_test)
    brier = brier_score_loss(y_test, s_test)

    # ---- 5. two-threshold triage ----
    test["triage"] = np.where(test["model_score"] < t_low, "confident negative (no referral)",
                     np.where(test["model_score"] >= t_high, "confident positive (refer)",
                              "uncertain (human grader)"))
    triage_rows = []
    for band, g in test.groupby("triage"):
        triage_rows.append({"band": band, "n": len(g), "fraction_of_images": len(g) / len(test),
                            "n_truly_referable": int(g["y_true"].sum()),
                            "fraction_truly_referable": float(g["y_true"].mean())})
    triage_df = pd.DataFrame(triage_rows)

    # ---- 6. error lists for error analysis ----
    test["error_type"] = np.where((test.y_true == 1) & (test.pred_referable == 0), "FN",
                         np.where((test.y_true == 0) & (test.pred_referable == 1), "FP", ""))
    errors = test[test["error_type"] != ""].sort_values(["error_type", "model_score"])
    cols = [c for c in ["image_id", "true_grade", "model_score", "error_type", "image_quality",
                        "truncated", "dup_group", "label_conflict"] if c in errors.columns]
    errors[cols].to_csv(os.path.join(args.out, "errors_FP_FN.csv"), index=False)
    # error rates by image quality / truncation / label conflict (among truly referable: FN rate; among non-referable: FP rate)
    for col in ["image_quality", "truncated", "label_conflict"]:
        if col in test.columns:
            t = test.assign(_g=test[col].fillna("unknown").astype(str))
            tab = t.groupby("_g").apply(lambda g: pd.Series({
                "n": len(g),
                "n_referable": int(g.y_true.sum()),
                "FN": int(((g.y_true == 1) & (g.pred_referable == 0)).sum()),
                "FN_rate_among_referable": float(((g.y_true == 1) & (g.pred_referable == 0)).sum() / max(g.y_true.sum(), 1)),
                "FP": int(((g.y_true == 0) & (g.pred_referable == 1)).sum()),
                "FP_rate_among_nonreferable": float(((g.y_true == 0) & (g.pred_referable == 1)).sum() / max((g.y_true == 0).sum(), 1)),
            }), include_groups=False).rename_axis(col).reset_index()
            tab.to_csv(os.path.join(args.out, f"errors_by_{col}.csv"), index=False)
            print(f"\nErrors by {col}:\n" + tab.round(3).to_string(index=False))

    # ---- save tables and plots ----
    summary = pd.DataFrame([
        {"setting": f"Chosen threshold (val sens >= {args.target_sens:.0%})", **m_test_main},
        {"setting": "Default 0.5 (for comparison only)", **m_test_05},
    ])
    summary.to_csv(os.path.join(args.out, "test_metrics.csv"), index=False)
    grade_df.to_csv(os.path.join(args.out, "referral_by_grade.csv"), index=False)
    cal_df.to_csv(os.path.join(args.out, "calibration_table.csv"), index=False)
    triage_df.to_csv(os.path.join(args.out, "triage.csv"), index=False)

    plot_roc(y_val, s_val, y_test, s_test, m_test_main, args.out, args.model_name)
    plot_confusion(m_test_main, args.out, f"Test set, threshold {t_main:.2f}")
    plot_calibration(cal_df, ece, args.out)
    plot_sens_by_grade(grade_df, args.out)

    report = {
        "model": args.model_name,
        "n_val": int(len(val)), "n_test": int(len(test)),
        "prevalence_referable_test": float(y_test.mean()),
        "thresholds_from_validation": {"main": t_main, "triage_low": t_low, "triage_high": t_high},
        "test_AUROC": auroc_test,
        "test_AUROC_grade1_vs_2to4": auroc_g1_test,
        "test_specificity_at_90pct_sensitivity": spec_at_90,
        "test_metrics_at_main_threshold": m_test_main,
        "test_95CI_bootstrap": ci,
        "calibration": {"ECE": ece, "Brier": brier},
        "test_without_label_conflicts": clean,
    }
    with open(os.path.join(args.out, "summary.json"), "w") as f:
        json.dump(report, f, indent=2)

    # ---- readable printout ----
    print(f"\n=== {args.model_name}: referable DR (grade >= 2) ===")
    print(f"Validation n={len(val)}, Test n={len(test)}, test prevalence={y_test.mean():.1%}")
    print(f"Thresholds chosen on VALIDATION: main={t_main:.3f}, triage low={t_low:.3f}, high={t_high:.3f}")
    print(f"\nTEST AUROC = {auroc_test:.3f} (95% CI {ci['AUROC'][0]:.3f}-{ci['AUROC'][1]:.3f})")
    for k in ["sensitivity", "specificity", "PPV_precision", "NPV", "F1"]:
        print(f"  {k:14s} = {m_test_main[k]:.3f} (95% CI {ci[k][0]:.3f}-{ci[k][1]:.3f})")
    print(f"  Confusion: TP={m_test_main['TP']} FP={m_test_main['FP']} TN={m_test_main['TN']} FN={m_test_main['FN']}")
    print(f"  Specificity at 90% sensitivity = {spec_at_90:.3f}")
    print(f"  Grade 1 vs 2-4 AUROC (no grade-0 shortcut) = {auroc_g1_test:.3f}")
    print(f"\nAt default 0.5: sens={m_test_05['sensitivity']:.3f}, spec={m_test_05['specificity']:.3f}")
    if clean:
        print(f"\nWithout label-conflict images (n={clean['n']}, {clean['n_excluded']} removed, same threshold): "
              f"AUROC={clean['AUROC']:.3f}, sens={clean['sensitivity']:.3f}, spec={clean['specificity']:.3f}")
    print(f"\nCalibration: ECE={ece:.3f}, Brier={brier:.3f}")
    print("\nReferral rate by original grade:\n" + grade_df.to_string(index=False))
    print("\nTriage:\n" + triage_df.to_string(index=False))
    print(f"\nErrors saved: {len(errors)} images -> errors_FP_FN.csv")
    print(f"All outputs in: {os.path.abspath(args.out)}")

    return {"model": model_name,
            "val_AUROC": float(roc_auc_score(y_val, s_val)),
            "test_AUROC": auroc_test,
            "test_AUROC_95CI": f"{ci['AUROC'][0]:.3f}-{ci['AUROC'][1]:.3f}",
            "val_AUROC_grade1_vs_2to4": auroc_g1_val,
            "test_AUROC_grade1_vs_2to4": auroc_g1_test,
            "threshold_from_val": t_main,
            "test_sensitivity": m_test_main["sensitivity"],
            "test_specificity": m_test_main["specificity"],
            "test_PPV": m_test_main["PPV_precision"],
            "test_NPV": m_test_main["NPV"],
            "test_FN_missed": m_test_main["FN"],
            "test_FP": m_test_main["FP"],
            "ECE": ece,
            **({"test_AUROC_no_conflicts": clean["AUROC"], "test_sens_no_conflicts": clean["sensitivity"],
                "test_spec_no_conflicts": clean["specificity"]} if clean else {})}


def main():
    p = argparse.ArgumentParser(description="BIOM9027 evaluation script (referable DR)")
    p.add_argument("csv", help="Results CSV: image_id/id_code, split, true_grade/diagnosis, model_score, [model]")
    p.add_argument("--out", default="eval_output")
    p.add_argument("--splits", default=None, help="Optional splits.csv to add dup_group / label_conflict to the error list")
    p.add_argument("--quality", default=None, help="Optional image_quality.csv from image_quality.py (adds image_quality / truncated)")
    p.add_argument("--val-only", action="store_true",
                   help="Report validation results only; the test rows are never read (use until all models are frozen)")
    p.add_argument("--target-sens", type=float, default=0.90, help="Sensitivity target for the main threshold")
    p.add_argument("--triage-low-sens", type=float, default=0.98, help="Sensitivity for the rule-out threshold")
    p.add_argument("--triage-high-spec", type=float, default=0.95, help="Specificity for the rule-in threshold")
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--model-name", default="model", help="Used only if the CSV has no 'model' column")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    # accept the repo's column names (splits.csv uses id_code and diagnosis)
    df = df.rename(columns={"id_code": "image_id", "diagnosis": "true_grade", "model_type": "model"})
    required = {"image_id", "split", "true_grade", "model_score"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"CSV is missing columns: {missing}")
    df["split"] = df["split"].astype(str).str.lower().str.strip()
    keep_splits = ["val", "validation"] if args.val_only else ["val", "validation", "test"]
    df = df[df["split"].isin(keep_splits)].copy()   # train rows (and test rows in --val-only mode) are dropped here
    df["y_true"] = (df["true_grade"] >= REFERABLE_MIN_GRADE).astype(int)

    if args.splits:
        sp = pd.read_csv(args.splits).rename(columns={"id_code": "image_id"})
        extra = [c for c in ["dup_group", "label_conflict"] if c in sp.columns]
        df = df.merge(sp[["image_id"] + extra], on="image_id", how="left")

    if args.quality:
        qq = pd.read_csv(args.quality).rename(columns={"id_code": "image_id"})
        keep = [c for c in ["image_quality", "truncated", "brightness", "sharpness"] if c in qq.columns]
        df = df.drop(columns=[c for c in keep if c in df.columns]).merge(qq[["image_id"] + keep], on="image_id", how="left")

    if "model" not in df.columns:
        df["model"] = args.model_name

    rows = []
    for model_name, g in df.groupby("model", sort=False):
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(model_name))
        if args.val_only:
            rows.append(evaluate_val_only(g.copy(), os.path.join(args.out, safe), args, str(model_name)))
        else:
            rows.append(evaluate_one(g.copy(), os.path.join(args.out, safe), args, str(model_name)))

    if args.val_only:
        comp = pd.DataFrame(rows).sort_values("val_AUROC", ascending=False)
        os.makedirs(args.out, exist_ok=True)
        comp.to_csv(os.path.join(args.out, "val_model_comparison.csv"), index=False)
        pd.set_option("display.width", 200)
        print("\n\n=========== VALIDATION-ONLY COMPARISON (test set not opened) ===========")
        print(comp.round(3).to_string(index=False))
        return

    comp = pd.DataFrame(rows).sort_values("val_AUROC", ascending=False)
    comp["best_by_val_AUROC"] = ""
    comp.iloc[0, comp.columns.get_loc("best_by_val_AUROC")] = "<- winner (chosen on VALIDATION)"
    os.makedirs(args.out, exist_ok=True)
    comp.to_csv(os.path.join(args.out, "model_comparison.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\n\n=========== MODEL COMPARISON (winner picked on validation AUROC, not test) ===========")
    print(comp.round(3).to_string(index=False))
    print(f"\nSaved: {os.path.join(os.path.abspath(args.out), 'model_comparison.csv')}")



if __name__ == "__main__":
    main()
