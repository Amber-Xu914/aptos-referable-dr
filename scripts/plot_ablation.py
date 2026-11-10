"""Plot the Part C ablation: AUROC per experiment with 95% bootstrap intervals.

Two panels: all validation images, and grade 1 vs grades 2-4 (which removes the
camera shortcut). Part A's best classical model is drawn as a reference line.

With several seeds, the point is the mean AUROC over seeds; each bootstrap
resample is scored on every seed and averaged, so the interval reflects
image-sampling uncertainty of that mean.

With --with-preprocessing, a strip of one example image per preprocessing mode
is drawn above the panels (slide 6 version).

Usage:
    python scripts/plot_ablation.py
    python scripts/plot_ablation.py --with-preprocessing --out results/part_c/figures/slide6_preprocessing_ablation.png
    python scripts/plot_ablation.py --out results/part_c/figures/ablation_auroc.png
"""

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.ablation import EXPERIMENTS  # noqa: E402

PREDS = ROOT / "results/part_c/preds"
REFERENCE = ROOT / "results/part_a/preds_A_rf_all.csv"
LABELS = {
    "E0_baseline": "E0  Baseline (geometry only)",
    "E1_clahe": "E1  + CLAHE",
    "E2_augment": "E2  + augmentation",
    "E3_classweight": "E3  + class weight (≈ B's model)",
    "E3b_oversample": "E3b + oversampling (alt. to E3)",
    "E4_bengraham": "E4  Ben Graham (alt. to CLAHE)",
}
RAW = ROOT / "data/raw/Aptos2019 retinal image database/train_images"
EXAMPLE = "6c3745a222da"  # grade 4, training split; also used in results/figures/preprocessing_comparison.png
STRIP = [
    ("Original photo", None),
    ("E0  Geometry only\ncrop, pad, mask", ROOT / "data/cache_rgb_512"),
    ("E1–E3  + CLAHE\ncontrast on lightness", ROOT / "data/cache_clahe_512"),
    ("E4  Ben Graham\nlocal mean subtracted", ROOT / "data/cache_bengraham_512"),
]
BLUE, GREY, INK, INK2, SURFACE = "#2a78d6", "#8a8984", "#0b0b0b", "#52514e", "#ffffff"


def val_scores(path):
    df = pd.read_csv(path, dtype={"image_id": str})
    df = df[df["split"] == "val"].sort_values("image_id").reset_index(drop=True)
    return df["true_grade"].to_numpy(), df["model_score"].to_numpy()


def auroc(grades, scores, hard):
    keep = grades >= 1 if hard else np.ones(len(grades), bool)
    return roc_auc_score(grades[keep] >= 2, scores[keep])


def summarise(grades, score_sets, hard, n_boot, rng):
    """Mean AUROC over seeds and a 95% bootstrap interval of that mean."""
    point = np.mean([auroc(grades, s, hard) for s in score_sets])
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, len(grades), len(grades))
        g = grades[i]
        if (g >= 2).all() or (g[g >= 1 if hard else slice(None)] < 2).all():
            continue
        boots.append(np.mean([auroc(g, s[i], hard) for s in score_sets]))
    return point, *np.percentile(boots, [2.5, 97.5])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(ROOT / "results/part_c/figures/ablation_auroc.png"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--with-preprocessing", action="store_true", help="add a strip of example images on top")
    a = p.parse_args()

    rows, n_seeds = [], set()
    for exp in EXPERIMENTS:
        files = sorted(PREDS.glob(f"preds_C_{exp.name}_s*.csv"))
        files = [f for f in files if f.stem.rsplit("_s", 1)[1].isdigit()]
        if not files:
            continue
        loaded = [val_scores(f) for f in files]
        grades = loaded[0][0]
        scores = [s for _, s in loaded]
        n_seeds.add(len(files))
        for hard in (False, True):
            rng = np.random.default_rng(9027)
            m, lo, hi = summarise(grades, scores, hard, a.n_boot, rng)
            rows.append({"exp": exp.name, "hard": hard, "auroc": m, "lo": lo, "hi": hi})
    res = pd.DataFrame(rows)
    ref_g, ref_s = val_scores(REFERENCE)
    ref = {False: auroc(ref_g, ref_s, False), True: auroc(ref_g, ref_s, True)}

    exps = list(dict.fromkeys(res["exp"]))
    y = np.arange(len(exps))[::-1]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})
    if a.with_preprocessing:
        fig = plt.figure(figsize=(13, 8.4), facecolor=SURFACE)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.45], hspace=0.16)
        top = gs[0].subgridspec(1, len(STRIP), wspace=0.06)
        for i, (label, folder) in enumerate(STRIP):
            ax = fig.add_subplot(top[i])
            if folder is None:
                img = cv2.cvtColor(cv2.imread(str(RAW / f"{EXAMPLE}.png")), cv2.COLOR_BGR2RGB)
            else:
                img = cv2.cvtColor(cv2.imread(str(folder / f"{EXAMPLE}.png")), cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            ax.set_title(label, fontsize=11.5, color=INK, linespacing=1.3)
            ax.axis("off")
        fig.text(0.01, 0.72, "Preprocessing\n(one grade 4\ntraining image)", fontsize=12, color=INK2,
                 va="center", linespacing=1.4)
        bottom = gs[1].subgridspec(1, 2, wspace=0.14)
        axes = [fig.add_subplot(bottom[0])]
        axes.append(fig.add_subplot(bottom[1], sharey=axes[0]))
        plt.setp(axes[1].get_yticklabels(), visible=False)
    else:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True, facecolor=SURFACE)
    panels = [(False, "All validation images", (0.94, 1.0)),
              (True, "Grade 1 vs grades 2–4 (no camera shortcut)", (0.80, 0.96))]
    for ax, (hard, title, xlim) in zip(axes, panels):
        d = res[res["hard"] == hard].set_index("exp").loc[exps]
        ax.set_facecolor(SURFACE)
        ax.axvline(ref[hard], color=GREY, lw=2, ls=(0, (4, 3)), zorder=1)
        ax.text(ref[hard], -0.55, f" Part A classical RF {ref[hard]:.3f} ",
                color=INK2, fontsize=10, va="bottom", ha="right",
                bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 1})
        ax.hlines(y, d["lo"], d["hi"], color=BLUE, lw=2, zorder=2)
        ax.scatter(d["auroc"], y, s=90, color=BLUE, edgecolor=SURFACE, linewidth=2, zorder=3)
        for yi, v in zip(y, d["auroc"]):
            ax.text(v, yi + 0.28, f"{v:.3f}", ha="center", va="bottom", fontsize=10, color=INK)
        ax.set_xlim(*xlim)
        ax.set_ylim(-0.75, len(exps) - 0.4)
        ax.set_title(title, loc="left", fontsize=13, color=INK, pad=10)
        ax.set_xlabel("Validation AUROC (95% bootstrap CI)", color=INK2)
        ax.grid(axis="x", color="#e6e5e0", lw=1)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color("#c9c8c2")
        ax.tick_params(colors=INK2, length=0)
    axes[0].set_yticks(y, [LABELS[e] for e in exps], color=INK)

    seeds = "/".join(str(n) for n in sorted(n_seeds))
    fig.suptitle("None of the data changes improved the CNN over the plain baseline",
                 x=0.01, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.text(0.01, -0.02,
             f"EfficientNet-B0 (ImageNet, fine-tuned), 384 px, 25 epochs; mean of {seeds} seeds. Validation split only "
             "(n = 549; 57 grade-1 images). Each row changes one thing from the row it builds on. Test set not used.",
             fontsize=10, color=INK2)
    if not a.with_preprocessing:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.24, right=0.99, top=0.9, bottom=0.08)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.out, dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(res.round(3).to_string(index=False))
    print(f"Wrote {a.out}")


if __name__ == "__main__":
    main()
