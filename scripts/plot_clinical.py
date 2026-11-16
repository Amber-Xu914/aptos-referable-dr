"""Figures for the medical-value slide (Part C).

    workflow.png          screening workflow in primary care
    referral_burden.png   unnecessary referrals per 1,000 people screened, by model

Referral burden is an illustration, not an economic evaluation:
1,000 people screened, assumed 10% referable prevalence, threshold set at 85%
sensitivity on validation, each unnecessary referral costed at MBS item 10915
($79.80). With --split test, the validation threshold is applied unchanged to
the test predictions (the group's rule; nothing is chosen on test). For models
with several seeds, sensitivity and specificity are averaged over seeds.

Usage:
    python scripts/plot_clinical.py               # validation numbers
    python scripts/plot_clinical.py --split test  # final test numbers (frozen models only)
"""

import argparse
import io
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.metrics import TARGET_SENS, load_preds, threshold_for_sensitivity  # noqa: E402

OUT = ROOT / "results/part_c/figures"
BLUE, BLUE_LIGHT, GREY, INK, INK2, SURFACE = "#2a78d6", "#e3eefb", "#8a8984", "#0b0b0b", "#52514e", "#ffffff"
BOX, BOX_EDGE = "#f4f3f0", "#c9c8c2"

N, PREVALENCE, SENS, FP_COST = 1000, 0.10, 0.85, 79.80


def box(ax, x, y, w, h, text, fill=BOX, edge=BOX_EDGE, bold=False, size=12, color=INK):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor=fill, edgecolor=edge, linewidth=1.5))
    ax.text(x, y, text, ha="center", va="center", fontsize=size, color=color,
            fontweight="bold" if bold else "normal", linespacing=1.35)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=16, color=GREY, linewidth=1.8,
                                 shrinkA=0, shrinkB=0))


def workflow(slide=False):
    """slide=True: no title and no weak-link note (the slide has its own), for slide 9a."""
    fig, ax = plt.subplots(figsize=(13, 4.2 if slide else 5.2), facecolor=SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(1.0 if slide else 0, 5.2)
    ax.axis("off")

    y = 3.4
    box(ax, 1.35, y, 2.3, 1.3, "Patient with diabetes\nat routine GP / AMS\nvisit")
    box(ax, 4.15, y, 2.5, 1.3, "Nurse or Aboriginal\nhealth worker takes\n1 photo per eye")
    box(ax, 6.95, y, 2.3, 1.3, "AI score\non site, in seconds", fill=BLUE_LIGHT, edge=BLUE, bold=True)
    box(ax, 9.55, y, 2.1, 1.3, "GP reviews result,\nexplains to patient")
    for x0, x1 in ((2.5, 2.9), (5.4, 5.8), (8.1, 8.5)):
        arrow(ax, (x0, y), (x1, y))

    outcomes = [
        (4.65, "Non-referable", "Rescreen in 2 years\n(1 year: Aboriginal and Torres\nStrait Islander / higher risk)"),
        (3.4, "Referable", "Refer to optometrist /\nophthalmologist,\ntrack attendance"),
        (2.15, "Ungradable", "Refer for full\neye examination"),
    ]
    for oy, title, body in outcomes:
        arrow(ax, (10.6, y), (11.05, oy))
        ax.text(11.15, oy + 0.22, title, ha="left", va="bottom", fontsize=12, fontweight="bold", color=INK)
        ax.text(11.15, oy + 0.18, body, ha="left", va="top", fontsize=10, color=INK2, linespacing=1.3)

    if slide:
        ax.text(0.2, 1.35, "Photos taken by trained staff under GP direction can be billed to MBS items 12325 / 12326.",
                fontsize=12, color=INK2, va="center")
        fig.savefig(OUT / "workflow_slide.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
        return
    ax.text(0.2, 1.3, "The weak link is after the AI", fontsize=12, fontweight="bold", color=INK, va="center")
    ax.text(0.2, 0.85, "In an Australian real-world AI screening study, only 9 of 28 referred patients "
            "(about 1 in 3) were confirmed to attend (Scheetz et al. 2021).",
            fontsize=12, color=INK2, va="center")
    ax.text(0.2, 0.35, "Photos taken by trained staff under GP direction can be billed to MBS items 12325 / 12326.",
            fontsize=10, color=INK2, va="center")
    fig.suptitle("Where the AI fits: a refer / don't-refer call during a visit the patient already makes",
                 x=0.015, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.savefig(OUT / "workflow.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)


B_MAIN = [ROOT / f"results/part_b/preds_B_efficientnet_b0_finetune_s{s}.csv" for s in (1, 2, 3)]


def read_text(path):
    """A prediction CSV's text, or (before Part B is merged) its contents from the Part_B_CNN branch."""
    if path.exists():
        return path.read_text()
    rel = path.relative_to(ROOT).as_posix()
    return subprocess.run(["git", "show", f"origin/Part_B_CNN:{rel}"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout


def operating_point(paths, split):
    """Mean sensitivity and specificity on `split` at each file's validation threshold for 85% sensitivity."""
    sens, spec = [], []
    for path in paths:
        text = read_text(path)
        val = load_preds(io.StringIO(text), "val")
        t = threshold_for_sensitivity(val["referable"].to_numpy(), val["model_score"].to_numpy(), TARGET_SENS)
        d = load_preds(io.StringIO(text), split)
        y, pos = d["referable"].to_numpy() == 1, d["model_score"].to_numpy() >= t
        sens.append((pos & y).sum() / y.sum())
        spec.append((~pos & ~y).sum() / (~y).sum())
    return sum(sens) / len(sens), sum(spec) / len(spec)


def referral_burden(split):
    models = [
        ("Simple colour baseline (Part A)", [ROOT / "results/part_a/preds_baseline_colour_logreg.csv"]),
        ("Classical random forest (Part A)", [ROOT / "results/part_a/preds_A_rf_all.csv"]),
        ("CNN, EfficientNet-B0 (Part B, 3 seeds)", B_MAIN),
    ]
    rows = []
    for name, paths in models:
        sens, spec = operating_point(paths, split)
        fp = N * (1 - PREVALENCE) * (1 - spec)
        tp = N * PREVALENCE * sens
        rows.append({"model": name, "sens": sens, "spec": spec, "fp": fp, "ppv": tp / (tp + fp), "cost": fp * FP_COST})
    df = pd.DataFrame(rows)

    def extra(r):
        return f"sens {r.sens:.2f}, " if split == "test" else ""

    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURFACE)
    y = range(len(df))[::-1]
    ax.barh(list(y), df["fp"], height=0.55, color=BLUE)
    for yi, r in zip(y, df.itertuples()):
        ax.text(r.fp + 4, yi, f"{r.fp:.0f} unnecessary referrals   ≈ ${round(r.cost, -2):,.0f}   ({extra(r)}spec {r.spec:.3f}, PPV {r.ppv:.0%})",
                va="center", fontsize=11, color=INK)
    ax.set_yticks(list(y), df["model"], fontsize=12, color=INK)
    ax.set_xlim(0, 480)
    ax.set_xlabel("Unnecessary referrals per 1,000 people screened " + ("(threshold from validation)" if split == "test" else "(at 85% sensitivity)"), color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BOX_EDGE)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color="#e6e5e0")
    ax.set_axisbelow(True)
    ratio = df["fp"].iloc[0] / df["fp"].iloc[-1]
    fig.suptitle(f"Specificity drives referral burden: about {ratio:.0f}× fewer unnecessary referrals than the simple baseline",
                 x=0.015, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.015, -0.06,
             "Illustration: 1,000 people screened, assumed 10% referable prevalence. Threshold set for 85% sensitivity on "
             f"validation{' and applied unchanged to test' if split == 'test' else ''}.\nCost = MBS item 10915 ($79.80) "
             "per unnecessary optometry visit only; excludes patient travel and time. "
             f"{'TEST-set results' if split == 'test' else 'Validation results'}; CNN = mean of 3 seeds.",
             fontsize=9.5, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(OUT / f"referral_burden{'_test' if split == 'test' else ''}.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["val", "test"], default="val")
    a = p.parse_args()
    workflow()
    workflow(slide=True)
    referral_burden(a.split)
    print(f"Wrote figures to {OUT}")
