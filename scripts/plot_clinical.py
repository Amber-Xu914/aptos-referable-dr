"""Figures for the medical-value slide (Part C).

    workflow.png          screening workflow in primary care
    referral_burden.png   unnecessary referrals per 1,000 people screened, by model

Referral burden is an illustration, not an economic evaluation:
1,000 people screened, assumed 10% referable prevalence, threshold set at 85%
sensitivity on validation, each unnecessary referral costed at MBS item 10915
($79.80). Specificities come from the validation predictions.

Usage:
    python scripts/plot_clinical.py
"""

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
from src.metrics import evaluate, load_preds  # noqa: E402

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


def workflow():
    fig, ax = plt.subplots(figsize=(13, 5.2), facecolor=SURFACE)
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5.2)
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

    ax.text(0.2, 1.3, "The weak link is after the AI", fontsize=12, fontweight="bold", color=INK, va="center")
    ax.text(0.2, 0.85, "In an Australian real-world AI screening study, only 9 of 28 referred patients "
            "(about 1 in 3) were confirmed to attend (Scheetz et al. 2021).",
            fontsize=12, color=INK2, va="center")
    ax.text(0.2, 0.35, "Photos taken by trained staff under GP direction can be billed to MBS items 12325 / 12326.",
            fontsize=10, color=INK2, va="center")
    fig.suptitle("Where the AI fits: a refer / don't-refer call during a visit the patient already makes",
                 x=0.015, ha="left", fontsize=15, fontweight="bold", color=INK)
    fig.savefig(OUT / "workflow.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)


B_MAIN = ROOT / "results/part_b/preds_B_efficientnet_b0_finetune.csv"


def read_preds(path):
    """Path to a prediction CSV, or (before Part B is merged) its contents from the Part_B_CNN branch."""
    if path.exists():
        return path
    rel = path.relative_to(ROOT).as_posix()
    text = subprocess.run(["git", "show", f"origin/Part_B_CNN:{rel}"], cwd=ROOT, check=True,
                          capture_output=True, text=True).stdout
    return io.StringIO(text)


def referral_burden():
    models = [
        ("Simple colour baseline (Part A)", ROOT / "results/part_a/preds_baseline_colour_logreg.csv"),
        ("Classical random forest (Part A)", ROOT / "results/part_a/preds_A_rf_all.csv"),
        ("CNN, EfficientNet-B0 (Part B)", B_MAIN),
    ]
    rows = []
    for name, path in models:
        spec = evaluate(load_preds(read_preds(path)), n_boot=0)["spec_at_sens85"]
        fp = N * (1 - PREVALENCE) * (1 - spec)
        tp = N * PREVALENCE * SENS
        rows.append({"model": name, "spec": spec, "fp": fp, "ppv": tp / (tp + fp), "cost": fp * FP_COST})
    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 3.6), facecolor=SURFACE)
    y = range(len(df))[::-1]
    ax.barh(list(y), df["fp"], height=0.55, color=BLUE)
    for yi, r in zip(y, df.itertuples()):
        ax.text(r.fp + 4, yi, f"{r.fp:.0f} unnecessary referrals   ≈ ${round(r.cost, -2):,.0f}   (spec {r.spec:.3f}, PPV {r.ppv:.0%})",
                va="center", fontsize=11, color=INK)
    ax.set_yticks(list(y), df["model"], fontsize=12, color=INK)
    ax.set_xlim(0, 480)
    ax.set_xlabel("Unnecessary referrals per 1,000 people screened (at 85% sensitivity)", color=INK2)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(BOX_EDGE)
    ax.tick_params(colors=INK2, length=0)
    ax.grid(axis="x", color="#e6e5e0")
    ax.set_axisbelow(True)
    fig.suptitle("Specificity drives referral burden: same sensitivity, about 7× fewer unnecessary referrals",
                 x=0.015, ha="left", fontsize=14, fontweight="bold", color=INK)
    fig.text(0.015, -0.06,
             "Illustration: 1,000 people screened, assumed 10% referable prevalence, threshold at 85% sensitivity "
             "(validation). Cost = MBS item 10915 ($79.80)\nper unnecessary optometry visit only; excludes patient "
             "travel and time. Specificities from validation predictions; not test-set results.",
             fontsize=9.5, color=INK2, va="top")
    fig.tight_layout()
    fig.savefig(OUT / "referral_burden.png", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans"})
    workflow()
    referral_burden()
    print(f"Wrote figures to {OUT}")
