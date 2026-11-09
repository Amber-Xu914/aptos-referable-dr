"""Collect all Part B runs into one comparison table, ROC/learning-curve figures and runtime notes.

    python part_b/compare.py
    python part_b/compare.py --extra results/part_a/preds_A_rf_all.csv    # add Part A to the ROC plot

Outputs (results/part_b/):
    comparison_val.csv        one row per run: parameters, val AUROC (95% CI), metrics at the val threshold, runtime
    roc_val.png               validation ROC curves of every run (and --extra prediction files)
    learning_curves.png       val AUROC and train loss per epoch
    runtime_hardware.md       runtime and hardware notes for the report, generated from the run summaries
Selection of the "winning" model for Grad-CAM and the test set uses validation AUROC only.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.common import add_path_args, get_paths, load_json


def main(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    paths = get_paths(a.root, a.cache)
    summaries = [load_json(f) for f in sorted(paths.results.glob("summary_*.json"))]
    debug = [s["run"] for s in summaries if s.get("args", {}).get("limit")]
    if debug and not a.include_quick:              # --quick / --limit runs are setup checks, not results
        print(f"ignoring quick/debug runs (trained with --limit): {', '.join(debug)}")
        summaries = [s for s in summaries if s["run"] not in debug]
    if not summaries:
        sys.exit("no summary_*.json found; train something first")

    rows = []
    for s in summaries:
        v, m, rt = s["val"], s["val"]["at_sens90"], s.get("runtime", {})
        rows.append(dict(
            run=s["run"], model=s.get("model", "hybrid"), mode=s.get("mode", s.get("best", {}).get("clf", "")),
            img_size=s.get("img_size"), params_M=round(s.get("params_total", 0) / 1e6, 2) or None,
            trainable_M=round(s.get("params_trainable", 0) / 1e6, 2) or None,
            val_auroc=round(v["auroc"], 4), ci_lo=round(v["auroc_ci"][0], 4), ci_hi=round(v["auroc_ci"][1], 4),
            threshold=round(m["threshold"], 4), sens=round(m["sensitivity"], 3), spec=round(m["specificity"], 3),
            precision=round(m["precision"], 3), f1=round(m["f1"], 3),
            best_epoch=s.get("best_epoch"), train_min=round(rt["train_s"] / 60, 1) if "train_s" in rt else None,
            epoch_s=rt.get("mean_epoch_s"), infer_ms=rt.get("inference_ms_per_image"),
            peak_gpu_gb=rt.get("peak_gpu_mem_gb"), gpu=s.get("hardware", {}).get("gpu")))
    table = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
    table.to_csv(paths.results / "comparison_val.csv", index=False)
    print(table.to_string(index=False))
    seeds = table.assign(model_run=table.run.str.replace(r"_s\d+$", "", regex=True))
    if (seeds.model_run != seeds.run).any():       # several seeds: mean +- SD per model
        agg = (seeds.groupby("model_run")[["val_auroc", "sens", "spec", "f1"]].agg(["mean", "std", "count"])
               .round(4).sort_values(("val_auroc", "mean"), ascending=False))
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.to_csv(paths.results / "comparison_val_seeds.csv")
        print("\nacross seeds (mean, SD, n):\n" + agg.to_string())
    winner = table.iloc[0]
    print(f"\nbest on validation: {winner.run} (AUROC {winner.val_auroc:.3f})")

    # ROC curves (validation only)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    files = [(paths.results / f"preds_{s['model_type']}.csv", s["run"]) for s in summaries]
    files += [(Path(f), Path(f).stem.replace("preds_", "")) for f in a.extra]
    for f, label in files:
        p = pd.read_csv(f).query("split == 'val'")
        y = (p.true_grade >= 2).astype(int)
        fpr, tpr, _ = roc_curve(y, p.model_score)
        auc = table.set_index("run").val_auroc.get(label)
        if auc is None:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y, p.model_score)
        ax.plot(fpr, tpr, lw=1.6, label=f"{label} ({auc:.3f})")
    ax.plot([0, 1], [0, 1], ls=":", c="grey", lw=1)
    ax.axhline(0.9, ls="--", c="grey", lw=0.8)
    ax.set(xlabel="1 - specificity", ylabel="sensitivity", title="Validation ROC, referable DR (grade >= 2)",
           xlim=(0, 1), ylim=(0, 1.01))
    ax.legend(fontsize=7, loc="lower right", title="run (AUROC)", title_fontsize=7)
    fig.tight_layout(); fig.savefig(paths.results / "roc_val.png", dpi=150); plt.close(fig)

    # Learning curves
    hist = {s["run"]: paths.results / f"history_{s['run']}.csv" for s in summaries}
    hist = {k: pd.read_csv(v) for k, v in hist.items() if v.exists()}
    if hist:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        for k, h in hist.items():
            ax1.plot(h.epoch, h.val_auroc, marker=".", label=k)
            ax2.plot(h.epoch, h.train_loss, marker=".", label=k)
        ax1.set(xlabel="epoch", ylabel="val AUROC", title="Validation AUROC per epoch")
        ax2.set(xlabel="epoch", ylabel="train loss (weighted BCE)", title="Training loss per epoch")
        ax1.legend(fontsize=7)
        fig.tight_layout(); fig.savefig(paths.results / "learning_curves.png", dpi=150); plt.close(fig)

    # Runtime and hardware notes
    hw = next((s["hardware"] for s in summaries if s.get("hardware")), {})
    lines = ["# Part B runtime and hardware notes (generated by part_b/compare.py)", "",
             f"- Hardware: GPU {hw.get('gpu')} ({hw.get('gpu_mem_gb', '?')} GB), {hw.get('cpu_count')} CPU cores, "
             f"{hw.get('ram_gb', '?')} GB RAM, Colab: {hw.get('colab')}",
             f"- Software: Python {hw.get('python')}, PyTorch {hw.get('torch')}, torchvision {hw.get('torchvision')}, "
             f"CUDA {hw.get('cuda_version')}, scikit-learn {hw.get('sklearn')}",
             "- Mixed precision (fp16 autocast) on GPU; images preloaded into RAM once per run; "
             "inference time includes flip TTA where enabled and is measured over the validation set (batch 32).", "",
             "| run | input px | params (M) | trainable (M) | epochs to best | mean epoch (s) | total train (min) | inference (ms/img) | peak GPU mem (GB) |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in table.itertuples():
        lines.append(f"| {r.run} | {r.img_size} | {r.params_M} | {r.trainable_M} | {r.best_epoch} | {r.epoch_s} | "
                     f"{r.train_min} | {r.infer_ms} | {r.peak_gpu_gb} |")
    lines += ["", "Limitations to report: free Colab GPUs vary between sessions (T4/P100/L4), so absolute times are "
              "indicative only; sessions time out after a few hours, which capped the number of epochs and "
              "ruled out repeated runs with different seeds or k-fold CV; images were downsized to "
              f"{table.img_size.dropna().iloc[0] if table.img_size.notna().any() else '?'} px, "
              "which can hide the smallest microaneurysms."]
    (paths.results / "runtime_hardware.md").write_text("\n".join(lines) + "\n")
    print(f"wrote comparison_val.csv, roc_val.png, learning_curves.png, runtime_hardware.md to {paths.results}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(p)
    p.add_argument("--extra", nargs="*", default=[], help="other prediction CSVs (e.g. Part A) for the ROC plot")
    p.add_argument("--include-quick", action="store_true", help="also list runs trained with --limit (run_all --quick)")
    main(p.parse_args())
