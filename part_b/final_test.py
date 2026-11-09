"""Score the TEST split once with the frozen models, at the thresholds chosen on validation.

    python part_b/final_test.py --runs smallcnn efficientnet_b0_frozen efficientnet_b0_finetune

Run this only after all choices (architecture, mode, epochs, threshold rule) were made on the
validation split. It appends test rows to results/part_b/preds_B_<run>.csv (same format as
Part A) and writes results/part_b/test_metrics.csv. Nothing is tuned here.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.common import (add_path_args, evaluate, get_paths, load_json, load_splits, make_dataset, preload,
                           save_json, write_preds)
from part_b.hardware import default_workers, device


def main(a):
    import torch
    from torch.utils.data import DataLoader
    from part_b.models import build_model
    from part_b.train import predict

    paths = get_paths(a.root, a.cache)
    dev = device()
    te = load_splits(paths.splits).query("split == 'test'").reset_index(drop=True)
    rows = []
    for run in a.runs:
        summ = load_json(paths.results / f"summary_{run}.json")
        ck = torch.load(paths.ckpt / f"{run}.pt", map_location=dev)
        model = build_model(ck["model"], pretrained=False).to(dev)
        model.load_state_dict(ck["state"])
        dl = DataLoader(make_dataset(preload(te, paths.cache, ck["size"]), te.referable, train=False),
                        batch_size=32, num_workers=a.workers)
        s = predict(model, dl, dev, tta=summ["runtime"]["tta"])
        write_preds(te, s, summ["model_type"], paths.results / f"preds_{summ['model_type']}.csv", append_test=True)
        res = evaluate(te.referable, s, summ["thresholds"])     # thresholds fixed on validation
        summ["test"] = res
        save_json(summ, paths.results / f"summary_{run}.json")
        for rule in summ["thresholds"]:
            rows.append(dict(run=run, threshold_rule=rule, auroc=res["auroc"], auroc_lo=res["auroc_ci"][0],
                             auroc_hi=res["auroc_ci"][1], **res[f"at_{rule}"]))
        print(f"{run}: test AUROC {res['auroc']:.3f} ({res['auroc_ci'][0]:.3f}-{res['auroc_ci'][1]:.3f})")
    out = pd.DataFrame(rows).round(4)
    out.to_csv(paths.results / "test_metrics.csv", index=False)
    print(out.to_string(index=False))
    base = out.run.str.replace(r"_s\d+$", "", regex=True)
    if (base != out.run).any():                    # several seeds: mean +- SD per model
        cols = ["auroc", "sensitivity", "specificity", "precision", "f1", "accuracy"]
        agg = out.assign(model_run=base).groupby(["model_run", "threshold_rule"])[cols].agg(["mean", "std", "count"]).round(4)
        agg.columns = ["_".join(c) for c in agg.columns]
        agg.to_csv(paths.results / "test_metrics_seeds.csv")
        print("\nacross seeds (mean, SD, n):\n" + agg.to_string())


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(p)
    p.add_argument("--runs", nargs="+", required=True)
    p.add_argument("--workers", type=int, default=default_workers())
    main(p.parse_args())
