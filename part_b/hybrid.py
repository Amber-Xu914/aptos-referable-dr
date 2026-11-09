"""Optional: hybrid model = CNN features -> PCA -> classical classifier.

    python part_b/hybrid.py --arch efficientnet_b0                       # ImageNet features, no fine-tuning
    python part_b/hybrid.py --run efficientnet_b0_finetune               # features of a fine-tuned run
    python part_b/hybrid.py --arch efficientnet_b0 --final-test          # once, after choices are fixed

1. One forward pass per image through the CNN backbone (global-average-pooled vector,
   1280-D for EfficientNet-B0, 2048-D for ResNet-50). Features are cached in outputs/part_b/.
2. StandardScaler -> PCA(k) -> logistic regression or RBF-SVM, fitted on TRAIN.
3. k, classifier and C are chosen on VALIDATION AUROC (grid below), like Part A.

This sits between Part A (hand-crafted features + classical classifier) and Part B
(end-to-end CNN): it asks whether learned features alone, with a classical classifier,
close the gap. Note: features from a fine-tuned run were learned on the training images,
so the classifier sees optimistic train features; the ImageNet-only variant has no such bias.
"""
import argparse
import itertools
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.common import (SEED, add_path_args, choose_threshold, evaluate, get_paths, load_json, load_splits,
                           make_dataset, preload, save_json, write_preds)
from part_b.hardware import amp, default_workers, device, hardware_info, timer

GRID = dict(k=[16, 32, 64, 128], clf=["logreg", "svm_rbf"], C=[0.01, 0.1, 1.0, 10.0])


def extract(model, df, paths, size, dev, workers):
    import torch
    from torch.utils.data import DataLoader
    dl = DataLoader(make_dataset(preload(df, paths.cache, size), df.referable, train=False),
                    batch_size=64, num_workers=workers)
    model.eval(); out = []
    with torch.no_grad():
        for x, _, _ in dl:
            with amp(dev):
                out.append(model.forward_features(x.to(dev)).float().cpu().numpy())
    return np.concatenate(out)


def make_clf(k, clf, C, n_max):
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    head = (LogisticRegression(C=C, max_iter=5000, class_weight="balanced") if clf == "logreg"
            else CalibratedClassifierCV(SVC(C=C, kernel="rbf", class_weight="balanced"), cv=3))
    return make_pipeline(StandardScaler(), PCA(n_components=min(k, n_max), random_state=SEED), head)


def main(a):
    import torch
    from sklearn.metrics import roc_auc_score
    from part_b.models import build_model

    paths = get_paths(a.root, a.cache)
    dev = device()
    if a.run:
        ck = torch.load(paths.ckpt / f"{a.run}.pt", map_location=dev)
        model = build_model(ck["model"], pretrained=False).to(dev); model.load_state_dict(ck["state"])
        src, size = a.run, ck["size"]
    else:
        model = build_model(a.arch, pretrained=not a.no_pretrained).to(dev)
        src, size = f"{a.arch}_imagenet", a.size
    name = f"hybrid_{src}"

    df = load_splits(paths.splits)
    splits = ["train", "val"] + (["test"] if a.final_test else [])
    feats, times = {}, {}
    for s in splits:
        f = paths.ckpt.parent / f"features_{src}_{size}_{s}.npy"
        sub = df[df.split == s].reset_index(drop=True)
        if f.exists():
            feats[s] = np.load(f)
        else:
            with timer(times, f"extract_{s}_s"):
                feats[s] = extract(model, sub, paths, size, dev, a.workers)
            np.save(f, feats[s])
        assert len(feats[s]) == len(sub)
    tr, va = (df[df.split == s].reset_index(drop=True) for s in ("train", "val"))

    rows = []
    for k, clf, C in itertools.product(*GRID.values()):
        m = make_clf(k, clf, C, min(feats["train"].shape)).fit(feats["train"], tr.referable)
        rows.append(dict(k=k, clf=clf, C=C, val_auroc=roc_auc_score(va.referable, m.predict_proba(feats["val"])[:, 1])))
        print(f"k={k:4d} {clf:8s} C={C:<6} val AUROC {rows[-1]['val_auroc']:.4f}")
    grid = pd.DataFrame(rows).sort_values("val_auroc", ascending=False)
    grid.to_csv(paths.results / f"grid_{name}.csv", index=False)
    best = grid.iloc[0]
    model_cls = make_clf(int(best.k), best.clf, float(best.C), min(feats["train"].shape)).fit(feats["train"], tr.referable)
    s_va = model_cls.predict_proba(feats["val"])[:, 1]
    model_type = f"B_{name}"
    write_preds(va, s_va, model_type, paths.results / f"preds_{model_type}.csv")
    thresholds = {r: choose_threshold(va.referable, s_va, r) for r in ("sens90", "youden")}
    summary = dict(run=name, model_type=model_type, feature_source=src, img_size=size,
                   n_features=int(min(feats["train"].shape)), best=dict(k=int(best.k), clf=best.clf, C=float(best.C)),
                   explained_variance=float(model_cls[1].explained_variance_ratio_.sum()),
                   thresholds=thresholds, val=evaluate(va.referable, s_va, thresholds),
                   runtime=times, hardware=hardware_info())
    print(f"best: {summary['best']} val AUROC {summary['val']['auroc']:.3f}, "
          f"PCA keeps {summary['explained_variance']:.1%} of variance")
    if a.final_test:
        te = df[df.split == "test"].reset_index(drop=True)
        s_te = model_cls.predict_proba(feats["test"])[:, 1]
        write_preds(te, s_te, model_type, paths.results / f"preds_{model_type}.csv", append_test=True)
        summary["test"] = evaluate(te.referable, s_te, thresholds)
        print(f"test AUROC {summary['test']['auroc']:.3f}")
    save_json(summary, paths.results / f"summary_{name}.json")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(p)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--run", help="use the backbone of this trained run")
    g.add_argument("--arch", default="efficientnet_b0", choices=["resnet18", "resnet50", "efficientnet_b0"])
    p.add_argument("--size", type=int, default=384)
    p.add_argument("--final-test", action="store_true", help="also score the test split (once)")
    p.add_argument("--no-pretrained", action="store_true", help="random weights (smoke test only)")
    p.add_argument("--workers", type=int, default=default_workers())
    main(p.parse_args())
