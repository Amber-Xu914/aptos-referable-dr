"""Part C ablation: add one data improvement at a time to the CNN and measure it on validation.

Each experiment changes exactly ONE thing compared with the experiment named
in its `base` column:

    E0_baseline      rgb preprocessing (geometry only), no augmentation, no imbalance fix
    E1_clahe         E0 + CLAHE preprocessing (clahe_rgb)
    E2_augment       E1 + augmentation (flips, rotation, brightness, contrast)
    E3_classweight   E2 + class weight in the loss
    E3b_oversample   E2 + oversampling instead of class weight (alternative to E3)
    E4_bengraham     E2 with Ben Graham preprocessing instead of CLAHE (alternative to E2)

Everything else (network, input size, epochs, optimiser, seed) is identical,
so a difference between two rows is caused by that one change (plus training
noise, which is why several seeds are recommended).

Only the train and val splits are used. The test split is never loaded here
except by `--predict-test`, which must only be run once the final model is frozen.

The network is a stand-in (ImageNet-pretrained ResNet-18) until B's CNN is
ready; B's model can be added to `build_model` and selected with --model.

Outputs:
    results/part_c/preds/preds_<experiment>_s<seed>.csv   val predictions, group CSV format
    results/part_c/ablation_table.csv                      one row per experiment (mean over seeds)
    results/part_c/ablation_runs.csv                       one row per experiment and seed
    outputs/part_c/checkpoints/<experiment>_s<seed>.pt      best-epoch weights (not in git)

Usage:
    python -m src.ablation                                   # all experiments, seed 9027
    python -m src.ablation --seeds 1 2 3 --epochs 12
    python -m src.ablation --only E0_baseline E1_clahe --subset 200 --epochs 1   # quick smoke test
    python -m src.ablation --predict-test E3_classweight --seeds 9027            # final model only, once
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from src.augment import Augmenter
from src.imbalance import oversampling_weights, pos_weight
from src.metrics import evaluate
from src.preprocess import cache_folder, normalize
from src.splits import ROOT, load_splits

RAW_DIR = ROOT / "data/raw/Aptos2019 retinal image database/train_images"
CACHE_DIRS = {
    "rgb": ROOT / "data/cache_rgb_512",
    "clahe_rgb": ROOT / "data/cache_clahe_512",
    "ben_graham": ROOT / "data/cache_bengraham_512",
}
RESULTS = ROOT / "results/part_c"
CKPT_DIR = ROOT / "outputs/part_c/checkpoints"


@dataclass(frozen=True)
class Experiment:
    name: str
    base: str
    change: str
    preprocess: str
    augment: bool
    imbalance: str  # "none", "class_weight" or "oversample"


EXPERIMENTS = [
    Experiment("E0_baseline", "-", "baseline", "rgb", False, "none"),
    Experiment("E1_clahe", "E0_baseline", "+ CLAHE preprocessing", "clahe_rgb", False, "none"),
    Experiment("E2_augment", "E1_clahe", "+ augmentation", "clahe_rgb", True, "none"),
    Experiment("E3_classweight", "E2_augment", "+ class weight", "clahe_rgb", True, "class_weight"),
    Experiment("E3b_oversample", "E2_augment", "+ oversampling (instead of class weight)", "clahe_rgb", True, "oversample"),
    Experiment("E4_bengraham", "E2_augment", "Ben Graham instead of CLAHE", "ben_graham", True, "none"),
]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ensure_cache(mode):
    """Build the 512 px cache for a preprocessing mode if it does not exist yet."""
    d = CACHE_DIRS[mode]
    if not d.is_dir() or not any(d.glob("*.png")):
        print(f"Cache for {mode!r} not found, building it in {d} (one-off, a few minutes)")
        cache_folder(RAW_DIR, d, size=512, mode=mode)
    return d


def load_images(ids, cache_dir, size):
    """Read cached images and resize to the network input size (uint8 RGB, N x H x W x 3)."""
    out = np.empty((len(ids), size, size, 3), np.uint8)
    for i, id_code in enumerate(ids):
        img = cv2.imread(str(cache_dir / f"{id_code}.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(cache_dir / f"{id_code}.png")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        out[i] = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return out


class FundusDataset(torch.utils.data.Dataset):
    """Images held in memory; augmentation (train only) happens before normalize()."""

    def __init__(self, images, labels, augmenter=None):
        self.images, self.labels, self.augmenter = images, labels, augmenter

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        img = self.images[i]
        if self.augmenter is not None:
            img = self.augmenter(img)
        x = torch.from_numpy(normalize(img)).permute(2, 0, 1)
        return x, torch.tensor(self.labels[i], dtype=torch.float32)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(name, pretrained=True):
    """Return a network with a single output logit (referable vs not).

    B: add your CNN here as another branch, then run with --model <name>.
    """
    import torchvision

    if name == "resnet18":
        weights = torchvision.models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = torchvision.models.resnet18(weights=weights)
        m.fc = torch.nn.Linear(m.fc.in_features, 1)
        return m
    raise ValueError(f"unknown model {name!r}")


def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    scores = []
    for x, _ in loader:
        scores.append(torch.sigmoid(model(x.to(device)).squeeze(1)).float().cpu().numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------------------
# One training run
# ---------------------------------------------------------------------------
def train_one(exp, seed, tr, va, args, device):
    torch.manual_seed(seed)
    np.random.seed(seed)

    cache = ensure_cache(exp.preprocess)
    t0 = time.time()
    x_tr, x_va = load_images(tr["id_code"], cache, args.size), load_images(va["id_code"], cache, args.size)
    y_tr, y_va = tr["referable"].to_numpy(), va["referable"].to_numpy()

    aug = Augmenter(seed=seed) if exp.augment else None
    ds_tr = FundusDataset(x_tr, y_tr, aug)
    ds_va = FundusDataset(x_va, y_va)

    g = torch.Generator().manual_seed(seed)
    if exp.imbalance == "oversample":
        sampler = torch.utils.data.WeightedRandomSampler(
            torch.as_tensor(oversampling_weights(y_tr)), num_samples=len(y_tr), replacement=True, generator=g)
        dl_tr = torch.utils.data.DataLoader(ds_tr, args.batch_size, sampler=sampler)
    else:
        dl_tr = torch.utils.data.DataLoader(ds_tr, args.batch_size, shuffle=True, generator=g)
    dl_va = torch.utils.data.DataLoader(ds_va, args.batch_size)

    model = build_model(args.model, pretrained=not args.no_pretrained).to(device)
    pw = torch.tensor(pos_weight(y_tr) if exp.imbalance == "class_weight" else 1.0, device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pw)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    best = {"auroc": -1.0}
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for x, y in dl_tr:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x).squeeze(1), y)
            loss.backward()
            opt.step()
            total += loss.item() * len(y)
        sched.step()
        s_va = predict(model, dl_va, device)
        auc = roc_auc_score(y_va, s_va)
        print(f"  {exp.name} seed {seed} epoch {epoch:2d}: train loss {total / len(y_tr):.4f}  val AUROC {auc:.4f}")
        if auc > best["auroc"]:
            best = {"auroc": auc, "epoch": epoch, "scores": s_va}
            CKPT_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CKPT_DIR / f"{exp.name}_s{seed}.pt")

    best["minutes"] = (time.time() - t0) / 60
    return best


def write_preds(df, scores, split, model_type, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "image_id": df["id_code"].to_numpy(),
        "split": split,
        "true_grade": df["diagnosis"].to_numpy(),
        "model_score": np.round(scores, 5),
        "model_type": model_type,
    }).to_csv(path, index=False)


def summarise(runs):
    """Mean (and SD over seeds) of each metric per experiment, in experiment order."""
    metric_cols = [c for c in runs.columns if c not in ("experiment", "base", "change", "seed")]
    order = [e.name for e in EXPERIMENTS if e.name in set(runs["experiment"])]
    g = runs.groupby("experiment", sort=False)
    table = g[metric_cols].mean()
    table.insert(0, "n_seeds", g.size())
    if runs["seed"].nunique() > 1:
        for c in ("auroc", "auroc_g1_vs_ref"):
            table.insert(table.columns.get_loc(c) + 1, f"{c}_sd", g[c].std())
    info = runs.drop_duplicates("experiment").set_index("experiment")[["base", "change"]]
    return info.join(table).loc[order].reset_index()


# ---------------------------------------------------------------------------
# Test predictions for the frozen final model (run once)
# ---------------------------------------------------------------------------
def predict_test(name, args, device):
    exp = next(e for e in EXPERIMENTS if e.name == name)
    te = load_splits(args.splits).query("split == 'test'").reset_index(drop=True)
    x_te = load_images(te["id_code"], ensure_cache(exp.preprocess), args.size)
    dl = torch.utils.data.DataLoader(FundusDataset(x_te, te["referable"].to_numpy()), args.batch_size)
    for seed in args.seeds:
        model = build_model(args.model, pretrained=False).to(device)
        model.load_state_dict(torch.load(CKPT_DIR / f"{name}_s{seed}.pt", map_location=device))
        path = RESULTS / "test_ready" / f"preds_C_{name}_s{seed}.csv"
        write_preds(te, predict(model, dl, device), "test", f"C_{name}", path)
        print(f"Wrote {path} (no metrics computed: evaluation is done by D once all models are frozen)")


def main():
    p = argparse.ArgumentParser(description="Part C ablation on the validation split.")
    p.add_argument("--only", nargs="+", help="run only these experiments")
    p.add_argument("--seeds", nargs="+", type=int, default=[9027])
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--size", type=int, default=224, help="network input size")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--model", default="resnet18")
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--subset", type=int, help="use only N train and N val images (smoke test)")
    p.add_argument("--splits", default=None)
    p.add_argument("--out-tag", default="", help="suffix for output tables, e.g. _smoke")
    p.add_argument("--predict-test", metavar="EXPERIMENT",
                   help="write test predictions for a trained experiment (final model only)")
    args = p.parse_args()

    device = pick_device()
    print(f"Device: {device}")
    if args.predict_test:
        return predict_test(args.predict_test, args, device)

    splits = load_splits(args.splits)
    conflict = frozenset(splits.loc[splits["label_conflict"] == 1, "id_code"])
    tr = splits.query("split == 'train'").reset_index(drop=True)
    va = splits.query("split == 'val'").reset_index(drop=True)
    if args.subset:
        tr = tr.sample(min(args.subset, len(tr)), random_state=0).reset_index(drop=True)
        va = va.sample(min(args.subset, len(va)), random_state=0).reset_index(drop=True)

    exps = [e for e in EXPERIMENTS if not args.only or e.name in args.only]
    runs_path = RESULTS / f"ablation_runs{args.out_tag}.csv"
    runs = []
    for exp in exps:
        for seed in args.seeds:
            print(f"\n=== {exp.name} ({exp.change}), seed {seed} ===")
            best = train_one(exp, seed, tr, va, args, device)
            write_preds(va, best["scores"], "val", f"C_{exp.name}",
                        RESULTS / "preds" / f"preds_C_{exp.name}_s{seed}{args.out_tag}.csv")
            va_eval = va.assign(image_id=va["id_code"], true_grade=va["diagnosis"], model_score=best["scores"])
            metrics = evaluate(va_eval, conflict)
            runs.append({"experiment": exp.name, "base": exp.base, "change": exp.change, "seed": seed,
                         **metrics, "best_epoch": best["epoch"], "minutes": best["minutes"]})
            pd.DataFrame(runs).to_csv(runs_path, index=False)  # save after every run

    table = summarise(pd.DataFrame(runs))
    table_path = RESULTS / f"ablation_table{args.out_tag}.csv"
    table.round(4).to_csv(table_path, index=False)
    cols = ["experiment", "change", "auroc", "auroc_g1_vs_ref", "auroc_no_conflict", "spec_at_sens85",
            "sens_g2", "sens_g3", "sens_g4"]
    print("\n" + table[cols].round(3).to_string(index=False))
    print(f"\nWrote {table_path} and {runs_path}")


if __name__ == "__main__":
    main()
