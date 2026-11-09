"""Train one Part B model on the training split and select it on the validation split.

Examples (Colab GPU):
    python part_b/train.py --model smallcnn --epochs 25                  # 2. CNN from scratch
    python part_b/train.py --model efficientnet_b0 --mode frozen --epochs 25         # 3. transfer: frozen backbone
    python part_b/train.py --model efficientnet_b0 --mode finetune --epochs 25        # 3. transfer: fine-tuning

What it does:
  * loads splits.csv, checks it (class counts, IDs, leakage) and stops on a hard failure
  * trains on `train` with augmentation, BCE loss (class-balanced via pos_weight), AdamW + cosine LR
  * after each epoch scores `val`; keeps the checkpoint with the best val AUROC; early stopping
  * writes results/part_b/preds_B_<run>.csv (val rows, same format as Part A),
    results/part_b/history_<run>.csv and results/part_b/summary_<run>.json
    (val metrics, thresholds chosen on val, parameter counts, runtime, hardware)

The test split is never loaded here. Use final_test.py once the models are frozen.
"""

import argparse
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.common import (SEED, add_path_args, check_splits, choose_threshold, evaluate, get_paths, load_splits,
                           make_dataset, preload, save_json, set_seed, write_preds)
from part_b.hardware import device, hardware_info, peak_gpu_mem_gb, sync, timer


def run_name(a):
    return a.name or (a.model if a.model == "smallcnn" else f"{a.model}_{a.mode}")


def predict(model, loader, dev, tta=True):
    """Return P(referable) for every image in loader order. TTA = average with the horizontal flip."""
    import torch
    model.eval()
    out = []
    with torch.no_grad():
        for x, _, _ in loader:
            x = x.to(dev, non_blocking=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                logit = model(x)
                if tta:
                    logit = (logit + model(torch.flip(x, dims=[3]))) / 2
            out.append(torch.sigmoid(logit.float()).cpu().numpy())
    return np.concatenate(out)


def main(a):
    import torch
    from sklearn.metrics import log_loss, roc_auc_score
    from torch.utils.data import DataLoader
    from part_b.models import build_model, count_params, keep_frozen_bn_eval, set_mode

    set_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    paths = get_paths(a.root, a.cache)
    name = run_name(a)
    dev = device()
    hw = hardware_info()
    print(f"run {name} on {hw.get('gpu')} | torch {hw.get('torch')}")

    # 1. data -----------------------------------------------------------------
    df = load_splits(paths.splits)
    ok, lines, _ = check_splits(df, paths.cache)
    print("\n".join(lines))
    if not ok:
        sys.exit("split check failed; fix the data before training")
    tr = df[df.split == "train"].reset_index(drop=True)
    va = df[df.split == "val"].reset_index(drop=True)
    if a.drop_conflicts:
        tr = tr[tr.label_conflict == 0].reset_index(drop=True)
    if a.limit:                                   # quick debugging only
        tr, va = tr.head(a.limit), va.head(max(a.limit // 2, 8))

    times = {}
    with timer(times, "load_images_s"):
        x_tr, x_va = preload(tr, paths.cache, a.size), preload(va, paths.cache, a.size)
    print(f"loaded {len(tr)} train / {len(va)} val images at {a.size}px in {times['load_images_s']}s")
    kw = dict(batch_size=a.batch_size, num_workers=a.workers, pin_memory=dev.type == "cuda",
              persistent_workers=a.workers > 0)
    dl_tr = DataLoader(make_dataset(x_tr, tr.referable, train=True), shuffle=True, drop_last=True, **kw)
    dl_va = DataLoader(make_dataset(x_va, va.referable, train=False), shuffle=False, **kw)

    # 2/3. model ---------------------------------------------------------------
    model = build_model(a.model, pretrained=not a.no_pretrained).to(dev)
    mode = "scratch" if a.model == "smallcnn" else a.mode
    groups = set_mode(model, mode, a.lr, a.backbone_lr_mult, a.weight_decay)
    n_total, n_train = count_params(model)
    print(f"{a.model} ({mode}): {n_total / 1e6:.2f} M parameters, {n_train / 1e6:.2f} M trainable")

    pos = tr.referable.mean()
    pos_weight = torch.tensor((1 - pos) / pos if a.pos_weight == "auto" else float(a.pos_weight), device=dev)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.AdamW(groups)
    steps = a.epochs * len(dl_tr)
    warm = len(dl_tr)                             # one epoch of linear warm-up
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, steps - warm))))
    scaler = torch.amp.GradScaler(dev.type, enabled=dev.type == "cuda")
    ckpt = paths.ckpt / f"{name}.pt"

    history, best, bad_epochs = [], -1.0, 0
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_train = time.perf_counter()
    for ep in range(1, a.epochs + 1):
        model.train(); keep_frozen_bn_eval(model)
        sync(); t0 = time.perf_counter()
        losses = []
        for x, y, _ in dl_tr:
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=dev.type == "cuda"):
                loss = loss_fn(model(x).float(), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update(); sched.step()
            losses.append(loss.item())
        sync(); ep_s = time.perf_counter() - t0
        s_va = predict(model, dl_va, dev, tta=False)
        auc = roc_auc_score(va.referable, s_va)
        vloss = log_loss(va.referable, np.clip(s_va, 1e-6, 1 - 1e-6))
        history.append(dict(epoch=ep, train_loss=np.mean(losses), val_loss=vloss, val_auroc=auc,
                            lr=opt.param_groups[0]["lr"], epoch_s=round(ep_s, 1)))
        improved = auc > best + 1e-4
        if improved:
            best, bad_epochs = auc, 0
            torch.save(dict(state=model.state_dict(), model=a.model, mode=mode, size=a.size, epoch=ep), ckpt)
        else:
            bad_epochs += 1
        print(f"ep {ep:3d}  train loss {np.mean(losses):.4f}  val loss {vloss:.4f}  val AUROC {auc:.4f}"
              f"{'  *' if improved else ''}  ({ep_s:.0f}s)", flush=True)
        if bad_epochs >= a.patience:
            print(f"early stop: no val AUROC gain for {a.patience} epochs")
            break
    train_s = time.perf_counter() - t_train

    # validation predictions from the best checkpoint ---------------------------
    model.load_state_dict(torch.load(ckpt, map_location=dev)["state"])
    t0 = time.perf_counter(); s_va = predict(model, dl_va, dev, tta=a.tta); sync()
    infer_ms = 1000 * (time.perf_counter() - t0) / len(va)
    model_type = f"B_{name}"
    write_preds(va, s_va, model_type, paths.results / f"preds_{model_type}.csv")
    pd.DataFrame(history).to_csv(paths.results / f"history_{name}.csv", index=False)

    thresholds = {r: choose_threshold(va.referable, s_va, r) for r in ("sens90", "youden")}
    val_metrics = evaluate(va.referable, s_va, thresholds)
    val_metrics["auroc_excl_label_conflict"] = evaluate(
        va.referable[va.label_conflict == 0], s_va[va.label_conflict == 0], {})["auroc"]
    summary = dict(
        run=name, model_type=model_type, model=a.model, mode=mode, img_size=a.size,
        params_total=n_total, params_trainable=n_train, best_epoch=int(pd.DataFrame(history).val_auroc.idxmax() + 1),
        epochs_run=len(history), thresholds=thresholds, val=val_metrics,
        runtime=dict(load_images_s=times["load_images_s"], train_s=round(train_s, 1),
                     mean_epoch_s=round(float(np.mean([h["epoch_s"] for h in history])), 1),
                     inference_ms_per_image=round(infer_ms, 2), tta=a.tta, batch_size=a.batch_size,
                     amp=dev.type == "cuda", peak_gpu_mem_gb=peak_gpu_mem_gb()),
        hardware=hw, args=vars(a))
    save_json(summary, paths.results / f"summary_{name}.json")
    m = val_metrics["at_sens90"]
    print(f"\n{name}: val AUROC {val_metrics['auroc']:.3f} (95% CI {val_metrics['auroc_ci'][0]:.3f}-"
          f"{val_metrics['auroc_ci'][1]:.3f}); at val threshold {m['threshold']:.3f}: "
          f"sens {m['sensitivity']:.3f} spec {m['specificity']:.3f}\n"
          f"train {train_s / 60:.1f} min, inference {infer_ms:.1f} ms/image")


def parse(argv=None):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(p)
    p.add_argument("--model", default="efficientnet_b0", choices=["smallcnn", "resnet18", "resnet50", "efficientnet_b0"])
    p.add_argument("--mode", default="finetune", choices=["frozen", "partial", "finetune"],
                   help="transfer models only; smallcnn is always trained from scratch")
    p.add_argument("--name", default=None, help="run name (default: <model>_<mode>)")
    p.add_argument("--size", type=int, default=384, help="input size; the 512 cache is downsized")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="head learning rate")
    p.add_argument("--backbone-lr-mult", type=float, default=0.1, help="backbone lr = lr x this (partial/finetune)")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--pos-weight", default="auto", help="'auto' = n_neg/n_pos on train, or a number (1 = none)")
    p.add_argument("--drop-conflicts", action="store_true", help="drop label-conflict duplicates from TRAIN only")
    p.add_argument("--no-tta", dest="tta", action="store_false", help="disable flip test-time augmentation")
    p.add_argument("--no-pretrained", action="store_true", help="random init (smoke test / ablation)")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--limit", type=int, default=0, help="use only N training images (debugging)")
    p.add_argument("--seed", type=int, default=SEED)
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse())