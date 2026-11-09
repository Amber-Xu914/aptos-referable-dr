"""Run the whole of Part B in one go, locally (e.g. VS Code: open this file and press Run).

Steps:
  0. find the images in the repository's data/ folder (preprocessed once if only raw images are there)
  1. data loader + split check                       5. hardware notes
  2. small CNN from scratch                          3. EfficientNet-B0 frozen and fine-tuned
  +  optional hybrid (CNN features -> PCA -> classical classifier)
  -  comparison on VALIDATION; winner = best validation AUROC
  4. Grad-CAM on the winner (validation examples)
  -  final TEST evaluation, once, at thresholds fixed on validation; Grad-CAM on test errors
  -  comparison table, ROC/learning curves and runtime notes refreshed

Finished runs are skipped when the script is started again (delete results/part_b/summary_<run>.json
or pass --force to retrain), so an interrupted session can simply be restarted.

Edit SETTINGS below, or override on the command line, e.g.
    python part_b/run_all.py
    python part_b/run_all.py --quick                 # 5-minute check that everything works (no real results)
    python part_b/run_all.py --seeds 1 2 3           # every model with three seeds
"""
import argparse
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# =============================================================================
# SETTINGS (edit here, or use the command-line options of the same name)
# =============================================================================
SETTINGS = dict(
    # Folder holding the downloaded images (cache_clahe_512/, cache_clahe_512.zip or raw train_images/,
    # at any depth). Empty = the "data" folder in the repository.
    data_dir="",
    size=384,                # input size (px); 224 is ~3x faster on weak GPUs/CPUs but loses small lesions
    epochs=25,               # every model is trained for this many epochs (no early stopping)
    seeds=[9027],            # e.g. [1, 2, 3] for the three-seed comparison
    hybrid=True,             # optional hybrid model
    final_test=True,         # score the test split once at the end
)
# The models (2. and 3.). Each entry: run name -> train.py arguments.
RUNS = {
    "smallcnn":                 ["--model", "smallcnn", "--lr", "1e-3"],
    "efficientnet_b0_frozen":   ["--model", "efficientnet_b0", "--mode", "frozen", "--lr", "1e-3"],
    "efficientnet_b0_finetune": ["--model", "efficientnet_b0", "--mode", "finetune", "--lr", "1e-3"],
}
# =============================================================================


class Runner:
    """Runs each step as its own process and prints a timeline of all planned steps before each one.

        Part B timeline  |  step 4 of 12  |  elapsed 18m 03s  |  training ~1h 05m left
        [x] Hardware notes                                     0m 04s
        [-] Train smallcnn                                     skipped (already trained)
        [>] Train efficientnet_b0_frozen                       running
        [ ] Train efficientnet_b0_finetune
    """

    def __init__(self, common, log, plan):
        self.common, self.log, self.t0 = common, log, time.perf_counter()
        self.steps = [dict(title=t, status="pending", dt=None, note="") for t in plan]
        self.times = []

    def _step(self, title):
        return next(st for st in self.steps if st["title"] == title and st["status"] in ("pending", "running"))

    def skip(self, title, note):
        st = self._step(title)
        st.update(status="skipped", note=note)

    def timeline(self):
        from part_b.common import fmt_time
        active = [st for st in self.steps if st["status"] != "skipped"]
        done = sum(st["status"] in ("done", "failed") for st in active)
        current = done + 1 if done < len(active) else len(active)
        # Training dominates the run time: estimate what is left from the training runs finished so far
        train_done = [st["dt"] for st in self.steps if st["title"].startswith("Train ") and st["status"] == "done"]
        train_left = sum(st["status"] in ("pending", "running") for st in self.steps if st["title"].startswith("Train "))
        eta = (f"  |  training ~{fmt_time(sum(train_done) / len(train_done) * train_left)} left"
               if train_done and train_left else "")
        lines = [f"Part B timeline  |  step {current} of {len(active)}  |  "
                 f"elapsed {fmt_time(time.perf_counter() - self.t0)}{eta}"]
        mark = dict(done="[x]", failed="[!]", skipped="[-]", running="[>]", pending="[ ]")
        for st in self.steps:
            info = (fmt_time(st["dt"]) if st["status"] == "done" else "FAILED" if st["status"] == "failed"
                    else st["note"] if st["status"] == "skipped" else "running" if st["status"] == "running" else "")
            lines.append(f"  {mark[st['status']]} {st['title']:52s} {info}")
        bar = "-" * 78
        print(f"\n{bar}\n" + "\n".join(lines) + f"\n{bar}", flush=True)

    def __call__(self, title, script, *args, required=True):
        st = self._step(title)
        st["status"] = "running"
        self.timeline()
        cmd = [sys.executable, str(HERE / script), *args, *self.common]
        print(f"$ python part_b/{script} {' '.join(args)}\n", flush=True)
        t0 = time.perf_counter()
        rc = subprocess.run(cmd).returncode
        dt = time.perf_counter() - t0
        st.update(status="done" if rc == 0 else "failed", dt=dt)
        self.times.append((title, dt, rc))
        with open(self.log, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {title:55s} {dt / 60:7.1f} min  exit {rc}\n")
        if rc and required:
            self.timeline()
            sys.exit(f"\nSTOPPED: '{title}' failed (exit {rc}). Fix the error above and run again; "
                     f"finished steps are skipped.")
        return rc == 0


REQUIRED = {"torch": "torch", "torchvision": "torchvision", "numpy": "numpy", "pandas": "pandas",
            "cv2": "opencv-python-headless", "sklearn": "scikit-learn", "matplotlib": "matplotlib"}


def check_packages():
    """Stop with clear instructions if a package is missing from the Python running this script."""
    import importlib.util
    missing = [pip for mod, pip in REQUIRED.items() if importlib.util.find_spec(mod) is None]
    if importlib.util.find_spec("tqdm") is None:
        print("note: tqdm is not installed, so there is no live batch progress bar (pip install tqdm)")
    if not missing:
        return
    py = f'& "{sys.executable}"' if sys.platform.startswith("win") else f'"{sys.executable}"'
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    msg = [f"Missing package(s): {', '.join(missing)}",
           f"This script is running with:  {sys.executable}",
           "  (not a virtual environment)" if not in_venv else "  (virtual environment)",
           "",
           "Install them into THIS Python (copy the whole line):",
           f'  {py} -m pip install -r part_b/requirments.txt']
    if "torch" in missing:
        msg += ["",
                "If the computer has an NVIDIA GPU, install the GPU build of PyTorch first, using the",
                "command from https://pytorch.org/get-started/locally/ with pip replaced by:",
                f'  {py} -m pip',
                "If pip answers 'No matching distribution found for torch', this Python version has no",
                "PyTorch build (too new, or 32-bit). Install a 64-bit Python version listed on pytorch.org,",
                "create the .venv with it and select it in VS Code. Check the version with:",
                f'  {py} --version']
    msg += ["",
            "If you already installed them, VS Code is probably using a different Python:",
            "  Ctrl+Shift+P (Cmd+Shift+P on Mac) -> 'Python: Select Interpreter' -> pick the one you installed",
            "  into (e.g. .venv), then open a NEW terminal and run again."]
    sys.exit("\n".join(msg))


def main(a):
    import pandas as pd
    from part_b.common import fmt_time, get_paths, load_json
    from part_b.local_data import find_images
    from part_b.hardware import device, gpu_advice, hardware_info

    paths = get_paths(a.root, a.cache)

    # 0. images ---------------------------------------------------------------
    print("Step 0: images")
    try:
        cache = find_images(paths, a.data_dir or None)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")
    paths = get_paths(a.root, cache)
    common = (["--root", a.root] if a.root else []) + ["--cache", str(cache)]

    hw = hardware_info()
    dev = device()
    print(f"\nDevice: {dev.type} | {hw.get('gpu')} | torch {hw.get('torch')} | {hw.get('cpu_count')} CPU cores, "
          f"{hw.get('ram_gb')} GB RAM")
    if dev.type == "cpu":
        advice = gpu_advice()
        print("\nNo usable GPU: training runs on the CPU." + (f"\n{advice}" if advice else ""))
        if not a.quick:
            print("On a CPU, fine-tuning EfficientNet-B0 at 384 px takes many hours (consider --size 224). "
                  "Fix the GPU setup above, or press Ctrl+C within 15 s to stop.", flush=True)
            time.sleep(15)

    # Plan every step up front, so the timeline can show what is still to come ----------
    epochs = ["--epochs", "1", "--limit", "64"] if a.quick else ["--epochs", str(a.epochs)]
    if a.quick:
        a.hybrid = False                               # feature extraction over all images is not "quick"
    train_jobs, names = [], []
    for seed in a.seeds:
        for base, args in RUNS.items():
            name = base if len(a.seeds) == 1 else f"{base}_s{seed}"
            names.append(name)
            summ = paths.results / f"summary_{name}.json"
            # finished = weights + summary from the same settings (a --quick run never counts as real;
            # a run with a different epoch count or image size is retrained)
            prev = load_json(summ)["args"] if summ.exists() else {}
            done = ((paths.ckpt / f"{name}.pt").exists() and bool(prev.get("limit")) == a.quick
                    and (a.quick or (prev.get("epochs") == a.epochs and prev.get("size") == a.size)))
            train_jobs.append((name, seed, args, done and not a.force))
    plan = ["Hardware notes", "Data loader and split check", *[f"Train {n}" for n in names]]
    plan += ["Hybrid: CNN features -> PCA -> classifier"] if a.hybrid else []
    plan += ["Compare runs on validation", "Grad-CAM on the winner (validation)"]
    if a.final_test:
        plan += ["Final test evaluation (thresholds from validation)"]
        plan += ["Hybrid on test"] if a.hybrid else []
        plan += ["Grad-CAM on the winner (test)", "Refresh comparison and runtime notes"]
    run = Runner(common, paths.results / "run_all_log.txt", plan)
    for name, _, _, skip in train_jobs:
        if skip:
            run.skip(f"Train {name}", "skipped (already trained; --force retrains)")

    run("Hardware notes", "hardware.py")
    run("Data loader and split check", "check_data.py")

    # 2/3. models ---------------------------------------------------------------
    for name, seed, args, skip in train_jobs:
        if skip:
            continue
        run(f"Train {name}", "train.py", *args, "--name", name, "--size", str(a.size), "--seed", str(seed), *epochs)

    hybrid_args = ["--arch", "efficientnet_b0", "--size", str(a.size)]
    if a.hybrid:
        run("Hybrid: CNN features -> PCA -> classifier", "hybrid.py", *hybrid_args, required=False)

    # Comparison on validation, winner, Grad-CAM -------------------------------------
    part_a = [str(p) for p in [paths.root / "results/part_a/preds_A_rf_all.csv"] if p.exists()]
    cmp_args = (["--extra", *part_a] if part_a else []) + (["--include-quick"] if a.quick else [])
    run("Compare runs on validation", "compare.py", *cmp_args)
    table = pd.read_csv(paths.results / "comparison_val.csv")
    cnn = table[table.run.isin(names)]
    winner = cnn.iloc[0].run
    print(f"\nWinner on validation AUROC: {winner} ({cnn.iloc[0].val_auroc:.3f})")
    cam_quick = ["--max-stats", "40"] if a.quick else []
    run("Grad-CAM on the winner (validation)", "gradcam.py", "--run", winner, "--split", "val", *cam_quick)

    # Final test, once -------------------------------------------------------------
    if a.final_test:
        run("Final test evaluation (thresholds from validation)", "final_test.py", "--runs", *names)
        if a.hybrid:
            run("Hybrid on test", "hybrid.py", *hybrid_args, "--final-test", required=False)
        run("Grad-CAM on the winner (test)", "gradcam.py", "--run", winner, "--split", "test", *cam_quick)
        run("Refresh comparison and runtime notes", "compare.py", *cmp_args)

    run.timeline()
    print(f"DONE in {fmt_time(time.perf_counter() - run.t0)}. Winner on validation: {winner}.")
    print(f"\nResults for the report: {paths.results}\n"
          "  comparison_val.csv, roc_val.png, learning_curves.png, runtime_hardware.md,\n"
          f"  gradcam_{winner}_val.png" + (", test_metrics.csv, gradcam_" + winner + "_test.png" if a.final_test else ""))
    if a.quick:
        print("\nQUICK MODE: these numbers are meaningless (1 epoch, 64 images). Run without --quick for real "
              "results; the quick runs are retrained, not reused.")


def parse():
    from part_b.common import add_path_args
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(p)
    p.add_argument("--data-dir", default=SETTINGS["data_dir"], help="folder with the downloaded images (default: data/)")
    p.add_argument("--size", type=int, default=SETTINGS["size"])
    p.add_argument("--epochs", type=int, default=SETTINGS["epochs"])
    p.add_argument("--seeds", type=int, nargs="+", default=SETTINGS["seeds"])
    p.add_argument("--no-hybrid", dest="hybrid", action="store_false", default=SETTINGS["hybrid"])
    p.add_argument("--no-final-test", dest="final_test", action="store_false", default=SETTINGS["final_test"])
    p.add_argument("--force", action="store_true", help="retrain runs that already finished")
    p.add_argument("--quick", action="store_true", help="1 epoch on 64 images: checks the setup, no real results")
    return p.parse_args()


if __name__ == "__main__":
    check_packages()
    main(parse())
