"""End-to-end test of every Part A script on synthetic images (takes 1-2 minutes).

Usage
  python part_a/smoke_test.py

A small synthetic project is built in a temporary folder: 96 colour "retinas" with more dark dots and
bright blobs at higher grades, a few exact duplicate files (one with a conflicting grade), a data
split and a preprocessed cache. Every script is then run on it and the prediction files are checked
against the shared format.

The test confirms that the code and the Python environment work. Its numbers carry no meaning, and
check_part_c.py reports WARN lines because the synthetic data do not reproduce APTOS's duplicate counts.
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
N_DARK = {0: 0, 1: 4, 2: 25, 3: 55, 4: 80}        # synthetic dark lesions per grade
N_BRIGHT = {0: 0, 1: 0, 2: 5, 3: 9, 4: 14}        # synthetic bright lesions per grade


# ---------------------------------------------------------------------------
# Synthetic fundus photograph
# ---------------------------------------------------------------------------
def make_raw(rng, grade, h=640, w=800):
    """Colour fundus-like image; the circle is cut off at top and bottom as in many APTOS photos."""
    yy, xx = np.mgrid[:h, :w]
    r = np.hypot(yy - h / 2, xx - w / 2)
    fov = r < 340
    g = (150 - 0.12 * r + rng.normal(0, 3, (h, w))).astype(np.float32)   # vignetted background with noise
    for _ in range(6):                                                    # vessels: random curved lines
        pt = np.array([w / 2 + rng.integers(-50, 50), h / 2 + rng.integers(-50, 50)], float)
        ang, pts = rng.uniform(0, 2 * np.pi), []
        for _ in range(90):
            ang += rng.normal(0, 0.15)
            pt = pt + 4 * np.array([np.cos(ang), np.sin(ang)])
            pts.append(pt.copy())
        cv2.polylines(g, [np.array(pts, np.int32).reshape(-1, 1, 2)], False, 100, 4)
    for _ in range(N_DARK[grade] + int(rng.integers(0, 4))):              # dark dots and blots
        a, rad = rng.uniform(0, 2 * np.pi), 290 * np.sqrt(rng.uniform())
        cv2.circle(g, (int(w / 2 + rad * np.cos(a)), int(h / 2 + rad * np.sin(a))),
                   int(rng.integers(3, 5 if grade < 3 else 9)), 95, -1)
    for _ in range(N_BRIGHT[grade]):                                      # bright blobs (exudates)
        a, rad = rng.uniform(0, 2 * np.pi), 270 * np.sqrt(rng.uniform())
        cv2.circle(g, (int(w / 2 + rad * np.cos(a)), int(h / 2 + rad * np.sin(a))),
                   int(rng.integers(5, 13)), 215, -1)
    cv2.circle(g, (int(rng.integers(470, 520)), int(rng.integers(290, 350))), 40, 225, -1)   # optic disc
    g = cv2.GaussianBlur(g, (0, 0), 1.5)
    bgr = np.stack([g * 0.3, g, np.clip(g * 1.4 + 30, 0, 255)], axis=2)
    bgr[~fov] = 0
    return np.clip(bgr, 0, 255).astype(np.uint8)


def run(*args, allow_fail=False):
    """Run one script, print the end of its output, and stop on failure."""
    cmd = [sys.executable, str(HERE / args[0]), *map(str, args[1:])]
    shown = " ".join("<tmp>" if isinstance(x, Path) else str(x) for x in args[1:])
    print(f"\n$ python part_a/{args[0]} {shown}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    print("\n".join(res.stdout.strip().splitlines()[-8:]))
    if res.returncode != 0 and not allow_fail:
        print(res.stderr[-2000:])
        sys.exit(f"FAILED: {args[0]}")
    return res


def main():
    # --- synthetic raw data ---
    root = Path(tempfile.mkdtemp(prefix="aptos_smoke_"))
    raw = root / "data" / "raw" / "train_images"
    raw.mkdir(parents=True)
    rng = np.random.default_rng(1)
    grades = rng.permutation(np.tile([0, 0, 0, 1, 2, 2, 3, 4], 12))
    rows = []
    for i, g in enumerate(grades):
        id_code = f"syn{i:03d}"
        if i in (5, 6, 7):                     # exact duplicates of images 0, 1 and 2
            shutil.copy(raw / f"syn{i - 5:03d}.png", raw / f"{id_code}.png")
            g = grades[i - 5] if i != 7 else (2 if grades[i - 5] < 2 else 0)   # one conflicting grade
        else:
            cv2.imwrite(str(raw / f"{id_code}.png"), make_raw(rng, int(g)))
        rows.append({"id_code": id_code, "diagnosis": int(g)})
    lab = pd.DataFrame(rows)
    lab.to_csv(root / "data" / "raw" / "train.csv", index=False)
    r = ["--root", root]

    # --- preprocessed cache and split, standing in for the shared ones ---
    run("make_green_cache.py", *r, "--workers", 2)
    shutil.copytree(root / "data" / "own_cache" / "green_512", root / "data" / "part_c" / "cache_green_512")
    sp = lab.assign(referable=(lab.diagnosis >= 2).astype(int),
                    split=["train" if i < 60 else ("val" if i < 78 else "test") for i in range(len(lab))],
                    label_conflict=[int(i in (2, 7)) for i in range(len(lab))])
    sp.to_csv(root / "data" / "part_c" / "splits.csv", index=False)

    # --- every script ---
    run("check_part_c.py", *r, "--n-images", 12)
    run("qa_overlays.py", *r, "--per-grade", 2, "--fov-check", 20)
    run("extract_features.py", *r, "--workers", 2, "--include-test")
    run("qa_features.py", *r)
    run("train_models.py", "ablation", *r)
    run("sweep_params.py", *r, "--param", "dark_thr", "--values", 10, 14, "--n-train", 40, "--workers", 2)
    run("train_models.py", "final", *r, "--model", "logreg", "--featset", "all", "--final-test")
    run("baseline_simple.py", *r, "--workers", 2, "--final-test")

    # --- prediction files follow the shared format ---
    pdir = root / "outputs" / "part_a" / "predictions"
    for f in ("preds_A_logreg_all.csv", "preds_baseline_colour_logreg.csv"):
        pred = pd.read_csv(pdir / f)
        assert list(pred.columns) == ["image_id", "split", "true_grade", "model_score", "model_type"], pred.columns
        assert set(pred["split"]) == {"val", "test"} and pred["model_score"].between(0, 1).all()
        assert pred["true_grade"].between(0, 4).all() and pred["model_type"].nunique() == 1
    assert pd.read_csv(pdir / "preds_A_logreg_all.csv")["model_type"].iloc[0] == "A_logreg_all"
    print(f"\nPASS - all scripts ran. Synthetic project and outputs are in {root}")


if __name__ == "__main__":
    main()
