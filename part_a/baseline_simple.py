"""Simple baseline: logistic regression on overall brightness and colour of the raw photographs.

No lesion detection and no preprocessing are used. Any more complex approach should outperform this
baseline to justify its complexity. A baseline score well above 0.5 also shows that the dataset
contains shortcuts (camera, exposure, clinic) that correlate with grade.

Two versions are fitted
  colour       mean and spread of red, green, blue and brightness inside the retina,
               retina area fraction and sharpness (the baseline used for comparison)
  colour_size  the same plus the original photo width and height (shortcut check only: photo size
               identifies the camera or clinic, not anything about the eye)

Usage
  python part_a/baseline_simple.py                 validation predictions
  python part_a/baseline_simple.py --final-test    also test predictions (final run only)

Output
  outputs/part_a/baseline_features.csv                       measurements per photo (reused on later runs)
  outputs/part_a/predictions/preds_baseline_<name>_logreg.csv
"""
import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import roc_auc_score

from common import get_paths, index_images, load_splits
from train_models import auc_ci, build, scores, tune, xy

COLOUR = ["bl_mean_r", "bl_mean_g", "bl_mean_b", "bl_std_r", "bl_std_g", "bl_std_b",
          "bl_brightness", "bl_fov_frac", "bl_sharp"]
SIZE = ["bl_width", "bl_height"]


# ---------------------------------------------------------------------------
# Measurements from one raw photograph
# ---------------------------------------------------------------------------
def raw_features(path):
    """Colour, brightness, retina fraction, sharpness and original size of one photo."""
    cv2.setNumThreads(1)
    w, h = Image.open(path).size                                  # original size (header only)
    bgr = cv2.imread(str(path), cv2.IMREAD_REDUCED_COLOR_4)      # quarter resolution is enough for averages
    bgr = cv2.resize(bgr, (512, int(512 * bgr.shape[0] / bgr.shape[1])), interpolation=cv2.INTER_AREA)
    fov = cv2.GaussianBlur(bgr.max(2), (0, 0), 2) > 10           # retina mask
    px = bgr[fov] if fov.any() else bgr.reshape(-1, 3)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    return {"bl_mean_b": px[:, 0].mean(), "bl_mean_g": px[:, 1].mean(), "bl_mean_r": px[:, 2].mean(),
            "bl_std_b": px[:, 0].std(), "bl_std_g": px[:, 1].std(), "bl_std_r": px[:, 2].std(),
            "bl_brightness": gray[fov].mean() if fov.any() else gray.mean(),
            "bl_fov_frac": fov.mean(),
            "bl_sharp": cv2.Laplacian(gray.astype(np.float32), cv2.CV_32F)[fov].var() if fov.any() else 0.0,
            "bl_width": w, "bl_height": h}


def _work(job):
    i, p = job
    try:
        return {"id_code": i, **raw_features(p)}
    except Exception as e:
        return {"id_code": i, "error": repr(e)}


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--final-test", action="store_true")
    a = ap.parse_args()
    paths = get_paths(a.root)
    paths.out.mkdir(parents=True, exist_ok=True)
    sp = load_splits(paths.splits)

    # measure every raw photo once and cache the result
    cache = paths.out / "baseline_features.csv"
    if cache.exists():
        feats = pd.read_csv(cache, dtype={"id_code": str})
    else:
        idx = index_images(paths.raw_images)
        jobs = [(i, str(idx[i])) for i in sp["id_code"] if i in idx]
        print(f"measuring {len(jobs)} raw photos...")
        with mp.Pool(a.workers) as pool:
            rows = list(pool.imap_unordered(_work, jobs, chunksize=8))
        bad = [r for r in rows if "error" in r]
        if bad:
            print(f"{len(bad)} failed, e.g. {bad[:2]}")
        feats = pd.DataFrame([r for r in rows if "error" not in r])
        feats.to_csv(cache, index=False)
    df = sp.merge(feats, on="id_code")
    tr, va, te = (df[df["split"] == s] for s in ("train", "val", "test"))

    # fit both versions on train, choose C on validation, write predictions
    pdir = paths.out / "predictions"
    pdir.mkdir(parents=True, exist_ok=True)
    for name, cols in (("colour", COLOUR), ("colour_size", COLOUR + SIZE)):
        auc, hp = tune("logreg", cols, tr, va)
        model = build("logreg", hp).fit(*xy(tr, cols))
        s_val = scores(model, xy(va, cols)[0])
        lo, hi = auc_ci(va["referable"], s_val)
        print(f"baseline '{name}': val AUROC {auc:.3f} [{lo:.3f}, {hi:.3f}]  (C={hp['C']})")
        parts = [va.assign(model_score=s_val)]
        if a.final_test:
            parts.append(te.assign(model_score=scores(model, xy(te, cols)[0])))
        out = pd.concat(parts)
        pd.DataFrame({"image_id": out["id_code"], "split": out["split"], "true_grade": out["diagnosis"],
                      "model_score": out["model_score"].round(6), "model_type": f"baseline_{name}_logreg"}
                     ).to_csv(pdir / f"preds_baseline_{name}_logreg.csv", index=False)
    print(f"\nfeature means by grade (train):\n{tr.groupby('diagnosis')[COLOUR + SIZE].mean().round(1).to_string()}")
    print(f"\nsaved predictions to {pdir}" + ("" if a.final_test else " (val only; add --final-test for the final run)"))
    print("A clearly higher score for 'colour_size' than for 'colour' indicates that photo size acts as a shortcut.")


if __name__ == "__main__":
    main()
