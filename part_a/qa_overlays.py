"""Visual quality assurance (QA) of the lesion detectors.

APTOS has no lesion outlines, so the detectors can only be checked by looking at what they find.
This script draws the detections on sample images and saves one contact sheet (a grid of examples)
per grade: outputs/part_a/qa/qa_grade0.png ... qa_grade4.png.

Each row shows one image in four panels:
  1. input: yellow = retina boundary, magenta = optic disc, cyan = fovea (both excluded from detection)
  2. vessel mask (blue): should follow the vessel tree, not lesions
  3. dark lesions (red circles): should sit on red dots and blots, not on vessels, the disc or the rim
  4. bright lesions (yellow circles): should sit on exudates, not on the disc, the rim or reflections

Usage
  python part_a/qa_overlays.py                                   4 training images per grade
  python part_a/qa_overlays.py --set dark_thr=15 bright_thr=25   try other detector settings
  python part_a/qa_overlays.py --fov-check 200                   also print field-of-view statistics
  python part_a/qa_overlays.py --ids 03a7f4a5786f 247e98aba610 --name missed_grade4   chosen images
"""
import argparse

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import green_dir, SEED, get_paths, index_images, load_green, load_labels, load_splits, parse_overrides
from detectors import make_params, run_detectors


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------
def draw_circles(img, L, colour):
    """Draw a circle slightly larger than each lesion."""
    for cy, cx, a in zip(L.cy, L.cx, L.area):
        r = int(np.sqrt(a / np.pi)) + 5
        cv2.circle(img, (int(cx), int(cy)), r, colour, 1, cv2.LINE_AA)


def panels(g, det):
    """The four RGB panels for one image (input with anatomy, vessels, dark lesions, bright lesions)."""
    base = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

    # panel 1: retina boundary, optic disc and fovea outlines
    p1 = base.copy()
    cnt, _ = cv2.findContours(det.fov.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(p1, cnt, -1, (255, 255, 0), 2)
    if det.disc_xy is not None:
        cnt, _ = cv2.findContours(det.disc.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(p1, cnt, -1, (255, 0, 255), 2)
    if det.fovea_xy is not None:
        cnt, _ = cv2.findContours(det.fovea.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(p1, cnt, -1, (0, 255, 255), 2)

    # panel 2: vessel mask blended in blue
    p2 = base.copy()
    p2[det.vessels] = (0.5 * p2[det.vessels] + 0.5 * np.array([0, 90, 255])).astype(np.uint8)

    # panels 3 and 4: lesion circles
    p3 = base.copy()
    draw_circles(p3, det.dark, (255, 40, 40))
    p4 = base.copy()
    draw_circles(p4, det.bright, (255, 255, 0))
    return [p1, p2, p3, p4]


# ---------------------------------------------------------------------------
# Field-of-view statistics
# ---------------------------------------------------------------------------
def fov_check(df, index, params, n, seed):
    """Print retina size, scale and background brightness for n random images, and list unusual ones."""
    rows = []
    for id_code in df.sample(min(n, len(df)), random_state=seed)["id_code"]:
        g = load_green(index[id_code])
        det = run_detectors(g, params)
        outside = g[~det.fov]
        rows.append({"id_code": id_code, "h": g.shape[0], "w": g.shape[1],
                     "fov_frac": det.fov.mean(), "scale": det.s,
                     "outside_median": float(np.median(outside)) if outside.size else np.nan,
                     "outside_max": float(outside.max()) if outside.size else np.nan,
                     "disc_x": det.disc_xy[0] if det.disc_xy else np.nan})
    r = pd.DataFrame(rows)
    print("\nFOV check on", len(r), "images")
    print(r[["fov_frac", "scale", "outside_median", "outside_max"]].describe().loc[["min", "50%", "max"]].round(3))
    odd = r[(r.fov_frac < 0.5) | (r.fov_frac > 0.95) | (r.scale < 0.85) | (r.scale > 1.15)]
    print(f"{len(odd)} image(s) with an unusual FOV (fraction <0.5 or >0.95, or scale far from 1):",
          odd["id_code"].tolist()[:10])
    print("image sizes seen:", sorted(set(zip(r.h, r.w)))[:5])


# ---------------------------------------------------------------------------
# Contact sheet for a chosen list of images
# ---------------------------------------------------------------------------
def draw_sheet(ids, labels, index, params, out):
    """Save one contact sheet with a row per image id."""
    grade = dict(zip(labels["id_code"], labels["diagnosis"]))
    fig, axes = plt.subplots(len(ids), 4, figsize=(17, 4.3 * len(ids)), squeeze=False)
    for r, id_code in enumerate(ids):
        g = load_green(index[id_code])
        det = run_detectors(g, params)
        burden = (det.dark.mask | det.bright.mask).sum() / max(det.fov_inner.sum(), 1) * 100
        titles = [f"grade {grade.get(id_code, '?')} | {id_code} | all lesions {burden:.2f}% of retina",
                  f"vessels {det.vessels.mean() * 100:.1f}% of image",
                  f"dark lesions n={det.dark.n}", f"bright lesions n={det.bright.n}"]
        for c, (im, t) in enumerate(zip(panels(g, det), titles)):
            axes[r, c].imshow(im)
            axes[r, c].set_title(t, fontsize=10)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print("saved", out)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--cache", choices=["auto", "c", "own"], default="auto",
                    help="green images: Part C's cache (c), the own cache (own), or Part C's if present (auto)")
    ap.add_argument("--split", default="train", help="split to sample from (use train while adjusting settings)")
    ap.add_argument("--per-grade", type=int, default=4)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--set", nargs="*", default=[], metavar="NAME=VALUE")
    ap.add_argument("--fov-check", type=int, default=0, metavar="N", help="print FOV statistics for N images")
    ap.add_argument("--ids", nargs="+", default=None, help="draw these specific images (any split)")
    ap.add_argument("--name", default="qa_selected", help="file name for the --ids sheet")
    a = ap.parse_args()

    # load labels and images (train.csv is used if splits.csv is not available yet)
    paths = get_paths(a.root)
    qa = paths.out / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    params = make_params(parse_overrides(a.set))
    if paths.splits.exists():
        df = load_splits(paths.splits)
        df = df[df["split"] == a.split]
    else:
        print(f"NOTE: {paths.splits} not found - sampling from all of train.csv instead of the train split")
        df = load_labels(paths.raw_csv)
    index = index_images(green_dir(paths, a.cache))
    all_df = (load_splits(paths.splits) if paths.splits.exists() else load_labels(paths.raw_csv))
    df = df[df["id_code"].isin(index)]
    if a.fov_check:
        fov_check(df, index, params, a.fov_check, a.seed)

    # chosen images only
    if a.ids:
        draw_sheet([i for i in a.ids if i in index], all_df, index, params, qa / f"{a.name}.png")
        return

    # one sheet per grade with randomly sampled images
    for grade in sorted(df["diagnosis"].unique()):
        d = df[df["diagnosis"] == grade]
        pick = d.sample(min(a.per_grade, len(d)), random_state=a.seed)["id_code"].tolist()
        fig, axes = plt.subplots(len(pick), 4, figsize=(17, 4.3 * len(pick)), squeeze=False)
        for r, id_code in enumerate(pick):
            g = load_green(index[id_code])
            det = run_detectors(g, params)
            burden = (det.dark.mask | det.bright.mask).sum() / max(det.fov_inner.sum(), 1) * 100
            titles = [f"grade {grade} | {id_code} | all lesions {burden:.2f}% of retina",
                      f"vessels {det.vessels.mean() * 100:.1f}% of image",
                      f"dark lesions n={det.dark.n}", f"bright lesions n={det.bright.n}"]
            for c, (im, t) in enumerate(zip(panels(g, det), titles)):
                axes[r, c].imshow(im)
                axes[r, c].set_title(t, fontsize=10)
                axes[r, c].axis("off")
        fig.tight_layout()
        out = qa / f"qa_grade{grade}.png"
        fig.savefig(out, dpi=110)
        plt.close(fig)
        print("saved", out)


if __name__ == "__main__":
    main()
