"""Image-quality tags for error analysis (Part D).

Uses the same functions as Part C (src/preprocess.py) and the same measurements as
notebooks/C_preprocessing_demo.ipynb (aspect after border crop, mean brightness inside
the field of view), plus a sharpness measure. Each image gets a quality label so the
evaluation can ask: "does the model miss more referable cases in poor-quality photos?"

Measurements (on the original photo, resized to 512 px wide, like the Part C notebook):
    aspect      width / height after cropping the black border (> 1.05 = top/bottom cut off)
    brightness  mean grey level inside the field of view (0-255)
    sharpness   variance of the Laplacian of the green channel inside the field of view
                (low = blurry / out of focus)

Labels: cut-offs are set on the TRAINING split only (5th / 95th percentiles), then applied
unchanged to val and test, so the test set is not used to make any decision.
    underexposed  brightness < train 5th percentile
    overexposed   brightness > train 95th percentile
    blurry        sharpness  < train 5th percentile
    good          none of the above
    (first matching label wins, in that order; `truncated` is a separate yes/no column,
     because more than half of APTOS images are truncated and that alone is not poor quality)

Usage (from the repo root, e.g. in Colab after mounting the group Drive):
    python evaluation/image_quality.py \
        --images "data/raw/Aptos2019 retinal image database/train_images" \
        --splits splits.csv --out results/image_quality.csv

Then pass it to the evaluation:
    python evaluation/evaluate.py results.csv --splits splits.csv --quality results/image_quality.csv
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.preprocess import crop_black_border, fov_mask, load_rgb  # noqa: E402

LOW_PCT, HIGH_PCT = 5, 95
TRUNCATED_ASPECT = 1.05


def measure(path):
    img = load_rgb(path)
    small = cv2.resize(img, (512, int(round(512 * img.shape[0] / img.shape[1]))), interpolation=cv2.INTER_AREA)
    crop = crop_black_border(small)
    aspect = crop.shape[1] / crop.shape[0]
    mask = fov_mask(small, erode_frac=0.02)          # shrink a little so the FOV edge is not counted as "sharp"
    if mask.sum() < 100:                              # almost black image
        return aspect, float("nan"), float("nan")
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    brightness = float(gray[mask].mean())
    lap = cv2.Laplacian(small[:, :, 1].astype(np.float64), cv2.CV_64F)
    sharpness = float(lap[mask].var())
    return aspect, brightness, sharpness


def label(row, cut):
    if np.isnan(row.brightness) or row.brightness < cut["brightness_low"]:
        return "underexposed"
    if row.brightness > cut["brightness_high"]:
        return "overexposed"
    if row.sharpness < cut["sharpness_low"]:
        return "blurry"
    return "good"


def main():
    p = argparse.ArgumentParser(description="Tag APTOS images by quality (cut-offs from train only)")
    p.add_argument("--images", required=True, help="Folder with the ORIGINAL APTOS images (<id_code>.png)")
    p.add_argument("--splits", default=str(ROOT / "splits.csv"))
    p.add_argument("--out", default=str(ROOT / "results" / "image_quality.csv"))
    p.add_argument("--train-sample", type=int, default=0,
                   help="Measure only this many random train images to set cut-offs (0 = all train images)")
    args = p.parse_args()

    splits = pd.read_csv(args.splits)
    train = splits[splits.split == "train"]
    if args.train_sample:
        train = train.sample(min(args.train_sample, len(train)), random_state=9027)
    todo = pd.concat([train, splits[splits.split.isin(["val", "test"])]])

    rows = []
    for k, r in enumerate(todo.itertuples(), 1):
        a, b, s = measure(Path(args.images) / f"{r.id_code}.png")
        rows.append({"id_code": r.id_code, "split": r.split, "diagnosis": r.diagnosis,
                     "aspect": round(a, 3), "brightness": round(b, 2), "sharpness": round(s, 2)})
        if k % 250 == 0:
            print(f"  measured {k}/{len(todo)}")
    q = pd.DataFrame(rows)

    tr = q[q.split == "train"]
    cut = {"brightness_low": float(np.nanpercentile(tr.brightness, LOW_PCT)),
           "brightness_high": float(np.nanpercentile(tr.brightness, HIGH_PCT)),
           "sharpness_low": float(np.nanpercentile(tr.sharpness, LOW_PCT)),
           "set_on": f"train split ({len(tr)} images), {LOW_PCT}th/{HIGH_PCT}th percentiles"}
    q["image_quality"] = q.apply(label, axis=1, cut=cut)
    q["truncated"] = (q.aspect > TRUNCATED_ASPECT).astype(int)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    q.to_csv(out, index=False)
    with open(out.with_suffix(".cutoffs.json"), "w") as f:
        json.dump(cut, f, indent=2)

    print("\nCut-offs (from train only):", json.dumps(cut, indent=2))
    print("\nQuality label counts:")
    print(pd.crosstab(q.split, q.image_quality, margins=True))
    print("\nTruncated field of view (fraction):")
    print(q.groupby("split").truncated.mean().round(3))
    print("\nReferable fraction by quality (val + test):")
    vt = q[q.split != "train"]
    print(vt.assign(referable=vt.diagnosis >= 2).groupby("image_quality").referable.agg(["count", "mean"]).round(3))
    print(f"\nSaved {out} and {out.with_suffix('.cutoffs.json')}")


if __name__ == "__main__":
    main()
