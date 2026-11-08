"""Verify the shared data split and the preprocessed green-channel images against the raw APTOS data.

Usage
  python part_a/check_part_c.py
  python part_a/check_part_c.py --n-images 300      check more cached images (default 100)

Inputs:  data/raw/train.csv, data/raw/train_images/, data/part_c/splits.csv, data/part_c/cache_green_512/
Outputs: outputs/part_c_check/report.txt, compare_raw_vs_c.png (and suspicious_pairs.png if needed)

Every result line starts with PASS, WARN or FAIL.
  FAIL  an error that must be corrected before training
  WARN  worth following up or stating as a limitation

Checks
  splits.csv
    every image appears exactly once and its grade matches train.csv    (a faulty merge mislabels images)
    referable equals grade >= 2                                          (the shared prediction target)
    split sizes, and the same grade mix in every split                   (stratification)
    exact duplicate photos, found by MD5 hash of the raw files, each
      kept within one split                                              (leakage)
    label_conflict flags agree with the recomputed grade conflicts       (label-noise flag)
  cache_green_512
    one file per image, 512 x 512, single channel, 8-bit                 (format the code expects)
    black background                                                     (field-of-view detection)
    retina round, not stretched into an oval                             (resize distortion)
    each cached file shows the raw photo with the same id               (file-naming errors)
"""
import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import SEED, get_paths, index_images, load_green, load_labels, load_splits
from make_green_cache import preprocess

# report lines and result counts
LINES = []
COUNT = {"PASS": 0, "WARN": 0, "FAIL": 0}


def say(level, msg):
    """Record and print one PASS / WARN / FAIL result."""
    COUNT[level] += 1
    line = f"{level:4s}  {msg}"
    print(line)
    LINES.append(line)


def info(msg):
    """Record and print an informational line."""
    print("      " + msg)
    LINES.append("      " + msg)


def md5(path, block=1 << 20):
    """MD5 hash of a file's bytes (identical files give identical hashes)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            h.update(chunk)
    return h.hexdigest()


# ======================================================================================
# splits.csv
# ======================================================================================
def check_splits(paths, labels, raw_idx):
    print("\n=== splits.csv ===")
    sp = load_splits(paths.splits)
    raw_cols = pd.read_csv(paths.splits, nrows=0).columns.str.strip().tolist()
    info(f"columns: {raw_cols}")
    has_conflict = "label_conflict" in sp.columns
    (say("PASS", "label_conflict column present") if has_conflict else
     say("WARN", "no label_conflict column"))

    # ids and labels
    dup_ids = sp["id_code"][sp["id_code"].duplicated()].tolist()
    say("FAIL" if dup_ids else "PASS", f"each id appears once in splits.csv ({len(dup_ids)} repeated)")
    missing = set(labels["id_code"]) - set(sp["id_code"])
    extra = set(sp["id_code"]) - set(labels["id_code"])
    say("FAIL" if (missing or extra) else "PASS",
        f"ids match train.csv ({len(sp)} rows; {len(missing)} missing, {len(extra)} unknown)")

    m = sp.merge(labels[["id_code", "diagnosis"]], on="id_code", suffixes=("", "_raw"))
    bad = (m["diagnosis"] != m["diagnosis_raw"]).sum()
    say("FAIL" if bad else "PASS", f"grades in splits.csv match train.csv ({bad} differ)")
    bad_ref = (sp["referable"] != (sp["diagnosis"] >= 2).astype(int)).sum()
    say("FAIL" if bad_ref else "PASS", f"referable = (grade >= 2) for every row ({bad_ref} wrong)")

    # split sizes and stratification
    bad_split = set(sp["split"]) - {"train", "val", "test"}
    say("FAIL" if bad_split else "PASS", f"split values are train/val/test (other: {sorted(bad_split)})")
    n = sp["split"].value_counts()
    info("counts: " + ", ".join(f"{k} {n.get(k, 0)} ({n.get(k, 0) / len(sp):.1%})" for k in ("train", "val", "test")))
    tab = pd.crosstab(sp["split"], sp["diagnosis"], normalize="index").reindex(["train", "val", "test"])
    info("grade mix per split (row %):\n" + (tab * 100).round(1).to_string().replace("\n", "\n      "))
    spread = (tab.max() - tab.min()).max() * 100
    say("PASS" if spread < 2 else "WARN", f"grade mix is the same across splits (largest gap {spread:.1f} points)")
    ref = sp.groupby("split")["referable"].mean().reindex(["train", "val", "test"])
    info("referable %: " + ", ".join(f"{k} {v:.1%}" for k, v in ref.items()))

    # exact duplicates, recomputed from the raw files
    print("      hashing raw images to find exact duplicates (takes a minute)...")
    hashes = {i: md5(raw_idx[i]) for i in sp["id_code"] if i in raw_idx}
    sp["md5"] = sp["id_code"].map(hashes)
    g = sp[sp["md5"].notna()].groupby("md5")
    groups = g.filter(lambda d: len(d) > 1)
    n_groups = groups["md5"].nunique()
    info(f"recomputed: {n_groups} duplicate groups, {len(groups)} images (expected 123 groups, 251 images)")
    say("PASS" if (n_groups, len(groups)) == (123, 251) else "WARN", "duplicate counts match the expected values")
    agg = groups.groupby("md5").agg(n_split=("split", "nunique"), n_grade=("diagnosis", "nunique"),
                                    n_ref=("referable", "nunique"))
    crossing = int((agg["n_split"] > 1).sum())
    say("FAIL" if crossing else "PASS", f"every duplicate group is inside one split ({crossing} groups cross splits)")

    # duplicates with conflicting grades
    n_conf, n_conf_ref = int((agg["n_grade"] > 1).sum()), int((agg["n_ref"] > 1).sum())
    info(f"conflicting grades: {n_conf} groups ({n_conf_ref} across the referable line); expected 30 (6)")
    say("PASS" if (n_conf, n_conf_ref) == (30, 6) else "WARN", "conflict counts match the expected values")
    if has_conflict:
        conflict_md5 = set(agg.index[agg["n_grade"] > 1])
        expected = sp["md5"].isin(conflict_md5).astype(int)
        flagged = sp["label_conflict"].fillna(0).astype(int)
        diff = int((expected != flagged).sum())
        say("PASS" if diff == 0 else "WARN", f"label_conflict flags match recomputed conflicts ({diff} rows differ)")
        info("label_conflict images per split: " + sp[flagged == 1]["split"].value_counts().to_dict().__str__())
    return sp


# ======================================================================================
# cache_green_512: image helpers
# ======================================================================================
def fov_shape(g):
    """Width/height ratio of an ellipse fitted to the retina's curved edge (1.0 = round), and retina fraction."""
    m = cv2.GaussianBlur(g, (0, 0), 3) > 15
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if not rows.size:
        return np.nan, np.nan
    # Use only the curved edge: many photos have the top and bottom of the circle cut off by the
    # camera, so straight edges and points on the image border are ignored.
    cnt = max(cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0], key=len)
    pts = cnt[:, 0, :]
    H, W = g.shape
    keep = ((pts[:, 1] > rows[0] + 3) & (pts[:, 1] < rows[-1] - 3) &
            (pts[:, 0] > 2) & (pts[:, 0] < W - 3) & (pts[:, 1] > 2) & (pts[:, 1] < H - 3))
    pts = pts[keep]
    if len(pts) < 50:
        return np.nan, m.mean()
    (_, _), (a1, a2), ang = cv2.fitEllipse(pts.reshape(-1, 1, 2).astype(np.float32))
    # horizontal and vertical axis of the fitted ellipse
    horiz = a1 if (ang % 180) < 45 or (ang % 180) > 135 else a2
    vert = a2 if horiz == a1 else a1
    return horiz / max(vert, 1e-6), m.mean()


def _keypoints(g):
    """ORB keypoints (vessel junctions, lesions) away from the retina's rim, which every photo shares."""
    fov = (cv2.GaussianBlur(g, (0, 0), 2) > 15).astype(np.uint8)
    fov = cv2.erode(fov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41)))
    return cv2.ORB_create(2000, fastThreshold=5).detectAndCompute(g, fov)


def match_score(fa, fb):
    """Number of keypoints that agree with one geometric transform (shift, scale, slight stretch).

    The same photo cropped or resized slightly differently gives many matches; a different photo gives few.
    """
    (ka, da), (kb, db) = fa, fb
    if da is None or db is None or len(ka) < 10 or len(kb) < 10:
        return 0
    pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da, db, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < 0.8 * p[1].distance]   # ratio test
    if len(good) < 6:
        return 0
    src = np.float32([ka[m.queryIdx].pt for m in good])
    dst = np.float32([kb[m.trainIdx].pt for m in good])
    _, inl = cv2.estimateAffine2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    return int(inl.sum()) if inl is not None else 0


def raw_green(path):
    """A raw photo and an independently made green + CLAHE version of it at 512 px, for comparison."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return bgr, preprocess(bgr, pad=True)


# ======================================================================================
# cache_green_512: checks
# ======================================================================================
def check_cache(paths, labels, raw_idx, n_images, out_dir):
    print("\n=== cache_green_512 ===")
    if not paths.green_c.is_dir():
        say("FAIL", f"folder not found: {paths.green_c}")
        return
    c_idx = index_images(paths.green_c)
    exts = sorted({p.suffix for p in c_idx.values()})
    info(f"{len(c_idx)} image files, extension(s) {exts}")
    missing = set(labels["id_code"]) - set(c_idx)
    say("FAIL" if missing else "PASS", f"one cached image per APTOS image ({len(missing)} missing)")

    # measure a random sample of cached images
    rng = np.random.default_rng(SEED)
    ids = [str(i) for i in rng.permutation(sorted(set(c_idx) & set(raw_idx)))[:n_images]]
    shapes, dtypes, chans, bg, ratio, fovfrac = set(), set(), set(), [], [], []
    same, other, pairs, shown = [], [], {}, []
    feats_raw = {}
    for k, i in enumerate(ids):
        raw_c = cv2.imread(str(c_idx[i]), cv2.IMREAD_UNCHANGED)
        shapes.add(raw_c.shape[:2]); dtypes.add(str(raw_c.dtype)); chans.add(1 if raw_c.ndim == 2 else raw_c.shape[2])
        g = load_green(c_idx[i])
        corners = np.concatenate([g[:8, :8].ravel(), g[:8, -8:].ravel(), g[-8:, :8].ravel(), g[-8:, -8:].ravel()])
        bg.append(np.median(corners))
        r, f = fov_shape(g)
        ratio.append(r); fovfrac.append(f)
        # keypoint match between the cached file and the raw photo with the same id
        bgr, mine = raw_green(raw_idx[i])
        feats_raw[i] = _keypoints(mine)
        fc = _keypoints(g)
        same.append(match_score(fc, feats_raw[i]))
        pairs[i] = (cv2.cvtColor(cv2.resize(bgr, (512, int(512 * bgr.shape[0] / bgr.shape[1]))), cv2.COLOR_BGR2RGB), g, fc)
        if k < 4:
            shown.append(i)
    # reference level: each cached file against a different raw photo
    for k, i in enumerate(ids):
        j = ids[(k + 1) % len(ids)]
        other.append(match_score(pairs[i][2], feats_raw[j]))

    # format, background and shape
    info(f"checked {len(ids)} random images: sizes {sorted(shapes)}, dtype {sorted(dtypes)}, channels {sorted(chans)}")
    say("PASS" if shapes == {(512, 512)} else "WARN", "all checked images are 512 x 512")
    say("PASS" if chans == {1} else "WARN",
        "single-channel images" + ("" if chans == {1} else " (3 channels: the loader uses the green one)"))
    say("PASS" if dtypes == {"uint8"} else "WARN", "8-bit images")
    bgm = float(np.median(bg))
    say("PASS" if bgm <= 8 else "WARN", f"background is black (median corner value {bgm:.0f}; field-of-view threshold is 10)")
    rmed = float(np.nanmedian(ratio))
    info(f"retina width/height ratio: median {rmed:.2f} (range {np.nanmin(ratio):.2f}-{np.nanmax(ratio):.2f}); 1.00 = round")
    say("PASS" if 0.95 <= rmed <= 1.05 else "WARN",
        "retina is round" + ("" if 0.95 <= rmed <= 1.05 else
                             " - appears stretched (cropped then resized to a square); lesion shapes are distorted"))
    info(f"retina fills {np.nanmedian(fovfrac):.0%} of the image (median)")

    # file-to-photo matching
    o_max = int(np.max(other))
    info(f"matching keypoints with the raw photo of the SAME id: median {np.median(same):.0f} (min {min(same)}); "
         f"with a DIFFERENT photo: median {np.median(other):.0f} (max {o_max})")
    cut = max(15, 2 * o_max)
    mism = [str(i) for i, s_ in zip(ids, same) if s_ < cut]
    say("PASS" if not mism else "FAIL",
        f"each cached file matches its own raw photo ({len(mism)} of {len(ids)} below {cut} matches"
        f"{': ' + str(mism[:8]) if mism else ''})")
    if mism:
        info("see suspicious_pairs.png: if both pictures in a row show the same eye (same vessels), the file is")
        info("correct and the check was misled (e.g. a very blurred or dark photo); otherwise the file is mislabelled.")

    def sheet(id_list, name, title):
        """Side-by-side figure of raw photos and the cached files with the same names."""
        id_list = id_list[:8]
        fig, ax = plt.subplots(len(id_list), 2, figsize=(7, 3.4 * len(id_list)), squeeze=False)
        for r, i in enumerate(id_list):
            ax[r, 0].imshow(pairs[i][0]); ax[r, 0].set_title(f"raw {i}", fontsize=9)
            ax[r, 1].imshow(pairs[i][1], cmap="gray"); ax[r, 1].set_title(title, fontsize=9)
            for x in ax[r]:
                x.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / name, dpi=100)
        plt.close(fig)

    sheet(shown, "compare_raw_vs_c.png", "Part C cache_green_512")
    if mism:
        sheet(mism, "suspicious_pairs.png", "Part C file with the same name")

    # files that do not belong to any APTOS image (e.g. duplicated downloads named "(1)")
    extra = sorted(set(c_idx) - set(labels["id_code"]))
    if extra:
        say("WARN", f"{len(extra)} extra file(s) in cache_green_512 that are not APTOS ids, e.g. {extra[:5]}")
        info("not used by Part A (only ids listed in splits.csv are read)")

    clahe = paths.green_c.parent / "cache_clahe_512"
    if clahe.is_dir():
        n_c = len(index_images(clahe))
        say("PASS" if n_c >= len(labels) else "WARN", f"cache_clahe_512 (colour cache) has {n_c} images")


# ======================================================================================
# Command line
# ======================================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--n-images", type=int, default=100)
    a = ap.parse_args()
    paths = get_paths(a.root)
    out_dir = paths.root / "outputs" / "part_c_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    labels = load_labels(paths.raw_csv)
    raw_idx = index_images(paths.raw_images)
    info(f"raw data: {len(labels)} rows in train.csv, {len(raw_idx)} images; grades {labels['diagnosis'].value_counts().sort_index().to_dict()}")
    say("PASS" if set(labels["id_code"]) <= set(raw_idx) else "FAIL", "every train.csv id has a raw image")

    if paths.splits.exists():
        check_splits(paths, labels, raw_idx)
    else:
        say("FAIL", f"splits.csv not found at {paths.splits}")
    check_cache(paths, labels, raw_idx, a.n_images, out_dir)

    print(f"\n{COUNT['PASS']} PASS, {COUNT['WARN']} WARN, {COUNT['FAIL']} FAIL")
    (out_dir / "report.txt").write_text("\n".join(LINES) + f"\n\n{COUNT}\n")
    print(f"report: {out_dir / 'report.txt'}   picture: {out_dir / 'compare_raw_vs_c.png'}")
    sys.exit(1 if COUNT["FAIL"] else 0)


if __name__ == "__main__":
    main()
