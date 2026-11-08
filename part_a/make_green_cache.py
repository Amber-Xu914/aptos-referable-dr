"""Create green-channel + CLAHE images from the raw APTOS photographs.

Uses
  1. Fallback input when the shared preprocessed cache is not available (the other scripts use it
     automatically with --cache auto).
  2. Reference images for check_part_c.py, which compares the shared cache against these to confirm
     that every file belongs to the correct image id.

Steps
  1. find the retina (pixels brighter than the black background) and crop to its bounding box
  2. pad to a square, so the round retina is not stretched into an oval
  3. resize to 512 x 512
  4. keep the green channel
  5. apply CLAHE (clip limit 2.0, 8 x 8 tiles)

Usage
  python part_a/make_green_cache.py            all images (a few minutes)
  python part_a/make_green_cache.py --n 50     first 50 images only

Output: data/own_cache/green_512/{id_code}.png (existing files are skipped)
"""
import argparse
import multiprocessing as mp
import os

import cv2
import numpy as np

from common import get_paths, index_images

SIZE = 512


def crop_and_square(bgr, thresh=10, pad=True):
    """Crop to the retina's bounding box, then pad (or, with pad=False, leave for stretching) to a square."""
    m = cv2.GaussianBlur(bgr.max(axis=2), (0, 0), 2) > thresh
    rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
    if rows.size and cols.size:
        bgr = bgr[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    h, w = bgr.shape[:2]
    if pad and h != w:
        s = max(h, w)
        out = np.zeros((s, s, 3), bgr.dtype)
        y, x = (s - h) // 2, (s - w) // 2
        out[y:y + h, x:x + w] = bgr
        bgr = out
    return bgr


def preprocess(bgr, pad=True, size=SIZE):
    """Raw BGR photo -> 512 x 512 green channel with CLAHE."""
    sq = crop_and_square(bgr, pad=pad)
    sq = cv2.resize(sq, (size, size), interpolation=cv2.INTER_AREA)
    g = sq[..., 1]                                           # OpenCV stores B, G, R, so index 1 is green
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g)


def _work(job):
    """Process one file; returns None on success or an error message."""
    src, dst = job
    try:
        cv2.setNumThreads(1)
        bgr = cv2.imread(src, cv2.IMREAD_COLOR)
        if bgr is None:
            return src
        cv2.imwrite(dst, preprocess(bgr))
        return None
    except Exception as e:
        return f"{src}: {e!r}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=None)
    ap.add_argument("--n", type=int, default=0, help="only the first N images (0 = all)")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    a = ap.parse_args()
    paths = get_paths(a.root)
    paths.green_own.mkdir(parents=True, exist_ok=True)
    idx = index_images(paths.raw_images)
    ids = sorted(idx)[: a.n or None]
    jobs = [(str(idx[i]), str(paths.green_own / f"{i}.png")) for i in ids
            if not (paths.green_own / f"{i}.png").exists()]
    print(f"{len(ids)} images requested, {len(jobs)} still to do -> {paths.green_own}")
    with mp.Pool(a.workers) as pool:
        fails = [r for r in pool.imap_unordered(_work, jobs, chunksize=4) if r]
    print("done" + (f"; {len(fails)} failed, e.g. {fails[:3]}" if fails else ""))


if __name__ == "__main__":
    main()
