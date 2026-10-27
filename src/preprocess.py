"""Shared fundus-image preprocessing for the APTOS referable-DR project (Part C).

All modes share the same geometry: crop the black border, pad to a square,
resize, and zero everything outside the camera's field of view. They differ
only in how contrast is handled:

    rgb         geometry only (control condition)
    green       green channel + CLAHE, single channel (classical pipeline, A)
    clahe_rgb   CLAHE on the LAB lightness channel, back to RGB (CNN, B)
    ben_graham  colour image minus its Gaussian blur, re-centred on grey

Outputs are uint8 so they can be cached as PNG. ImageNet normalisation for the
CNN is applied separately with `normalize`, after augmentation.

Usage:
    from src.preprocess import preprocess
    img = preprocess("train_images/000c1434d8d7.png", size=512, mode="clahe_rgb")

    # Cache a whole folder once so nobody recomputes it:
    python -m src.preprocess IN_DIR OUT_DIR --size 512 --mode clahe_rgb
"""

import argparse
from pathlib import Path

import cv2
import numpy as np

MODES = ("rgb", "green", "clahe_rgb", "ben_graham")

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def load_rgb(path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def crop_black_border(img, tol=7):
    """Crop to the bounding box of pixels brighter than `tol`.

    Falls back to the original image if the threshold removes everything
    (e.g. a very dark, underexposed photo).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    mask = gray > tol
    if not mask.any():
        return img
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    return img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]


def pad_to_square(img):
    """Pad with black so the fundus is not stretched by a later resize."""
    h, w = img.shape[:2]
    side = max(h, w)
    top = (side - h) // 2
    left = (side - w) // 2
    return cv2.copyMakeBorder(
        img, top, side - h - top, left, side - w - left,
        cv2.BORDER_CONSTANT, value=0,
    )


def fov_mask(img, tol=7, erode_frac=0.0):
    """Boolean mask of the camera's field of view (the visible fundus).

    Many APTOS photos have the top and bottom of the circle cut off, so after
    padding a geometric circle would include black bars; thresholding follows
    the real boundary instead. `erode_frac` shrinks it by a fraction of the width.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if img.ndim == 3 else img
    mask = (gray > tol).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    k = int(round(img.shape[0] * erode_frac))
    if k > 0:
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * k + 1, 2 * k + 1)))
    return mask.astype(bool)


def apply_clahe(channel, clip_limit=2.0, tile_grid=8):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_grid, tile_grid))
    return clahe.apply(channel)


def ben_graham(img, mask, sigma_frac=1 / 30):
    """Local-average colour subtraction (Ben Graham, Kaggle DR 2015).

    Removes slow illumination changes and highlights small lesions.
    `sigma_frac` sets the blur width relative to image size. The local average
    is taken over the field of view only (blur(img*mask) / blur(mask)), so the
    black background does not create a bright halo at the edge.
    """
    sigma = img.shape[0] * sigma_frac
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(img.astype(np.float32) * m[:, :, None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)[:, :, None]
    blur = num / np.maximum(den, 1e-3)
    out = 4 * img.astype(np.float32) - 4 * blur + 128
    return np.clip(out, 0, 255).astype(np.uint8)


def preprocess(path_or_img, size=512, mode="clahe_rgb", clip_limit=2.0, tile_grid=8):
    """Load (if given a path) and preprocess one fundus image.

    Returns uint8, shape (size, size, 3) for colour modes or (size, size) for "green".
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")

    img = load_rgb(path_or_img) if isinstance(path_or_img, (str, Path)) else path_or_img
    img = crop_black_border(img)
    img = pad_to_square(img)
    img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    mask = fov_mask(img)

    if mode == "green":
        out = apply_clahe(img[:, :, 1], clip_limit, tile_grid)
    elif mode == "clahe_rgb":
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = apply_clahe(lab[:, :, 0], clip_limit, tile_grid)
        out = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    elif mode == "ben_graham":
        out = ben_graham(img, mask)
    else:
        out = img

    # A slightly eroded mask hides the thin bright rim that enhancement leaves at the edge.
    if mode != "rgb":
        mask = fov_mask(img, erode_frac=0.01)
    out = out.copy()
    out[~mask] = 0
    return out


def normalize(img):
    """uint8 RGB -> float32 HWC with ImageNet mean/std, for pretrained CNNs.

    Single-channel input is repeated to three channels first.
    """
    x = img.astype(np.float32) / 255.0
    if x.ndim == 2:
        x = np.repeat(x[:, :, None], 3, axis=2)
    return (x - IMAGENET_MEAN) / IMAGENET_STD


def cache_folder(in_dir, out_dir, size=512, mode="clahe_rgb", exts=(".png", ".jpg", ".jpeg")):
    """Preprocess every image in `in_dir` and write PNGs with the same stem to `out_dir`."""
    in_dir, out_dir = Path(in_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in in_dir.iterdir() if p.suffix.lower() in exts)
    for i, p in enumerate(paths, 1):
        out = preprocess(p, size=size, mode=mode)
        if out.ndim == 3:
            out = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
        cv2.imwrite(str(out_dir / f"{p.stem}.png"), out)
        if i % 200 == 0 or i == len(paths):
            print(f"{i}/{len(paths)} done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache preprocessed fundus images.")
    parser.add_argument("in_dir")
    parser.add_argument("out_dir")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--mode", choices=MODES, default="clahe_rgb")
    args = parser.parse_args()
    cache_folder(args.in_dir, args.out_dir, size=args.size, mode=args.mode)
