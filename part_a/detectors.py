"""Classical lesion detectors for retinal fundus images.

Input:  a 2-D uint8 green-channel image (green channel + CLAHE, 512 x 512).
Output: masks and measurements for the field of view (FOV), optic disc, fovea, vessels,
        dark lesions (microaneurysms, haemorrhages) and bright lesions (exudates, cotton-wool spots).

Processing order (each step uses the results of the steps before it):
    FOV -> scale -> local background -> raw vessel mask -> main vessel tree -> optic disc
        -> fovea -> large dark blots -> bright lesions -> cleaned vessel mask -> small dark lesions

Design rules
  1. Thresholds are fixed contrasts in gray levels. CLAHE puts every photograph on a similar
     contrast scale, so "12 gray levels darker than the surroundings" means the same in every image.
     Thresholds set relative to each image (percentiles, or multiples of the image's own noise level)
     give clean, healthy images the lowest thresholds and therefore the most detections, so they are
     not used. A noise guard (k x the image's noise level) only stops very grainy images being flooded.
  2. All sizes are given in pixels for a 512-pixel-wide retina and multiplied by the scale factor
     s = FOV diameter / 512, so the same settings work for retinas of slightly different size.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage import measure
from skimage.filters import frangi, threshold_otsu

cv2.setNumThreads(1)        # one thread per process; parallelism comes from the worker pool


# ======================================================================================
# Parameters
# Every value can be overridden from the command line with  --set name=value
# Sizes are in pixels at 512 px; contrasts are in gray levels (0-255).
# ======================================================================================
@dataclass
class Params:
    # --- field of view and background ---
    fov_thresh: float = 10.0          # a blurred pixel brighter than this counts as retina
    fov_erode: float = 10.0           # margin ignored inside the retina's edge (bright rim artefact)
    bg_ksize: float = 51.0            # median-filter window for the local background

    # --- optic disc: bright, and where the vessels converge ---
    use_disc_mask: bool = True
    disc_blur_sigma: float = 14.0     # blur used to find the bright disc region
    disc_vessel_weight: float = 1.0   # weight of vessel density relative to brightness
    disc_radius: float = 45.0         # radius of the masked disc circle

    # --- fovea: normal dark centre of the macula, excluded from dark-lesion detection ---
    use_fovea_mask: bool = True
    fovea_dist: tuple = (130.0, 240.0)  # search band: distance from the disc centre
    fovea_max_dy: float = 80.0        # search band: maximum vertical offset from the disc
    fovea_radius: float = 22.0        # radius of the masked fovea circle

    # --- vessels: Frangi vesselness for dark tubular structures ---
    vessel_sigmas: tuple = (1.5, 2.5, 3.5)   # vessel widths the filter responds to
    vessel_thresh: float = 0.10       # keep pixels above this fraction of the image's strongest response
    vessel_min_area: float = 60.0     # drop vessel fragments smaller than this
    vessel_dilate: float = 2.5        # margin around vessels excluded from dark-lesion detection
    vessel_tree_min_area: float = 1500.0  # vessel pieces at least this large form the main vessel tree

    # --- dark lesions: microaneurysms and haemorrhages ---
    dark_smooth: float = 1.0          # light blur before detection
    dark_line_len: float = 15.0       # small path: structures shorter than this in every direction count
    dark_thr: float = 12.0            # small path: minimum darkness (chosen by validation sweep)
    dark_large_ksize: float = 101.0   # large path: background window
    dark_large_thr: float = 22.0      # large path: minimum darkness
    dark_large_min_area: float = 80.0 # large path: minimum area
    dark_large_open_r: float = 5.0    # large path: blots must be about 2r+1 px wide (wider than vessels)
    dark_noise_k: float = 4.0         # noise guard: threshold at least k x the image's noise level
    dark_close_r: float = 1.0         # closing that joins fragments of one lesion
    dark_min_area: float = 4.0        # minimum lesion area
    dark_max_aspect: float = 3.0      # maximum length/width for small lesions (drops vessel pieces)

    # --- bright lesions: hard exudates and cotton-wool spots ---
    bright_smooth: float = 1.0        # light blur before detection
    bright_thr: float = 20.0          # minimum brightness above background (chosen by validation sweep)
    bright_noise_k: float = 4.0       # noise guard, as for dark lesions
    bright_rim: float = 20.0          # extra margin ignored at the retina's edge (bright halo)
    bright_disc_margin: float = 12.0  # extra margin ignored around the optic disc
    bright_dark_margin: float = 6.0   # margin ignored around large dark blots
    bright_open_r: float = 1.0        # opening removes 1-2 px specks and thin reflexes
    bright_close_r: float = 2.0       # closing joins fragments of one lesion
    bright_min_area: float = 8.0      # minimum lesion area
    bright_max_aspect: float = 4.0    # maximum length/width (drops streak-like reflexes)
    bright_vessel_frac: float = 0.3   # drop spots with more than this fraction along a main vessel
                                      # (light reflex of the nerve-fibre layer, common in young eyes)

    # --- feature settings (areas in px at 512 px) ---
    small_max_area: float = 25.0      # upper size of a microaneurysm-sized lesion
    medium_max_area: float = 150.0    # upper size of a dot/blot haemorrhage
    quad_min_count: int = 5           # lesions needed for a quadrant to count as "active"

    def __post_init__(self):
        self.vessel_sigmas = tuple(self.vessel_sigmas)
        self.fovea_dist = tuple(self.fovea_dist)


def make_params(overrides=None) -> Params:
    """Default parameters with optional overrides; unknown names raise an error."""
    overrides = overrides or {}
    valid = {f.name for f in dataclasses.fields(Params)}
    bad = set(overrides) - valid
    if bad:
        raise KeyError(f"Unknown parameter(s) {sorted(bad)}. Valid names: {sorted(valid)}")
    return Params(**overrides)


# ======================================================================================
# Morphology and mask helpers
# ======================================================================================
def disk(r) -> np.ndarray:
    """Disc-shaped structuring element of radius r."""
    r = max(1, int(round(r)))
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))


def line_kernels(length, n_angles=12):
    """Thin line structuring elements at n_angles orientations (used by the linear top-hat)."""
    L = max(3, int(round(length)) | 1)
    c = L // 2
    out = []
    for a in np.arange(n_angles) * np.pi / n_angles:
        k = np.zeros((L, L), np.uint8)
        dx, dy = np.cos(a) * c, np.sin(a) * c
        cv2.line(k, (int(round(c - dx)), int(round(c - dy))), (int(round(c + dx)), int(round(c + dy))), 1, 1)
        out.append(k)
    return out


def _odd(k):
    """Nearest odd integer of at least 3 (median filters need odd window sizes)."""
    k = max(3, int(round(k)))
    return k + (k % 2 == 0)


def _erode(mask, r):
    return mask.copy() if r <= 0 else cv2.erode(mask.astype(np.uint8), disk(r)).astype(bool)


def _dilate(mask, r):
    return mask.copy() if r <= 0 else cv2.dilate(mask.astype(np.uint8), disk(r)).astype(bool)


def _largest(mask):
    """Keep only the largest connected component of a mask."""
    lab, n = ndi.label(mask)
    if n <= 1:
        return mask
    sizes = ndi.sum(mask, lab, index=np.arange(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def _noise(x):
    """Robust noise level of a filter response: median absolute deviation scaled to a standard deviation."""
    if x.size == 0:
        return 1.0
    med = float(np.median(x))
    return max(1.4826 * float(np.median(np.abs(x - med))), 0.5)


def _remove_small(mask, min_area):
    """Remove connected components smaller than min_area pixels."""
    lab, n = ndi.label(mask)
    if n == 0:
        return mask
    keep = np.bincount(lab.ravel()) >= min_area
    keep[0] = False
    return keep[lab]


def _clean(mask, open_r, close_r):
    """Morphological opening (removes specks) followed by closing (fills small gaps)."""
    m = mask.astype(np.uint8)
    if open_r > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(open_r))
    if close_r > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, disk(close_r))
    return m.astype(bool)


def _blur(x, sigma):
    """Gaussian blur returning float32."""
    return cv2.GaussianBlur(x.astype(np.float32), (0, 0), max(float(sigma), 0.5))


# ======================================================================================
# Connected-component measurement
# ======================================================================================
@dataclass
class Lesions:
    """Detected lesions of one type: a mask plus one value per lesion."""
    mask: np.ndarray
    area: np.ndarray       # pixels
    cy: np.ndarray         # centroid row
    cx: np.ndarray         # centroid column
    aspect: np.ndarray     # major axis / minor axis
    contrast: np.ndarray   # mean detector response inside the lesion (gray levels)

    @property
    def n(self) -> int:
        return int(self.area.size)

    @classmethod
    def empty(cls, shape):
        z = np.zeros(0)
        return cls(np.zeros(shape, bool), z, z, z, z, z)


def extract_components(mask, intensity, min_area, max_aspect=None, aspect_max_area=None) -> Lesions:
    """Label connected components, measure them, and drop tiny or elongated ones.

    If aspect_max_area is given, the elongation filter applies only to components smaller than it,
    so large irregular lesions are kept.
    """
    lab = measure.label(mask, connectivity=2)
    n = int(lab.max())
    if n == 0:
        return Lesions.empty(mask.shape)
    t = measure.regionprops_table(
        lab, properties=("label", "area", "centroid", "major_axis_length", "minor_axis_length"))
    labels = t["label"]
    area = t["area"].astype(float)
    aspect = np.maximum(t["major_axis_length"], 1.0) / np.maximum(t["minor_axis_length"], 1.0)

    # size and shape filters
    keep = area >= min_area
    if max_aspect is not None:
        ok = aspect <= max_aspect
        if aspect_max_area is not None:
            ok |= area >= aspect_max_area
        keep &= ok
    if not keep.any():
        return Lesions.empty(mask.shape)

    # mean response per component, and a mask of the kept components only
    contrast = np.atleast_1d(ndi.mean(intensity, labels=lab, index=labels))
    keep_lab = np.zeros(n + 1, bool)
    keep_lab[labels[keep]] = True
    return Lesions(keep_lab[lab], area[keep], t["centroid-0"][keep], t["centroid-1"][keep],
                   aspect[keep], contrast[keep])


# ======================================================================================
# Field of view
# ======================================================================================
def compute_fov(g, p: Params):
    """Retina mask: pixels brighter than the black background, holes filled, largest region kept."""
    blur = cv2.GaussianBlur(g, (0, 0), 3)
    m = _largest(ndi.binary_fill_holes(blur > p.fov_thresh))
    if m.mean() < 0.2:                      # fixed threshold failed (very dark image): use Otsu
        try:
            m = _largest(ndi.binary_fill_holes(blur > threshold_otsu(blur)))
        except ValueError:
            pass
    if m.mean() < 0.05:                     # still nothing found: use the whole image
        m = np.ones(g.shape, bool)
    return m


def fov_geometry(fov):
    """Scale factor s (FOV diameter / 512), FOV centre (row, col) and FOV radius."""
    rows = np.flatnonzero(fov.any(axis=1))
    cols = np.flatnonzero(fov.any(axis=0))
    diam = float(max(rows[-1] - rows[0] + 1, cols[-1] - cols[0] + 1))
    s = float(np.clip(diam / 512.0, 0.5, 2.0))
    centre = ((rows[0] + rows[-1]) / 2.0, (cols[0] + cols[-1]) / 2.0)
    return s, centre, diam / 2.0


def fill_outside(g, fov):
    """Replace pixels outside the FOV with the nearest retina pixel, so filters see no hard black edge."""
    if fov.all():
        return g
    idx = ndi.distance_transform_edt(~fov, return_distances=False, return_indices=True)
    return np.ascontiguousarray(g[idx[0], idx[1]])


# ======================================================================================
# Anatomy: vessels, optic disc, fovea
# ======================================================================================
def compute_vessels(g_f, fov_inner, s, p: Params):
    """Vessel mask from Frangi vesselness (dark ridges).

    The threshold is a fraction of the image's strongest vessel response (99.5th percentile).
    Returns the mask and the raw vesselness map.
    """
    gf = g_f.astype(np.float32) / 255.0
    v = frangi(gf, sigmas=[x * s for x in p.vessel_sigmas], black_ridges=True)
    v = np.nan_to_num(v).astype(np.float32)
    v[~fov_inner] = 0
    vals = v[fov_inner]
    ref = float(np.percentile(vals, 99.5)) if vals.size else 0.0
    if ref <= 0:
        return np.zeros(g_f.shape, bool), v
    m = _remove_small(v > p.vessel_thresh * ref, max(p.vessel_min_area * s * s, 1))
    return m, v


def main_tree(vessels_raw, s, p: Params):
    """Thick parts of the large connected vessel pieces.

    Small Frangi fragments are left out; these often come from the dark gaps between clustered
    exudates or from the edges of haemorrhages rather than from real vessels.
    """
    tree = _remove_small(vessels_raw, p.vessel_tree_min_area * s * s)
    return cv2.morphologyEx(tree.astype(np.uint8), cv2.MORPH_OPEN, disk(2 * s)).astype(bool)


def compute_disc(g_f, fov, vessels_raw, s, p: Params):
    """Optic disc: the location that is both bright and where vessels converge.

    Score = standardised brightness (heavily blurred) + weight x standardised vessel density.
    Brightness alone is fooled by exudate patches; vessel density from the full vessel mask is more
    reliable than from the main tree only. Dense exudate clusters can still attract the disc.
    Returns a circular mask and the disc centre (x, y).
    """
    empty = np.zeros(g_f.shape, bool)
    if not p.use_disc_mask:
        return empty, None
    search = _erode(fov, 8 * s)
    if search.sum() < 1000:
        search = fov
    br = _blur(g_f, p.disc_blur_sigma * s)
    vd = _blur(vessels_raw, 20 * s)

    def z(x):
        v = x[search]
        return (x - v.mean()) / (v.std() + 1e-6)

    score = z(br) + p.disc_vessel_weight * z(vd)
    score[~search] = -1e9
    y, x = np.unravel_index(int(np.argmax(score)), score.shape)
    mask = np.zeros(g_f.shape, np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(round(p.disc_radius * s)), 1, -1)
    return mask.astype(bool), (int(x), int(y))


def compute_fovea(g_f, fov_inner, disc_xy, s, p: Params):
    """Fovea: the strongest local dark dip in a band beside the optic disc.

    The fovea lies about 2.5 disc diameters from the disc and roughly level with it. Local darkness
    (small blur minus large blur) is used so that dark corners and vignetting are ignored.
    Returns a circular mask and the fovea centre (x, y).
    """
    empty = np.zeros(g_f.shape, bool)
    if not p.use_fovea_mask or disc_xy is None:
        return empty, None
    # search band around the disc
    yy, xx = np.mgrid[:g_f.shape[0], :g_f.shape[1]]
    d = np.hypot(xx - disc_xy[0], yy - disc_xy[1])
    ring = ((d >= p.fovea_dist[0] * s) & (d <= p.fovea_dist[1] * s)
            & (np.abs(yy - disc_xy[1]) <= p.fovea_max_dy * s) & _erode(fov_inner, 20 * s))
    if ring.sum() < 500:
        return empty, None
    # darkest local dip inside the band
    sm = _blur(g_f, 8 * s) - _blur(g_f, 40 * s)
    sm[~ring] = 1e9
    y, x = np.unravel_index(int(np.argmin(sm)), sm.shape)
    mask = np.zeros(g_f.shape, np.uint8)
    cv2.circle(mask, (int(x), int(y)), int(round(p.fovea_radius * s)), 1, -1)
    return mask.astype(bool), (int(x), int(y))


# ======================================================================================
# Lesion detectors
# ======================================================================================
def detect_bright(g_f, bg, fov_inner, disc, tree, large_dark, s, p: Params) -> Lesions:
    """Bright lesions (exudates, cotton-wool spots).

    A pixel is a candidate if it is clearly brighter than the local background. Candidates are
    cleaned with opening and closing, and spots that are streak-shaped or lie along a main vessel
    (light reflex) are dropped.

    The background is re-estimated after lifting dark structures up to the background level, so normal
    retina between large haemorrhages does not appear bright only because its surroundings are dark.
    Pixels right next to large dark blots are skipped for the same reason.
    """
    # brightness relative to a background that ignores dark structures
    gs = _blur(g_f, p.bright_smooth * s)
    lifted = np.maximum(g_f, np.clip(bg, 0, 255).astype(np.uint8))
    bg_b = cv2.medianBlur(lifted, _odd(p.bg_ksize * s)).astype(np.float32)
    diff = gs - bg_b

    # region searched: away from the rim, the optic disc and large dark blots
    valid = (_erode(fov_inner, p.bright_rim * s) & ~_dilate(disc, p.bright_disc_margin * s)
             & ~_dilate(large_dark, p.bright_dark_margin * s))
    if valid.sum() < 1000:
        return Lesions.empty(g_f.shape)

    # threshold, then opening and closing
    thr = max(p.bright_thr, p.bright_noise_k * _noise(diff[valid]))
    cand = _clean((diff > thr) & valid, p.bright_open_r * s, p.bright_close_r * s) & valid

    # drop spots lying mostly along the main vessel tree (light reflex, not exudate)
    if p.bright_vessel_frac < 1:
        near = _dilate(tree, 8 * s)
        lab, n = ndi.label(cand)
        if n:
            frac = ndi.mean(near.astype(np.float32), labels=lab, index=np.arange(1, n + 1))
            drop = np.concatenate([[True], np.atleast_1d(frac) > p.bright_vessel_frac])
            cand = ~drop[lab]
    return extract_components(cand, diff, p.bright_min_area * s * s, p.bright_max_aspect)


def detect_large_dark(g_f, fov_inner, disc, fovea, s, p: Params):
    """Large haemorrhages: darker than a wide background, and wide.

    Blots must survive an opening with a disc of radius dark_large_open_r, which removes every
    vessel because vessels are thinner. Returns the blot mask and the darkness response.
    """
    gs = _blur(g_f, p.dark_smooth * s)
    bg_l = cv2.medianBlur(g_f, _odd(p.dark_large_ksize * s)).astype(np.float32)
    resp = bg_l - gs
    region = fov_inner & ~(disc | fovea)
    if region.sum() < 1000:
        return np.zeros(g_f.shape, bool), resp
    thr = max(p.dark_large_thr, p.dark_noise_k * _noise(resp[region]))
    m = ((resp > thr) & region).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, disk(p.dark_large_open_r * s)).astype(bool)
    return _remove_small(m, p.dark_large_min_area * s * s), resp


def detect_dark(g_f, bg, fov_inner, vessels, disc, fovea, bright_mask, large, large_resp, s, p: Params) -> Lesions:
    """Dark lesions (microaneurysms and haemorrhages), combining two paths.

    Step 0  Bright structures are flattened to the background (minimum of image and background),
            so the dark gaps between exudates cannot look like dark lesions.
    Step 1  Small lesions (microaneurysms, dot and blot haemorrhages) by a linear top-hat:
            invert the image so lesions and vessels are bright, then take the largest opening with
            a thin line in 12 directions. This keeps anything at least dark_line_len long in some
            direction (vessels of any width or faintness). Subtracting it leaves compact blobs only.
    Step 2  Large blots from detect_large_dark. They are exempt from the shape filter because large
            haemorrhages are often irregular or elongated.
    Vessels, the optic disc and the fovea are excluded.
    """
    # step 0: flatten bright structures and invert
    gs = _blur(g_f, p.dark_smooth * s)
    flat = np.minimum(gs, bg)
    inv = 255.0 - flat

    # step 1: linear top-hat (image minus the largest opening over 12 line orientations)
    lin = np.zeros_like(inv)
    for k in line_kernels(p.dark_line_len * s):
        lin = np.maximum(lin, cv2.morphologyEx(inv, cv2.MORPH_OPEN, k))
    small_resp = inv - lin

    # region searched: retina minus vessels, disc, fovea and bright lesions
    valid = fov_inner & ~(_dilate(vessels, p.vessel_dilate * s) | disc | fovea | _dilate(bright_mask, 2 * s))
    if valid.sum() < 1000:
        return Lesions.empty(g_f.shape)
    thr_s = max(p.dark_thr, p.dark_noise_k * _noise(small_resp[valid]))
    small = (small_resp > thr_s) & valid

    # step 2: add large blots, join fragments, fill hollow blots
    cand = small | large
    if p.dark_close_r > 0:
        cand = cv2.morphologyEx(cand.astype(np.uint8), cv2.MORPH_CLOSE, disk(p.dark_close_r * s)).astype(bool)
    cand = ndi.binary_fill_holes(cand) & fov_inner & ~(disc | fovea)
    resp = np.maximum(small_resp, large_resp)
    return extract_components(cand, resp, p.dark_min_area * s * s, p.dark_max_aspect,
                              aspect_max_area=p.dark_large_min_area * s * s)


# ======================================================================================
# Full pipeline
# ======================================================================================
@dataclass
class Detections:
    """Everything the detectors produce for one image."""
    g: np.ndarray            # input image
    g_filled: np.ndarray     # input with the outside of the FOV filled in
    fov: np.ndarray
    fov_inner: np.ndarray    # FOV shrunk away from the rim
    s: float                 # scale factor
    centre: tuple            # FOV centre (row, col)
    radius: float            # FOV radius
    disc: np.ndarray
    disc_xy: tuple | None
    fovea: np.ndarray
    fovea_xy: tuple | None
    vessels: np.ndarray      # cleaned vessel mask
    vesselness: np.ndarray   # raw Frangi response
    dark: Lesions
    bright: Lesions


def run_detectors(g: np.ndarray, p: Params | None = None) -> Detections:
    """Run every detection step on one green-channel image."""
    p = p or Params()

    # field of view, scale and background
    fov = compute_fov(g, p)
    s, centre, radius = fov_geometry(fov)
    fov_inner = _erode(fov, p.fov_erode * s)
    if fov_inner.sum() < 1000:
        fov_inner = fov
    g_f = fill_outside(g, fov)
    bg = cv2.medianBlur(g_f, _odd(p.bg_ksize * s)).astype(np.float32)

    # anatomy
    vessels_raw, vesselness = compute_vessels(g_f, fov_inner, s, p)
    tree = main_tree(vessels_raw, s, p)
    disc, disc_xy = compute_disc(g_f, fov, vessels_raw, s, p)
    fovea, fovea_xy = compute_fovea(g_f, fov_inner, disc_xy, s, p)

    # lesions
    large, large_resp = detect_large_dark(g_f, fov_inner, disc, fovea, s, p)
    bright = detect_bright(g_f, bg, fov_inner, disc, tree, large, s, p)
    # Frangi also responds to the gaps between exudates and the edges of large haemorrhages,
    # so vessel pixels next to bright lesions or inside large dark blots are removed
    vessels = _remove_small(vessels_raw & ~_dilate(bright.mask, 3 * s) & ~_dilate(large, 2 * s),
                            max(p.vessel_min_area * s * s, 1))
    dark = detect_dark(g_f, bg, fov_inner, vessels, disc, fovea, bright.mask, large, large_resp, s, p)

    return Detections(g, g_f, fov, fov_inner, s, centre, radius, disc, disc_xy, fovea, fovea_xy,
                      vessels, vesselness, dark, bright)
