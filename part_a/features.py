"""Convert detector output into one row of 38 numerical features per image.

Feature groups (by name prefix)
  gl_  global image statistics (exposure, contrast, sharpness, framing); no lesion detection
  dk_  dark lesions: microaneurysms and haemorrhages
  br_  bright lesions: hard exudates and cotton-wool spots
  vs_  vessels
  ls_  all lesions combined (dark + bright)

Rationale
  The ICDR grading scale is defined by how many red lesions there are, where they are (the 4-2-1 rule
  for severe NPDR counts quadrants), how large they are, and whether bright lesions or new vessels are
  present. Each feature is a measurable proxy for one of these criteria.

  The combined ls_ features exist because a referable eye shows some lesion, but not always the same
  kind: one eye may show mainly exudates, another mainly haemorrhages. The total lesion burden is the
  more consistent signal, while the separate dk_ and br_ features describe which kind is present.
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

from detectors import Detections, Lesions, Params, run_detectors


# ---------------------------------------------------------------------------
# Spatial helpers
# ---------------------------------------------------------------------------
def _quadrants(L: Lesions, centre, min_count):
    """Lesion counts in the four quadrants around the FOV centre.

    Returns (fewest in any quadrant, most in any quadrant, number of quadrants with at least min_count).
    """
    if L.n == 0:
        return 0, 0, 0
    q = (L.cy >= centre[0]).astype(int) * 2 + (L.cx >= centre[1]).astype(int)
    c = np.bincount(q, minlength=4)
    return int(c.min()), int(c.max()), int((c >= min_count).sum())


def _spread(L: Lesions, radius):
    """Spatial spread of lesions relative to the retina radius (0 = one cluster, higher = scattered)."""
    if L.n < 2:
        return 0.0
    return float(np.sqrt(L.cy.var() + L.cx.var()) / radius)


# ---------------------------------------------------------------------------
# Global image statistics (gl_)
# ---------------------------------------------------------------------------
def _global_features(g, det: Detections):
    """Exposure, contrast, texture, sharpness and framing of the retina region."""
    vals = g[det.fov_inner]
    if vals.size == 0:
        vals = g.ravel()
    pr = np.bincount(vals, minlength=256).astype(float)          # gray-level histogram
    pr = pr[pr > 0] / pr.sum()
    lap = cv2.Laplacian(det.g_filled.astype(np.float32), cv2.CV_32F)
    p5, p95 = np.percentile(vals, [5, 95])
    return {
        "gl_fov_frac": float(det.fov.mean()),                      # retina area / image area
        "gl_mean": float(vals.mean()),                              # mean brightness
        "gl_std": float(vals.std()),                                # brightness spread
        "gl_entropy": float(-(pr * np.log2(pr)).sum()),             # histogram entropy
        "gl_contrast": float(p95 - p5),                             # 95th minus 5th percentile
        "gl_sharp": float(lap[det.fov_inner].var()),                # variance of the Laplacian (focus)
    }


# ---------------------------------------------------------------------------
# Lesion features (dk_ and br_)
# ---------------------------------------------------------------------------
def _lesion_features(prefix, L: Lesions, det: Detections, p: Params, dark: bool):
    """Count, size, contrast and spatial distribution of one lesion type."""
    s2 = det.s ** 2
    fov_area = float(det.fov_inner.sum())
    area = L.area / s2                      # areas rescaled to a 512-px retina
    n = L.n
    q_min, q_max, q_active = _quadrants(L, det.centre, p.quad_min_count)

    # features shared by dark and bright lesions
    f = {
        f"{prefix}_n": n,
        f"{prefix}_area_frac": float(L.area.sum() / fov_area) if n else 0.0,
        f"{prefix}_mean_area": float(area.mean()) if n else 0.0,
        f"{prefix}_max_area": float(area.max()) if n else 0.0,
        f"{prefix}_mean_contrast": float(L.contrast.mean()) if n else 0.0,
        f"{prefix}_q_max": q_max,
        f"{prefix}_n_quad_active": q_active,
        f"{prefix}_spread": _spread(L, det.radius),
    }

    # features specific to each lesion type
    if dark:
        f.update({
            "dk_n_small": int((area <= p.small_max_area).sum()),                                   # microaneurysm-sized
            "dk_n_medium": int(((area > p.small_max_area) & (area <= p.medium_max_area)).sum()),   # dot/blot haemorrhage
            "dk_n_large": int((area > p.medium_max_area).sum()),                                   # large haemorrhage
            "dk_mean_aspect": float(L.aspect.mean()) if n else 0.0,
            "dk_frac_elong": float((L.aspect > 2.5).mean()) if n else 0.0,                         # flame-like shapes
            "dk_q_min": q_min,                                                                      # 4-2-1 rule proxy
        })
    else:
        f.update({
            "br_n_small": int((area <= p.small_max_area).sum()),      # punctate hard exudates
            "br_n_large": int((area > p.medium_max_area).sum()),      # confluent patches, cotton-wool spots, laser scars
        })
    return f


# ---------------------------------------------------------------------------
# Vessel features (vs_)
# ---------------------------------------------------------------------------
def _vessel_features(det: Detections):
    """Vessel area, contrast, length and branching from the vessel mask and its skeleton."""
    v = det.vessels
    fov_area = float(det.fov_inner.sum())
    if not v.any():
        return {"vs_area_frac": 0.0, "vs_mean_vesselness": 0.0, "vs_skel_frac": 0.0,
                "vs_branch_density": 0.0, "vs_end_density": 0.0}

    # skeleton (1-px centre lines) and the number of skeleton neighbours of each pixel
    sk = skeletonize(v).astype(np.uint8)
    nb = cv2.filter2D(sk, -1, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT).astype(int)
    ends = (sk == 1) & (nb == 2)               # one neighbour: vessel end point
    branch = (sk == 1) & (nb >= 4)             # three or more neighbours: branch point
    n_branch = ndi.label(branch, structure=np.ones((3, 3)))[1]
    per = 1e4 / fov_area                       # densities are per 10,000 retina pixels
    return {
        "vs_area_frac": float(v.sum() / fov_area),
        "vs_mean_vesselness": float(det.vesselness[v].mean()),
        "vs_skel_frac": float(sk.sum() / fov_area),
        "vs_branch_density": float(n_branch * per),
        "vs_end_density": float(ends.sum() * per),
    }


# ---------------------------------------------------------------------------
# All features for one image
# ---------------------------------------------------------------------------
def compute_features(g: np.ndarray, params: Params | None = None, return_detections: bool = False):
    """Run the detectors on one image and return its feature dictionary (and optionally the detections)."""
    p = params or Params()
    det = run_detectors(g, p)
    f = {}
    f.update(_global_features(g, det))
    f.update(_lesion_features("dk", det.dark, det, p, dark=True))
    f.update(_lesion_features("br", det.bright, det, p, dark=False))
    f.update(_vessel_features(det))
    f.update(_combined_features(det, p))
    return (f, det) if return_detections else f


# ---------------------------------------------------------------------------
# Combined lesion features (ls_)
# ---------------------------------------------------------------------------
def _combined_features(det: Detections, p: Params):
    """Total lesion count, total lesion area and number of quadrants with lesions of any kind."""
    fov_area = float(det.fov_inner.sum())
    both = det.dark.mask | det.bright.mask
    q = 0
    if det.dark.n + det.bright.n:
        cy = np.concatenate([det.dark.cy, det.bright.cy]); cx = np.concatenate([det.dark.cx, det.bright.cx])
        quad = (cy >= det.centre[0]).astype(int) * 2 + (cx >= det.centre[1]).astype(int)
        q = int((np.bincount(quad, minlength=4) >= p.quad_min_count).sum())
    return {
        "ls_n": det.dark.n + det.bright.n,                 # all lesions
        "ls_area_frac": float(both.sum() / fov_area),      # total lesion burden
        "ls_n_quad_active": q,                             # quadrants with at least quad_min_count lesions
    }
