"""Training-time augmentation for preprocessed fundus images (Part C).

Works on the uint8 images from `src/preprocess.py` (colour HxWx3 or green HxW),
before `normalize()`. Use it on the TRAINING split only; val and test images
are never augmented.

Transforms (each applied at random, with its own probability):

    flip        horizontal and/or vertical. A fundus has no fixed "up", and a
                horizontal flip turns a left eye into a right eye.
    rotate      any angle, about the image centre. Rotation keeps the retina
                inside its circle, so no tissue is lost.
    brightness  add a value to every pixel (exposure differences between photos)
    contrast    scale pixel values about the image mean (camera and lighting differences)

Brightness and contrast are changed only inside the field of view, so the black
background stays black and does not become a fake signal.

Default ranges are deliberately mild: strong colour or intensity changes can
hide or imitate lesions (e.g. turning a haemorrhage into a normal-looking dark area).

Usage:
    from src.augment import Augmenter
    aug = Augmenter(seed=9027)
    img_aug = aug(img)            # uint8 in, uint8 out, same shape
"""

import cv2
import numpy as np

# Pixels at or below this value are treated as background (outside the field of view).
BACKGROUND_MAX = 0


class Augmenter:
    def __init__(
        self,
        p_hflip=0.5,
        p_vflip=0.5,
        p_rotate=1.0,
        max_rotate=180.0,
        p_brightness=0.5,
        brightness=0.10,
        p_contrast=0.5,
        contrast=0.10,
        seed=None,
    ):
        """
        max_rotate   rotation angle is drawn from [-max_rotate, +max_rotate] degrees
        brightness   shift as a fraction of 255, drawn from [-brightness, +brightness]
        contrast     scale factor drawn from [1 - contrast, 1 + contrast]
        seed         for reproducible augmentation; each DataLoader worker should get its own
        """
        self.p_hflip, self.p_vflip = p_hflip, p_vflip
        self.p_rotate, self.max_rotate = p_rotate, max_rotate
        self.p_brightness, self.brightness = p_brightness, brightness
        self.p_contrast, self.contrast = p_contrast, contrast
        self.rng = np.random.default_rng(seed)

    def __call__(self, img):
        if img.dtype != np.uint8:
            raise TypeError(f"expected a uint8 image, got {img.dtype}")
        out = img
        if self.rng.random() < self.p_hflip:
            out = out[:, ::-1]
        if self.rng.random() < self.p_vflip:
            out = out[::-1, :]
        if self.max_rotate > 0 and self.rng.random() < self.p_rotate:
            out = rotate(out, self.rng.uniform(-self.max_rotate, self.max_rotate))
        do_b = self.brightness > 0 and self.rng.random() < self.p_brightness
        do_c = self.contrast > 0 and self.rng.random() < self.p_contrast
        if do_b or do_c:
            shift = self.rng.uniform(-self.brightness, self.brightness) * 255 if do_b else 0.0
            scale = self.rng.uniform(1 - self.contrast, 1 + self.contrast) if do_c else 1.0
            out = brightness_contrast(out, shift, scale)
        return np.ascontiguousarray(out)


def rotate(img, angle):
    """Rotate about the centre by `angle` degrees; corners are filled with black."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D(((w - 1) / 2, (h - 1) / 2), angle, 1.0)
    return cv2.warpAffine(img, m, (w, h), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def brightness_contrast(img, shift, scale):
    """Apply `scale * (x - mean) + mean + shift` inside the field of view only."""
    gray = img if img.ndim == 2 else img.max(axis=2)
    fov = gray > BACKGROUND_MAX
    if not fov.any():
        return img
    x = img.astype(np.float32)
    mean = x[fov].mean(axis=0)
    y = scale * (x - mean) + mean + shift
    y = np.clip(y, 1, 255)          # 1, not 0: keep the retina distinguishable from the background
    y[~fov] = 0
    return y.astype(np.uint8)
