# Preprocessing parameters: choices and rationale (Part C)

Notes for the "preprocessing" part of the individual report. Code: `src/preprocess.py`. Figure: `results/figures/preprocessing_comparison.png`.

## Why preprocessing is needed for APTOS

APTOS 2019 images come from several clinics and cameras. In the dataset there are 17 different image sizes (from 474 × 358 to 4288 × 2848 pixels; the most common is 1050 × 1050, 974 images), the black border around the fundus varies, many photos have the top and bottom of the circular field of view cut off, and illumination ranges from underexposed to overexposed. On a fixed random sample of 400 training images, mean brightness inside the field of view ranged from 34 to 155 (grey levels, 0–255). Preprocessing puts every image into the same geometry and reduces illumination differences that are unrelated to disease.

## Pipeline order

1. Crop the black border (threshold `tol = 7`).
2. Pad to a square with black (no stretching).
3. Resize to 512 × 512 (`INTER_AREA`, which avoids aliasing when shrinking).
4. Detect the field of view (FOV) mask from the resized image.
5. Contrast step depending on mode (`rgb`, `green`, `clahe_rgb`, `ben_graham`).
6. Set pixels outside the FOV to 0 (mask eroded by 1 % of the width for the enhanced modes).

## Parameter choices

### Image size: 512 × 512

- DR lesions such as microaneurysms are very small; reducing a 2000–4000 px image straight to 224 px makes them only one or two pixels wide or removes them entirely. 512 keeps enough detail for the classical pipeline (A), which detects lesions directly with thresholding and morphology.
- 512 is still small enough to cache all 3662 images and train quickly.
- The CNN (B) can shrink the cached 512 images further to its input size (224 or 384). Downsizing from the cache is cheap; going the other way would need reprocessing from the originals.
- Trade-off: a fixed 512 is a compromise. Grading at full resolution might help small-lesion detection but is computationally impractical for this project.

### Pad to square instead of stretching

Many APTOS fundus images are wider than they are tall after cropping (width/height up to about 1.33 in our sample). Resizing directly to a square would flatten the optic disc and lesions; padding keeps the true shape.

### Border threshold: `tol = 7`

- The background outside the fundus is close to black but not exactly 0 (JPEG/PNG noise, camera text marks), so a threshold slightly above 0 is needed.
- 7 out of 255 is low enough to keep the dark edges of underexposed fundus images: the darkest image in our 400-image sample still had a mean FOV brightness of 34, well above the threshold.
- Checked on a random sample of 300 images: no failed or mis-cropped images. If the threshold removes everything (a completely dark image), the code falls back to the uncropped image instead of crashing.
- Limitation: chosen by inspection, not by systematic tuning.

### Field-of-view mask from the image (not a geometric circle)

Because many images have the top and bottom of the fundus cut off, a geometric circle would include the black padding bars. The mask is computed by thresholding the image itself (plus a small morphological opening to remove noise), so it follows the real edge.

### CLAHE: `clipLimit = 2.0`, `tileGridSize = 8 × 8`

- CLAHE (Contrast Limited Adaptive Histogram Equalisation) equalises contrast locally in tiles, so a dark periphery and a bright centre are both improved. Global histogram equalisation would be dominated by the large uniform background.
- `tileGridSize = 8 × 8` on a 512 px image gives tiles of 64 × 64 px: small enough to adapt to uneven illumination across the fundus, large enough that each tile contains meaningful structure.
- `clipLimit = 2.0` limits how much contrast can be amplified in a tile. A higher value would also amplify noise and JPEG artefacts in flat areas of the retina; a lower value gives little enhancement.
- Both values are the commonly used defaults (8 × 8 is the OpenCV default). They were checked visually on normal, lesion, truncated, under- and overexposed images, but **not tuned against model performance**. Whether CLAHE helps is tested in the ablation table instead.

### CLAHE in LAB space, not on each RGB channel

- Applying CLAHE separately to R, G and B changes each channel's histogram independently, so their ratios change and colours shift (e.g. yellow exudates or red haemorrhages change hue).
- LAB separates lightness (L) from colour (A, B). Applying CLAHE only to L improves contrast while keeping colour information intact.
- This matters for the CNN: ImageNet-pretrained networks expect natural-looking colour images, and colour is a clinical cue (yellow exudates vs. red haemorrhages).

### Green channel for the classical pipeline

In fundus photographs, the green channel has the best contrast for vessels, microaneurysms and haemorrhages (the red channel is often saturated and the blue channel is dark and noisy). The classical pipeline works on a single channel, so the green channel with CLAHE is the natural input.

### Ben Graham mode (optional comparison)

`4 × image − 4 × Gaussian blur + 128`, with σ = image width / 30. It removes slow illumination changes and highlights small, high-contrast structures; it was used by the winner of the 2015 Kaggle diabetic-retinopathy competition. Our version calculates the local average only inside the FOV (normalised blur), because a plain Gaussian blur pulls in the black background and creates a bright halo and stripes at the edge, especially on truncated images.

## Normalisation (CNN only)

The cached images stay uint8. ImageNet mean/std normalisation (`normalize()`) is applied after augmentation, so brightness/contrast augmentation works on ordinary image values and the cache can be reused for any model.

## Limitations

- Parameters were chosen by inspection and common practice, not optimised. The effect of preprocessing on performance is measured in the ablation table, on the validation set only.
- Fixed thresholds may behave differently on images from a new camera (generalisability).
- Downsizing to 512 can still remove the smallest microaneurysms.
- 44 images are smaller than 512 px (474 × 358 and 640 × 480) and are upscaled; upscaling adds no detail, so these images stay lower quality than the rest.
- Preprocessing cannot recover information lost in very poor-quality images (blur, severe over/underexposure).
