# aptos-referable-dr
 BIOM9027 group project: referable diabetic retinopathy (grade 2+) detection from APTOS fundus images, comparing classical image processing and deep learning.

## Data and splits

- Raw images: `data/raw/Aptos2019 retinal image database/train_images/<id_code>.png` (not in git; get it from the group Google Drive).
- **Use `splits.csv` for everything.** It is the fixed 70/15/15 partition (stratified by grade, seed 9027, exact-duplicate images kept in the same split). Do not re-split.

| column | meaning |
|---|---|
| `id_code` | image file name without `.png` |
| `diagnosis` | original grade 0–4 |
| `referable` | 1 if grade ≥ 2, else 0 (our target label) |
| `split` | `train` / `val` / `test` |
| `dup_group` | ID shared by byte-identical images (empty if unique) |
| `label_conflict` | 1 if copies of the same image have different grades |

```python
import pandas as pd
splits = pd.read_csv("splits.csv")
train = splits[splits.split == "train"]
val   = splits[splits.split == "val"]
# test: only once, for the final models. Never tune on it.
```

## Preprocessing (Part C)

`src/preprocess.py` crops the black border, pads to a square, resizes, and blacks out everything outside the camera's field of view. Modes differ only in contrast handling:

| mode | output | for |
|---|---|---|
| `rgb` | 3-channel, geometry only | baseline row in the ablation table |
| `green` | **1-channel**, green channel + CLAHE | **A** – classical pipeline |
| `clahe_rgb` | 3-channel, CLAHE on LAB lightness | **B** – CNN (default) |
| `ben_graham` | 3-channel, local mean subtracted | optional comparison (C) |

All outputs are **uint8** (0–255), shape `(512, 512, 3)` or `(512, 512)` for `green`. Comparison figure: `results/figures/preprocessing_comparison.png` (made in `notebooks/C_preprocessing_demo.ipynb`).

### Cached images (use these, don't reprocess)

Already preprocessed at 512 × 512, one PNG per image, same file name as the original:

- `data/cache_clahe_512/<id_code>.png` – `clahe_rgb`
- `data/cache_green_512/<id_code>.png` – `green` (single-channel PNG)

On Colab, mount the group Drive and point to the same folder names. The caches contain all 3662 images; select train/val/test with `splits.csv`.

To regenerate (e.g. another mode):

```bash
python -m src.preprocess "data/raw/Aptos2019 retinal image database/train_images" data/cache_clahe_512 --mode clahe_rgb
```

### Using it

**A (classical):** read the green cache as greyscale; it is ready for thresholding / morphology.

```python
img = cv2.imread(f"data/cache_green_512/{id_code}.png", cv2.IMREAD_GRAYSCALE)
```

**B (CNN):** order matters: **read → resize (optional) → augment (train only) → `normalize()` → tensor**. `normalize()` applies ImageNet mean/std, so it must come after augmentation (brightness/contrast augmentation expects 0–255 or 0–1 images, not standardised ones). Val/test get no augmentation, only resize + `normalize()`.

```python
import cv2, torch
from src.preprocess import normalize

img = cv2.cvtColor(cv2.imread(f"data/cache_clahe_512/{id_code}.png"), cv2.COLOR_BGR2RGB)  # uint8 RGB
img = cv2.resize(img, (384, 384), interpolation=cv2.INTER_AREA)  # if the network needs a smaller input
# img = augment(img)                                               # train split only
x = torch.from_numpy(normalize(img)).permute(2, 0, 1)             # float32, CHW
```

`normalize()` also accepts the 1-channel `green` image and repeats it to 3 channels.

Requirements: `pip install -r requirements.txt` (numpy, opencv-python-headless; Colab already has both).
