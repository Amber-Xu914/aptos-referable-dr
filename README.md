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
# img = aug(img)            # aug = src.augment.Augmenter(seed=...); train split only
x = torch.from_numpy(normalize(img)).permute(2, 0, 1)             # float32, CHW
```

`normalize()` also accepts the 1-channel `green` image and repeats it to 3 channels.

Requirements: `pip install -r requirements.txt` (Colab already has all of them).

## Augmentation, class imbalance and ablation (Part C)

| file | what it does |
|---|---|
| `src/augment.py` | `Augmenter`: random flips, rotation, mild brightness/contrast (inside the field of view only). Train split only. |
| `src/imbalance.py` | `pos_weight()` for `BCEWithLogitsLoss` (class weight) and `oversampling_weights()` for `WeightedRandomSampler`. Use one, not both. |
| `src/metrics.py` | Validation metrics for any prediction CSV in the group format: AUROC (+ bootstrap CI), **grade 1 vs 2–4 AUROC**, AUROC without `label_conflict` images, specificity at 85 % sensitivity, sensitivity by grade. |
| `src/ablation.py` | Trains the CNN once per experiment, changing one thing at a time, and writes the ablation table. |

Experiments (each row changes one thing compared with its `base`):

| experiment | base | change |
|---|---|---|
| E0_baseline | – | rgb preprocessing, no augmentation, no imbalance fix |
| E1_clahe | E0 | + CLAHE preprocessing |
| E2_augment | E1 | + augmentation |
| E3_classweight | E2 | + class weight |
| E3b_oversample | E2 | + oversampling (instead of class weight) |
| E4_bengraham | E2 | Ben Graham instead of CLAHE |

```bash
python -m src.ablation                                        # all experiments, seed 9027, 10 epochs
python -m src.ablation --seeds 1 2 3                          # repeat with several seeds
python -m src.ablation --only E0_baseline --subset 100 --epochs 1 --out-tag _smoke   # quick check
python -m src.metrics results/part_a/preds_A_rf_all.csv       # metrics for any prediction file (val rows)
```

Figures for slides 6 and 9 (`results/part_c/figures/`):

```bash
python scripts/plot_ablation.py                       # ablation chart (validation)
python scripts/plot_ablation.py --with-preprocessing --out results/part_c/figures/slide6_preprocessing_ablation.png
python scripts/plot_clinical.py --split test          # workflow + unnecessary referrals per 1,000 (test, D's threshold rule)
```

Outputs: `results/part_c/ablation_table.csv` (mean over seeds), `results/part_c/ablation_runs.csv` (per seed), and validation predictions in `results/part_c/preds/` (group CSV format: `image_id, split, true_grade, model_score, model_type`). Checkpoints go to `outputs/part_c/` (not in git). Missing caches (`data/cache_rgb_512`, `data/cache_bengraham_512`) are built automatically from the raw images.

Notes:
- The final ablation uses B's main model and settings: `--model efficientnet_b0 --size 384 --batch-size 32 --lr 1e-4 --epochs 25` (ImageNet weights, all layers fine-tuned), seeds 1, 2, 3. An earlier pilot with ResNet-18 at 224 px (1 seed, 10 epochs) is kept in `results/part_c/resnet18_pilot/`. Other networks can be added to `build_model()` in `src/ablation.py`.
- Final result (mean of 3 seeds, grade 1 vs 2–4 AUROC): E0 0.912, E1 0.902, E2 0.895, E3 0.893, E3b 0.896, E4 0.885. No change beat the baseline; paired bootstrap differences vs E0 all include zero except Ben Graham (−0.028, 95% CI −0.056 to 0.000). Figure: `results/part_c/figures/ablation_auroc.png`.
- Difference from B's main model: B uses lr 1e-3 for the classification head (1e-4 for the pretrained layers) and a frozen-backbone warm-up; the ablation fine-tunes all layers at 1e-4 from the start. B's model matches the E3 row (val AUROC 0.982 vs 0.980; grade 1 vs 2–4 0.897 vs 0.893).
- The best epoch is chosen by validation AUROC, so absolute validation numbers are slightly optimistic; the comparison between rows is fair because every row is treated the same way.
- Overall AUROC is inflated by the camera shortcut (grade 0 photos come mostly from different cameras, see Part A). Judge the improvements mainly by **`auroc_g1_vs_ref`**.
- The test split is never used here. `--predict-test <experiment>` writes test predictions for the frozen final model only, without computing metrics (D evaluates them once).
