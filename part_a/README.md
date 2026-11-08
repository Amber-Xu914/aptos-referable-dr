# Part A: classical image-processing pipeline for referable diabetic retinopathy

Detects dark lesions (microaneurysms, haemorrhages) and bright lesions (hard exudates, cotton-wool spots) in APTOS 2019 fundus photographs using thresholding and morphology. It summarises each image as 38 features and classifies referable DR (grade 2 or above) with logistic regression, an SVM and a random forest.

## Folder layout

```
project_root/
  data/                      not in the repository (images are too large)
    raw/
      train.csv              APTOS labels (id_code, diagnosis)
      train_images/          APTOS photographs
    part_c/
      splits.csv             shared train/val/test split
      cache_green_512/       green channel + CLAHE, 512 x 512
  part_a/                    scripts
  outputs/part_a/            everything the scripts write
```

All scripts locate the data relative to the folder that contains `part_a/`. A different location can be given with `--root /path` or the `APTOS_ROOT` environment variable.

## Setup

```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python part_a/smoke_test.py        should end with PASS
```

## Pipeline

| Step | Command | Output |
|---|---|---|
| 1. Verify the shared split and cache | `python part_a/check_part_c.py` | `outputs/part_c_check/report.txt` |
| 2. Visual QA of the detectors | `python part_a/qa_overlays.py` | `outputs/part_a/qa/qa_grade0.png` to `qa_grade4.png` |
| 3. Simple baseline | `python part_a/baseline_simple.py` | `predictions/preds_baseline_colour_logreg.csv` |
| 4. Feature extraction | `python part_a/extract_features.py` | `features_A.csv` (train and val) |
| 5. Feature QA | `python part_a/qa_features.py` | `feature_trends.csv`, `feature_boxplots.png` |
| 6. Ablation | `python part_a/train_models.py ablation` | `ablation_table.csv` |
| 7. Parameter check (optional) | `python part_a/sweep_params.py --param dark_thr --values 10 12 15` | `sweep_dark_thr.csv` |
| 8. Final models | `python part_a/train_models.py final --model rf --featset all` | `predictions/preds_A_rf_all.csv` |
| 9. Test run (once only) | step 4 with `--include-test`, then steps 3 and 8 with `--final-test` | test rows added to the prediction files |

QA (quality assurance) steps do not change any results. They check that each stage behaves as intended before its output is used.

## Reading the contact sheets

`qa_overlays.py` draws the detections on sample training images, one row per image:

1. Input image. Yellow outlines the retina, magenta the optic disc and cyan the fovea. Disc and fovea are excluded from lesion detection.
2. Vessel mask in blue. It should follow the vessel tree.
3. Dark lesions as red circles. They should sit on red dots and blots, not on vessels, the disc or the rim.
4. Bright lesions as yellow circles. They should sit on exudates, not on the disc, the rim or reflections.

Detector settings can be changed without editing code, for example `python part_a/qa_overlays.py --set dark_thr=15 bright_thr=25`. Specific images can be drawn with `--ids <id> <id> --name <file name>`.

## Prediction file format

| Column | Content |
|---|---|
| `image_id` | image id, identical to `id_code` in `splits.csv` |
| `split` | `val` or `test` |
| `true_grade` | original grade, 0 to 4 |
| `model_score` | predicted probability of referable DR |
| `model_type` | `A_<model>_<featset>`, e.g. `A_rf_all`, or `baseline_colour_logreg` |

All models are fitted on the training split. Feature sets, classifiers and hyperparameters are selected on validation AUROC. The test split is used only with `--final-test`.

## Files

| File | Purpose |
|---|---|
| `common.py` | paths, constants and data loaders |
| `detectors.py` | image processing: field of view, vessels, optic disc, fovea, dark and bright lesions; all detector settings |
| `features.py` | 38 features per image from the detector output |
| `extract_features.py` | runs detectors and features on every image |
| `train_models.py` | ablation ladder, final models, prediction files, permutation importance |
| `baseline_simple.py` | brightness-and-colour baseline from the raw photographs |
| `qa_overlays.py` | contact sheets of the detections |
| `qa_features.py` | feature trends by grade and single-feature AUROC |
| `sweep_params.py` | effect of one detector setting on validation AUROC |
| `check_part_c.py` | verification of the shared split and preprocessed images |
| `make_green_cache.py` | green + CLAHE images from the raw photographs (fallback and reference) |
| `smoke_test.py` | end-to-end test on synthetic images |
