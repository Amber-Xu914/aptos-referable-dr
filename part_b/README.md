# Part B: deep learning for referable diabetic retinopathy

Deep-learning side of the project: a small CNN trained from scratch (baseline), an
ImageNet-pretrained network used frozen versus fine-tuned (transfer learning), Grad-CAM on
the winning model, runtime/hardware reporting, and an optional hybrid (CNN features → PCA →
classical classifier). Target: **referable DR = grade ≥ 2**; the original 0–4 grade is kept
in every table and prediction file.

Input is Part C's `clahe_rgb` cache (`data/cache_clahe_512/`) and the fixed `splits.csv`.
Prediction files use the **same format as Part A**, so both pipelines go through one evaluation.

## Run it locally in VS Code (one execution)

`part_b/run_all.py` does everything in order: finds the images in `data/`, checks the data, trains the three models, runs the hybrid, compares them on
validation, draws Grad-CAM for the winner, scores the test split once and writes the
runtime notes. Finished models are skipped when you start it again, so an interrupted run
just continues.

**One-time setup**

1. Install Python 3.10+ and the VS Code Python extension; open the repository folder in VS Code.
2. Create an environment and install the packages (VS Code terminal):
   ```bash
   python -m venv .venv
   ```
   Activate it on VSCode use ctrl+shift+P go to Python:Select Interpreter and use venv to setup enviroment, then:
   ```bash
   pip install -r part_b/requirments.txt
   ```
   **NVIDIA GPU on Windows/Linux:** install PyTorch first with the CUDA command from
   <https://pytorch.org/get-started/locally/>, otherwise pip may install a CPU-only build.
   Apple-silicon Macs use the built-in GPU (MPS) automatically.
3. Put the downloaded files in a folder named `data` in the repository (it is git-ignored, so
   the images are never committed). The script searches `data/` for, in this order:
   - `cache_clahe_512/` with Part C's preprocessed PNGs (used in place), e.g.
     `data/cache_clahe_512/000c1434d8d7.png`;
   - `cache_clahe_512.zip` (unzipped once to `data/cache_clahe_512/`);
   - the raw APTOS `train_images/` folder (preprocessed once with Part C's `src/preprocess.py`,
     `clahe_rgb` at 512 px, into `data/cache_clahe_512/`; takes a few minutes).

   Sub-folders are fine (e.g. `data/raw/Aptos2019 retinal image database/train_images/`). Every
   image in `splits.csv` must be there, otherwise the script stops and lists what is missing.
   Check with `python part_b/local_data.py`.

**Run**

- VS Code: *Run and Debug* → **Part B: quick setup check** first (about 5 minutes: 1 epoch on
  64 images, checks the images, GPU and every step; numbers are meaningless), then
  **Part B: run everything**. Or open `part_b/run_all.py` and press ▶.
- Terminal: `python part_b/run_all.py` (options: `--quick`, `--seeds 1 2 3`, `--size 224`,
  `--no-hybrid`, `--no-final-test`, `--force`, `--data-dir ...`).

**GPU or CPU.** The GPU is used whenever PyTorch can run on it (NVIDIA CUDA, or Apple-silicon
MPS), otherwise the CPU; a GPU that fails a small test calculation also falls back to the CPU.
If the computer has an NVIDIA GPU but training runs on the CPU, the script says why. Most often
the installed PyTorch is the **CPU-only build** (`torch.__version__` ends in `+cpu`). Installing
the CUDA toolkit does not fix that; the GPU build of PyTorch brings its own CUDA and only needs
the NVIDIA driver. Replace it (the script prints these commands for your Python):
```bash
python -m pip uninstall -y torch torchvision
```
```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```
Use the CUDA version that pytorch.org/get-started/locally lists, no higher than the "CUDA Version"
shown by `nvidia-smi`. Check with `python part_b/hardware.py` (prints `training device: cuda`).

**Progress in the terminal.** Before each step a timeline lists every planned step
(`[x]` done, `[>]` running, `[ ]` to do, `[-]` skipped) with elapsed time and, once a model has
finished, an estimate of the training time left. During training a live bar shows the batches
of the current epoch and the running loss, and each epoch ends with a line like
`[########............] epoch 10/25 ... val AUROC 0.9412 *best | 48s/epoch, elapsed 8m 02s, ~12m 04s left`.

Results land in `results/part_b/`; the step times are in `run_all_log.txt` and at the end of the output.

On a CPU only, fine-tuning EfficientNet-B0 at 384 px takes many hours; use a GPU, or
`--size 224` with fewer runs. The seed now fixes weights, batch order and augmentation, so a
run with the same `--seed` repeats exactly on the same machine.

**Test split:** `run_all.py` scores it once at the end, with every choice (models, epochs,
threshold rule, winner) fixed in advance and the winner picked on validation. Changing
settings after looking at the test results and running again would be tuning on the test set.

## Run it on Colab (alternative *Not tested myself)

Open `notebooks/B_deep_learning_colab.ipynb` (Runtime → Change runtime type → GPU) and run
top to bottom. It copies the image cache from the group Drive to the Colab disk, runs every
step below and copies `results/part_b/` back to Drive.

## Individual steps

| Step | Command | Output (`results/part_b/`) |
|---|---|---|
| 0. Smoke test (CPU, synthetic) | `python part_b/smoke_test.py` | prints PASS |
| 1. Data loader + split check | `python part_b/check_data.py` | `data_check.txt`, `class_counts.csv` |
| 2. CNN from scratch | `python part_b/train.py --model smallcnn` | `preds_B_smallcnn.csv`, `history_*.csv`, `summary_*.json` |
| 3a. Transfer, frozen | `python part_b/train.py --model efficientnet_b0 --mode frozen` | `..._efficientnet_b0_frozen.*` |
| 3b. Transfer, fine-tuned | `python part_b/train.py --model efficientnet_b0 --mode finetune` | `..._efficientnet_b0_finetune.*` |
| 3c. (optional) last stage only | `python part_b/train.py --model efficientnet_b0 --mode partial` | `..._efficientnet_b0_partial.*` |
| Compare on validation | `python part_b/compare.py --extra results/part_a/preds_A_rf_all.csv` | `comparison_val.csv`, `roc_val.png`, `learning_curves.png`, `runtime_hardware.md` |
| 4. Grad-CAM (winner) | `python part_b/gradcam.py --run <best run>` | `gradcam_<run>_val.png`, `gradcam_stats_*.csv` |
| Optional hybrid | `python part_b/hybrid.py --arch efficientnet_b0` | `preds_B_hybrid_*.csv`, `grid_hybrid_*.csv` |
| **Test, once** | `python part_b/final_test.py --runs smallcnn efficientnet_b0_frozen efficientnet_b0_finetune` (+ `hybrid.py ... --final-test`) | test rows appended to the prediction files, `test_metrics.csv` |
| 4b. Grad-CAM on test errors | `python part_b/gradcam.py --run <best run> --split test` | `gradcam_<run>_test.png` |
| 5. Hardware | `python part_b/hardware.py` | `hardware.json` (also stored in every summary) |

Model weights go to `outputs/part_b/checkpoints/` (not in git; ~20–100 MB each). Copy them
to Drive if they are needed later (the notebook does).
## Files

| File | Purpose |
|---|---|
| `common.py` | paths, `load_splits`, `check_splits`, augmentation, dataset, metrics, threshold, prediction files |
| `check_data.py` | step 1 report |
| `models.py` | `SmallCNN`, `TransferNet` (ResNet-18/50, EfficientNet-B0), frozen/partial/fine-tune setup |
| `train.py` | training + validation selection, runtime logging |
| `compare.py` | validation comparison table, ROC and learning curves, runtime notes |
| `gradcam.py` | Grad-CAM figure and saliency statistics |
| `hybrid.py` | CNN features → PCA → classical classifier |
| `final_test.py` | one-off test evaluation at validation thresholds |
| `hardware.py` | hardware/software info, device choice (CUDA → Apple MPS → CPU), timers |
| `local_data.py` | finds the images in `data/` (cache folder, zip or raw images) |
| `run_all.py` | the whole pipeline in one execution; settings at the top |
| `smoke_test.py` | end-to-end test on synthetic data |
