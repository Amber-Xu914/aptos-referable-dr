"""Shared paths, data loading, split checks, dataset, augmentation and metrics for Part B (deep learning).

Inputs (produced by Part C, see the top-level README):
    splits.csv                     fixed 70/15/15 partition (id_code, diagnosis, referable, split, dup_group, label_conflict)
    data/cache_clahe_512/<id>.png  colour + CLAHE on LAB lightness, 512 x 512, uint8

Outputs:
    results/part_b/                small files that go into git (prediction CSVs, tables, figures, run summaries)
    outputs/part_b/checkpoints/    model weights (not in git)

A different project root can be given with --root or the APTOS_ROOT environment variable,
and a different image folder with --cache.
"""
import json
import os
import random
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pandas as pd

SEED = 9027                    # same seed as the group's split
REFERABLE_MIN_GRADE = 2        # referable DR = moderate NPDR (grade 2) or worse
SPLITS = ("train", "val", "test")
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
PROJECT_DEFAULT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
def get_paths(root=None, cache=None):
    root = Path(root or os.environ.get("APTOS_ROOT") or PROJECT_DEFAULT).resolve()
    p = SimpleNamespace(
        root=root,
        splits=root / "splits.csv",
        cache=Path(cache).resolve() if cache else root / "data" / "cache_clahe_512",
        results=root / "results" / "part_b",
        ckpt=root / "outputs" / "part_b" / "checkpoints",
    )
    p.results.mkdir(parents=True, exist_ok=True)
    p.ckpt.mkdir(parents=True, exist_ok=True)
    return p


def add_path_args(parser):
    parser.add_argument("--root", default=None, help="project root (default: repository folder or $APTOS_ROOT)")
    parser.add_argument("--cache", default=None, help="folder of preprocessed images (default: data/cache_clahe_512)")


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


# ---------------------------------------------------------------------------
# 1. Data loader and split checks
# ---------------------------------------------------------------------------
def load_splits(path):
    """Read the supplied partition.

    Keeps the original 0-4 grade in `grade` (and `diagnosis`, as supplied) and adds the
    binary target `referable` = grade >= 2. If the file already has a `referable` column
    it must agree with the grade, otherwise this raises.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = {"id_code", "diagnosis", "split"} - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing column(s) {sorted(missing)}")
    df["id_code"] = df["id_code"].astype(str).str.strip().str.replace(r"(?i)\.(png|jpe?g)$", "", regex=True)
    df["split"] = df["split"].astype(str).str.strip().str.lower()
    df["grade"] = df["diagnosis"].astype(int)
    derived = (df["grade"] >= REFERABLE_MIN_GRADE).astype(int)
    if "referable" in df.columns and not (df["referable"].astype(int) == derived).all():
        bad = int((df["referable"].astype(int) != derived).sum())
        raise ValueError(f"`referable` disagrees with grade >= {REFERABLE_MIN_GRADE} in {bad} rows")
    df["referable"] = derived
    df["dup_group"] = df.get("dup_group", pd.Series("", index=df.index)).fillna("").astype(str)
    df["label_conflict"] = df.get("label_conflict", pd.Series(0, index=df.index)).fillna(0).astype(int)
    return df


def check_splits(df, image_dir=None):
    """Check class counts, ID uniqueness, leakage between splits and missing images.

    Returns (ok, report_lines, counts_table). `ok` is False only for hard failures
    (unknown split names, repeated IDs, an ID or duplicate group in more than one split,
    missing images). Everything else is reported for the write-up.
    """
    lines, ok = [], True

    def say(status, msg):
        nonlocal ok
        ok &= status != "FAIL"
        lines.append(f"[{status}] {msg}")

    unknown = set(df["split"]) - set(SPLITS)
    say("FAIL" if unknown else "PASS", f"split names are train/val/test{' (unknown: ' + str(unknown) + ')' if unknown else ''}")

    # Class counts: grade x split, plus referable prevalence
    counts = pd.crosstab(df["split"], df["grade"]).reindex(index=list(SPLITS), columns=range(5), fill_value=0)
    counts.columns = [f"grade_{g}" for g in counts.columns]
    counts["n"] = counts.sum(axis=1)
    counts["referable"] = df.groupby("split")["referable"].sum().reindex(SPLITS).fillna(0).astype(int)
    counts["prevalence"] = (counts["referable"] / counts["n"]).round(3)
    lines.append("Class counts (original grade 0-4 and binary referable label):")
    lines += ["    " + s for s in counts.to_string().splitlines()]
    prev = counts["prevalence"]
    spread = float(prev.max() - prev.min())
    say("PASS" if spread < 0.03 else "WARN", f"referable prevalence similar across splits (range {spread:.3f})")

    # ID checks. APTOS 2019 has no patient identifiers, so the image ID is the finest key available.
    n_dup_ids = int(df["id_code"].duplicated().sum())
    say("FAIL" if n_dup_ids else "PASS", f"every id_code appears once ({n_dup_ids} repeats)")
    by_split = {s: set(df.loc[df.split == s, "id_code"]) for s in SPLITS}
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        n = len(by_split[a] & by_split[b])
        say("FAIL" if n else "PASS", f"no image ID shared by {a} and {b} ({n})")

    # Exact-duplicate images (same file bytes) must sit in one split, otherwise the test set leaks.
    g = df[df["dup_group"] != ""].groupby("dup_group")["split"].nunique()
    n_cross = int((g > 1).sum())
    say("FAIL" if n_cross else "PASS",
        f"duplicate-image groups kept within one split ({len(g)} groups, "
        f"{int((df['dup_group'] != '').sum())} images; {n_cross} cross splits)")
    conf = df[df["label_conflict"] == 1]
    conf_groups = conf.groupby("dup_group")["referable"].nunique()
    lines.append(f"[INFO] label conflicts: {len(conf)} images in {conf['dup_group'].nunique()} groups, "
                 f"{int((conf_groups > 1).sum())} groups disagree on the referable label; "
                 f"per split {conf['split'].value_counts().reindex(SPLITS).fillna(0).astype(int).to_dict()}")
    lines.append("[INFO] APTOS 2019 provides no patient IDs, so both eyes of one person may fall in different "
                 "splits. Leakage control is limited to unique image IDs and byte-identical duplicates.")

    if image_dir is not None:
        image_dir = Path(image_dir)
        if not image_dir.is_dir():
            say("FAIL", f"image folder not found: {image_dir}")
        else:
            have = {p.stem for p in image_dir.iterdir() if p.suffix.lower() == ".png"}
            missing = sorted(set(df["id_code"]) - have)
            say("FAIL" if missing else "PASS",
                f"all {len(df)} images present in {image_dir.name} ({len(missing)} missing"
                f"{': ' + ', '.join(missing[:5]) if missing else ''})")
    return ok, lines, counts


# ---------------------------------------------------------------------------
# Images, augmentation and the PyTorch dataset
# ---------------------------------------------------------------------------
def read_rgb(path, size=None):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"could not read {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if size and img.shape[0] != size:
        img = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)
    return img


def augment(img, rng):
    """Train-time augmentation on a uint8 RGB fundus image (before normalisation).

    Fundus photos have no fixed orientation between cameras/eyes, so flips and full
    rotations are safe. Small zoom and brightness/contrast jitter mimic camera and
    exposure differences. Colour hue is not changed: it is a clinical cue.
    """
    h, w = img.shape[:2]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1, :]
    angle = rng.uniform(-180, 180)
    scale = rng.uniform(0.95, 1.10)
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    img = cv2.warpAffine(np.ascontiguousarray(img), m, (w, h), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    alpha = rng.uniform(0.85, 1.15)       # contrast
    beta = rng.uniform(-15, 15)           # brightness
    fov = img.max(axis=2) > 0
    out = img.astype(np.float32) * alpha + beta
    out[~fov] = 0                         # keep the background black
    return np.clip(out, 0, 255).astype(np.uint8)


def normalize(img):
    """uint8 RGB HWC -> float32 CHW with ImageNet mean/std (same as src.preprocess.normalize)."""
    x = (img.astype(np.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def preload(df, image_dir, size):
    """Read all images of `df` once into one uint8 array (N, size, size, 3).

    About 1.3 GB for the whole dataset at 384 px. Reading 2,500 PNGs from Google Drive
    every epoch is the main bottleneck on Colab, so this is done once per run.
    """
    arr = np.empty((len(df), size, size, 3), dtype=np.uint8)
    for i, id_code in enumerate(df["id_code"]):
        arr[i] = read_rgb(Path(image_dir) / f"{id_code}.png", size)
    return arr


def make_dataset(images, labels, train, seed=SEED):
    import torch
    from torch.utils.data import Dataset

    class FundusDataset(Dataset):
        def __init__(self):
            self.images, self.labels, self.train = images, np.asarray(labels, np.float32), train
            self.seed = seed

        def __len__(self):
            return len(self.images)

        def __getitem__(self, i):
            img = self.images[i]
            if self.train:
                # Different random stream per worker and call; reproducible enough for a course project.
                img = augment(img, np.random.default_rng())
            return torch.from_numpy(normalize(img)), torch.tensor(self.labels[i]), i

    return FundusDataset()


# ---------------------------------------------------------------------------
# Metrics and threshold
# ---------------------------------------------------------------------------
def auroc_ci(y, s, n_boot=2000, seed=SEED):
    """AUROC with a 95 % percentile bootstrap interval (same approach as Part A)."""
    from sklearn.metrics import roc_auc_score
    y, s = np.asarray(y), np.asarray(s)
    auc = roc_auc_score(y, s)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() != y[idx].max():
            boots.append(roc_auc_score(y[idx], s[idx]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(auc), float(lo), float(hi)


def choose_threshold(y, s, rule="sens90"):
    """Pick the decision threshold on the VALIDATION set only.

    sens90: highest threshold that still gives sensitivity >= 0.90. Screening for referable
            DR is a rule-out test: a missed referable case is worse than an unnecessary
            referral, so sensitivity is fixed first (cf. common screening targets of >= 80-90 %).
    youden: maximises sensitivity + specificity - 1 (reported for comparison).
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, s)
    thr = np.clip(thr, 0, 1)
    if rule == "youden":
        return float(thr[np.argmax(tpr - fpr)])
    if rule.startswith("sens"):
        target = int(rule[4:]) / 100
        return float(thr[np.argmax(tpr >= target)])
    raise ValueError(rule)


def metrics_at(y, s, thr):
    y, pred = np.asarray(y).astype(int), (np.asarray(s) >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    fp = int(((pred == 1) & (y == 0)).sum()); fn = int(((pred == 0) & (y == 1)).sum())
    div = lambda a, b: a / b if b else float("nan")
    sens, spec, prec = div(tp, tp + fn), div(tn, tn + fp), div(tp, tp + fp)
    return dict(threshold=float(thr), sensitivity=sens, specificity=spec, precision=prec,
                npv=div(tn, tn + fn), f1=div(2 * prec * sens, prec + sens),
                accuracy=div(tp + tn, len(y)), tp=tp, fp=fp, tn=tn, fn=fn)


def evaluate(y, s, thresholds):
    auc, lo, hi = auroc_ci(y, s)
    return dict(auroc=auc, auroc_ci=[lo, hi],
                **{f"at_{name}": metrics_at(y, s, t) for name, t in thresholds.items()})


# ---------------------------------------------------------------------------
# Prediction files (same format as Part A, so the evaluation code is shared)
# ---------------------------------------------------------------------------
PRED_COLUMNS = ["image_id", "split", "true_grade", "model_score", "model_type"]


def write_preds(df, scores, model_type, path, append_test=False):
    out = pd.DataFrame({"image_id": df["id_code"].values, "split": df["split"].values,
                        "true_grade": df["grade"].values, "model_score": np.round(scores, 5),
                        "model_type": model_type})
    path = Path(path)
    if append_test and path.exists():
        old = pd.read_csv(path)
        out = pd.concat([old[old.split != "test"], out], ignore_index=True)
    out[PRED_COLUMNS].to_csv(path, index=False)
    return path


def save_json(obj, path):
    Path(path).write_text(json.dumps(obj, indent=2, default=float))


def load_json(path):
    return json.loads(Path(path).read_text())