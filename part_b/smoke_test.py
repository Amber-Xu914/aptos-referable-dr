"""End-to-end test of Part B on synthetic fundus-like images (CPU, ~1-2 minutes, no downloads).

    python part_b/smoke_test.py

Builds a temporary project (splits.csv + 64 px images in which referable cases have bright
spots), then runs check_data, train (smallcnn and a random-init resnet18 in frozen and
finetune mode), compare, gradcam, hybrid and final_test. Nothing is written to the repository.
Ends with PASS if every step exits cleanly and the outputs exist.
"""
import subprocess
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent


def make_project(root, n=96, size=96, seed=0):
    rng = np.random.default_rng(seed)
    cache = root / "data" / "cache_clahe_512"
    cache.mkdir(parents=True)
    rows = []
    for i in range(n):
        grade = int(rng.integers(0, 5))
        img = np.zeros((size, size, 3), np.uint8)
        cv2.circle(img, (size // 2, size // 2), size // 2 - 2, (150, 70, 40), -1)
        if grade >= 2:                                  # "exudates"
            for _ in range(3 + grade):
                c = tuple(int(v) for v in rng.integers(size // 4, 3 * size // 4, 2))
                cv2.circle(img, c, 3, (230, 220, 90), -1)
        img = np.clip(img + rng.normal(0, 6, img.shape), 0, 255).astype(np.uint8)
        id_code = f"img{i:04d}"
        cv2.imwrite(str(cache / f"{id_code}.png"), img[:, :, ::-1])
        split = "train" if i < 0.6 * n else "val" if i < 0.8 * n else "test"
        rows.append(dict(id_code=id_code, diagnosis=grade, referable=int(grade >= 2), split=split,
                         dup_group="dup001" if i in (0, 1) else "", label_conflict=0))
    pd.DataFrame(rows).to_csv(root / "splits.csv", index=False)


def run(*args):
    cmd = [sys.executable, str(HERE / args[0]), *args[1:]]
    print("\n$", " ".join(cmd[1:]), flush=True)
    r = subprocess.run(cmd)
    if r.returncode:
        sys.exit(f"FAIL: {' '.join(args)} exited with {r.returncode}")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_project(root)
        R = ["--root", str(root)]
        common = [*R, "--size", "64", "--epochs", "2", "--batch-size", "16", "--workers", "0", "--patience", "5"]
        run("check_data.py", *R)
        run("train.py", "--model", "smallcnn", *common)
        run("train.py", "--model", "resnet18", "--mode", "frozen", "--no-pretrained", *common)
        run("train.py", "--model", "resnet18", "--mode", "finetune", "--no-pretrained", *common)
        run("compare.py", *R)
        run("gradcam.py", *R, "--run", "resnet18_finetune", "--n", "2")
        run("gradcam.py", *R, "--run", "smallcnn", "--n", "2")
        run("hybrid.py", *R, "--arch", "resnet18", "--no-pretrained", "--size", "64", "--workers", "0", "--final-test")
        run("final_test.py", *R, "--runs", "smallcnn", "resnet18_finetune", "--workers", "0")
        res = root / "results" / "part_b"
        expected = ["data_check.txt", "comparison_val.csv", "roc_val.png", "learning_curves.png", "runtime_hardware.md",
                    "gradcam_resnet18_finetune_val.png", "test_metrics.csv", "preds_B_smallcnn.csv",
                    "preds_B_hybrid_resnet18_imagenet.csv"]
        missing = [f for f in expected if not (res / f).exists()]
        p = pd.read_csv(res / "preds_B_smallcnn.csv")
        assert set(p.split) == {"val", "test"}, p.split.unique()
        assert list(p.columns) == ["image_id", "split", "true_grade", "model_score", "model_type"]
        if missing:
            sys.exit(f"FAIL: missing outputs {missing}")
        print("\nPASS")
