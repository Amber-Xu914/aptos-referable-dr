"""Create the shared, fixed train/val/test partition for APTOS 2019 (70/15/15).

- Stratified by original grade (0-4), so every split has the same grade mix.
- Byte-identical images (same MD5) are grouped and always placed in the same
  split, so a duplicate cannot leak from train into val/test.
- Duplicate groups whose copies carry different grades are flagged
  (`label_conflict`) but kept, so the group can decide how to handle them.
- Fixed seed, so re-running gives exactly the same file.

Usage:
    python scripts/make_splits.py \
        --csv "data/raw/Aptos2019 retinal image database/train.csv" \
        --images "data/raw/Aptos2019 retinal image database/train_images" \
        --out splits.csv
"""

import argparse
import csv
import hashlib
from collections import defaultdict
from pathlib import Path

import numpy as np

SEED = 9027
FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(csv_path, image_dir, out_path, seed=SEED):
    with open(csv_path, newline="") as f:
        rows = [{"id_code": r["id_code"], "diagnosis": int(r["diagnosis"])} for r in csv.DictReader(f)]

    # Group byte-identical images.
    groups = defaultdict(list)
    for r in rows:
        r["md5"] = md5(Path(image_dir) / f"{r['id_code']}.png")
        groups[r["md5"]].append(r)

    # Stratify groups by grade. A conflicting group uses its highest grade,
    # so it is counted with the more severe (referable) side.
    strata = defaultdict(list)
    for members in groups.values():
        strata[max(m["diagnosis"] for m in members)].append(members)

    rng = np.random.default_rng(seed)
    for grade in sorted(strata):
        group_list = strata[grade]
        rng.shuffle(group_list)
        n_total = sum(len(g) for g in group_list)
        targets = {"val": round(FRACTIONS["val"] * n_total), "test": round(FRACTIONS["test"] * n_total)}
        counts = {"val": 0, "test": 0}
        for members in group_list:
            # Fill val, then test, then the rest goes to train.
            split = next((s for s in ("val", "test") if counts[s] < targets[s]), "train")
            if split != "train":
                counts[split] += len(members)
            for m in members:
                m["split"] = split

    dup_ids = {}
    for i, (h, members) in enumerate(sorted(g for g in groups.items() if len(g[1]) > 1), 1):
        conflict = len({m["diagnosis"] for m in members}) > 1
        for m in members:
            dup_ids[m["id_code"]] = (f"dup{i:03d}", int(conflict))

    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id_code", "diagnosis", "referable", "split", "dup_group", "label_conflict"])
        for r in rows:
            dup_group, conflict = dup_ids.get(r["id_code"], ("", 0))
            w.writerow([r["id_code"], r["diagnosis"], int(r["diagnosis"] >= 2), r["split"], dup_group, conflict])

    # Summary
    print(f"Wrote {out_path} (seed={seed})")
    for s in FRACTIONS:
        sub = [r for r in rows if r["split"] == s]
        grades = [sum(r["diagnosis"] == g for r in sub) for g in range(5)]
        ref = sum(r["diagnosis"] >= 2 for r in sub)
        print(f"  {s:5s} n={len(sub):4d}  grades 0-4={grades}  referable={ref} ({ref / len(sub):.1%})")
    n_dup_groups = len({v[0] for v in dup_ids.values()})
    n_conflict = len({v[0] for v in dup_ids.values() if v[1]})
    print(f"  duplicate groups: {n_dup_groups} ({len(dup_ids)} images), label conflicts: {n_conflict}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--csv", required=True)
    p.add_argument("--images", required=True)
    p.add_argument("--out", default="splits.csv")
    p.add_argument("--seed", type=int, default=SEED)
    a = p.parse_args()
    main(a.csv, a.images, a.out, a.seed)
