""" Data loader and split check.

    python part_b/check_data.py                      # uses splits.csv and data/cache_clahe_512
    python part_b/check_data.py --cache /content/cache_clahe_512

Loads the supplied partition, derives the referable label (grade >= 2) while keeping the
original 0-4 grade, and checks class counts, image IDs, duplicate-image leakage and missing
images. Writes results/part_b/data_check.txt and results/part_b/class_counts.csv.
Exit code 1 on a hard failure.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.common import add_path_args, check_splits, get_paths, load_splits

if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(p)
    p.add_argument("--no-images", action="store_true", help="skip the image-folder check")
    a = p.parse_args()
    paths = get_paths(a.root, a.cache)
    df = load_splits(paths.splits)
    ok, lines, counts = check_splits(df, None if a.no_images else paths.cache)
    text = "\n".join(lines)
    print(text)
    (paths.results / "data_check.txt").write_text(text + f"\n\nOVERALL: {'PASS' if ok else 'FAIL'}\n")
    counts.to_csv(paths.results / "class_counts.csv")
    print(f"\nOVERALL: {'PASS' if ok else 'FAIL'}")
    sys.exit(0 if ok else 1)