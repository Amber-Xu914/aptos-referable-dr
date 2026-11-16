"""Find the images in the repository's `data/` folder (downloaded by hand; data/ is git-ignored).

Searched below data/ (any depth up to 4), in this order:
    1. a folder `cache_clahe_512/` of Part C's preprocessed PNGs     -> used in place
    2. a file   `cache_clahe_512.zip`                                -> unzipped to data/cache_clahe_512/
    3. the raw APTOS `train_images/` folder                          -> preprocessed once with Part C's
                                                                        src/preprocess.py (clahe_rgb, 512 px)
                                                                        into data/cache_clahe_512/
Every image listed in splits.csv must be present, otherwise the script stops.

    python part_b/local_data.py                 # report what was found
    python part_b/local_data.py --data-dir D:/aptos_data
"""
import argparse
import os
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from part_b.common import add_path_args, get_paths, load_splits

CACHE_NAME = "cache_clahe_512"
RAW_NAME = "train_images"


def _search(top, max_depth=4):
    found = {"cache": None, "zip": None, "raw": None}
    level = [Path(top)]
    for _ in range(max_depth + 1):
        nxt = []
        for d in level:
            try:
                entries = sorted(os.scandir(d), key=lambda e: e.name)
            except OSError:
                continue
            for e in entries:
                if e.name.startswith("."):
                    continue
                if e.is_dir():
                    key = "cache" if e.name == CACHE_NAME else "raw" if e.name == RAW_NAME else None
                    if key and not found[key]:
                        found[key] = Path(e.path)
                    elif not key:
                        nxt.append(e.path)
                elif e.name == f"{CACHE_NAME}.zip" and not found["zip"]:
                    found["zip"] = Path(e.path)
        level = nxt
    return found


def _missing(folder, ids):
    have = {p.stem for p in Path(folder).glob("*.png")} if folder and Path(folder).is_dir() else set()
    return [i for i in ids if i not in have]


def _unzip(zip_path, dest, ids):
    print(f"unzipping {zip_path} -> {dest}")
    dest.mkdir(parents=True, exist_ok=True)
    wanted = set(ids)
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            stem = Path(name).stem
            if name.lower().endswith(".png") and stem in wanted and not (dest / f"{stem}.png").exists():
                (dest / f"{stem}.png").write_bytes(z.read(name))


def _build_from_raw(raw_dir, dest, ids):
    import cv2
    from src.preprocess import preprocess
    dest.mkdir(parents=True, exist_ok=True)
    todo = [i for i in _missing(dest, ids) if (raw_dir / f"{i}.png").exists()]
    print(f"no preprocessed cache found; preprocessing {len(todo)} raw images from {raw_dir} "
          f"(Part C clahe_rgb, 512 px) into {dest}. This runs once.")
    for n, i in enumerate(todo, 1):
        out = preprocess(raw_dir / f"{i}.png", size=512, mode="clahe_rgb")
        cv2.imwrite(str(dest / f"{i}.png"), cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
        if n % 250 == 0 or n == len(todo):
            print(f"  {n}/{len(todo)}", flush=True)


def find_images(paths, data_dir=None):
    """Return the folder of preprocessed images that covers every image in splits.csv.

    An explicit --cache wins. Otherwise data/ is searched as described in the module docstring.
    Raises FileNotFoundError with a clear message when the images are not (all) there.
    """
    ids = list(load_splits(paths.splits)["id_code"])
    if paths.cache.is_dir() and not _missing(paths.cache, ids):
        print(f"images: all {len(ids)} found in {paths.cache}")
        return paths.cache

    data_dir = Path(data_dir) if data_dir else paths.root / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"data folder not found: {data_dir}\n"
                                f"Put the downloaded files in a folder named 'data' in the repository.")
    f = _search(data_dir)
    dest = paths.root / "data" / CACHE_NAME
    if f["cache"] and not _missing(f["cache"], ids):
        src = f["cache"]
    elif f["zip"]:
        _unzip(f["zip"], dest, ids); src = dest
    elif f["raw"]:
        _build_from_raw(f["raw"], dest, ids); src = dest
    elif f["cache"]:
        src = f["cache"]                                  # incomplete cache: report below
    else:
        raise FileNotFoundError(
            f"no '{CACHE_NAME}' folder, '{CACHE_NAME}.zip' or raw '{RAW_NAME}' folder found under {data_dir}.\n"
            f"Expected e.g. data/{CACHE_NAME}/<id_code>.png")
    missing = _missing(src, ids)
    if missing:
        raise FileNotFoundError(f"{len(missing)} of {len(ids)} images in splits.csv are not in {src}, "
                                f"e.g. {', '.join(missing[:5])}")
    print(f"images: all {len(ids)} found in {src}")
    return src


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    add_path_args(ap)
    ap.add_argument("--data-dir", default=None, help="folder to search (default: <repository>/data)")
    a = ap.parse_args()
    try:
        find_images(get_paths(a.root, a.cache), a.data_dir)
    except FileNotFoundError as e:
        sys.exit(f"ERROR: {e}")
