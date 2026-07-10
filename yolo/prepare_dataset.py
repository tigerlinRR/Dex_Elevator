"""Build the elevator-button YOLO dataset from one or more Roboflow exports.

By default this is **multi-class**: each floor symbol keeps its own label (``1``,
``2``, ``B1``, ``G``, ``open``…), so the trained detector both *finds* a button and
*identifies which floor it is* in one shot. Pass ``--collapse`` to instead flatten
every class to a single ``button`` (localization only — identification handled
elsewhere); that mode can merge sources with different taxonomies.

No in-house elevator dataset exists yet, so we bootstrap from public CC BY datasets
on Roboflow Universe, then fine-tune on our own chest-335 captures later. See
``DATASETS.md`` (repo root) for attribution.

Pipeline::

    N source exports  ->  keep original labels (or --collapse to `button`)
                      ->  perceptual-hash de-dup (kills fork/re-export overlap
                          and train/val leakage)
                      ->  merged dataset under data/datasets/buttons/

Inputs (mix freely, each repeatable):
  --zip PATH            a Roboflow "YOLOv8" export .zip (download from the UI)  [preferred]
  --src DIR             an already-extracted YOLO dataset dir (train/valid/test)
  --roboflow WS/PROJ/V  auto-download via REST (needs ROBOFLOW_API_KEY + requests) [best-effort]

``--zip``/``--src`` are preferred on the robot: they need nothing beyond cv2/numpy
(already in the training env), whereas the ``roboflow`` pip package can pull
``numpy>=2``. If ``--roboflow`` fails, download the export zip from the Roboflow UI
("Download Dataset" -> YOLOv8) and pass it with ``--zip``.

MULTI-CLASS (default) keeps the source's own class ids + names, so it needs a
**single** source (different datasets number their classes differently; merging by
id would corrupt labels). Use ``--collapse`` to merge several sources into one
``button`` class.

De-dup keeps split priority train > val > test: if the same image appears in several
splits (common when two sources fork one upstream set), the higher-priority copy is
kept and the rest dropped -- so no image lands in both our train and val.

NOTE: pHash catches identical / re-encoded / rescaled duplicates. It does NOT catch
augmentation variants (flips/rotations Roboflow bakes into train) -- acceptable for a
baseline; the real domain fix is in-house fine-tuning.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from core.config import REPO_ROOT

OUT_DIR = REPO_ROOT / "data" / "datasets" / "buttons"

# Roboflow uses "valid"; ultralytics wants "val". Canonicalize on the way in.
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SPLIT_PRIORITY = ["train", "val", "test"]  # de-dup keeps the earliest-priority copy
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# --------------------------------------------------------------------------- #
# perceptual hash (DCT pHash, cv2-only -- no imagehash/PIL dependency)
# --------------------------------------------------------------------------- #
def phash(path: Path, hash_size: int = 8) -> int | None:
    """64-bit DCT perceptual hash of an image, or ``None`` if it can't be read."""
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    size = hash_size * 4
    resized = cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(resized)[:hash_size, :hash_size]
    bits = (low > float(np.median(low))).flatten()
    h = 0
    for b in bits:
        h = (h << 1) | int(b)
    return h


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


# --------------------------------------------------------------------------- #
# source discovery / class names / label copy
# --------------------------------------------------------------------------- #
def locate_root(base: Path) -> Path:
    """Find the dataset root inside an extracted archive (handles a nested folder)."""
    for c in [base, *(p for p in base.iterdir() if p.is_dir())]:
        if (c / "data.yaml").exists() or (c / "train").is_dir() or (c / "images").is_dir():
            return c
    return base


def read_names(root: Path) -> list[str] | None:
    """Class names from a source's ``data.yaml`` (list, or dict keyed by id)."""
    p = Path(root) / "data.yaml"
    if not p.exists():
        return None
    d = yaml.safe_load(p.read_text()) or {}
    names = d.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names, key=lambda x: int(x))]
    return list(names) if names else None


def find_pairs(root: Path) -> list[tuple[str, Path, Path | None]]:
    """Yield ``(canonical_split, image, label_or_None)`` for a YOLO export dir.

    Supports both the Roboflow layout (``<split>/images``, ``<split>/labels``) and
    the ultralytics layout (``images/<split>``, ``labels/<split>``).
    """
    pairs: list[tuple[str, Path, Path | None]] = []
    layouts = [
        lambda s: (root / s / "images", root / s / "labels"),      # roboflow
        lambda s: (root / "images" / s, root / "labels" / s),      # ultralytics
    ]
    for raw_split, split in SPLIT_ALIASES.items():
        for layout in layouts:
            img_dir, lbl_dir = layout(raw_split)
            if not img_dir.is_dir():
                continue
            for img in sorted(img_dir.iterdir()):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lbl = lbl_dir / (img.stem + ".txt")
                pairs.append((split, img, lbl if lbl.exists() else None))
    return pairs


def write_label(src_lbl: Path | None, dst_lbl: Path, collapse: bool = False) -> None:
    """Copy a YOLO label file.

    ``collapse=False`` (default) keeps the original class ids (multi-class: each
    floor symbol stays its own class). ``collapse=True`` forces class 0 (single
    ``button`` class). Works for bbox (5 tokens) and segmentation (>5 tokens). An
    empty or missing label becomes an empty file (a valid background image).
    """
    out_lines: list[str] = []
    if src_lbl and src_lbl.exists():
        for line in src_lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            if collapse:
                parts[0] = "0"
            out_lines.append(" ".join(parts))
    dst_lbl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))


# --------------------------------------------------------------------------- #
# optional Roboflow REST download (best-effort)
# --------------------------------------------------------------------------- #
def download_roboflow(spec: str, dest: Path) -> Path:
    try:
        import requests  # lazy: only needed for --roboflow
    except ImportError:
        sys.exit("--roboflow needs `requests`. Install it, or download the export "
                 "zip from the Roboflow UI and pass it with --zip.")
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        sys.exit("Set ROBOFLOW_API_KEY (free account) for --roboflow, or use --zip "
                 "with a manually downloaded export.")
    try:
        ws, proj, ver = spec.split("/")
    except ValueError:
        sys.exit(f"--roboflow expects WORKSPACE/PROJECT/VERSION, got: {spec!r}")
    api = f"https://api.roboflow.com/{ws}/{proj}/{ver}/yolov8?api_key={key}"
    r = requests.get(api, timeout=120)
    r.raise_for_status()
    link = (r.json().get("export") or {}).get("link")
    if not link:
        raise RuntimeError(f"no export link in Roboflow response: {r.text[:200]}")
    zip_path = dest / f"{ws}_{proj}_{ver}.zip"
    with requests.get(link, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        with open(zip_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1 << 20):
                f.write(chunk)
    return zip_path


def _tag(name: str) -> str:
    """Filesystem-safe short tag used to namespace merged filenames."""
    return "".join(c if c.isalnum() else "_" for c in Path(name).stem)[:40] or "src"


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", action="append", default=[], metavar="PATH",
                    help="Roboflow YOLOv8 export .zip (repeatable) -- preferred")
    ap.add_argument("--src", action="append", default=[], metavar="DIR",
                    help="already-extracted YOLO dataset dir (repeatable)")
    ap.add_argument("--roboflow", action="append", default=[], metavar="WS/PROJ/VER",
                    help="auto-download via REST; needs ROBOFLOW_API_KEY (repeatable)")
    ap.add_argument("--collapse", action="store_true",
                    help="flatten ALL classes to a single `button` (localization only). "
                         "Default keeps original labels (multi-class: detect + identify).")
    ap.add_argument("--dup-hamming", type=int, default=5,
                    help="pHash Hamming distance treated as a duplicate (0 = exact). Default 5.")
    ap.add_argument("--val-frac", type=float, default=0.15,
                    help="fraction carved from train for val IF no source provides a val split")
    ap.add_argument("--out", default=str(OUT_DIR), help="output dataset dir")
    ap.add_argument("--clean", action="store_true", help="wipe the output dir first")
    args = ap.parse_args()

    if not (args.zip or args.src or args.roboflow):
        ap.error("give at least one source: --zip / --src / --roboflow")

    out = Path(args.out).resolve()
    work = Path(tempfile.mkdtemp(prefix="buttons_prep_"))
    try:
        # 1. resolve every input to an on-disk dataset root ---------------- #
        sources: list[tuple[str, Path]] = []
        for spec in args.roboflow:
            args.zip.append(str(download_roboflow(spec, work)))
        for z in args.zip:
            zp = Path(z)
            dest = work / _tag(zp.name)
            dest.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp) as zf:
                zf.extractall(dest)
            sources.append((_tag(zp.name), locate_root(dest)))
        for d in args.src:
            sources.append((_tag(Path(d).name), locate_root(Path(d))))

        # multi-class (default): keep the single source's own class taxonomy.
        names = None
        if not args.collapse:
            if len(sources) != 1:
                sys.exit("multi-class needs exactly ONE source (different datasets number "
                         "their classes differently; merging by id would corrupt labels).\n"
                         "Use one --src/--zip, or pass --collapse to flatten all to `button`.")
            names = read_names(sources[0][1])
            if not names:
                sys.exit("could not read `names` from the source data.yaml; "
                         "pass --collapse to train a single-class detector instead.")
            print(f"multi-class: {len(names)} classes from {sources[0][0]}")
        else:
            print("collapse: all classes -> single `button`")

        # 2. gather (split, image, label) records across all sources ------- #
        records: list[dict] = []
        for tag, root in sources:
            pairs = find_pairs(root)
            if not pairs:
                print(f"  ! WARNING: no train/valid/test images found under {root}")
            for split, img, lbl in pairs:
                records.append({"tag": tag, "split": split, "img": img, "lbl": lbl})
        if not records:
            sys.exit("no images found in any source -- check the export layout")

        print(f"Found {len(records)} images across {len(sources)} source(s). Hashing...")
        for r in records:
            r["hash"] = phash(r["img"])

        # 3. de-dup, keeping split priority train > val > test ------------- #
        records.sort(key=lambda r: SPLIT_PRIORITY.index(r["split"]))
        kept: list[dict] = []
        kept_hashes: list[int] = []
        exact: set[int] = set()
        dropped = 0
        for r in records:
            h = r["hash"]
            if h is None:
                dropped += 1
                continue
            if h in exact or (args.dup_hamming > 0
                              and any(hamming(h, kh) <= args.dup_hamming for kh in kept_hashes)):
                dropped += 1
                continue
            kept.append(r)
            kept_hashes.append(h)
            exact.add(h)

        # 3b. carve a val split if none of the sources had one ------------- #
        if not any(r["split"] == "val" for r in kept):
            cut = int(args.val_frac * 100)
            for r in kept:
                if r["split"] == "train" and (r["hash"] % 100) < cut:
                    r["split"] = "val"
            print(f"  no source val split -> carved ~{cut}% of train into val (by pHash)")

        # 4. write the merged dataset ------------------------------------- #
        if args.clean and out.exists():
            shutil.rmtree(out)
        for split in SPLIT_PRIORITY:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for r in kept:
            split = r["split"]
            name = f'{r["tag"]}__{r["img"].name}'
            shutil.copy2(r["img"], out / "images" / split / name)
            write_label(r["lbl"], out / "labels" / split / f"{Path(name).stem}.txt",
                        collapse=args.collapse)
            counts[split] = counts.get(split, 0) + 1

        # 5. data.yaml (absolute path -> works regardless of cwd) + provenance
        data_yaml: dict = {
            "path": str(out),
            "train": "images/train",
            "val": "images/val",
            "nc": len(names) if names else 1,
            "names": names if names else ["button"],
        }
        if counts.get("test"):
            data_yaml["test"] = "images/test"
        (out / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False))
        (out / "SOURCES.json").write_text(json.dumps({
            "sources": [{"tag": t, "root": str(root)} for t, root in sources],
            "mode": "single-class `button`" if args.collapse else f"multi-class ({len(names)} classes)",
            "dup_hamming": args.dup_hamming,
            "dropped_duplicates": dropped,
            "counts": counts,
        }, indent=2))

        print("\n=== done ===")
        print(f"  sources           : {', '.join(t for t, _ in sources)}")
        print(f"  mode              : {'single-class button' if args.collapse else str(len(names)) + '-class (detect+identify)'}")
        print(f"  duplicates dropped : {dropped}")
        print(f"  train / val / test : {counts.get('train', 0)} / "
              f"{counts.get('val', 0)} / {counts.get('test', 0)}")
        print(f"  dataset            : {out}")
        print(f"  train with         : python yolo/train_buttons.py")
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
