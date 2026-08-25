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
# --------------------------------------------------------------------------- #
# Class-name canonicalisation across sources
#
# Public elevator datasets do NOT agree on what to call a button. Merging them by
# class id silently corrupts labels; merging by NAME needs this table first:
#
#   floor 3   ->  "3" (sun_moon)      "three" (annotations/saga)   "button-3" (entc)
#   ground    ->  "G"                 —                            "button-g"
#
# Anything not listed passes through unchanged, so the 368-class taxonomy of the
# largest source stays the canonical vocabulary and the small sets fold into it.
# --------------------------------------------------------------------------- #
CLASS_ALIASES: dict[str, str] = {
    # spelled-out digits (annotations, saga)
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
    # "button-" prefixed (entc)
    "button-1": "1", "button-2": "2", "button-3": "3", "button-4": "4",
    "button-5": "5", "button-g": "G", "button-up": "up", "button-down": "down",
    "button-open": "open", "button-close": "close", "button-alarm": "alarm",
}

# Classes that are NOT buttons, or whose meaning is ambiguous enough that folding
# them in would inject wrong labels. Dropped with a count so the loss is visible.
#   closed-door  — a door STATE, not a button (door state is a depth problem here)
#   floor-*      — entc has BOTH `button-1` and `floor-1`; the latter is most likely
#                  the floor INDICATOR display, and guessing wrong mislabels a button
#   red-sqr      — undocumented; only in one 542-image set
CLASS_DROP: set[str] = {
    "closed-door", "floor-1", "floor-2", "floor-3", "floor-ground", "red-sqr",
}


def canonical(name: str) -> str | None:
    """Canonical class name, or ``None`` if this class should be dropped."""
    n = name.strip()
    if n in CLASS_DROP:
        return None
    return CLASS_ALIASES.get(n.lower(), n)


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


def write_label(src_lbl: Path | None, dst_lbl: Path, collapse: bool = False,
                remap: dict[int, int] | None = None) -> tuple[int, int]:
    """Copy a YOLO label file, returning ``(kept, dropped)`` box counts.

    ``collapse=True`` forces class 0 (single ``button`` class). Otherwise ``remap``
    translates this source's class ids into the merged vocabulary's ids; an id
    missing from ``remap`` is a class this merge drops, and its boxes go with it.
    With neither, the original ids are kept (single-source multi-class).

    Works for bbox (5 tokens) and segmentation (>5 tokens). An empty or missing
    label becomes an empty file (a valid background image).
    """
    out_lines: list[str] = []
    dropped = 0
    if src_lbl and src_lbl.exists():
        for line in src_lbl.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            if collapse:
                parts[0] = "0"
            elif remap is not None:
                tgt = remap.get(int(parts[0]))
                if tgt is None:
                    dropped += 1
                    continue
                parts[0] = str(tgt)
            out_lines.append(" ".join(parts))
    dst_lbl.write_text("\n".join(out_lines) + ("\n" if out_lines else ""))
    return len(out_lines), dropped


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

        # Multi-class (default). Sources are merged by canonical class NAME, never
        # by class id — different datasets number their classes differently, so an
        # id-based merge silently relabels every box. Sources are taken
        # largest-vocabulary-first so the richest taxonomy defines the ids and the
        # smaller sets fold into it rather than the other way round.
        names: list[str] | None = None
        remaps: dict[str, dict[int, int]] = {}
        if not args.collapse:
            per_source: list[tuple[str, Path, list[str]]] = []
            for tag, root in sources:
                n = read_names(root)
                if not n:
                    sys.exit(f"could not read `names` from {tag}'s data.yaml; "
                             "pass --collapse to train a single-class detector instead.")
                per_source.append((tag, root, n))
            per_source.sort(key=lambda t: -len(t[2]))

            vocab: dict[str, int] = {}
            for tag, _root, n in per_source:
                for raw in n:
                    c = canonical(raw)
                    if c is not None and c not in vocab:
                        vocab[c] = len(vocab)
            names = [c for c, _ in sorted(vocab.items(), key=lambda kv: kv[1])]

            print(f"multi-class: {len(names)} canonical classes from "
                  f"{len(per_source)} source(s)")
            for tag, _root, n in per_source:
                m, dropped_names = {}, []
                for i, raw in enumerate(n):
                    c = canonical(raw)
                    if c is None:
                        dropped_names.append(raw)
                    else:
                        m[i] = vocab[c]
                remaps[tag] = m
                renamed = sum(1 for raw in n
                              if canonical(raw) is not None and canonical(raw) != raw)
                print(f"    {tag:<22} {len(n):>4} classes -> {len(m):>4} mapped"
                      f"  ({renamed} renamed, {len(dropped_names)} dropped"
                      + (f": {dropped_names}" if dropped_names else "") + ")")
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

        # 3b. make sure val is big enough to MEAN anything -------------------- #
        # Trigger on the val FRACTION, not on val being absent. Merging several
        # sources reliably produces a token val split — saga ships 5 val images,
        # annotations 10 — and "some source had a val split" was enough to skip the
        # carve entirely. That left 194 val images for a 369-class problem, i.e.
        # well under one example per class, so every mAP number would have been
        # noise while still looking like a real measurement.
        n_val = sum(1 for r in kept if r["split"] == "val")
        want = args.val_frac * len(kept)
        if n_val < 0.5 * want:
            need = want - n_val
            pool = [r for r in kept if r["split"] == "train"]
            # deterministic and independent of source order: rank by pHash
            pool.sort(key=lambda r: r["hash"] % 100003)
            for r in pool[:int(need)]:
                r["split"] = "val"
            print(f"  val was {n_val} ({n_val/len(kept):.1%}) -> carved "
                  f"{int(need)} more from train, now "
                  f"{sum(1 for r in kept if r['split'] == 'val')} "
                  f"(~{args.val_frac:.0%} target)")

        # 4. write the merged dataset ------------------------------------- #
        if args.clean and out.exists():
            shutil.rmtree(out)
        for split in SPLIT_PRIORITY:
            (out / "images" / split).mkdir(parents=True, exist_ok=True)
            (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        boxes_kept = boxes_dropped = 0
        for r in kept:
            split = r["split"]
            name = f'{r["tag"]}__{r["img"].name}'
            shutil.copy2(r["img"], out / "images" / split / name)
            k, d = write_label(r["lbl"], out / "labels" / split / f"{Path(name).stem}.txt",
                               collapse=args.collapse,
                               remap=remaps.get(r["tag"]))
            boxes_kept += k
            boxes_dropped += d
            counts[split] = counts.get(split, 0) + 1
        # Report what the merge threw away — a silent drop here looks exactly like
        # "that class was never in the data".
        print(f"  boxes: {boxes_kept} kept"
              + (f", {boxes_dropped} dropped (classes in CLASS_DROP)" if boxes_dropped else ""))

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
