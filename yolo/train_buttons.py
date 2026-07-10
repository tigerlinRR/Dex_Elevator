"""Train the single-class ``button`` detector (YOLO11) on the prepared dataset.

Baseline bootstrap from public data -- run :mod:`yolo.prepare_dataset` first (see
that module + ``DATASETS.md``). The best checkpoint is copied to the path in
``configs/pipeline.yaml:button_yolo.weights`` (default ``data/weights/buttons.pt``),
so :class:`~yolo.button_detector.ButtonDetector` and the pipeline pick it up unchanged.

Run on the robot's GPU (the ``richtech-v3`` env has ultralytics + cuda)::

    python yolo/train_buttons.py                 # defaults: yolo11n, imgsz from config
    python yolo/train_buttons.py --model yolo11s.pt --epochs 150
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from core.config import REPO_ROOT, load_pipeline

DEFAULT_DATA = REPO_ROOT / "data" / "datasets" / "buttons" / "data.yaml"


def main() -> None:
    cfg = load_pipeline().get("button_yolo", {})
    default_out = REPO_ROOT / (cfg.get("weights") or "data/weights/buttons.pt")

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=str(DEFAULT_DATA),
                    help="dataset data.yaml (default: the prepared button dataset)")
    ap.add_argument("--model", default="yolo11n.pt",
                    help="base weights to fine-tune (yolo11n/s/m...; -seg for masks)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=int(cfg.get("imgsz", 1280)))
    ap.add_argument("--batch", type=int, default=-1, help="-1 = auto-fit to VRAM")
    ap.add_argument("--device", default=str(cfg.get("device", "cuda") or "cuda"))
    ap.add_argument("--name", default="buttons", help="ultralytics run name")
    ap.add_argument("--out", default=str(default_out),
                    help="where to copy best.pt (pipeline weights path)")
    args = ap.parse_args()

    if not Path(args.data).exists():
        raise SystemExit(f"dataset not found: {args.data}\n"
                         f"run `python yolo/prepare_dataset.py --zip <export.zip>` first")

    from ultralytics import YOLO  # heavy import; keep out of module load

    model = YOLO(args.model)
    results = model.train(
        data=args.data, epochs=args.epochs, imgsz=args.imgsz,
        batch=args.batch, device=args.device, name=args.name,
    )

    # locate the best checkpoint and copy it to the pipeline's expected path
    best = getattr(getattr(model, "trainer", None), "best", None)
    if not best:
        best = Path(getattr(results, "save_dir", f"runs/detect/{args.name}")) / "weights" / "best.pt"
    best = Path(best)
    if not best.exists():
        raise SystemExit(f"training finished but best.pt not found at {best}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out)
    print(f"\ncopied {best}  ->  {out}")
    print("pipeline will use this automatically (configs/pipeline.yaml:button_yolo.weights)")


if __name__ == "__main__":
    main()
