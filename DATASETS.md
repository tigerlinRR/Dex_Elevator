# Third-party datasets

Training data used to **bootstrap** the elevator-button detector. No in-house
elevator dataset exists yet, so the *baseline* single-class `button` localizer is
trained on public datasets and later fine-tuned on our own chest-335 captures.

**Processing applied to every source** (see `yolo/prepare_dataset.py`): all classes
are collapsed to a single class `button` (localization only — the original
floor-symbol labels are discarded); near-duplicate images are removed via
perceptual hash; the data is re-split. These are **modifications** in the CC BY
sense, disclosed here.

> Attribution is required by CC BY 4.0 and costs nothing, so we credit sources here
> regardless of whether the trained weights are legally a "derivative". Confirm each
> source's license on its page before using it in a commercial build.

---

## elevator-buttons (yolov7ncku)

- **Source:** https://universe.roboflow.com/yolov7ncku/elevator-buttons-scpv6
- **License:** CC BY 4.0 — https://creativecommons.org/licenses/by/4.0/  ✅ confirmed
- **Author:** YOLOv7NCKU (Roboflow Universe), 2023.
- **Modifications:** classes remapped to single `button`; perceptual-hash de-dup; re-split.

```bibtex
@misc{ elevator-buttons-scpv6_dataset,
  title = { elevator-buttons Dataset },
  type = { Open Source Dataset },
  author = { YOLOv7NCKU },
  howpublished = { \url{ https://universe.roboflow.com/yolov7ncku/elevator-buttons-scpv6 } },
  url = { https://universe.roboflow.com/yolov7ncku/elevator-buttons-scpv6 },
  journal = { Roboflow Universe },
  publisher = { Roboflow },
  year = { 2023 },
  month = { mar },
  note = { visited on 2026-07-09 },
}
```

## elevator-button-recognition (Sun Moon University)

- **Source:** https://universe.roboflow.com/sun-moon-university/elevator-button-recognition (v1)
- **License:** **CC BY 4.0** ✅ confirmed (export `README.roboflow.txt` + `data.yaml: license: CC BY 4.0`).
- **Author:** Sun Moon University (Roboflow Universe), exported 2023-04-27.
- **Contents:** 2019 images (1412/405/202 train/valid/test), **368 classes**, bounding-box labels.
  Preprocessed to 416×416 (stretch), no augmentation.
- **Modifications:** all 368 classes remapped to single `button`; perceptual-hash de-dup; re-split.

<!-- Add new sources above this line: source URL, license (confirmed), author, modifications. -->
