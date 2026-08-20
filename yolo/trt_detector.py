"""Button detection on the Orin's GPU, through TensorRT directly — no torch.

Why not Ultralytics? On this robot ``torch`` is the generic ``2.10.0+cpu`` aarch64
wheel (``torch.cuda.is_available()`` is False), so Ultralytics runs the model on the
CPU: 491 ms per frame at imgsz 640. The GPU fix is NOT to replace that torch —
``~/.local`` is shared and ``~/mmdetection`` imports the same install, so swapping in
NVIDIA's JetPack build would change another project's environment underneath it.

JetPack already ships TensorRT 10.3 with working Python bindings, and TensorRT has
nothing to do with torch. So the model goes ``buttons.pt -> ONNX -> .engine`` once,
offline, and inference here talks to the engine directly:

    measured on the AGX Orin, yolo11m @ 640, FP16
        Ultralytics + CPU torch   491 ms   (2.0 FPS)
        this module (TensorRT)    6.4 ms   (157 FPS)     ~77x

Build the engine (once per machine — engines are NOT portable, they are tuned for the
specific GPU, and this takes ~10 min):

    PYTHONPATH=~/Dex_Elevator/.pydeps-onnx python3 -c \\
      'from ultralytics import YOLO; YOLO("data/weights/buttons.pt").export(
           format="onnx", imgsz=640, opset=17, simplify=False, dynamic=False)'
    /usr/src/tensorrt/bin/trtexec --onnx=data/weights/buttons.onnx \\
        --saveEngine=data/weights/buttons_fp16.engine --fp16

The class names do not survive into the engine, so they are written alongside it as
``buttons_names.json`` (see ``export_names`` below).
"""
from __future__ import annotations

import ctypes
import json
from pathlib import Path
from typing import Optional

import numpy as np

from core.config import REPO_ROOT
from core.types import Detection

try:  # both are JetPack-only, like every other SDK in this repo
    import tensorrt as trt  # type: ignore
    from cuda import cudart  # type: ignore

    _TRT_AVAILABLE = True
except ImportError:  # pragma: no cover - laptop / no JetPack
    _TRT_AVAILABLE = False

DEFAULT_ENGINE = "data/weights/buttons_fp16.engine"
DEFAULT_NAMES = "data/weights/buttons_names.json"


def _check(ret):
    """cuda-python returns ``(err, ...)`` from every call; raise on a non-zero err."""
    err, *rest = ret if isinstance(ret, tuple) else (ret,)
    if int(err) != 0:
        raise RuntimeError(f"CUDA call failed with error {int(err)}")
    return rest[0] if len(rest) == 1 else rest


def letterbox(bgr: np.ndarray, size: int = 640) -> tuple[np.ndarray, float, int, int]:
    """Resize keeping aspect ratio and pad to ``size`` x ``size`` (grey 114, as trained).

    Returns the padded image plus the scale and offsets needed to map boxes back to
    the original pixel coordinates.
    """
    import cv2

    h, w = bgr.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    out[dy:dy + nh, dx:dx + nw] = resized
    return out, scale, dx, dy


class TrtButtonDetector:
    """YOLO button detector running on the GPU through a TensorRT engine.

    Same output contract as :class:`yolo.button_detector.ButtonDetector` — a list of
    :class:`~core.types.Detection` in ORIGINAL image pixels — so it drops into the
    same call sites.

    ``detect`` takes **BGR** (OpenCV order), matching how the model was trained.
    ``CameraFrame.rgb`` is RGB, so convert before calling.
    """

    def __init__(
        self,
        engine_path: str | Path = DEFAULT_ENGINE,
        names_path: str | Path | None = DEFAULT_NAMES,
        conf: float = 0.25,
        iou: float = 0.45,
        zero_copy: bool = True,
    ):
        if not _TRT_AVAILABLE:
            raise ImportError(
                "tensorrt / cuda-python not importable; run this on the Orin "
                "(JetPack ships TensorRT; cuda-python is a --user install)."
            )
        self.engine_path = Path(engine_path)
        if not self.engine_path.is_absolute():
            self.engine_path = REPO_ROOT / self.engine_path
        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"{self.engine_path} not found. Engines are per-GPU and must be built "
                "on this machine — see the module docstring for the two commands."
            )
        self.conf = conf
        self.iou = iou
        self.zero_copy = zero_copy

        self.names: dict[int, str] = {}
        if names_path:
            p = Path(names_path)
            if not p.is_absolute():
                p = REPO_ROOT / p
            if p.exists():
                self.names = {int(k): v for k, v in json.loads(p.read_text()).items()}

        logger = trt.Logger(trt.Logger.WARNING)
        runtime = trt.Runtime(logger)
        self.engine = runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self.engine is None:
            raise RuntimeError(
                f"could not deserialize {self.engine_path}. An engine built for a "
                "different GPU or TensorRT version fails exactly like this — rebuild it."
            )
        self.ctx = self.engine.create_execution_context()
        self.stream = _check(cudart.cudaStreamCreate())

        # One input, one output for a plain detect model. Shapes are static (the
        # engine was built with dynamic=False), so buffers are allocated once.
        #
        # HOW THE HOST BUFFERS ARE ALLOCATED IS WORTH ~16 ms. Measured on this Orin,
        # median of 200 runs after 120+ warm-up frames — the warm-up is not optional:
        # the CPU governor is schedutil and the GPU idles at 306 MHz, so whatever is
        # measured first pays for the clock ramp and everything after it looks faster.
        # (Two numbers earlier in this session were wrong for exactly that reason.)
        #
        #   plain malloc + cudaMemcpy   42 ms   pageable copies move all 17 MB by hand
        #   zero-copy (mapped)          26 ms   no copy at all
        #   pinned host + device mem    28 ms   A/B'd twice, interleaved: 44.9 / 45.9 ms
        #                                      end-to-end vs 43.2 / 43.2 for zero-copy
        #
        # Zero-copy wins, but only just, and NOT for the reason it looks like: it does
        # remove the copies, yet the GPU then reads and writes UNCACHED host memory,
        # which costs it 13.2 ms of compute instead of the 6.4 ms trtexec measures
        # against device memory. The copy it saves and the slowdown it causes nearly
        # cancel. Pinned+device was expected to win clearly and did not — hence the
        # flag, so the comparison can be re-run on a model with a smaller output.
        self.bindings: dict[str, dict] = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = np.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            nbytes = int(np.prod(shape)) * dtype.itemsize
            flag = (cudart.cudaHostAllocMapped if self.zero_copy
                    else cudart.cudaHostAllocDefault)
            host_ptr = _check(cudart.cudaHostAlloc(nbytes, flag))
            if self.zero_copy:
                dev_ptr, dev_owned = _check(
                    cudart.cudaHostGetDevicePointer(host_ptr, 0)), False
            else:
                dev_ptr, dev_owned = _check(cudart.cudaMalloc(nbytes)), True
            host = np.ctypeslib.as_array(
                ctypes.cast(int(host_ptr),
                            ctypes.POINTER(np.ctypeslib.as_ctypes_type(dtype))),
                shape)
            self.ctx.set_tensor_address(name, int(dev_ptr))
            self.bindings[name] = {
                "shape": shape, "dtype": dtype, "host": host, "host_ptr": host_ptr,
                "dev_ptr": dev_ptr, "dev_owned": dev_owned, "nbytes": nbytes,
                "is_input": self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT,
            }
        self.in_name = next(n for n, b in self.bindings.items() if b["is_input"])
        self.out_name = next(n for n, b in self.bindings.items() if not b["is_input"])
        self.imgsz = int(self.bindings[self.in_name]["shape"][-1])

    # -- inference ----------------------------------------------------------
    def infer_raw(self, bgr: np.ndarray) -> np.ndarray:
        """Run the engine on one BGR image; return the raw ``(4+nc, N)`` output.

        The returned array is a VIEW of the output buffer, so it is overwritten by
        the next call — copy it if it has to outlive one inference.
        """
        padded, self._scale, self._dx, self._dy = letterbox(bgr, self.imgsz)
        bin_, bout = self.bindings[self.in_name], self.bindings[self.out_name]
        # The normalisation writes straight into the host buffer, so the conversion
        # doubles as the staging step — nothing is copied twice.
        np.divide(padded[:, :, ::-1].transpose(2, 0, 1), 255.0, out=bin_["host"][0],
                  dtype=np.float32, casting="unsafe")
        if not self.zero_copy:
            _check(cudart.cudaMemcpyAsync(
                bin_["dev_ptr"], bin_["host_ptr"], bin_["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, self.stream))
        if not self.ctx.execute_async_v3(self.stream):
            raise RuntimeError("TensorRT execute_async_v3 failed")
        if not self.zero_copy:
            _check(cudart.cudaMemcpyAsync(
                bout["host_ptr"], bout["dev_ptr"], bout["nbytes"],
                cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, self.stream))
        _check(cudart.cudaStreamSynchronize(self.stream))
        return bout["host"][0]

    def detect(self, bgr: np.ndarray) -> list[Detection]:
        """Detect buttons in a BGR image. Boxes come back in ORIGINAL image pixels."""
        import cv2

        pred = self.infer_raw(bgr)                      # (4 + nc, N)
        boxes_xywh = pred[:4].T                         # (N, 4) centre-x, centre-y, w, h
        scores_all = pred[4:].T                         # (N, nc)
        cls = scores_all.argmax(1)
        conf = scores_all[np.arange(len(cls)), cls]
        keep = conf >= self.conf
        if not keep.any():
            return []
        boxes_xywh, cls, conf = boxes_xywh[keep], cls[keep], conf[keep]

        # NMS with OpenCV (already a dependency) rather than pulling in torchvision.
        xywh_int = np.stack([
            boxes_xywh[:, 0] - boxes_xywh[:, 2] / 2,
            boxes_xywh[:, 1] - boxes_xywh[:, 3] / 2,
            boxes_xywh[:, 2], boxes_xywh[:, 3],
        ], 1)
        idx = cv2.dnn.NMSBoxes(xywh_int.tolist(), conf.astype(float).tolist(),
                               self.conf, self.iou)
        if len(idx) == 0:
            return []
        idx = np.array(idx).reshape(-1)

        out: list[Detection] = []
        h, w = bgr.shape[:2]
        for i in idx:
            cx, cy, bw, bh = boxes_xywh[i]
            # undo the letterbox: remove the pad, then the scale
            x1 = (cx - bw / 2 - self._dx) / self._scale
            y1 = (cy - bh / 2 - self._dy) / self._scale
            x2 = (cx + bw / 2 - self._dx) / self._scale
            y2 = (cy + bh / 2 - self._dy) / self._scale
            out.append(Detection(
                label=self.names.get(int(cls[i]), str(int(cls[i]))),
                confidence=float(conf[i]),
                bbox_xyxy=np.array([np.clip(x1, 0, w), np.clip(y1, 0, h),
                                    np.clip(x2, 0, w), np.clip(y2, 0, h)], float),
                class_id=int(cls[i]),
            ))
        return out

    def close(self) -> None:
        for b in self.bindings.values():
            if b["dev_owned"]:
                cudart.cudaFree(b["dev_ptr"])
            cudart.cudaFreeHost(b["host_ptr"])
        cudart.cudaStreamDestroy(self.stream)
        self.bindings.clear()

    def __enter__(self) -> "TrtButtonDetector":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def export_names(weights: str = "data/weights/buttons.pt",
                 out: str = DEFAULT_NAMES) -> int:
    """Write the .pt's class-name table next to the engine (the engine has no names)."""
    from ultralytics import YOLO  # only needed for this offline step

    names = YOLO(str(REPO_ROOT / weights)).names
    path = REPO_ROOT / out
    path.write_text(json.dumps({int(k): v for k, v in names.items()}))
    return len(names)
