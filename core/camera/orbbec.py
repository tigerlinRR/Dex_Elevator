"""Orbbec Gemini 335L driver (pyorbbecsdk v2).

Wraps the Orbbec SDK behind the :class:`Camera` interface. The color stream's
format/fps are **selected from what the device actually offers** (the 335L, for
example, only exposes MJPG at 1280x720, capped at 15 fps — raw RGB is limited to
lower resolutions), then decoded to RGB. Depth is aligned to color and converted
to meters; the organized point cloud is computed on demand. Depth/align are
best-effort so color-only intrinsic calibration still runs if they are
unavailable.

Install: https://github.com/orbbec/pyorbbecsdk  (v2 release wheel — PyPI's is a
broken v1 build).
"""

from __future__ import annotations

import numpy as np

from core.camera.base import Camera
from core.types import CameraFrame, CameraIntrinsics

try:
    import cv2  # type: ignore
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from pyorbbecsdk import (  # type: ignore
        Config,
        Context,
        OBAlignMode,
        OBFormat,
        OBSensorType,
        Pipeline,
    )

    _SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - SDK is an optional runtime dep
    _SDK_AVAILABLE = False


# Color formats we can decode to RGB, in preference order (most direct first).
_FORMAT_PREFERENCE = ("RGB", "BGR", "MJPG", "YUYV", "RGBA", "BGRA")

# One process-wide SDK Context (it owns the device manager). It must outlive any
# DeviceList/Device handed out by query_devices() — a per-call Context gets GC'd
# and later calls fail with "NULL pointer ... deviceMgr" — and the SDK expects a
# single context per process anyway.
_CONTEXT = None


def _shared_context():
    global _CONTEXT
    if _CONTEXT is None:
        _CONTEXT = Context()
    return _CONTEXT


class OrbbecCamera(Camera):
    """Single Orbbec camera. With >1 unit on the bus, select by ``serial`` (exact,
    preferred) or ``match_name`` (device name suffix, e.g. "335" vs "335L")."""

    def __init__(
        self,
        camera_id: str,
        serial: str | None = None,
        match_name: str | None = None,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        intrinsics: CameraIntrinsics | None = None,
    ):
        super().__init__(camera_id)
        if not _SDK_AVAILABLE:
            raise ImportError(
                "pyorbbecsdk is not installed. See "
                "https://github.com/orbbec/pyorbbecsdk for build/install steps."
            )
        if cv2 is None:
            raise ImportError("opencv-contrib-python is required to decode color frames.")
        self.serial = serial
        self.match_name = match_name
        self.width = width
        self.height = height
        self.fps = fps
        self._pipeline: Pipeline | None = None
        self._depth_enabled = False
        # Pre-loaded (Part 1) intrinsics shared from the calibration file; when
        # set, this is authoritative and we do NOT re-read the SDK factory values.
        self._intrinsics: CameraIntrinsics | None = intrinsics

    # -- lifecycle ----------------------------------------------------------
    def _resolve_device(self):
        """The connected device this camera should open, or ``None`` for the SDK default.

        Two Orbbec units are on the DEX robot (Gemini 335 chest + 335L head), so
        ``Pipeline()`` alone would grab whichever the SDK enumerates first. Bind
        explicitly, in order of robustness:
          1. ``serial`` — exact match, read by index (no device open needed);
          2. ``match_name`` — the device whose reported name ENDS WITH the string
             ("335" selects the 335 but not the 335L); reading names opens each
             device briefly, so prefer a serial when a camera is already streaming.
        """
        if self.serial:
            devices = _shared_context().query_devices()
            serials = [devices.get_device_serial_number_by_index(i)
                       for i in range(devices.get_count())]
            if self.serial not in serials:
                raise RuntimeError(
                    f"[{self.camera_id}] no connected Orbbec device with serial "
                    f"{self.serial!r}; connected: {serials or 'none'}. "
                    "Fix the serial in configs/cameras.yaml or plug the camera in."
                )
            return devices.get_device_by_serial_number(self.serial)

        if self.match_name:
            devices = _shared_context().query_devices()
            entries = []  # (name, serial, device)
            for i in range(devices.get_count()):
                dev = devices.get_device_by_index(i)
                name = dev.get_device_info().get_name().strip()
                serial = devices.get_device_serial_number_by_index(i)
                entries.append((name, serial, dev))
            target = self.match_name.strip().lower()
            # Exact-suffix first so "335" doesn't also catch "335L"; then substring.
            picks = [e for e in entries if e[0].lower().endswith(target)] \
                or [e for e in entries if target in e[0].lower()]
            listing = [(n, s) for n, s, _ in entries] or "none"
            if not picks:
                raise RuntimeError(
                    f"[{self.camera_id}] no Orbbec device whose name matches "
                    f"{self.match_name!r}; connected: {listing}."
                )
            if len(picks) > 1:
                raise RuntimeError(
                    f"[{self.camera_id}] match {self.match_name!r} is ambiguous: "
                    f"{[(n, s) for n, s, _ in picks]}. Use an exact serial instead."
                )
            name, serial, dev = picks[0]
            print(f"[{self.camera_id}] matched {name!r} (serial {serial}) by name "
                  f"{self.match_name!r}")
            return dev

        return None

    def start(self) -> None:
        # Bind a specific device by serial so two 335Ls don't collide; with no
        # serial configured, fall back to the SDK's default (first) device.
        device = self._resolve_device()
        self._pipeline = Pipeline(device) if device is not None else Pipeline()
        config = Config()

        color_profiles = self._pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        color_profile = self._select_color_profile(color_profiles)
        # Adopt the profile we actually got (resolution/fps may differ from request).
        self.width = color_profile.get_width()
        self.height = color_profile.get_height()
        self.fps = color_profile.get_fps()
        self._color_format = color_profile.get_format()
        print(f"[{self.camera_id}] color: {self.width}x{self.height} "
              f"{self._color_format} @{self.fps}")
        config.enable_stream(color_profile)

        # Depth + alignment are best-effort: intrinsic calibration needs only
        # color, so a depth/align failure must not block it.
        try:
            depth_profiles = self._pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
            config.enable_stream(depth_profiles.get_default_video_stream_profile())
            config.set_align_mode(OBAlignMode.SW_MODE)
            self._depth_enabled = True
        except Exception as e:  # pragma: no cover - hardware/firmware dependent
            print(f"[{self.camera_id}] depth/align unavailable ({e}); color-only.")

        self._pipeline.start(config)

        # If shared (Part 1) intrinsics were provided, keep them authoritative.
        # Otherwise fall back to the SDK's factory intrinsics + distortion.
        if self._intrinsics is None:
            intr = color_profile.get_intrinsic()  # SDK: OBCameraIntrinsic
            self._intrinsics = CameraIntrinsics(
                fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy,
                width=self.width, height=self.height,
                dist=self._read_rgb_distortion(),
            )

    def _select_color_profile(self, color_profiles):
        """Pick a decodable color profile at the requested resolution.

        Prefers the requested (width, height); among those a decodable format
        (:data:`_FORMAT_PREFERENCE`) and an fps closest to (but not above) the
        request. Falls back to any decodable profile, then the device default.
        """
        vids = []
        for i in range(color_profiles.get_count()):
            sp = color_profiles.get_stream_profile_by_index(i)
            if sp.is_video_stream_profile():
                vids.append(sp.as_video_stream_profile())

        def fmt_name(v) -> str:
            return str(v.get_format()).rsplit(".", 1)[-1]

        decodable = [v for v in vids if fmt_name(v) in _FORMAT_PREFERENCE]
        at_res = [v for v in decodable
                  if v.get_width() == self.width and v.get_height() == self.height]
        pool = at_res or decodable
        if not pool:
            raise RuntimeError(
                f"[{self.camera_id}] no decodable color profile "
                f"(formats: {sorted({fmt_name(v) for v in vids})})"
            )

        def key(v):
            f = v.get_fps()
            return (
                _FORMAT_PREFERENCE.index(fmt_name(v)),  # best decodable format
                0 if f <= self.fps else 1,              # prefer <= requested fps
                abs(self.fps - f),                      # then closest fps
            )

        return min(pool, key=key)

    def _read_rgb_distortion(self) -> np.ndarray | None:
        """RGB distortion ``[k1, k2, p1, p2, k3]`` from the SDK, or None."""
        try:  # SDK: get_camera_param().rgb_distortion (OBCameraDistortion)
            cam_param = self._pipeline.get_camera_param()
            d = cam_param.rgb_distortion
            return np.array([d.k1, d.k2, d.p1, d.p2, d.k3], dtype=np.float64)
        except Exception:
            return None

    def stop(self) -> None:
        if self._pipeline is not None:
            self._pipeline.stop()
            self._pipeline = None

    def intrinsics(self) -> CameraIntrinsics:
        if self._intrinsics is None:
            raise RuntimeError("Camera not started; call start() first.")
        return self._intrinsics

    # -- capture ------------------------------------------------------------
    def capture(self, retries: int = 10) -> CameraFrame:
        if self._pipeline is None:
            raise RuntimeError("Camera not started; call start() first.")

        color_frame = None
        frames = None
        for _ in range(retries):  # warm-up frames can be dropped right after start
            frames = self._pipeline.wait_for_frames(1000)
            if frames is not None:
                color_frame = frames.get_color_frame()
                if color_frame is not None:
                    break
        if color_frame is None:
            raise RuntimeError(f"[{self.camera_id}] no color frame after {retries} tries")

        rgb = self._decode_color(color_frame)
        h, w = rgb.shape[:2]

        depth_m = np.zeros((h, w), dtype=np.float32)
        if self._depth_enabled:
            depth_frame = frames.get_depth_frame()
            if depth_frame is not None:
                depth_raw = np.frombuffer(depth_frame.get_data(), dtype=np.uint16).reshape(
                    depth_frame.get_height(), depth_frame.get_width()
                )
                depth_m = depth_raw.astype(np.float32) * depth_frame.get_depth_scale() / 1000.0

        return CameraFrame(
            camera_id=self.camera_id,
            rgb=rgb,
            depth=depth_m,
            intrinsics=self.intrinsics(),
            timestamp=color_frame.get_timestamp(),
        )

    def _decode_color(self, color_frame) -> np.ndarray:
        """Decode an Orbbec color frame to a contiguous (H, W, 3) RGB uint8 array."""
        fmt = color_frame.get_format()
        w, h = color_frame.get_width(), color_frame.get_height()
        buf = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
        if fmt == OBFormat.MJPG:
            bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"[{self.camera_id}] MJPG decode failed")
            return np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if fmt == OBFormat.RGB:
            return buf.reshape(h, w, 3).copy()
        if fmt == OBFormat.BGR:
            return np.ascontiguousarray(cv2.cvtColor(buf.reshape(h, w, 3), cv2.COLOR_BGR2RGB))
        if fmt == OBFormat.RGBA:
            return np.ascontiguousarray(cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_RGBA2RGB))
        if fmt == OBFormat.BGRA:
            return np.ascontiguousarray(cv2.cvtColor(buf.reshape(h, w, 4), cv2.COLOR_BGRA2RGB))
        if fmt == OBFormat.YUYV:
            return np.ascontiguousarray(cv2.cvtColor(buf.reshape(h, w, 2), cv2.COLOR_YUV2RGB_YUYV))
        raise RuntimeError(f"[{self.camera_id}] unsupported color format: {fmt}")

    def compute_point_cloud(self, frame: CameraFrame) -> np.ndarray:
        """Organized (H, W, 3) point cloud in the camera frame, from aligned depth."""
        intr = frame.intrinsics
        h, w = frame.depth.shape
        us, vs = np.meshgrid(np.arange(w), np.arange(h))
        z = frame.depth
        x = (us - intr.cx) * z / intr.fx
        y = (vs - intr.cy) * z / intr.fy
        return np.stack([x, y, z], axis=-1).astype(np.float32)


def list_orbbec_devices() -> list[dict]:
    """Connected Orbbec devices as ``{index, name, serial, pid}`` — for filling
    ``configs/cameras.yaml``. Run: ``python -m core.camera.orbbec``."""
    if not _SDK_AVAILABLE:
        raise ImportError("pyorbbecsdk is not installed.")
    devices = _shared_context().query_devices()
    out = []
    for i in range(devices.get_count()):
        info = devices.get_device_by_index(i).get_device_info()
        out.append({
            "index": i,
            "name": info.get_name().strip(),
            "serial": devices.get_device_serial_number_by_index(i),
            "pid": hex(info.get_pid()) if hasattr(info, "get_pid") else None,
        })
    return out


if __name__ == "__main__":  # `python -m core.camera.orbbec` -> list connected devices
    devs = list_orbbec_devices()
    if not devs:
        print("no Orbbec devices found")
    for d in devs:
        print(f"[{d['index']}] name={d['name']!r}  serial={d['serial']}  pid={d['pid']}")
