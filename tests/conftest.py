"""
conftest.py — Session-wide module stubs for offline testing.

Injected into sys.modules BEFORE any project module is imported so that
heavy third-party packages (redis, ultralytics, cv2, etc.) and the project's
own Redis-connecting code are fully mocked. Lets the entire test suite run
without Docker, GPU, or a live Redis instance.
"""
import sys
import types
import unittest.mock as _mock


# ---------------------------------------------------------------------------
# Helper — register a plain MagicMock as a module so dotted imports work
# ---------------------------------------------------------------------------
def _stub_module(dotted_name: str, **attrs):
    """Create a plain MagicMock module and register it + its parents."""
    parts = dotted_name.split(".")
    for i in range(len(parts)):
        key = ".".join(parts[:i + 1])
        if key not in sys.modules:
            sys.modules[key] = _mock.MagicMock()
    mod = _mock.MagicMock()
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[dotted_name] = mod
    return mod


# ---------------------------------------------------------------------------
# Third-party stubs (Docker-only packages)
# ---------------------------------------------------------------------------
_THIRD_PARTY_STUBS = [
    "redis",
    "redis.client",
    "rapidocr_onnxruntime",
    "paddleocr",
    "paddle",
    "faiss",
]
for _mod in _THIRD_PARTY_STUBS:
    _stub_module(_mod)

# ultralytics — must expose a callable YOLO class
_yolo_cls = _mock.MagicMock(name="YOLO")
_stub_module("ultralytics", YOLO=_yolo_cls)
_stub_module("ultralytics.models")

# cv2 — needs real constants + controlled return values
_cv2 = _mock.MagicMock()
_cv2.FONT_HERSHEY_SIMPLEX = 0
_cv2.LINE_AA = 16
_img_buf = _mock.MagicMock()
_img_buf.tobytes.return_value = b"\xff\xd8\xff"   # minimal JPEG header
_cv2.imencode.return_value = (True, _img_buf)
_cv2.rectangle.return_value = None
_cv2.putText.return_value = None
_cv2.getTextSize.return_value = ((80, 14), 4)     # (text_w, text_h), baseline
sys.modules["cv2"] = _cv2

# ---------------------------------------------------------------------------
# Shapely — use the real library if installed, otherwise stub it out
# ---------------------------------------------------------------------------
try:
    from shapely.geometry import Point as _Point, Polygon as _Polygon, box as _box   # noqa
    _shape_ns = types.SimpleNamespace(Point=_Point, Polygon=_Polygon, box=_box)
    sys.modules["shapely"] = _mock.MagicMock(_SHAPELY_AVAILABLE=True)
    sys.modules["shapely.geometry"] = _shape_ns
except ImportError:
    sys.modules["shapely"] = _mock.MagicMock(_SHAPELY_AVAILABLE=False)
    sys.modules["shapely.geometry"] = _mock.MagicMock()


