#!/usr/bin/env python3
"""Ensure YOLO / FastSAM weights exist for Docker build or local dev."""

from __future__ import annotations

import os
import shutil
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

WEIGHTS = ("yolov8n.pt", "yolov8n-pose.pt", "FastSAM-s.pt", "osnet_x0_25.onnx")
MIN_BYTES = 100_000
# OSNet is optional — journeys fall back to OpenCV embeddings if missing
OPTIONAL_WEIGHTS = frozenset({"osnet_x0_25.onnx"})
RELEASE_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0"
MIRROR_URLS: dict[str, tuple[str, ...]] = {
    "yolov8n.pt": (
        "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n.pt",
        f"{RELEASE_BASE}/yolov8n.pt",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",
    ),
    "yolov8n-pose.pt": (
        "https://huggingface.co/Ultralytics/YOLOv8/resolve/main/yolov8n-pose.pt",
        f"{RELEASE_BASE}/yolov8n-pose.pt",
        "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n-pose.pt",
    ),
    "FastSAM-s.pt": (
        f"{RELEASE_BASE}/FastSAM-s.pt",
        "https://huggingface.co/Ultralytics/FastSAM/resolve/main/FastSAM-s.pt",
    ),
    "osnet_x0_25.onnx": (
        "https://huggingface.co/onnxmodelzoo/osnet_x0_25_msmt17/resolve/main/osnet_x0_25_msmt17.onnx",
        "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x0_25_msmt17.pth",
    ),
}


def _in_docker_build() -> bool:
    if os.environ.get("BUILD_IN_DOCKER", "").lower() in ("1", "true", "yes"):
        return True
    return Path("/.dockerenv").is_file()


def _ok(path: Path, name: str = "") -> bool:
    min_b = 10_000 if name.endswith(".onnx") or path.suffix == ".onnx" else MIN_BYTES
    return path.is_file() and path.stat().st_size >= min_b


def _missing_names(root: Path, models_dir: Path, *, required_only: bool = False) -> list[str]:
    missing: list[str] = []
    for name in WEIGHTS:
        if required_only and name in OPTIONAL_WEIGHTS:
            continue
        if _ok(root / name, name) or _ok(models_dir / name, name):
            continue
        missing.append(name)
    return missing


def _copy_bundled(root: Path, models_dir: Path) -> None:
    for name in WEIGHTS:
        target = root / name
        if _ok(target, name):
            continue
        bundled = models_dir / name
        if _ok(bundled, name):
            shutil.copy2(bundled, target)
            print(f"Copied bundled weight: {bundled} -> {target}")


def _direct_download(dest: Path, name: str, attempts: int = 3) -> bool:
    urls = MIRROR_URLS.get(name, (f"{RELEASE_BASE}/{name}",))
    min_b = 10_000 if name.endswith(".onnx") else MIN_BYTES
    for url in urls:
        if name.endswith(".onnx") and not url.lower().endswith(".onnx"):
            continue  # skip .pth mirrors for ONNX target
        for attempt in range(1, attempts + 1):
            try:
                print(f"Direct download {name} from {url} (attempt {attempt}/{attempts})...")
                tmp = dest.with_suffix(dest.suffix + ".part")
                req = urllib.request.Request(url, headers={"User-Agent": "DRP-CV-weights/1.0"})
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = resp.read()
                if len(data) < min_b:
                    raise OSError(f"too small ({len(data)} bytes)")
                tmp.write_bytes(data)
                tmp.replace(dest)
                print(f"OK: {dest} ({len(data)} bytes)")
                return True
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                print(f"Direct download failed {name}: {e}", file=sys.stderr)
                if attempt < attempts:
                    time.sleep(min(15, 3 * attempt))
    return False


def _ultralytics_download(root: Path, name: str) -> bool:
    if name in OPTIONAL_WEIGHTS or name.endswith(".onnx"):
        return False
    try:
        from ultralytics import FastSAM, YOLO

        print(f"Ultralytics download {name}...")
        if name == "FastSAM-s.pt":
            FastSAM(str(root / name) if _ok(root / name, name) else name)
        else:
            YOLO(str(root / name) if _ok(root / name, name) else name)
        return _ok(root / name, name)
    except Exception as e:
        print(f"Ultralytics download failed {name}: {e}", file=sys.stderr)
        return False


def _fetch_to_models(models_dir: Path, name: str) -> bool:
    dest = models_dir / name
    if _ok(dest, name):
        return True
    if _direct_download(dest, name):
        return True
    return _ultralytics_download(models_dir, name) and _ok(dest, name)


def main() -> int:
    root = Path(os.environ.get("APP_ROOT", "/app" if Path("/app").is_dir() else ".")).resolve()
    models_dir = root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    missing_required = _missing_names(root, models_dir, required_only=True)
    missing_optional = [n for n in OPTIONAL_WEIGHTS if n in _missing_names(root, models_dir)]

    if missing_required and _in_docker_build():
        msg = (
            "Bundled weights missing in models/: "
            + ", ".join(missing_required)
            + ".\n"
            "Docker build does not download from GitHub (network is unreliable inside build).\n"
            "On the host, run:\n"
            "  powershell -ExecutionPolicy Bypass -File scripts\\download_ultralytics_weights.ps1\n"
            "Then run docker-rebuild.bat again."
        )
        print(msg, file=sys.stderr)
        return 1

    if missing_required:
        allow_skip = os.environ.get("SKIP_WEIGHT_DOWNLOAD", "").lower() in ("1", "true", "yes")
        if allow_skip:
            print("SKIP_WEIGHT_DOWNLOAD=1 — continuing without:", ", ".join(missing_required), file=sys.stderr)
            return 0
        for name in list(missing_required):
            _fetch_to_models(models_dir, name)
        _copy_bundled(root, models_dir)

    # Best-effort optional Re-ID weights (OpenCV fallback if missing)
    for name in missing_optional:
        if not _in_docker_build():
            _fetch_to_models(models_dir, name)
        _copy_bundled(root, models_dir)
        if not (_ok(root / name, name) or _ok(models_dir / name, name)):
            print(f"Optional weight missing (OK): {name} — PersonReID uses OpenCV fallback", file=sys.stderr)

    _copy_bundled(root, models_dir)
    still_missing = _missing_names(root, models_dir, required_only=True)

    if still_missing:
        msg = (
            "Missing model weights: "
            + ", ".join(still_missing)
            + ".\n"
            "Run on the host:\n"
            "  powershell -ExecutionPolicy Bypass -File scripts\\download_ultralytics_weights.ps1\n"
            "Then rebuild."
        )
        print(msg, file=sys.stderr)
        return 1

    from ultralytics import FastSAM, YOLO

    YOLO(str(root / "yolov8n.pt"))
    YOLO(str(root / "yolov8n-pose.pt"))
    FastSAM(str(root / "FastSAM-s.pt"))
    print("All ultralytics weights verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())