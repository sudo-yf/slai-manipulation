"""Image-domain metrics for matching simulator renders to real cameras."""

from __future__ import annotations

from typing import Any

import numpy as np


def compare_images(real: object, sim: object) -> tuple[dict[str, Any], np.ndarray]:
    """Return image metrics and a ``real | sim | heatmap`` visualization."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for sim-real image evaluation") from exc
    first = np.asarray(real, dtype=np.uint8)
    second = np.asarray(sim, dtype=np.uint8)
    if first.ndim != 3 or first.shape[2] != 3 or second.ndim != 3 or second.shape[2] != 3:
        raise ValueError("real and sim images must be BGR HxWx3 arrays")
    if second.shape[:2] != first.shape[:2]:
        second = cv2.resize(second, (first.shape[1], first.shape[0]), interpolation=cv2.INTER_AREA)
    difference = cv2.absdiff(first, second)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    heatmap = cv2.applyColorMap(gray, cv2.COLORMAP_TURBO)
    first_gray = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY).astype(np.float64)
    second_gray = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY).astype(np.float64)
    mse = float(np.mean((first_gray - second_gray) ** 2))
    report = {
        "resolution": {"width": int(first.shape[1]), "height": int(first.shape[0])},
        "mean_absolute_error_8bit": float(np.mean(difference)),
        "rmse_gray_8bit": float(np.sqrt(mse)),
        "psnr_gray_db": float("inf") if mse == 0 else float(20 * np.log10(255 / np.sqrt(mse))),
    }
    return report, np.hstack((first, second, heatmap))
