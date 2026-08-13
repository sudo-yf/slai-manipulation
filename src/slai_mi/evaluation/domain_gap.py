"""Aggregate image statistics for visual-domain comparisons."""

from __future__ import annotations

import numpy as np


def jensen_shannon_divergence(left: object, right: object) -> float:
    first = np.asarray(left, dtype=np.float64)
    second = np.asarray(right, dtype=np.float64)
    if first.shape != second.shape or first.ndim != 1 or np.any(first < 0) or np.any(second < 0):
        raise ValueError("histograms must be matching non-negative vectors")
    if first.sum() <= 0 or second.sum() <= 0:
        raise ValueError("histograms must have positive mass")
    first, second = first / first.sum(), second / second.sum()
    midpoint = (first + second) / 2

    def divergence(values: np.ndarray) -> float:
        mask = values > 0
        return float(np.sum(values[mask] * np.log2(values[mask] / midpoint[mask])))

    return (divergence(first) + divergence(second)) / 2


def summarize_images(images: object, *, bins: int = 32) -> dict[str, object]:
    array = np.asarray(images)
    if array.ndim != 4 or array.shape[-1] != 3 or array.shape[0] == 0 or bins <= 0:
        raise ValueError("images must be a non-empty NxHxWx3 RGB array and bins positive")
    pixels = array.astype(np.float64).reshape(-1, 3) / 255.0
    luminance = pixels @ np.asarray([0.2126, 0.7152, 0.0722])
    histograms = [np.histogram(pixels[:, i], bins=bins, range=(0, 1))[0] for i in range(3)]
    luminance_histogram = np.histogram(luminance, bins=bins, range=(0, 1))[0]
    return {
        "frames": int(array.shape[0]),
        "rgb_mean": pixels.mean(axis=0).tolist(),
        "rgb_std": pixels.std(axis=0).tolist(),
        "luminance_mean": float(luminance.mean()),
        "rgb_histograms": [(item / item.sum()).tolist() for item in histograms],
        "luminance_histogram": (luminance_histogram / luminance_histogram.sum()).tolist(),
    }


def compare_summaries(reference: dict[str, object], candidate: dict[str, object]) -> dict[str, object]:
    reference_rgb = reference["rgb_histograms"]
    candidate_rgb = candidate["rgb_histograms"]
    rgb_jsd = [
        jensen_shannon_divergence(left, right)
        for left, right in zip(reference_rgb, candidate_rgb, strict=True)  # type: ignore[arg-type]
    ]
    return {
        "rgb_histogram_jsd": rgb_jsd,
        "rgb_histogram_jsd_mean": float(np.mean(rgb_jsd)),
        "luminance_histogram_jsd": jensen_shannon_divergence(
            reference["luminance_histogram"], candidate["luminance_histogram"]
        ),
        "rgb_mean_absolute_delta": np.abs(
            np.asarray(reference["rgb_mean"]) - np.asarray(candidate["rgb_mean"])
        ).tolist(),
        "luminance_mean_absolute_delta": abs(
            float(reference["luminance_mean"]) - float(candidate["luminance_mean"])
        ),
    }
