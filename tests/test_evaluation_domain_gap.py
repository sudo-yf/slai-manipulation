import numpy as np

from slai_mi.evaluation import compare_summaries, jensen_shannon_divergence, summarize_images


def test_identical_image_distributions_have_zero_gap() -> None:
    images = np.full((2, 8, 8, 3), 127, dtype=np.uint8)
    summary = summarize_images(images, bins=8)
    gap = compare_summaries(summary, summary)
    assert gap["rgb_histogram_jsd_mean"] == 0.0
    assert gap["luminance_mean_absolute_delta"] == 0.0


def test_js_divergence_detects_disjoint_histograms() -> None:
    assert jensen_shannon_divergence([1, 0], [0, 1]) == 1.0
