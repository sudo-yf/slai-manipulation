"""Pure OpenPI/PI0.5 observation and action transforms."""

from __future__ import annotations

import numpy as np


def parse_rgb(value: object) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"expected HxWx3 RGB, got {image.shape}")
    return np.ascontiguousarray(image, dtype=np.uint8)


def make_pi05_observation(
    *, primary_rgb: object, secondary_rgb: object, state: object, prompt: str
) -> dict[str, object]:
    state_array = np.asarray(state, dtype=np.float32)
    if state_array.ndim != 1 or not np.isfinite(state_array).all():
        raise ValueError("state must be a finite vector")
    primary = parse_rgb(primary_rgb)
    secondary = parse_rgb(secondary_rgb)
    if primary.shape != secondary.shape:
        raise ValueError("policy images must have matching shapes")
    return {
        "state": state_array,
        "image": {
            "base_0_rgb": primary,
            "left_wrist_0_rgb": secondary,
            "right_wrist_0_rgb": np.zeros_like(primary),
        },
        "image_mask": {
            "base_0_rgb": np.True_,
            "left_wrist_0_rgb": np.True_,
            "right_wrist_0_rgb": np.False_,
        },
        "prompt": prompt,
    }


def hand_position_to_delta(actions: object, state: object, *, arm_dim: int = 6) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32).copy()
    current = np.asarray(state, dtype=np.float32)
    hand_dim = result.shape[-1] - arm_dim
    if hand_dim <= 0 or current.shape[-1] < hand_dim:
        raise ValueError("state/action dimensions do not contain a hand segment")
    result[..., arm_dim:] -= current[-hand_dim:]
    return result


def hand_delta_to_position(
    actions: object, state: object, *, action_dim: int, arm_dim: int = 6
) -> np.ndarray:
    result = np.asarray(actions, dtype=np.float32)[..., :action_dim].copy()
    current = np.asarray(state, dtype=np.float32)
    hand_dim = action_dim - arm_dim
    if hand_dim <= 0 or current.shape[-1] < hand_dim:
        raise ValueError("state/action dimensions do not contain a hand segment")
    result[..., arm_dim:] += current[-hand_dim:]
    return result
