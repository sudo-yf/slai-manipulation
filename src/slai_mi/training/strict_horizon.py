"""Filter dataset anchors to complete, single-episode future horizons."""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import numpy as np


class CompleteActionHorizonDataset:
    def __init__(self, dataset: Any, action_horizon: int, action_keys: Sequence[str]) -> None:
        if action_horizon <= 0:
            raise ValueError("action_horizon must be positive")
        self._dataset = dataset
        episode_owner = dataset
        episode_index = getattr(episode_owner, "episode_data_index", None)
        while episode_index is None and hasattr(episode_owner, "_dataset"):
            episode_owner = episode_owner._dataset
            episode_index = getattr(episode_owner, "episode_data_index", None)
        if episode_index is None:
            raise ValueError("strict horizon filtering requires episode_data_index")
        self._action_keys = tuple(action_keys)
        valid = []
        for start, end in zip(episode_index["from"], episode_index["to"], strict=True):
            stop = int(end) - action_horizon + 1
            if stop > int(start):
                valid.append(np.arange(int(start), stop, dtype=np.int64))
        self._indices = np.concatenate(valid) if valid else np.empty(0, dtype=np.int64)
        if not len(self._indices):
            raise ValueError("dataset has no anchors with a complete action horizon")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, index: object) -> Any:
        mapped = int(self._indices[operator.index(index)])
        item = self._dataset[mapped]
        for key in self._action_keys:
            pad_key = f"{key}_is_pad"
            if pad_key not in item or bool(np.asarray(item[pad_key]).any()):
                raise ValueError(f"invalid or missing horizon padding mask {pad_key!r}")
        return item


def install_openpi_strict_horizon_filter() -> None:
    """Patch OpenPI only when explicitly requested by a training entry point."""
    try:
        from openpi.training import data_loader
    except ImportError as exc:
        raise RuntimeError("OpenPI is required to install its dataset filter") from exc
    original = data_loader.create_torch_dataset
    if getattr(original, "_slai_strict_horizon", False):
        return

    def create_strict_dataset(data_config: Any, action_horizon: int, model_config: Any) -> Any:
        dataset = original(data_config, action_horizon, model_config)
        if data_config.repo_id == "fake":
            return dataset
        return CompleteActionHorizonDataset(
            dataset, action_horizon, data_config.action_sequence_keys
        )

    create_strict_dataset._slai_strict_horizon = True
    data_loader.create_torch_dataset = create_strict_dataset
