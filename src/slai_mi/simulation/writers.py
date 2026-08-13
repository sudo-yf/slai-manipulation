"""Small local simulation writer; production datasets may inject another writer."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np


class NpzEpisodeWriter:
    def __init__(self, root: Path) -> None:
        self.root = root / "simulation_npz"
        self._frames: list[Mapping[str, Any]] = []
        self._index = -1
        self._seed = 0

    def begin_episode(self, *, index: int, seed: int) -> None:
        self._index, self._seed = index, seed
        self._frames = []

    def add_frame(self, frame: Mapping[str, Any]) -> None:
        self._frames.append(frame)

    def end_episode(self, *, success: bool) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        keys = set.intersection(*(set(frame) for frame in self._frames)) if self._frames else set()
        arrays = {key: np.asarray([frame[key] for frame in self._frames]) for key in keys}
        np.savez_compressed(
            self.root / f"episode_{self._index:06d}.npz",
            **arrays,
            seed=np.asarray(self._seed),
            success=np.asarray(success),
        )


def create_npz_writer(*, root: Path, **_kwargs: Any) -> NpzEpisodeWriter:
    return NpzEpisodeWriter(root)
