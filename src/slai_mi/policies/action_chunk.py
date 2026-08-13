"""Framework-neutral action chunk execution."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

import numpy as np


class Policy(Protocol):
    def infer(self, observation: Mapping[str, object]) -> Mapping[str, object]: ...


class ActionChunkPolicy:
    """Cache a bounded number of steps from each policy action chunk."""

    def __init__(self, policy: Policy, action_dim: int, open_loop_horizon: int) -> None:
        if action_dim <= 0 or open_loop_horizon <= 0:
            raise ValueError("action_dim and open_loop_horizon must be positive")
        self._policy = policy
        self._action_dim = action_dim
        self._horizon = open_loop_horizon
        self.reset()

    def reset(self) -> None:
        self._chunk = np.empty((0, self._action_dim), dtype=np.float32)
        self._index = 0

    def infer(self, observation: Mapping[str, Any]) -> np.ndarray:
        if self._index >= min(len(self._chunk), self._horizon):
            result = self._policy.infer(observation)
            chunk = np.asarray(result.get("actions"), dtype=np.float32)
            if chunk.ndim != 2 or chunk.shape[1] != self._action_dim:
                raise ValueError(f"expected actions[N,{self._action_dim}], got {chunk.shape}")
            if not len(chunk) or not np.isfinite(chunk).all():
                raise ValueError("policy returned an empty or non-finite action chunk")
            self._chunk = chunk
            self._index = 0
        action = self._chunk[self._index].copy()
        self._index += 1
        return action
