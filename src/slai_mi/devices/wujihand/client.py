"""Vendor-neutral WujiHand client and lazy SDK factory."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Protocol


class HandBackend(Protocol):
    def read_positions(self) -> Sequence[float]: ...
    def write_positions(self, positions: Sequence[float]) -> None: ...
    def disable(self) -> None: ...


class WujiHandClient:
    """Small lifecycle wrapper around an injected vendor backend."""

    def __init__(self, backend_factory: Callable[[], HandBackend], *, expected_joints: int = 20):
        if expected_joints <= 0:
            raise ValueError("expected_joints must be positive")
        self._factory, self.expected_joints, self._backend = backend_factory, expected_joints, None

    @property
    def connected(self) -> bool:
        return self._backend is not None

    def connect(self) -> None:
        if self.connected:
            raise RuntimeError("WujiHand is already connected")
        self._backend = self._factory()
        self.read_positions()

    def read_positions(self) -> tuple[float, ...]:
        if self._backend is None:
            raise RuntimeError("WujiHand is not connected")
        values = tuple(float(item) for item in self._backend.read_positions())
        if len(values) != self.expected_joints:
            raise RuntimeError(f"WujiHand returned {len(values)} joints, expected {self.expected_joints}")
        return values

    def write_positions(self, positions: Sequence[float]) -> None:
        if self._backend is None:
            raise RuntimeError("WujiHand is not connected")
        values = tuple(float(item) for item in positions)
        if len(values) != self.expected_joints:
            raise ValueError("WujiHand command has the wrong dimension")
        self._backend.write_positions(values)

    def close(self) -> None:
        if self._backend is not None:
            self._backend.disable()
        self._backend = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def vendor_backend_factory(module_name: str, factory_name: str, **kwargs):
    """Return a factory that imports a site-specific hand SDK only on connect."""
    def create():
        import importlib

        try:
            module = importlib.import_module(module_name)
        except ImportError as exc:
            raise RuntimeError(f"WujiHand vendor SDK {module_name!r} is not installed") from exc
        return getattr(module, factory_name)(**kwargs)

    return create
