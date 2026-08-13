"""Configuration validation and lazy inference backend loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
ALLOWED_TARGETS = frozenset({"offline", "simulation"})


def _path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def _factory(spec: str) -> Callable[..., object]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("backend must use the 'module:factory' form")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"inference backend factory is not callable: {spec}")
    return factory


def _run(plugin: object) -> object:
    run = getattr(plugin, "run", None)
    if callable(run):
        return run()
    if callable(plugin):
        return plugin()
    raise TypeError("inference backend factory must return a callable or an object with run()")


@dataclass(frozen=True)
class InferencePlan:
    config_path: Path
    checkpoint: Path
    backend: str | None
    target: str
    config: Mapping[str, Any]

    def as_dict(self, *, mode: str) -> dict[str, object]:
        return {
            "app": "inference",
            "mode": mode,
            "config": str(self.config_path),
            "checkpoint": str(self.checkpoint),
            "checkpoint_exists": self.checkpoint.exists(),
            "backend": self.backend,
            "target": self.target,
        }


def build_inference_plan(
    config_path: Path, *, checkpoint_override: str | None = None
) -> InferencePlan:
    resolved = _path(config_path)
    payload = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"configuration must be a mapping: {resolved}")
    if payload.get("schema_version") != 1:
        raise ValueError("inference.schema_version must be 1")
    target = payload.get("target")
    if target == "real":
        raise ValueError(
            "real inference is forbidden in this entry point; use an apps real workflow "
            "with the real-hardware safety supervisor"
        )
    if target not in ALLOWED_TARGETS:
        raise ValueError("inference.target must be 'offline' or 'simulation'")
    checkpoint = checkpoint_override or payload.get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint.strip():
        raise ValueError("inference.checkpoint must be a non-empty path")
    backend = payload.get("backend")
    if backend is not None and (not isinstance(backend, str) or ":" not in backend):
        raise ValueError("inference.backend must be null or use the 'module:factory' form")
    return InferencePlan(resolved, _path(checkpoint), backend, target, payload)


def execute_inference(plan: InferencePlan) -> dict[str, object]:
    if plan.target not in ALLOWED_TARGETS:
        raise RuntimeError("only offline and simulation inference may run here")
    if plan.backend is None:
        raise RuntimeError("inference backend is not configured; set backend to module:factory")
    if not plan.checkpoint.exists():
        raise ValueError(f"checkpoint does not exist: {plan.checkpoint}")
    plugin = _factory(plan.backend)(
        config=dict(plan.config), checkpoint=plan.checkpoint, target=plan.target
    )
    return {**plan.as_dict(mode="execute"), "backend_result": _run(plugin)}
