"""Configuration validation and lazy training backend loading."""

from __future__ import annotations

import importlib
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from slai_mi.datasets.lerobot_v3 import validate_dataset_root
from slai_mi.training.strict_horizon import install_openpi_strict_horizon_filter

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def _path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base / path).resolve()


def _load_mapping(path: Path) -> dict[str, Any]:
    path = _path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"configuration must be a mapping: {path}")
    return payload


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _plugin_factory(spec: str) -> Callable[..., object]:
    module_name, separator, attribute = spec.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("backend must use the 'module:factory' form")
    factory = getattr(importlib.import_module(module_name), attribute)
    if not callable(factory):
        raise TypeError(f"training backend factory is not callable: {spec}")
    return factory


def _run_plugin(plugin: object) -> object:
    run = getattr(plugin, "run", None)
    if callable(run):
        return run()
    if callable(plugin):
        return plugin()
    raise TypeError("training backend factory must return a callable or an object with run()")


@dataclass(frozen=True)
class TrainingPlan:
    config_path: Path
    dataset_root: Path
    backend: str | None
    action_horizon: int
    action_keys: tuple[str, ...]
    config: Mapping[str, Any]
    install_openpi_filter: bool

    def as_dict(self, *, mode: str) -> dict[str, object]:
        return {
            "app": "train",
            "mode": mode,
            "config": str(self.config_path),
            "dataset": str(self.dataset_root),
            "dataset_exists": self.dataset_root.is_dir(),
            "backend": self.backend,
            "action_horizon": self.action_horizon,
            "action_keys": list(self.action_keys),
        }


def build_training_plan(
    config_path: Path, *, dataset_override: str | None = None
) -> TrainingPlan:
    resolved_config = _path(config_path)
    config = _load_mapping(resolved_config)
    if config.get("schema_version") != 1:
        raise ValueError("training.schema_version must be 1")
    dataset = dataset_override or config.get("dataset", {}).get("root")
    if not isinstance(dataset, str) or not dataset.strip():
        raise ValueError("training.dataset.root must be a non-empty path")
    policy = config.get("policy")
    if not isinstance(policy, dict):
        raise TypeError("training.policy must be a mapping")
    horizon = _positive_int(policy.get("action_horizon"), "policy.action_horizon")
    keys = policy.get("action_keys")
    if not isinstance(keys, list) or not keys or not all(isinstance(key, str) and key for key in keys):
        raise ValueError("policy.action_keys must be a non-empty list of strings")
    backend = config.get("backend")
    if backend is not None and (not isinstance(backend, str) or ":" not in backend):
        raise ValueError("training.backend must be null or use the 'module:factory' form")
    strict = config.get("strict_horizon", {})
    if not isinstance(strict, dict):
        raise TypeError("training.strict_horizon must be a mapping")
    return TrainingPlan(
        config_path=resolved_config,
        dataset_root=_path(dataset),
        backend=backend,
        action_horizon=horizon,
        action_keys=tuple(keys),
        config=config,
        install_openpi_filter=bool(strict.get("install_openpi_filter", False)),
    )


def execute_training(plan: TrainingPlan) -> dict[str, object]:
    if plan.backend is None:
        raise RuntimeError("training backend is not configured; set backend to module:factory")
    dataset_summary = validate_dataset_root(plan.dataset_root)
    if plan.install_openpi_filter:
        install_openpi_strict_horizon_filter()
    plugin = _plugin_factory(plan.backend)(config=dict(plan.config), dataset_root=plan.dataset_root)
    return {
        **plan.as_dict(mode="execute"),
        "dataset_validation": dataset_summary,
        "backend_result": _run_plugin(plugin),
    }
