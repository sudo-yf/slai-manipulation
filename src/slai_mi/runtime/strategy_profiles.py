"""Declarative control-strategy profiles for real station workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_STRATEGY_ROOT = PROJECT_ROOT / "configs" / "strategies"
READY_STATUS = "ready"
COMMISSIONING_STATUS = "commissioning"
VALID_STATUSES = frozenset((READY_STATUS, COMMISSIONING_STATUS))


class StrategyProfileError(ValueError):
    """Raised when a strategy profile is invalid or incompatible with a workflow."""


@dataclass(frozen=True)
class StrategyProfile:
    """Validated description of one complete real-hardware operating mode."""

    id: str
    label: str
    status: str
    supported_apps: tuple[str, ...]
    required_hardware: tuple[str, ...]
    disabled_hardware: tuple[str, ...]
    input_devices: tuple[str, ...]
    output_devices: tuple[str, ...]
    control_modes: Mapping[str, str]
    physical_dof: int
    recorded_dof: int | None
    state_schema: str | None
    input_schema: str | None
    synthetic_channels: tuple[str, ...]
    source: Path

    @property
    def commissioned(self) -> bool:
        return self.status == READY_STATUS

    def validate_for(
        self,
        app: str,
        *,
        hardware: Mapping[str, Any],
        task: Mapping[str, Any] | None = None,
        execute: bool = False,
    ) -> None:
        """Fail before opening hardware when a selected strategy cannot run safely."""
        if app not in self.supported_apps:
            supported = ", ".join(self.supported_apps)
            raise StrategyProfileError(
                f"strategy {self.id!r} does not support {app}; supported apps: {supported}"
            )
        missing = [
            name
            for name in self.required_hardware
            if not isinstance(hardware.get(name), Mapping)
            or hardware[name].get("enabled") is not True
        ]
        if missing:
            raise StrategyProfileError(
                f"strategy {self.id!r} requires enabled hardware: {', '.join(missing)}"
            )
        if task is not None and self.state_schema is not None and self.commissioned:
            selected_schema = str(task.get("state_schema") or "").strip()
            if selected_schema != self.state_schema:
                raise StrategyProfileError(
                    f"strategy {self.id!r} requires task state_schema={self.state_schema!r}; "
                    f"got {selected_schema!r}"
                )
        if execute and not self.commissioned:
            raise StrategyProfileError(
                f"strategy {self.id!r} is still commissioning and cannot control real hardware"
            )

    def plan_fields(self) -> dict[str, Any]:
        """Return stable, beginner-readable fields for CLI dry-run plans."""
        return {
            "strategy": self.id,
            "strategy_label": self.label,
            "strategy_status": self.status,
            "physical_dof": self.physical_dof,
            "recorded_dof": self.recorded_dof,
            "inputs": list(self.input_devices),
            "outputs": list(self.output_devices),
            "control_modes": dict(self.control_modes),
            "synthetic_channels": list(self.synthetic_channels),
            "dataset_state_schema": self.state_schema,
            "input_schema": self.input_schema,
            "required_hardware": list(self.required_hardware),
            "disabled_hardware": list(self.disabled_hardware),
        }

    def configure_hardware(self, hardware: Mapping[str, Any]) -> dict[str, Any]:
        """Return an isolated config with devices outside this group explicitly disabled."""
        configured = deepcopy(dict(hardware))
        for name in self.disabled_hardware:
            section = configured.get(name)
            if isinstance(section, Mapping):
                configured[name] = {**section, "enabled": False}
        if self.input_schema is not None:
            configured["input_schema"] = self.input_schema
        return configured


def strategy_path(value: str | Path, *, root: Path = DEFAULT_STRATEGY_ROOT) -> Path:
    """Resolve either a short strategy id or an explicit YAML path."""
    raw = Path(value).expanduser()
    if len(raw.parts) == 1 and raw.suffix in ("", ".yaml", ".yml"):
        filename = raw if raw.suffix else raw.with_suffix(".yaml")
        return root / filename
    return raw if raw.is_absolute() else PROJECT_ROOT / raw


def load_strategy_profile(
    value: str | Path, *, root: Path = DEFAULT_STRATEGY_ROOT
) -> StrategyProfile:
    path = strategy_path(value, root=root).resolve()
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as exc:
        raise StrategyProfileError(f"strategy profile does not exist: {path}") from exc
    except yaml.YAMLError as exc:
        raise StrategyProfileError(f"invalid strategy YAML in {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise StrategyProfileError(f"strategy profile must contain a mapping: {path}")
    if payload.get("schema_version") != 1:
        raise StrategyProfileError(f"strategy profile schema_version must be 1: {path}")
    strategy = _mapping(payload, "strategy", path)
    dataset = payload.get("dataset")
    if dataset is not None and not isinstance(dataset, Mapping):
        raise StrategyProfileError(f"strategy dataset must be a mapping: {path}")
    dataset = dataset or {}

    strategy_id = _text(strategy, "id", path)
    if strategy_id != path.stem:
        raise StrategyProfileError(f"strategy id {strategy_id!r} must match filename {path.stem!r}")
    status = _text(strategy, "status", path)
    if status not in VALID_STATUSES:
        raise StrategyProfileError(
            f"strategy status must be one of {sorted(VALID_STATUSES)}: {path}"
        )
    physical_dof = _positive_int(strategy, "physical_dof", path)
    recorded_dof = dataset.get("recorded_dof")
    if recorded_dof is not None:
        recorded_dof = _positive_int(dataset, "recorded_dof", path)
    state_schema = dataset.get("state_schema")
    if state_schema is not None:
        state_schema = str(state_schema).strip() or None
    input_schema = dataset.get("input_schema")
    if input_schema is not None:
        input_schema = str(input_schema).strip() or None

    required_hardware = _text_list(strategy, "required_hardware", path)
    disabled_hardware = _optional_text_list(strategy, "disabled_hardware", path)
    overlap = sorted(set(required_hardware) & set(disabled_hardware))
    if overlap:
        raise StrategyProfileError(
            f"strategy hardware cannot be both required and disabled: {overlap}: {path}"
        )

    return StrategyProfile(
        id=strategy_id,
        label=_text(strategy, "label", path),
        status=status,
        supported_apps=_text_list(strategy, "supported_apps", path),
        required_hardware=required_hardware,
        disabled_hardware=disabled_hardware,
        input_devices=_text_list(strategy, "inputs", path),
        output_devices=_text_list(strategy, "outputs", path),
        control_modes=MappingProxyType(
            dict(_text_mapping(_mapping(strategy, "control", path), path))
        ),
        physical_dof=physical_dof,
        recorded_dof=recorded_dof,
        state_schema=state_schema,
        input_schema=input_schema,
        synthetic_channels=_optional_text_list(dataset, "synthetic_channels", path),
        source=path,
    )


def available_strategy_profiles(
    *, root: Path = DEFAULT_STRATEGY_ROOT
) -> tuple[StrategyProfile, ...]:
    return tuple(load_strategy_profile(path, root=root) for path in sorted(root.glob("*.yaml")))


def _mapping(parent: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    value = parent.get(key)
    if not isinstance(value, Mapping):
        raise StrategyProfileError(f"strategy {key} must be a mapping: {path}")
    return value


def _text(parent: Mapping[str, Any], key: str, path: Path) -> str:
    value = str(parent.get(key) or "").strip()
    if not value:
        raise StrategyProfileError(f"strategy {key} must be a non-empty string: {path}")
    return value


def _text_list(parent: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    value = parent.get(key)
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or not value
        or any(not str(item).strip() for item in value)
    ):
        raise StrategyProfileError(f"strategy {key} must be a non-empty string list: {path}")
    result = tuple(str(item).strip() for item in value)
    if len(set(result)) != len(result):
        raise StrategyProfileError(f"strategy {key} contains duplicates: {path}")
    return result


def _optional_text_list(parent: Mapping[str, Any], key: str, path: Path) -> tuple[str, ...]:
    if key not in parent:
        return ()
    value = parent.get(key)
    if value == []:
        return ()
    return _text_list(parent, key, path)


def _text_mapping(value: Mapping[str, Any], path: Path) -> tuple[tuple[str, str], ...]:
    result = tuple((str(key).strip(), str(item).strip()) for key, item in value.items())
    if not result or any(not key or not item for key, item in result):
        raise StrategyProfileError(f"strategy control must map names to modes: {path}")
    return result


def _positive_int(parent: Mapping[str, Any], key: str, path: Path) -> int:
    value = parent.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        raise StrategyProfileError(f"strategy {key} must be a positive integer: {path}")
    return value
