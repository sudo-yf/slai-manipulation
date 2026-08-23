"""Named, grouped snapshots of measured robot joint positions."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml


def local_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def safe_filename(value: str) -> str:
    cleaned = re.sub(r"[^\w.-]+", "-", value.strip(), flags=re.UNICODE).strip("-._")
    return cleaned or "record"


@dataclass
class HoldGestureDetector:
    """Emit one event after each configured button has been held long enough."""

    buttons: dict[str, int]
    hold_seconds: float = 0.8
    _pressed_at: dict[int, float] = field(default_factory=dict, init=False)
    _fired: set[int] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.hold_seconds <= 0.0:
            raise ValueError("hold_seconds must be positive")
        if len(set(self.buttons.values())) != len(self.buttons):
            raise ValueError("gesture buttons must be unique")

    def update(self, states: dict[int, bool], now: float) -> list[str]:
        events: list[str] = []
        for name, code in self.buttons.items():
            if states.get(code, False):
                started = self._pressed_at.setdefault(code, now)
                if code not in self._fired and now - started >= self.hold_seconds:
                    self._fired.add(code)
                    events.append(name)
            else:
                self._pressed_at.pop(code, None)
                self._fired.discard(code)
        return events

    def reset(self) -> None:
        self._pressed_at.clear()
        self._fired.clear()


class PoseJournal:
    def __init__(
        self,
        path: Path,
        *,
        recording_name: str,
        joint_names: tuple[str, ...],
    ) -> None:
        self.path = path
        self.joint_names = joint_names
        created_at = local_timestamp()
        self.payload: dict[str, Any] = {
            "schema_version": 1,
            "kind": "slai_named_pose_recording",
            "recording": {
                "name": recording_name,
                "created_at": created_at,
                "updated_at": created_at,
            },
            "source": {
                "type": "physical_station_measured",
                "state_schema": "real_v1",
                "dimension": len(joint_names),
                "units": "rad",
                "joint_names": list(joint_names),
                "missing_from_simulation_v1": ["wrist_pitch_joint", "wrist_yaw_joint"],
            },
            "active_group": None,
            "groups": [],
        }

    @classmethod
    def create(
        cls,
        root: Path,
        *,
        recording_name: str,
        joint_names: tuple[str, ...],
        exact_filename: bool = False,
    ) -> PoseJournal:
        if exact_filename:
            filename = f"{safe_filename(recording_name)}.yaml"
        else:
            stamp = datetime.now().astimezone().strftime("%Y%m%dT%H%M%S-%f")
            filename = f"{safe_filename(recording_name)}-{stamp}.yaml"
        path = root.expanduser().resolve() / filename
        if exact_filename and path.exists():
            return cls.load(path, joint_names=joint_names)
        journal = cls(
            path,
            recording_name=recording_name,
            joint_names=joint_names,
        )
        journal.save()
        return journal

    @classmethod
    def load(cls, path: Path, *, joint_names: tuple[str, ...]) -> PoseJournal:
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ValueError(f"cannot load pose recording {path}: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("kind") != "slai_named_pose_recording":
            raise ValueError(f"not a SLaI named pose recording: {path}")
        source = payload.get("source")
        if (
            not isinstance(source, dict)
            or source.get("dimension") != len(joint_names)
            or source.get("joint_names") != list(joint_names)
        ):
            raise ValueError(f"pose recording joint schema does not match: {path}")
        groups = payload.get("groups")
        if not isinstance(groups, list) or not all(isinstance(group, dict) for group in groups):
            raise ValueError(f"pose recording groups are invalid: {path}")
        journal = cls.__new__(cls)
        journal.path = path
        journal.joint_names = joint_names
        journal.payload = payload
        return journal

    @property
    def active_group(self) -> dict[str, Any]:
        name = self.payload["active_group"]
        if name is None:
            raise RuntimeError("先在终端输入分组名称并回车")
        return next(group for group in self.payload["groups"] if group["name"] == name)

    def select_group(self, name: str) -> bool:
        name = name.strip()
        if not name:
            raise ValueError("group name cannot be empty")
        existing = next(
            (group for group in self.payload["groups"] if group["name"] == name), None
        )
        created = existing is None
        if created:
            self.payload["groups"].append(
                {"name": name, "created_at": local_timestamp(), "poses": []}
            )
        self.payload["active_group"] = name
        self._touch()
        return created

    def next_pose_name(self) -> str:
        return f"pose_{len(self.active_group['poses']) + 1:03d}"

    def record(self, name: str, positions: Any) -> dict[str, Any]:
        name = name.strip()
        if not name:
            raise ValueError("pose name cannot be empty")
        values = np.asarray(positions, dtype=float)
        if values.shape != (len(self.joint_names),) or not np.isfinite(values).all():
            raise ValueError(
                f"measured pose must contain {len(self.joint_names)} finite joint positions"
            )
        pose = {
            "name": name,
            "captured_at": local_timestamp(),
            "joint_positions": values.tolist(),
        }
        self.active_group["poses"].append(pose)
        self._touch()
        return pose

    def _touch(self) -> None:
        self.payload["recording"]["updated_at"] = local_timestamp()

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(
            yaml.safe_dump(self.payload, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
