"""Lazy Isaac launcher and task-scene plugin loading."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def load_factory(spec: str):
    try:
        module_name, attribute = spec.split(":", 1)
    except ValueError as exc:
        raise ValueError("scene plugin must use 'module:factory' syntax") from exc
    module = importlib.import_module(module_name)
    factory = getattr(module, attribute, None)
    if not callable(factory):
        raise TypeError(f"scene plugin factory is not callable: {spec}")
    return factory


def launch_scene(
    *,
    plugin: str,
    task_config: Mapping[str, Any],
    project_root: Path,
    headless: bool,
):
    """Start Isaac before importing any plugin that imports Isaac/Omniverse modules."""
    from isaaclab.app import AppLauncher

    launcher = AppLauncher(headless=headless)
    simulation_app = launcher.app
    try:
        # This import is deliberately after AppLauncher creation. Scene plugins may
        # import torch, isaaclab.sim, and Omniverse modules in their factory module.
        factory = load_factory(plugin)
        scene = factory(
            simulation_app=simulation_app,
            task_config=task_config,
            project_root=project_root,
        )
        return simulation_app, scene
    except Exception:
        simulation_app.close()
        raise


def close_app(simulation_app: Any, *, headless: bool) -> None:
    simulation_app.close(skip_cleanup=headless)
