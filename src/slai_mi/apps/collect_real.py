"""Collect demonstrations from the physical robot system."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from slai_mi.runtime import CollectionDependencies, RealCollectionWorkflow
from slai_mi.runtime.adapters import (
    AdapterPluginError,
    adapter_plugin_spec,
    build_adapter_dependencies,
)

from ._common import (
    enabled_devices,
    load_yaml,
    print_plan,
    project_path,
    require_real_robot_confirmation,
)

_dependencies_factory = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--task", default="configs/tasks/block_into_box.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--execute-real", action="store_true", help="Allow real robot commands")
    parser.add_argument("--confirm", help="Required physical-motion confirmation phrase")
    parser.add_argument("--adapter-plugin", help="Site adapter factory as module:factory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    hardware = load_yaml(args.hardware_config)
    dataset = load_yaml(args.dataset_config)
    task = load_yaml(args.task)
    require_real_robot_confirmation(args.execute_real, args.confirm)
    print_plan(
        {
            "app": "collect_real",
            "mode": "execute" if args.execute_real else "dry-run",
            "task": task.get("task", {}).get("id"),
            "episodes": args.episodes,
            "dataset_format": dataset.get("format"),
            "dataset_root": str(project_path(dataset.get("root", "data/lerobot"))),
            "enabled_devices": enabled_devices(hardware),
        }
    )
    if args.execute_real:
        spec = adapter_plugin_spec(args.adapter_plugin, hardware)
        if spec is None and _dependencies_factory is None:
            raise SystemExit(
                "Real collection adapters are unavailable in this installation; no motion was sent."
            )
        try:
            dependencies: CollectionDependencies = (
                build_adapter_dependencies(
                    spec, CollectionDependencies, hardware, dataset, task
                )
                if spec is not None
                else _dependencies_factory(hardware, dataset, task)
            )
        except AdapterPluginError as exc:
            raise SystemExit(f"Real adapter plugin failed: {exc}") from exc
        try:
            RealCollectionWorkflow(
                hardware, dataset, task, dependencies, episode_limit=args.episodes
            ).run()
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Real collection failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
