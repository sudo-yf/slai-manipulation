"""Teleoperate the physical UR5 and Wujihand system."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from slai_mi.runtime import (
    RealTeleopWorkflow,
    StrategyProfileError,
    TeleopDependencies,
    load_strategy_profile,
)
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
DEFAULT_STRATEGY = "ur5e_wujihand_retargeting"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--task", default="configs/tasks/block_into_box.yaml")
    parser.add_argument(
        "--strategy",
        default=DEFAULT_STRATEGY,
        help="Strategy id from configs/strategies or an explicit YAML path",
    )
    parser.add_argument("--execute-real", action="store_true", help="Allow real robot commands")
    parser.add_argument("--confirm", help="Required physical-motion confirmation phrase")
    parser.add_argument("--adapter-plugin", help="Site adapter factory as module:factory")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hardware = load_yaml(args.hardware_config)
    task = load_yaml(args.task)
    require_real_robot_confirmation(args.execute_real, args.confirm)
    try:
        strategy = load_strategy_profile(args.strategy)
        strategy.validate_for(
            "teleop_real", hardware=hardware, task=task, execute=args.execute_real
        )
        hardware = strategy.configure_hardware(hardware)
    except StrategyProfileError as exc:
        raise SystemExit(f"Real teleoperation strategy failed: {exc}") from exc
    plan = {
        "app": "teleop_real",
        "mode": "execute" if args.execute_real else "dry-run",
        "hardware_config": str(project_path(args.hardware_config)),
        "task": task.get("task", {}).get("id"),
        "enabled_devices": enabled_devices(hardware),
        **strategy.plan_fields(),
    }
    print_plan(plan)
    if args.execute_real:
        spec = adapter_plugin_spec(args.adapter_plugin, hardware)
        if spec is None and _dependencies_factory is None:
            raise SystemExit(
                "Real adapters are unavailable in this installation; no motion was sent."
            )
        try:
            dependencies: TeleopDependencies = (
                build_adapter_dependencies(spec, TeleopDependencies, hardware, task)
                if spec is not None
                else _dependencies_factory(hardware, task)
            )
        except AdapterPluginError as exc:
            raise SystemExit(f"Real adapter plugin failed: {exc}") from exc
        try:
            RealTeleopWorkflow(hardware, dependencies).run()
        except (OSError, RuntimeError, ValueError) as exc:
            raise SystemExit(f"Real teleoperation failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
