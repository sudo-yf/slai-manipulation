"""Teleoperate the simulated UR5 and Wujihand system."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from slai_mi.devices.spacemouse.device import SpaceMouse
from slai_mi.devices.spacemouse.mapping import (
    SpeedSettings,
    build_hardware_twist,
    select_speed_limits,
)
from slai_mi.simulation.runtime import SimulationCommand, run_teleoperation

from ._common import load_yaml, print_plan, project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", default="configs/tasks/block_into_box.yaml")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scene-plugin",
        default="slai_mi.simulation.isaac.robot_scene:create_scene",
        help="Isaac scene factory as module:callable",
    )
    parser.add_argument("--run", action="store_true", help="Launch the Isaac simulation")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    task = load_yaml(args.task)
    print_plan(
        {
            "app": "teleop_sim",
            "mode": "run" if args.run else "dry-run",
            "task": task.get("task", {}).get("id"),
            "task_config": str(project_path(args.task)),
            "headless": args.headless,
        }
    )
    if args.run:
        from slai_mi.simulation.isaac.runtime import close_app, launch_scene

        simulation_app, scene = launch_scene(
            plugin=args.scene_plugin,
            task_config=task,
            project_root=project_path("."),
            headless=args.headless,
        )
        settings = SpeedSettings()
        try:
            with SpaceMouse() as spacemouse:
                def command_source() -> SimulationCommand:
                    motion, buttons = spacemouse.state()
                    mapped = build_hardware_twist(
                        motion, buttons, select_speed_limits(buttons, settings)
                    )
                    return SimulationCommand(mapped.twist, buttons)

                run_teleoperation(
                    scene, command_source, should_continue=simulation_app.is_running
                )
        finally:
            close_app(simulation_app, headless=args.headless)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
