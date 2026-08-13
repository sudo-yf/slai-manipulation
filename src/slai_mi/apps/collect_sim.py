"""Collect demonstrations from simulation."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from slai_mi.simulation.runtime import collect_episodes

from ._common import load_yaml, print_plan, project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-config", default="configs/dataset.yaml")
    parser.add_argument("--task", default="configs/tasks/block_into_box.yaml")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=900)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument(
        "--scene-plugin",
        default="slai_mi.simulation.isaac.robot_scene:create_scene",
        help="Isaac scene factory as module:callable",
    )
    parser.add_argument(
        "--writer-plugin",
        default="slai_mi.simulation.writers:create_npz_writer",
        help="Writer factory as module:callable",
    )
    parser.add_argument("--run", action="store_true", help="Launch collection in Isaac Sim")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.episodes < 1:
        raise SystemExit("--episodes must be at least 1")
    if args.max_steps < 1:
        raise SystemExit("--max-steps must be at least 1")
    dataset = load_yaml(args.dataset_config)
    task = load_yaml(args.task)
    print_plan(
        {
            "app": "collect_sim",
            "mode": "run" if args.run else "dry-run",
            "task": task.get("task", {}).get("id"),
            "episodes": args.episodes,
            "headless": args.headless,
            "dataset_format": dataset.get("format"),
            "dataset_root": str(project_path(dataset.get("root", "data/lerobot"))),
        }
    )
    if args.run:
        from slai_mi.simulation.isaac.runtime import close_app, launch_scene, load_factory

        simulation_app, scene = launch_scene(
            plugin=args.scene_plugin,
            task_config=task,
            project_root=project_path("."),
            headless=args.headless,
        )
        try:
            writer = load_factory(args.writer_plugin)(
                dataset_config=dataset,
                root=project_path(dataset.get("root", "data/lerobot")),
                task_config=task,
            )
            collect_episodes(
                scene,
                writer,
                episodes=args.episodes,
                seed=args.seed,
                max_steps=args.max_steps,
            )
        finally:
            close_app(simulation_app, headless=args.headless)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
