"""Prepare datasets and train PI0.5 for the UR5/Wujihand pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from ._common import load_yaml, project_path


def _settings(config: dict) -> dict:
    if config.get("schema_version") != 1:
        raise ValueError("PI0.5 config schema_version must be 1")
    dataset, policy, training = config["dataset"], config["policy"], config["training"]
    openpi_root = training.get("openpi_root") or os.environ.get("OPENPI_ROOT")
    v21_python = training.get("lerobot_v21_python") or os.environ.get("LEROBOT_V21_PYTHON")
    v3_python = training.get("lerobot_v3_python") or os.environ.get("LEROBOT_V3_PYTHON")
    return {
        **policy,
        **training,
        "source": project_path(dataset["source"]),
        "converted": project_path(dataset["converted"]),
        "repo_id": dataset["repo_id"],
        "source_fps": int(dataset["source_fps"]),
        "policy_fps": int(dataset["policy_fps"]),
        "openpi_root": openpi_root,
        "v21_python": v21_python,
        "v3_python": project_path(v3_python) if v3_python else None,
        "assets_dir": project_path(training["assets_dir"]),
        "checkpoints_dir": project_path(training["checkpoints_dir"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("convert", "config", "norm", "train"))
    parser.add_argument("--config", default="configs/pi05.yaml")
    parser.add_argument("--source", type=Path, help="One v3 root or a directory of v3 roots")
    parser.add_argument("--target", type=Path, help="Output LeRobot v2.1 root")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = _settings(load_yaml(args.config))
        if args.source is not None:
            settings["source"] = args.source.expanduser().resolve()
        if args.target is not None:
            settings["converted"] = args.target.expanduser().resolve()
        plan = {
            "app": "pi05",
            "command": args.command,
            "mode": "execute" if args.execute else "dry-run",
            "source": str(settings["source"]),
            "converted": str(settings["converted"]),
            "repo_id": settings["repo_id"],
            "openpi_root": settings["openpi_root"],
            "v21_python": settings["v21_python"],
            "v3_python": str(settings["v3_python"]) if settings["v3_python"] else None,
            "smoke": args.smoke,
        }
        print(json.dumps(plan, indent=2))
        if not args.execute:
            return 0
        # OpenPI has a tightly pinned Python/JAX/LeRobot environment. Re-enter
        # this module with that interpreter so ``uv run slai-pi05 ...`` cannot
        # accidentally use the lightweight project environment.
        if args.command in ("config", "norm", "train") and os.environ.get("SLAI_PI05_REEXEC") != "1":
            if not settings["v21_python"]:
                raise ValueError("set training.lerobot_v21_python or LEROBOT_V21_PYTHON")
            env = os.environ.copy()
            env["SLAI_PI05_REEXEC"] = "1"
            subprocess.run(
                [str(settings["v21_python"]), "-m", "slai_mi.apps.pi05", *sys.argv[1:]],
                check=True,
                env=env,
            )
            return 0
        if args.command == "convert":
            if not settings["v21_python"]:
                raise ValueError("set training.lerobot_v21_python or LEROBOT_V21_PYTHON")
            if not settings["v3_python"]:
                raise ValueError("set training.lerobot_v3_python or LEROBOT_V3_PYTHON")
            subprocess.run(
                [
                    str(settings["v3_python"]),
                    "-m",
                    "slai_mi.datasets.pi05_convert_entry",
                    str(settings["source"]),
                    str(settings["converted"]),
                    "--repo-id",
                    settings["repo_id"],
                    "--v21-python",
                    str(settings["v21_python"]),
                ],
                check=True,
            )
        elif args.command == "config":
            from slai_mi.training.pi05 import make_train_config

            print(make_train_config(settings, smoke=args.smoke))
        else:
            if not settings["openpi_root"]:
                raise ValueError("set training.openpi_root or OPENPI_ROOT")
            from slai_mi.training.pi05 import run_openpi

            run_openpi(args.command, settings, smoke=args.smoke, max_frames=args.max_frames)
        return 0
    except (ImportError, KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"PI0.5 {args.command} failed: {exc}") from exc


if __name__ == "__main__":
    sys.exit(main())
