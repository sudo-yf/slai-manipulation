"""Run the local iPhone Pose Hub bridge."""

from __future__ import annotations

import argparse
import asyncio
import os
from collections.abc import Sequence

from slai_mi.devices.iphone.pose_hub import PoseHubBridge

from ._common import print_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://6d.leai.me")
    parser.add_argument("--session", required=True)
    parser.add_argument("--port", type=int, default=5005, help="local pose output port")
    parser.add_argument("--robot-state-port", type=int, default=5006)
    parser.add_argument("--max-age", type=float, default=0.25)
    parser.add_argument("--run", action="store_true", help="Open network connections")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    print_plan(
        {
            "app": "pose_hub_bridge",
            "mode": "run" if args.run else "dry-run",
            "url": args.url,
            "session": args.session,
            "pose_port": args.port,
            "robot_state_port": args.robot_state_port,
            "max_age_s": args.max_age,
        }
    )
    if not args.run:
        return 0
    token = os.environ.get("POSE_HUB_BRIDGE_TOKEN", "")
    if not token:
        raise SystemExit("POSE_HUB_BRIDGE_TOKEN must be set")
    bridge = PoseHubBridge(
        args.url,
        args.session,
        token,
        args.port,
        args.robot_state_port,
        args.max_age,
    )
    asyncio.run(bridge.run())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
