"""Run the public iPhone pose viewer and robot IK service."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from ._common import print_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8706)
    parser.add_argument("--run", action="store_true", help="Start the HTTP/WebSocket server")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 1 <= args.port <= 65535:
        raise SystemExit("--port must be between 1 and 65535")
    print_plan(
        {
            "app": "pose_hub",
            "mode": "run" if args.run else "dry-run",
            "host": args.host,
            "port": args.port,
        }
    )
    if not args.run:
        return 0
    try:
        import uvicorn
    except ImportError as exc:
        raise SystemExit("Pose Hub requires the pose-hub optional dependencies") from exc
    uvicorn.run(
        "slai_mi.ui.pose_hub.server:app",
        host=args.host,
        port=args.port,
        proxy_headers=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
