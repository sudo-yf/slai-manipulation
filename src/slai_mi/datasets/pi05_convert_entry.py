"""LeRobot v3-side entry point for the PI0.5 converter."""

from __future__ import annotations

import argparse
from pathlib import Path

from .pi05 import convert_v3_to_v21


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--v21-python", type=Path, required=True)
    args = parser.parse_args()
    convert_v3_to_v21(
        args.source,
        args.target,
        repo_id=args.repo_id,
        source_fps=30,
        policy_fps=15,
        v21_python=args.v21_python,
    )


if __name__ == "__main__":
    main()
