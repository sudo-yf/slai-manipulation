"""Inspect the real-hardware strategy groups without opening any device."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from slai_mi.runtime import (
    StrategyProfileError,
    available_strategy_profiles,
    load_strategy_profile,
)

from ._common import print_plan


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("strategy", nargs="?", help="Strategy id or explicit YAML path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.strategy:
            profile = load_strategy_profile(args.strategy)
            print_plan({"app": "strategies", **profile.plan_fields(), "source": profile.source})
        else:
            profiles = available_strategy_profiles()
            print_plan(
                {
                    "app": "strategies",
                    "strategies": [profile.plan_fields() for profile in profiles],
                }
            )
    except StrategyProfileError as exc:
        raise SystemExit(f"Strategy inspection failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
