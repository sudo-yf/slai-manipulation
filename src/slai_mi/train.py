"""Safe, backend-neutral training command line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from slai_mi.training.runtime import build_training_plan, execute_training


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/training.yaml")
    parser.add_argument("--dataset", help="Override the dataset root from the config")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Validate the complete dataset and invoke the configured training backend",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_training_plan(Path(args.config), dataset_override=args.dataset)
        result = execute_training(plan) if args.execute else plan.as_dict(mode="dry-run")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
