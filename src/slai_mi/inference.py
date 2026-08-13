"""Safe, backend-neutral policy inference command line entry point."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from slai_mi.policies.runtime import build_inference_plan, execute_inference


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/inference.yaml")
    parser.add_argument("--checkpoint", help="Override the checkpoint from the config")
    parser.add_argument(
        "--execute", action="store_true", help="Invoke the configured offline/simulation backend"
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        plan = build_inference_plan(Path(args.config), checkpoint_override=args.checkpoint)
        result = execute_inference(plan) if args.execute else plan.as_dict(mode="dry-run")
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
