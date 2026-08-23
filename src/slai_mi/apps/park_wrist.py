"""Park the two-axis wrist and verify torque-hold before power-off."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from ._common import print_plan, require_real_robot_confirmation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--openrb-port", default="auto")
    parser.add_argument("--baud", type=int, default=115200)
    parser.add_argument("--velocity-raw", type=int, default=80)
    parser.add_argument("--acceleration-raw", type=int, default=25)
    parser.add_argument("--timeout-s", type=float, default=30.0)
    parser.add_argument("--tolerance-raw", type=int, default=25)
    parser.add_argument("--execute-real", action="store_true")
    parser.add_argument("--confirm")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    require_real_robot_confirmation(args.execute_real, args.confirm)
    print_plan({
        "app": "park_wrist",
        "mode": "execute" if args.execute_real else "dry-run",
        "operation": "HOME_ALL -> verify -> HOLD_ALL",
        "port": args.openrb_port,
        "power_off": "safe only after SAFE_TO_POWER_OFF",
    })
    if not args.execute_real:
        return 0

    bridge_root = Path(__file__).resolve().parents[3] / "third_party" / "02_Python_Client_CLI"
    if str(bridge_root) not in sys.path:
        sys.path.insert(0, str(bridge_root))
    from openrb_bridge.pc_client.openrb_client import OpenRBClient
    from openrb_bridge.serial_ports import resolve_openrb_port

    client = OpenRBClient(resolve_openrb_port(args.openrb_port), args.baud, timeout=1.0).connect()
    try:
        client.stop_motion()
        client.stop_all_velocity()
        client.set_arm_profile(args.velocity_raw, args.acceleration_raw)
        home = client.home_all()
        print(home.raw_lines[-1])
        deadline = time.monotonic() + args.timeout_s
        while time.monotonic() < deadline:
            status = client.get_motion_status(retries=2)
            if status.fields.get("active") == "0":
                if status.fields.get("status") != "done":
                    raise RuntimeError(status.raw_lines[-1])
                break
            time.sleep(0.05)
        else:
            raise TimeoutError("wrist HOME_ALL timed out")
        state = client.read_joints()
        actual = (int(state.fields["j1_pos"]), int(state.fields["j2_pos"]))
        limits = client.get_limits().fields
        target = (int(limits["j1_zero"]), int(limits["j2_zero"]))
        if max(abs(actual[0] - target[0]), abs(actual[1] - target[1])) > args.tolerance_raw:
            raise RuntimeError(f"HOME verification failed: target={target} actual={actual}")
        held = client.hold_all()
        print(held.raw_lines[-1])
        print(f"SAFE_TO_POWER_OFF: wrist parked and holding at raw={actual}")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
