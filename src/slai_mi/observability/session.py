"""Atomic session manifests for the combined hardware supervisor."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import platform
import re
import shutil
import socket
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from slai_mi import __version__

SCHEMA_VERSION = 2
MANIFEST_NAME = "manifest.json"
HISTORY_DIRECTORY = "history"
MAX_LOG_TAIL_BYTES = 256 * 1024

INCIDENT_PATTERNS = (
    (
        "wuji_over_temperature",
        "wuji",
        re.compile(r"(?:runtime Wuji temperature limit|joint temperature exceeds)", re.IGNORECASE),
        "Wuji temperature protection stopped control.",
        "Let the hand cool below the operating limit; use lyf --slow for the next run.",
    ),
    (
        "realsense_alignment_lost",
        "realsense",
        re.compile(r"lost live RealSense hand during startup alignment", re.IGNORECASE),
        "RealSense lost the hand target during Wuji alignment.",
        "Keep the full hand visible and well lit while the hand safely opens and reacquires.",
    ),
    (
        "wuji_usb_unavailable",
        "wuji",
        re.compile(r"cannot open Wuji USB|Failed to init.*Wuji|ERROR_ACCESS", re.IGNORECASE),
        "The Wuji USB device could not be opened.",
        "Reconnect Wuji 12V power and USB, then confirm no HMI or other controller is open.",
    ),
    (
        "ur5_unreachable",
        "ur5",
        re.compile(
            r"No route to host|Could not connect to:.*(?:30004|UR5)|"
            r"RTDE receive did not connect",
            re.IGNORECASE,
        ),
        "The UR5 RTDE endpoint was unreachable.",
        "Check the robot IP, Ethernet carrier, route, and that the UR5 is powered on.",
    ),
    (
        "spacemouse_no_events",
        "spacemouse",
        re.compile(r"no SpaceMouse motion events were received", re.IGNORECASE),
        "SpaceMouse input stopped producing motion events.",
        "Reconnect the SpaceMouse and restart spacenavd before retrying.",
    ),
    (
        "spacemouse_preflight_failed",
        "spacemouse",
        re.compile(r"spacemouse_preflight_failed|spacenavd_restart_failed", re.IGNORECASE),
        "SpaceMouse or spacenavd preflight failed.",
        "Check the USB binding and spacenavd service status.",
    ),
    (
        "supervisor_disappeared",
        "supervisor",
        re.compile(r"supervisor_disappeared", re.IGNORECASE),
        "The lyf supervisor disappeared before cleanup completed.",
        "Inspect kernel and power logs, then verify no controller process remains.",
    ),
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="milliseconds")


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.chmod(0o600)
    os.replace(temporary, path)


def _read_manifest(run_directory: Path) -> dict[str, Any] | None:
    path = run_directory / MANIFEST_NAME
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    return value if isinstance(value, dict) else None


def _read_log_tail(path: Path) -> str:
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - MAX_LOG_TAIL_BYTES))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _duration_seconds(manifest: dict[str, Any]) -> float | None:
    try:
        started = datetime.fromisoformat(manifest["started_at"])
        finished = datetime.fromisoformat(manifest["finished_at"])
    except (KeyError, TypeError, ValueError):
        return None
    return round(max(0.0, (finished - started).total_seconds()), 3)


def classify_incident(
    run_directory: Path,
    *,
    reason: str,
    exit_code: int | None,
) -> dict[str, str] | None:
    """Turn raw controller output into a stable, actionable incident code."""
    if exit_code == 0 or reason.startswith("signal_"):
        return None

    text_parts = [reason]
    for name in ("wuji.log", "ur5.log", "supervisor.log"):
        text_parts.append(_read_log_tail(run_directory / name))
    text = "\n".join(text_parts)
    failure_lines = re.findall(r"^(?:UR5|Wuji) teleoperation failed:\s*(.+)$", text, re.MULTILINE)

    for code, component, pattern, summary, action in INCIDENT_PATTERNS:
        if pattern.search(text):
            message = failure_lines[-1].strip() if failure_lines else summary
            return {
                "action": action,
                "code": code,
                "component": component,
                "message": message[:500],
                "summary": summary,
            }

    component = "supervisor"
    if reason.startswith("ur5_"):
        component = "ur5"
    elif reason.startswith("wuji_"):
        component = "wuji"
    message = failure_lines[-1].strip() if failure_lines else reason.replace("_", " ")
    return {
        "action": "Inspect the session summary and component logs before retrying.",
        "code": reason,
        "component": component,
        "message": message[:500],
        "summary": f"The {component} session ended unexpectedly.",
    }


def _archive_history(state_root: Path, manifest: dict[str, Any]) -> Path:
    history_directory = state_root / HISTORY_DIRECTORY
    history_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    history_directory.chmod(0o700)
    result = manifest.get("result") if isinstance(manifest.get("result"), dict) else {}
    record = {
        "duration_seconds": manifest.get("duration_seconds"),
        "finished_at": manifest.get("finished_at"),
        "incident": result.get("incident"),
        "profile": manifest.get("profile"),
        "result": {
            "children": result.get("children", {}),
            "exit_code": result.get("exit_code"),
            "reason": result.get("reason"),
        },
        "schema_version": 1,
        "session_id": manifest.get("session_id"),
        "started_at": manifest.get("started_at"),
        "status": manifest.get("status"),
    }
    destination = history_directory / f"{manifest['session_id']}.json"
    _write_json(destination, record)
    return destination


def _process_exists(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def mark_abandoned_sessions(runs_root: Path) -> None:
    """Close manifests whose supervisor disappeared without running its EXIT trap."""
    if not runs_root.is_dir():
        return
    for run_directory in runs_root.iterdir():
        if not run_directory.is_dir():
            continue
        manifest = _read_manifest(run_directory)
        if not manifest or manifest.get("status") != "running":
            continue
        if _process_exists(manifest.get("supervisor_pid")):
            continue
        manifest["status"] = "aborted"
        manifest["finished_at"] = _timestamp()
        manifest["result"] = {
            "children": {},
            "exit_code": None,
            "reason": "supervisor_disappeared",
        }
        manifest["duration_seconds"] = _duration_seconds(manifest)
        manifest["result"]["incident"] = classify_incident(
            run_directory,
            reason="supervisor_disappeared",
            exit_code=None,
        )
        manifest["schema_version"] = SCHEMA_VERSION
        _write_json(run_directory / MANIFEST_NAME, manifest)
        _archive_history(runs_root.parent, manifest)


def _update_latest_link(state_root: Path, run_directory: Path) -> None:
    link = state_root / "latest"
    temporary = state_root / f".latest.{os.getpid()}.tmp"
    with contextlib.suppress(FileNotFoundError):
        temporary.unlink()
    temporary.symlink_to(run_directory)
    os.replace(temporary, link)


def begin_session(
    run_directory: Path,
    *,
    session_id: str,
    profile: str,
    max_runtime_s: int,
    ur5_command: str,
    wuji_command: str,
    supervisor_pid: int,
) -> dict[str, Any]:
    """Create one running manifest and make it the latest session."""
    state_root = run_directory.parent.parent
    runs_root = run_directory.parent
    state_root.mkdir(parents=True, exist_ok=True)
    runs_root.mkdir(parents=True, exist_ok=True)
    mark_abandoned_sessions(runs_root)
    run_directory.mkdir(mode=0o700, exist_ok=False)
    manifest: dict[str, Any] = {
        "commands": {"ur5": ur5_command, "wuji": wuji_command},
        "files": {
            "kernel": "kernel.log",
            "manifest": MANIFEST_NAME,
            "spacenavd": "spacenavd.log",
            "summary": "summary.txt",
            "supervisor": "supervisor.log",
            "system_before": "system_before.txt",
            "system_after": "system_after.txt",
            "ur5": "ur5.log",
            "wuji": "wuji.log",
            "wuji_sdk_directory": "wuji_sdk",
        },
        "finished_at": None,
        "host": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": platform.python_version(),
        },
        "max_runtime_s": max_runtime_s,
        "profile": profile,
        "project_version": __version__,
        "result": None,
        "schema_version": SCHEMA_VERSION,
        "session_id": session_id,
        "started_at": _timestamp(),
        "status": "running",
        "supervisor_pid": supervisor_pid,
    }
    _write_json(run_directory / MANIFEST_NAME, manifest)
    _update_latest_link(state_root, run_directory)
    return manifest


def finish_session(
    run_directory: Path,
    *,
    exit_code: int,
    reason: str,
    children: dict[str, int | None],
) -> dict[str, Any]:
    """Atomically close a running session manifest."""
    manifest = _read_manifest(run_directory)
    if manifest is None:
        raise RuntimeError(f"session manifest is missing or invalid: {run_directory}")
    if reason.startswith("signal_"):
        status = "interrupted"
    elif exit_code == 0:
        status = "completed"
    else:
        status = "failed"
    manifest["finished_at"] = _timestamp()
    manifest["result"] = {
        "children": children,
        "exit_code": exit_code,
        "incident": classify_incident(
            run_directory,
            reason=reason,
            exit_code=exit_code,
        ),
        "reason": reason,
    }
    manifest["status"] = status
    manifest["duration_seconds"] = _duration_seconds(manifest)
    manifest["schema_version"] = SCHEMA_VERSION
    _write_json(run_directory / MANIFEST_NAME, manifest)
    _archive_history(run_directory.parent.parent, manifest)
    return manifest


def backfill_history(state_root: Path) -> int:
    """Classify and archive completed manifests created by earlier logger versions."""
    runs_root = state_root / "runs"
    if not runs_root.is_dir():
        return 0
    archived = 0
    for run_directory in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        manifest = _read_manifest(run_directory)
        if not manifest or manifest.get("status") == "running":
            continue
        result = manifest.get("result")
        if not isinstance(result, dict):
            continue
        reason = str(result.get("reason") or "unknown_failure")
        exit_code = result.get("exit_code")
        if not isinstance(exit_code, int):
            exit_code = None
        result["incident"] = classify_incident(
            run_directory,
            reason=reason,
            exit_code=exit_code,
        )
        manifest["duration_seconds"] = _duration_seconds(manifest)
        manifest["schema_version"] = SCHEMA_VERSION
        _write_json(run_directory / MANIFEST_NAME, manifest)
        _archive_history(state_root, manifest)
        archived += 1
    return archived


def read_history(state_root: Path) -> list[dict[str, Any]]:
    history_directory = state_root / HISTORY_DIRECTORY
    if not history_directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in history_directory.glob("*.json"):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(record, dict):
            records.append(record)
    return sorted(records, key=lambda item: str(item.get("started_at") or ""), reverse=True)


def render_history(state_root: Path, *, limit: int = 20) -> str:
    records = read_history(state_root)
    if not records:
        return f"No lyf history found under {state_root}.\n"
    incidents = [item.get("incident") for item in records if isinstance(item.get("incident"), dict)]
    counts = Counter(str(item.get("code") or "unknown") for item in incidents)
    failed = sum(item.get("status") in {"failed", "aborted"} for item in records)
    lines = [
        f"Sessions: {len(records)} total, {failed} failed/aborted",
        "",
        "Failure counts:",
    ]
    if counts:
        for code, count in counts.most_common():
            sample = next(item for item in incidents if item.get("code") == code)
            lines.append(f"  {count:>4}  {code:<30} {sample.get('summary', '')}")
            lines.append(f"        Action: {sample.get('action', '')}")
    else:
        lines.append("     0  No classified failures")
    lines.extend(("", "Recent sessions:"))
    for record in records[:limit]:
        incident = record.get("incident")
        code = incident.get("code") if isinstance(incident, dict) else "-"
        reason = record.get("result", {}).get("reason") or "-"
        lines.append(
            f"  {record.get('session_id', '?'):<24} "
            f"{record.get('status', 'unknown'):<12} {code:<30} {reason}"
        )
    return "\n".join(lines) + "\n"


def prune_sessions(runs_root: Path, keep: int, *, exclude: Path | None = None) -> list[Path]:
    """Remove oldest completed sessions while preserving running and excluded runs."""
    if keep < 0:
        raise ValueError("keep must be non-negative")
    if keep == 0 or not runs_root.is_dir():
        return []
    directories = sorted(
        (path for path in runs_root.iterdir() if path.is_dir()),
        key=lambda path: path.name,
        reverse=True,
    )
    removable: list[Path] = []
    retained = 0
    excluded = exclude.resolve() if exclude else None
    for directory in directories:
        if excluded is not None and directory.resolve() == excluded:
            continue
        manifest = _read_manifest(directory)
        if manifest and manifest.get("status") == "running":
            continue
        if retained < keep:
            retained += 1
        else:
            removable.append(directory)
    for directory in removable:
        shutil.rmtree(directory)
    return removable


def _parse_child_status(values: list[str]) -> dict[str, int | None]:
    statuses: dict[str, int | None] = {}
    for value in values:
        name, separator, raw_status = value.partition("=")
        if not separator or not name:
            raise ValueError(f"invalid child status: {value}")
        statuses[name] = None if raw_status == "unknown" else int(raw_status)
    return statuses


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="action", required=True)

    begin = commands.add_parser("begin", help="create a running session manifest")
    begin.add_argument("--run-directory", type=Path, required=True)
    begin.add_argument("--session-id", required=True)
    begin.add_argument("--profile", required=True)
    begin.add_argument("--max-runtime-s", type=int, required=True)
    begin.add_argument("--ur5-command", required=True)
    begin.add_argument("--wuji-command", required=True)
    begin.add_argument("--supervisor-pid", type=int, required=True)

    finish = commands.add_parser("finish", help="close a session manifest")
    finish.add_argument("--run-directory", type=Path, required=True)
    finish.add_argument("--exit-code", type=int, required=True)
    finish.add_argument("--reason", required=True)
    finish.add_argument("--child", action="append", default=[])

    prune = commands.add_parser("prune", help="apply completed-session retention")
    prune.add_argument("--runs-root", type=Path, required=True)
    prune.add_argument("--keep", type=int, required=True)
    prune.add_argument("--exclude", type=Path)

    backfill = commands.add_parser("backfill", help="archive and classify existing sessions")
    backfill.add_argument("--state-root", type=Path, required=True)

    history = commands.add_parser("history", help="summarize durable session history")
    history.add_argument("--state-root", type=Path, required=True)
    history.add_argument("--limit", type=int, default=20)
    return parser


def main(argv: list[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.action == "begin":
        begin_session(
            arguments.run_directory,
            session_id=arguments.session_id,
            profile=arguments.profile,
            max_runtime_s=arguments.max_runtime_s,
            ur5_command=arguments.ur5_command,
            wuji_command=arguments.wuji_command,
            supervisor_pid=arguments.supervisor_pid,
        )
    elif arguments.action == "finish":
        finish_session(
            arguments.run_directory,
            exit_code=arguments.exit_code,
            reason=arguments.reason,
            children=_parse_child_status(arguments.child),
        )
    elif arguments.action == "prune":
        prune_sessions(arguments.runs_root, arguments.keep, exclude=arguments.exclude)
    elif arguments.action == "backfill":
        print(f"Archived {backfill_history(arguments.state_root)} sessions.")
    else:
        if arguments.limit < 1 or arguments.limit > 10000:
            raise ValueError("history limit must be in [1, 10000]")
        print(render_history(arguments.state_root, limit=arguments.limit), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
