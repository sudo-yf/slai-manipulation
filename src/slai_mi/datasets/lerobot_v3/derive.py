"""Compatibility guard for the removed non-canonical 15 Hz v3 derivative."""

from __future__ import annotations

from pathlib import Path


def derive_15hz_dataset(source_root: Path, target_root: Path, *, repo_id: str) -> Path:
    del source_root, target_root, repo_id
    raise RuntimeError(
        "15 Hz LeRobot v3 derivation is disabled: canonical capture is fixed at 30 Hz. "
        "Use scripts/convert_real_vla_for_pi05.sh to create the 15 Hz LeRobot v2.1 "
        "PI0.5 training view."
    )
