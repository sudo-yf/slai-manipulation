# Record Pipeline

## Executed Code

`python record/main.py` does not represent the whole implementation size. The
current executed path contains approximately 5,375 source lines:

| Area | Purpose | Lines |
| --- | --- | ---: |
| `record/` | Producers, synchronization, episode control, quality audit | 1,013 |
| Schema, writer, local telemetry, SpaceMouse input | Dataset and input boundary | 685 |
| UR5 control/configuration | Physical arm control and safety | 818 |
| Wuji control/configuration/safety/client | Physical hand control and safety | 2,254 |
| Hardware supervisor and launch wrappers | Preflight, process lifecycle, diagnostics | 605 |

Imports do not make those dependencies free. The UR5/Wuji safety code remains
because removing tracking-error, thermal, joint-limit, disconnect, and process
lifecycle handling would make the physical system less safe. Record-only logic
is kept in `record/main.py` and `record/sync.py`.

## Time Streams

Six producers run independently and retain five seconds of samples:

| Source | Device timestamp | Sequence |
| --- | --- | --- |
| Primary D435 RGB | RealSense color-frame timestamp | RealSense frame number |
| D405 RGB | RealSense color-frame timestamp | RealSense frame number |
| Secondary D435 RGB | RealSense color-frame timestamp | RealSense frame number |
| UR5 | RTDE controller timestamp | Recorder receive sequence |
| Wuji | SDK read completion monotonic time | Telemetry packet sequence |
| SpaceMouse | Input event monotonic arrival time | Recorder sample sequence |

RealSense and RTDE clocks are continuously mapped to `CLOCK_MONOTONIC` using
`host_time = slope * device_time + offset`. Wuji and SpaceMouse timestamps are
already on the host monotonic clock. Each sample also retains host receive time.

The primary RGB stream defines the 30 Hz dataset timeline. A primary frame is
held for 100 ms before alignment so later-arriving sources can enter their
buffers.

## Fallback Policy

| Condition | Action | Data modification |
| --- | --- | --- |
| D435 RGB sequence skips | Continue from the next real frame | No image created; gap count recorded |
| Either secondary RGB missing once | Use its nearest real RGB frame within 20 ms | A real frame may be reused; sequence and age expose it |
| Either secondary RGB farther than 20 ms | Skip this training frame | No image created |
| UR5 samples bracket RGB within 50 ms | Linear interpolation of actual joints | State is interpolated |
| Wuji samples bracket RGB within 75 ms | Linear interpolation of actual joints | State is interpolated; 75 ms tolerates one 30 Hz packet loss |
| Robot command | Use command immediately before RGB time | Zero-order hold; never interpolated |
| SpaceMouse axis/button event is old | Hold last state and set its validity bit to 0 | Zero-order hold; age recorded |
| One synchronization failure | Skip only that frame | Episode and hardware continue |
| Source API throws after startup | Close and reconnect with bounded backoff | Restart counter recorded |
| 30 consecutive synchronization failures | Discard the current episode | Dataset session and hardware continue |
| No primary RGB for 1 second | Discard the current episode | Dataset session and hardware continue |
| No UR5 or Wuji state for 200 ms | Discard the current episode | Dataset session and hardware continue |
| Hardware supervisor exits | Stop collection | Safety controller owns this decision |

RGB pixels are never averaged, generated, optical-flow interpolated, or copied
from the previous primary frame. Such synthesis could silently alter the visual
task and is therefore not an automatic fallback.

## Audit Data

Every committed frame stores:

- `telemetry.camera_skew_ms`
- `telemetry.source_age_ms`
- `telemetry.device_timestamps_s`
- `telemetry.host_receive_timestamps_s`
- `telemetry.source_sequence_numbers`
- `telemetry.source_dropped_before`
- `telemetry.source_restart_counts`
- `telemetry.validity_mask`
- `telemetry.spacemouse_axes`
- `telemetry.spacemouse_buttons`

Each dataset also contains `meta/record_policy.json`, which records the active
thresholds, and `meta/record_quality.jsonl`, which records one row for every
saved, discarded, or invalid episode attempt. It includes rejection reasons,
P95/max skew, source ages, drops, reconnects, secondary-frame reuse, stale
SpaceMouse frames, and the number of synthesized RGB frames (always zero).

The complete process transcript is written outside the dataset at
`~/.local/state/robot_teleoperation/record_latest.log`.

Datasets recorded before this synchronized schema, including
`/home/user/shiyi/task/block_into_box/datasets/20260724_232632`, contain a different feature
set. Keep them as legacy evidence, but do not merge them directly with new
sessions. A deliberate migration or feature projection is required first.
