# PI0.5 Data Contract For UR5 + Wuji Block Placement

## Decision

Keep a lossless LeRobot Dataset v3 capture as the source of truth. Derive a
15 Hz LeRobot v2.1 training view because the pinned OpenPI checkout currently
loads `lerobot.common.datasets` v2.1. Do not change or weaken the canonical v3
dataset merely to satisfy a temporary training-loader dependency.

The learned arm action is the controller's target Cartesian TCP twist reported
by RTDE while the runtime executes `speedL`. The learned hand action is the
final 20-joint position target. This matches the operator interface and the two
real controllers used by this task. The RTDE target is deliberately used
instead of claiming that the dataset contains the application's pre-controller
SpaceMouse command.

## Canonical 30 Hz LeRobot v3 capture

| Feature | Shape | Semantics |
| --- | ---: | --- |
| `observation.images.primary_rgb` | `480x640x3` | Primary RGB video |
| `observation.images.secondary_rgb` | `480x640x3` | Secondary RGB video |
| `observation.state` | `26` | UR5 measured `q[6]` + Wuji measured `q[20]`, rad |
| `observation.tcp_pose` | `6` | Actual TCP XYZ metres + rotation vector radians, base frame |
| `action` | `26` | Target TCP twist `[6]` + Wuji position target `[20]` |
| `telemetry.ur5_target_qd` | `6` | Controller target joint velocity, rad/s |
| `telemetry.actual_tcp_speed` | `6` | Measured TCP twist |

The arm action slice is explicitly `[vx, vy, vz, wx, wy, wz]`, with linear
units m/s and angular units rad/s in the UR base frame. It must never be called
six joint angles or passed through a joint-position delta transform.

Keep all existing timestamps, sequence numbers, validity masks, restart/drop
counters, SpaceMouse state, and synchronization diagnostics.

## PI0.5 15 Hz training view

OpenPI's PI0.5 configuration natively uses `action_dim=32`, but its model
transforms already pad physical action vectors. Use all 32 state dimensions for
real measured features, while keeping action at its true 26 dimensions:

```text
state[0:6]    = actual TCP pose
state[6:12]   = actual UR5 joint angles
state[12:32]  = actual Wuji joint angles

actions[0:6]  = target TCP twist
actions[6:26] = Wuji target joint angles
```

The project transform subtracts `state[12:32]` only from `actions[6:26]`.
Therefore only absolute finger targets become deltas relative to measured
finger angles. TCP velocity is not subtracted from state. OpenPI then pads the
26-D transformed action to its internal width of 32. PI0.5 quantile
normalization is computed from the physical values before model padding. TCP
pose is intentionally included with joint state: it aligns with the Cartesian
arm action, while joint angles retain posture, limit, and singularity context.
The 26-D joint-only state remains a documented ablation, not the default.

Store two resized `224x224` RGB video streams and the task string alongside
these vectors. The deterministic conversion selects every other canonical
30 Hz frame, producing 15 Hz data and a 15-step (one second) action horizon.

## Required end-to-end checks

1. Validate v3 metadata, frame counts, finite values, video decoding, and
   monotonic 30 Hz timestamps.
2. Replay recorded controller-target TCP and hand commands through the same
   bounded deployment adapter at low speed before training.
3. Convert to the v2.1 OpenPI view and validate 32-D state, 26-D physical action,
   15-step action chunks, task prompts, and RGB shapes.
4. Compute new PI0.5 quantile normalization statistics; do not reuse DROID or
   another robot's state/action statistics.
5. Run one LoRA optimizer step and require finite loss and gradient norm.
6. Before real evaluation, enforce workspace, singularity, joint-speed,
   joint-limit, hand-slew, watchdog, and ignored-padding checks.

## Existing dataset status

`20260725_005416` is structurally valid v3 data, but its arm action is
`target_qd`, not TCP twist, and it does not contain TCP pose. It can test the
old joint-velocity contract, but it cannot validate or train the selected final
contract without lossy reconstruction. New demonstrations are required after
the recorder schema changes.

## Evidence

See `findings_vla_practice.md`, `findings_control_evidence.md`, and
`findings_practitioner_reports.md` in this directory for the original papers,
official repositories, controller documentation, and direct deployment reports.
