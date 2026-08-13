# iPhone Pose Hub

The native client lives in `clients/ios/IPhonePoseStreamer`. The cloud service,
local bridge and robot handoff remain inside the package ownership boundaries:

- `slai_mi.ui.pose_hub`: authenticated WebSocket service, viewer and robot IK.
- `slai_mi.devices.iphone.pose_hub`: local bridge and measured-state handoff.
- `slai_mi.apps.pose_hub`: public service CLI.
- `slai_mi.apps.pose_hub_bridge`: 4090 bridge CLI.

Both CLIs default to a JSON dry-run plan. Production startup must pass `--run`.
Tokens and session identifiers belong in machine-local environment files and
must not be committed.

## Local contract

- `127.0.0.1:5005`: newline-delimited iPhone pose packets for the teleoperator.
- `127.0.0.1:5006`: newline-delimited measured robot joint states for reference
  binding, forward kinematics and robot-coordinate calibration.

Robot runtimes should reuse one handoff instance in their feedback loop:

```python
from slai_mi.devices.iphone import RobotStateHandoff

handoff = RobotStateHandoff()
handoff.publish(joint_names, measured_joint_positions_rad)
```

Publish fresh samples at 30-100 Hz. The joint names must follow the checked-in
URDF. Real hardware state is 26-DoF; the two physical wrist values must only be
included when a measured and documented source exists. Missing values are never
silently padded by collection code.

## Coordinate binding

The viewer first calibrates the operator frame from iPhone motion. Robot frame
calibration then samples the current palm origin, an operator-right palm point
and an operator-forward palm point. The resulting right-handed rotation maps
operator translation and rotation into `base_link`. Calibration only reads
joint state and never commands hardware.
