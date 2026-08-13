# iPhone 6DoF to Simulated UR5 TCP

This chain never imports RTDE and never connects to the physical UR5:

```text
iPhone pose -> isolated input worker -> relative pose mapper
             -> bounded TCP target -> differential IK -> Isaac UR5
```

The default Record3D transport uses the existing USB Streaming purchase. Start
Record3D with USB Streaming enabled, then launch Isaac:

```bash
cd /home/user/shiyi/Robot_Teleoperation
conda activate robot_teaching_isaaclab
scripts/run_iphone_pose_sim.sh
```

When Isaac prints `In Record3D ... tap Record now`, tap Record on the iPhone.
The first valid frame becomes the phone neutral pose and the current simulated
TCP becomes the robot neutral pose. The default mapping uses that first phone
orientation, so Record3D's session heading cannot rotate the robot axes.
Restart the simulation to recenter.

For the best ARKit initialization in Record3D:

1. Switch from the Record tab to Settings and back to Record to reset tracking.
2. Point the rear camera at a bright, static scene with textured objects.
3. Move the phone slowly for a few seconds, then hold it upright in the desired
   neutral orientation and tap Record.

Avoid blank walls, darkness, reflective-only views, rapid shaking, and covering
the camera. ARKit visual-inertial odometry needs stable imagery with recognizable
features; Record3D does not expose ARKit's tracking-quality state over its USB API.

The default mapping is:

| iPhone motion | Simulated UR base motion |
| --- | --- |
| Forward, along camera `-Z` | `+X` |
| Screen-right, `+X` | `-Y` |
| Gravity-up, `+Y` | `+Z` |

Translation defaults to a `0.5` control scale and is bounded to `+/- 0.25 m` on
each base axis. Relative rotation is bounded to `60 deg`, and the target is
rate-limited to `0.20 m/s` and `0.75 rad/s`. A pose older than `0.30 s` freezes
the target. Translation and rotation use `0.06 s` and `0.08 s` low-pass time
constants plus small deadbands to reject stationary jitter.

Each status line reports the full target `xyz` and `rpy_deg`, actual-to-target
IK error, and `clamped`. For example, `clamped=XY` means the phone command is
beyond the configured X and Y workspace. Recenter or lower the scale instead of
moving farther.

For finer control:

```bash
scripts/run_iphone_pose_sim.sh --translation-scale 0.25
```

For a faster response with less smoothing:

```bash
scripts/run_iphone_pose_sim.sh \
  --translation-filter-time-constant-s 0.03 \
  --rotation-filter-time-constant-s 0.04
```

Use `--mapping-frame arkit-world` only to compare against the old session-world
mapping. It depends on Record3D's ARKit initialization heading and is not the
recommended teleoperation mode.

Define a tool center point relative to `right_palm_link` with:

```bash
scripts/run_iphone_pose_sim.sh \
  --tool-offset-m 0 0 0.12 \
  --tool-offset-rpy-deg 0 0 0
```

The marker shows the configured TCP, while differential IK commands the palm
pose required to place that TCP at the marker.

## TCP Input Transport

The custom iOS app can replace Record3D. The launcher starts the USB `iproxy`
tunnel automatically:

```bash
IPHONE_POSE_SIM_BACKEND=tcp scripts/run_iphone_pose_sim.sh
```

Both transports produce the same validated `world_from_camera` input and use
the same simulated TCP mapping.

## Hardware-Free Self-Test

```bash
scripts/run_iphone_pose_sim.sh \
  --self-test --headless --max-runtime-s 5
```

A passing run ends with `IPHONE_POSE_SIM_CONTROL_OK`.
