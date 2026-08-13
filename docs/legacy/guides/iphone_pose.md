# iPhone ARKit 6DoF

This path acquires pose only. It does not connect to or command the UR5. The
current default uses Record3D over USB; the repository's own ARKit app remains
available as a no-purchase alternative.

## What Is Streamed

Record3D publishes a world-space quaternion and position over its USB protocol;
the custom app publishes ARKit's `world_from_camera` transform over TCP port
`5005`. The Linux receiver converts either source to the same validated rigid
transform and reports:

- position `x, y, z` in metres;
- rotation vector `rx, ry, rz` in radians;
- quaternion `x, y, z, w`;
- roll, pitch, yaw in degrees for operator inspection;
- the complete homogeneous transform.

The six-vector is in the local ARKit world frame. It is not yet in the UR5 base
frame and must not be sent directly to the robot.

## Custom App Installation

Apple requires every native ARKit app to be signed. A regular Apple Account and
Xcode Personal Team can sign this app for on-device testing at no cost. The
free provisioning profile expires after seven days, so the app must be rebuilt
and installed periodically. On a Mac with Xcode:

```bash
cd /home/user/shiyi/Robot_Teleoperation/ios/IPhonePoseStreamer
open IPhonePoseStreamer.xcodeproj
```

In Xcode:

1. Open the `IPhonePoseStreamer` target and select Signing & Capabilities.
2. Select your Apple development team. A free Apple ID is sufficient for local testing.
3. Connect this iPhone and select it as the run destination.
4. Press Run and accept the camera permission on the phone.
5. If prompted, enable Developer Mode under Settings > Privacy & Security.

Keep the app open in the foreground while acquiring pose.

## Receive From the Custom App

With the app open on the iPhone, run on this Linux computer:

```bash
cd /home/user/shiyi/Robot_Teleoperation
IPHONE_POSE_BACKEND=custom scripts/stream_iphone_pose.sh
```

Capture 300 validated frames to JSON Lines and stop automatically:

```bash
IPHONE_POSE_BACKEND=custom scripts/stream_iphone_pose.sh \
  --samples 300 \
  --json \
  --output-jsonl outputs/iphone_pose.jsonl
```

The launcher selects the single trusted USB iPhone and starts `iproxy` from
local port `5005` to app port `5005`. Set `IPHONE_UDID` if more than one Apple
device is connected.

The phone screen should show `USB 接收端已连接`, and the terminal should print
changing `xyz_m`, `rotvec_rad`, and `rpy_deg` values as the phone moves.

## Record3D USB

With USB Streaming enabled and recording active in Record3D, run:

```bash
scripts/stream_iphone_pose.sh
```

For a lightweight visual check without Isaac Sim or any robot connection, run:

```bash
scripts/view_iphone_pose.sh
```

The viewer scans for device index 0 automatically and retries every second if
the phone is connected late or the stream disconnects. Only one viewer can own
the Record3D USB stream; a second launch exits with an explicit message.

The solid phone is the raw, unsmoothed Record3D pose relative to the first
frame; the gray wireframe is the reset pose. At reset, place the phone flat
with its screen up and its top edge pointing toward the monitor. The tabletop
frame is then `+X` toward the operator's right, `+Y` toward the monitor, and
`+Z` vertically upward. The virtual camera looks from the seated operator's
side toward `+Y`, so physical and on-screen left/right motion agree. The view
uses a 1:1 metric scale.
Press `Space` or `R` to reset the origin, `S` to measure five seconds of
stationary position/orientation repeatability, and `Q` to close the window.
The still test measures repeatability, not absolute ground-truth accuracy.
