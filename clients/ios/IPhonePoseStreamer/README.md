# iPhone Pose Streamer

Native ARKit app that streams the iPhone camera pose over local TCP and an
authenticated Pose Hub WebSocket. The transform is `world_from_camera`, in
metres, with ARKit's session origin and gravity-aligned world frame.

The local stream uses latest-only delivery on TCP `5005`. The cloud stream uses
latest-only delivery, reconnects automatically and includes a hold-to-move
deadman state plus an engagement epoch for reference rebinding.

This is the default no-purchase path. It can be installed for personal testing
with a free Xcode Personal Team; see `docs/iphone_pose.md`.

Open the checked-in project on a Mac:

```bash
cd clients/ios/IPhonePoseStreamer
open IPhonePoseStreamer.xcodeproj
```

`project.yml` is also included for teams that prefer regenerating projects with
XcodeGen.

In Xcode, select a personal development team under Signing & Capabilities,
connect the iPhone, select it as the run destination, and press Run. Accept the
camera permission on first launch. If iOS asks for Developer Mode, enable it in
Settings > Privacy & Security > Developer Mode and restart the phone.

The app is acquisition-only. Holding the on-screen control publishes permission
state but never commands a robot directly; final robot safety remains in the
4090 teleoperation runtime.
