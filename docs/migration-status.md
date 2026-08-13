# Robot_Teleoperation migration status

This document audits capabilities from the legacy `Robot_Teleoperation` repository against
this repository. It is a migration ledger, not a claim that similarly named files are
behaviorally equivalent.

Status meanings:

- **migrated**: implementation and hardware-independent tests exist in the new package.
- **partial**: useful implementation exists, but an entry point, integration, dependency, or
  parity test is still missing.
- **deferred**: intentionally left for a later milestone.
- **excluded**: generated, machine-local, duplicated, or historical material that should not be
  migrated as source.

## Capability map

| Capability | Legacy source | New destination | Status | Remaining work |
| --- | --- | --- | --- | --- |
| SpaceMouse input, buttons, mapping, diagnostics | `src/robot_teleoperation/spacemouse/` | `src/slai_mi/devices/spacemouse/` | migrated | Validate supported device models on the collection workstation. |
| UR5 configuration, geometry, IK, runtime and zero pose | `src/robot_teleoperation/ur5/` | `src/slai_mi/devices/ur5/` | migrated | Hardware acceptance test remains mandatory before commanding a real arm. |
| Wujihand client, filters, runtime and safety | `src/robot_teleoperation/wuji/` | `src/slai_mi/devices/wujihand/` | partial | Pose conversion, grasp-pose persistence and landmark gating are migrated. Real SDK acceptance tests remain; do not weaken limits, temperature checks, disconnect handling, or command validation. |
| RealSense RGB/RGB-D acquisition | `collection/realsense_rgb*.py` | `devices/cameras/`, `collection/realsense_rgb*.py` | partial | Camera models and capture helpers exist; three-camera orchestration, calibration and end-to-end timestamp tests remain. |
| Stereo hand retargeting / MANO | `multiview_mano/`, `wrist_tracking/`, `realsense_only/` | `retargeting/`, `devices/wrist_sensor/`, `third_party/wuji-retargeting/` | partial | Pure geometry, robust extrinsic calibration and process boundaries exist; fitter worker and live device path need integration. |
| iPhone pose and Record3D | `iphone_pose/`, `ios/IPhonePoseStreamer/` | `devices/iphone/`, `clients/ios/IPhonePoseStreamer/`, `ui/pose_hub/` | partial | Protocol/client, Pose Hub bridge, measured-state handoff, coordinate calibration, relative TCP mapping, rate limiting, static-pose metrics and native client are present; real-hardware acceptance and supervised command integration remain. |
| Episode lifecycle and operator controls | `collection/episode.py`, `operator_control.py`, `home_control.py` | `collection/`, `runtime/real_workflows.py` | partial | The supervised save/discard lifecycle is wired and tested with injected adapters. Production device, synchronizer, recorder and LeRobot dataset factories are still missing. |
| Multi-device synchronization and continuity | record pipeline and `collection/continuity.py` | `collection/`, `datasets/lerobot_v3/continuity.py` | partial | Clock fitting, bounded queues, primary-camera frame association, interpolation/ZOH and sync diagnostics are tested. Production adapters still need to feed the synchronizer. |
| LeRobot Dataset v3 schema, contract, writer, derive and merge | `collection/vla_v3*.py` | `datasets/lerobot_v3/` | migrated | Keep 26-DoF real and 28-DoF simulation schemas explicit; run dataset compatibility tests against the pinned LeRobot environment before release. |
| Task start pose and combined zero pose | `collection/task_start_pose.py`, `combined_zero_pose.py`, task files | `configs/tasks/`, `configs/poses/`, `datasets/lerobot_v3/` | migrated | Replace unconfigured defaults only through machine-specific calibration; never copy an absolute legacy path into runtime config. |
| Session logs and manifests | `observability/session.py` | `observability/session.py` | migrated | Integrate with each runnable app and define retention policy. Runtime logs themselves are excluded. |
| Collection dashboard | `apps/device_dashboard.py`, `device_dashboard_static/`, `record/dashboard_static/` | `ui/`, `ui/static/` | partial | The read-only HTTP server, status/camera API, provider lifecycle and failure isolation are tested. The CLI defaults to offline status because a production live-provider factory is not yet available. |
| Isaac Sim articulation and drawer scene | `integrations/isaac/`, simulation apps | `simulation/`, `simulation/isaac/` | partial | Backend-neutral loops, lazy Isaac launch, robot scene plugin and NPZ writer are wired. Isaac Sim 5.1 headless loaded the articulation, ran two steps and wrote an NPZ episode on RTX 4090. Task-specific sensor/action validation remains. |
| OpenPI / PI0.5 pipeline | `integrations/openpi/`, conversion/training/evaluation tools | `policies/openpi.py`, `datasets/pi05*.py`, `training/pi05.py`, `apps/pi05.py`, `evaluation/heldout.py` | partial | v3→v2.1 conversion, 15 Hz episode-safe sampling, norm stats, LoRA training configuration and held-out horizon selection are migrated. A maintained external OpenPI checkout, checkpoint smoke test, serving and real-policy bridge remain. |
| Other policies (XVLA, PI0-FAST) | bridge apps | `policies/` | deferred | Add only when a maintained model dependency and evaluation target are selected. |
| Training and inference entry points | legacy scripts/apps and `inference/` | `train.py`, `inference.py`, `training/runtime.py`, `policies/runtime.py` | partial | Validated dry-run plans and lazy `module:factory` execution are wired. No production training/inference backend, pinned model stack, checkpoint compatibility test or real-robot inference adapter is included yet. |
| RL environment and RSL-RL configuration | `rl/`, train/eval apps | future `training/rl/` | deferred | Keep out of the base install until simulation training is a current milestone. |
| Sim-real evaluation | `simreal/`, analysis apps/reports | `evaluation/` | partial | Action/domain-gap metrics, camera landmark fitting and a runnable offline evaluation CLI exist. Dataset-scale reports and task-specific acceptance thresholds remain. |
| iOS client | `ios/IPhonePoseStreamer/` | `clients/ios/IPhonePoseStreamer/` | migrated | Build on a supported Xcode/iOS toolchain; it is intentionally outside the Python package. |
| Vendored retargeting library | `third_party/wuji-retargeting/` | `third_party/wuji-retargeting/` | migrated | Preserve its license and upstream provenance. Decide whether to pin as a submodule/package before modifying vendor code. |
| Robot URDF and meshes | `assets/` | `assets/robots/ur5_wrist_wujihand/` | partial | URDF and meshes are present and represent 28 movable joints. Redistribution licenses/provenance for UR5, wrist and Wujihand assets are not documented and must be resolved before public release. |
| Legacy guides and research notes | `docs/`, `research_*` | `docs/legacy/` | migrated | Treat as historical context, not current operating instructions. |
| Legacy tests | `tests/` | `tests/` | partial | App CLI, real supervisor, real synchronization, simulation runtime, dashboard provider, device parity helpers and train/inference plugin contracts have hardware-independent tests. Heavy SDK/model, iPhone build and full system acceptance tests remain. |

## Runnable workflow status

The five top-level workflows now have safe CLIs and hardware-independent runtime contracts. Their
status below distinguishes a tested orchestration path from a production deployment:

| Workflow | Entry point | Status |
| --- | --- | --- |
| Real teleoperation | `slai_mi.apps.teleop_real` | dry-run and supervised injected runtime; production adapter factory missing |
| Simulation teleoperation | `slai_mi.apps.teleop_sim` | runnable plugin CLI; Isaac Sim 5.1 headless smoke passed |
| Real collection | `slai_mi.apps.collect_real` | dry-run and supervised injected runtime; production adapter factory missing |
| Simulation collection | `slai_mi.apps.collect_sim` | Isaac headless two-step NPZ smoke passed; LeRobot writer plugin pending |
| Collection frontend | `slai_mi.ui.collection_frontend` | runnable read-only server; production live-provider factory missing |

Training and inference likewise have validated dry-run and backend-plugin CLIs, but execute only
when a caller supplies a maintained backend. Migration is not production-complete until real
factories compose the lower-level modules and heavy SDK/model environments pass smoke and
acceptance tests.

## Safety and data invariants

The following are release gates, not optional cleanup:

1. Real hardware starts disabled and unconfigured. No repository default may contain an active
   robot address, calibration, zero pose, or command-enable flag.
2. UR5 workspace/joint checks and Wujihand joint, temperature and disconnect protections must be
   applied at the final command boundary, including policy-driven commands.
3. Home pose, task start pose and hand semantic poses are separate concepts. Each stored pose must
   state joint order, units, device identity or calibration provenance, and dimension.
4. Real 26-DoF state and simulated 28-DoF state must use distinct declared schemas. Missing wrist
   values must never be silently inserted or removed.
5. Every recorded sample needs a documented clock domain and source timestamp. Episode commit must
   be atomic; discard and interrupted-write recovery must be tested.
6. Dataset/video/checkpoint paths are relative configuration or CLI inputs. Legacy absolute paths
   are documentation evidence only and must not become runtime defaults.

## Excluded generated and local material

Do not migrate `.cache/`, `.nv/`, `.pytest_cache/`, `.ruff_cache/`, `.venvs/`, `__pycache__/`,
`*.egg-info/`, `.swanlab/`, `swanlog/`, `logs/`, `outputs/`, `tmp/`, captured datasets,
checkpoints, or machine-generated reports as source. The new `.gitignore` covers the principal
runtime outputs; add tool-specific entries when a tool is adopted.

Large legacy `Datasets/`, `record/` outputs and analysis artifacts should be retained in managed
data storage with checksums and metadata, not copied into Git. Duplicate dashboard bundles and
historical reproduction scripts should only return if a current test or operating procedure needs
them.

## Recommended completion order

1. Implement production real-hardware adapter factories and verify final-command safety gates.
2. Implement the real synchronizer/dataset adapters and verify atomic LeRobot v3 episode commit.
3. Add a production live dashboard provider without granting motion control through the read-only API.
4. Add task-specific sensor/action validation on top of the passing Isaac headless smoke test.
5. Select and pin one maintained model backend, then run training, checkpoint and inference tests.
6. Run hardware acceptance, dataset compatibility, Isaac, iOS and license/provenance checks in
   their respective environments before declaring migration complete.
