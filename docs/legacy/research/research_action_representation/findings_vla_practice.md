# VLA / imitation-learning action representation findings

## Bottom line

There is no single action representation prescribed by a VLA architecture. The
same DROID demonstrations are consumed as **joint velocity + absolute gripper
position** by PI's `pi05-droid` path, but as **base-frame Cartesian velocity +
gripper position** by OpenVLA. In practice, checkpoint pretraining, dataset
conversion, and the deployment controller must use the same semantics.

## Primary-source evidence

### OpenPI / pi0.5 DROID

- PI's official DROID-to-LeRobot converter defines an 8-D action as seven joint
  velocities plus one gripper position. Its comment explicitly says joint
  velocity is used because `pi05-droid` was pretrained on that representation.
- PI's official rollout creates the DROID environment with
  `action_space="joint_velocity"` and `gripper_action_space="position"`, expects
  action chunks shaped `[10, 8]`, clips them, then passes them directly to
  `env.step` at 15 Hz.
- OpenPI itself is not limited to that representation: its generic transforms
  provide a per-dimension boolean mask that converts absolute actions to deltas
  by subtracting state, and reverses that transform at inference. This supports
  mixed semantics, provided the mask and state dimensions are defined correctly.

Sources:

- [Official OpenPI DROID converter](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/examples/droid/convert_droid_data_to_lerobot.py)
- [Official pi0.5 DROID rollout](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/examples/droid/main.py)
- [OpenPI `DeltaActions` and `AbsoluteActions`](https://github.com/Physical-Intelligence/openpi/blob/15a9616a00943ada6c20a0f158e3adb39df2ccac/src/openpi/transforms.py#L203-L243)

### DROID as used by OpenVLA

- OpenVLA defines `EEF_POS` as end-effector delta XYZ + delta roll/pitch/yaw +
  gripper, and `JOINT_POS` separately as delta joint position + gripper.
- Its DROID configuration selects `EEF_POS`, not joint action, and uses Cartesian
  pose plus gripper as proprioception.
- The actual DROID transform takes `action_dict.cartesian_velocity` (translation
  and rotation) in the robot base frame and appends gripper position. Thus the
  label is a Cartesian velocity/delta-like control signal, not six robot joint
  angles or joint velocities.
- OpenVLA's official troubleshooting guidance recommends replaying recorded
  actions through the real execution pipeline before training. It also reports
  that this non-action-chunked model often works better around 5-10 Hz and can be
  harmed by many idle/small actions. These are model-specific collection
  constraints, not universal robot-control laws.

Sources:

- [OpenVLA action encodings and DROID config](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/oxe/configs.py)
- [OpenVLA DROID Cartesian-action transform](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/prismatic/vla/datasets/rlds/oxe/utils/droid_utils.py#L58-L83)
- [OpenVLA official data-collection troubleshooting](https://github.com/openvla/openvla/blob/c8f03f48af692657d3060c19588038c7220e9af9/README.md#vla-performance-troubleshooting)

### Octo / Open X-Embodiment

- Octo standardizes many robot datasets to a common seven-dimensional action:
  six end-effector motion values plus one gripper value. RT-1/Kuka transforms,
  for example, concatenate `world_vector`, `rotation_delta`, and an absolute
  gripper command.
- For Bridge, Octo calls `relabel_actions`: it replaces the first six logged
  actions with the next reached end-effector state minus the current reached
  state, while preserving the gripper action. This is explicitly a reached-pose
  delta. A reasonable inference is that the relabeling makes the learned label
  describe observed robot motion rather than only the requested command; the
  code does not claim that this is universally superior.

Sources:

- [Octo OXE standardization transforms](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/oxe/oxe_standardization_transforms.py)
- [Octo `relabel_actions` implementation](https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/utils/data_utils.py#L396-L414)

## Implications for UR5 + Wuji

1. If the target checkpoint is specifically `pi05-droid`, six UR5 joint
   velocities match its pretrained arm semantics better than Cartesian velocity,
   but the 20-DOF absolute Wuji hand remains a new, custom output contract.
2. If deployment keeps UR5 `speedL`, the most faithful behavior-cloning label is
   the six-dimensional Cartesian command actually sent to `speedL`; replacing it
   with RTDE `target_qd` changes the learned interface to joint velocity.
3. A hybrid action is technically defensible: UR5 Cartesian velocity `[6]` plus
   Wuji absolute target angles `[20]`. Store measured UR5 joints, TCP pose,
   measured hand joints, and the exact low-level commands as separate raw fields;
   construct the policy action explicitly rather than overloading a generic
   `26DOF` label.
4. Do not subtract state from velocity dimensions. If OpenPI delta transforms are
   used, their mask should cover only dimensions that are truly absolute target
   positions. Normalize arm velocity and hand angle groups independently.
5. Before committing to either representation, replay several recorded action
   sequences through the exact deployment adapter and verify task completion,
   units, frame convention, update rate, clipping, and latency. One successful
   file-format check is not evidence that the action contract is trainable.

## Limitation

These sources establish what the official implementations use; they do not
provide a controlled Cartesian-versus-joint ablation for this particular UR5 +
20-DOF hand. The final choice therefore has to be validated on this hardware and
the selected pretrained checkpoint.
