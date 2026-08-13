# Major real-robot pipelines: synchronization, drops, and validation

Research date: 2026-07-25. Only primary sources are used: official papers,
official repositories, and official format documentation.

## Executive finding

Real-to-real robot datasets do encounter asynchronous reads, control-loop jitter,
missing/corrupt camera recordings, and incomplete trajectories. The established
pipelines do **not** solve those problems merely by writing a standard dataset
format. Their responsibilities are split:

1. acquisition records source timing and keeps sensors independently recoverable;
2. post-processing validates and rejects or relabels bad trajectories;
3. the training loader assumes each episode is already a valid ordered sequence and
   uses masks only for model padding and episode boundaries.

Therefore, deleting rejected frames and replacing physical time with
`frame_index / nominal_fps` is not a documented DROID/OXE/Octo synchronization
strategy. It makes a real time gap invisible to the learner.

## DROID

### Acquisition and timestamps

- The DROID paper says that all released data is recorded at 15 Hz and contains
  three synchronized stereo RGB streams, robot joint/end-effector state, and robot
  commands. It also distinguishes roughly 76k successful episodes from roughly 16k
  collector-labeled unsuccessful trajectories.
- Its public acquisition loop is paced to `env.control_hz` by sleeping only when the
  observation/policy work finishes early. If a loop iteration overruns, it does not
  invent missed timesteps or catch up; the next iteration simply occurs late. The
  loop stores `step_start`, `policy_start`, `sleep_start`, `control_start`, and
  `step_end` timestamps with every timestep.
- Robot state reads store `read_start` and `read_end`. Every ZED read stores host
  `read_start`/`read_end`, the ZED image timestamp (`frame_received`), and an
  `estimated_capture` value obtained by subtracting an estimated camera latency.
- The three cameras are read sequentially; their order is randomized on each call.
  Thus, the code does not claim that Python reads occur at the same instant. It
  preserves per-camera timestamps so the timing difference is observable.
- Camera streams are also recorded separately as one native ZED SVO file per
  camera. During replay, the SVO reader checks that the frame's ZED timestamp is
  exactly the timestamp stored at acquisition. A mismatch or failed read returns
  failure rather than silently pairing a different frame.

Important limitation: the public code does not expose a general cross-camera skew
threshold or interpolation scheme. DROID controls the hardware/software platform,
records detailed per-source timing, and verifies correspondence. That is different
from proving that every modality was captured at one identical instant.

### Validation and filtering

- Collection is refused unless all three stereo cameras (six image feeds) are
  present.
- A trajectory is first written under `failure/` and moved into `success/` only when
  the collector marks it successful.
- The official post-processing pipeline is explicitly fail-fast. It checks readable
  HDF5 data, exactly three SVO files, metadata completeness, and successful SVO-to-MP4
  conversion. Missing/corrupt camera recordings in a nominally successful trajectory
  are handled by repairing them when possible or relabeling the whole trajectory as
  failure; they are not documented as being filled with duplicated frames.
- DROID training examples default to removing failure trajectories. The paper's
  policy experiments likewise exclude trajectories marked unsuccessful.

This is direct evidence that large real-robot projects expect hardware/data failures
and maintain rejection/quarantine paths. DROID's public post-processing is mainly
trajectory-level validation, not arbitrary per-frame gap repair.

Primary sources:

- DROID paper: https://arxiv.org/abs/2403.12945
- DROID paced control loop and per-step timing:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/trajectory_utils/misc.py#L55-L125
- DROID robot-state read timestamps:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/robot_env.py#L70-L107
- DROID ZED timestamps and latency estimate:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/camera_utils/camera_readers/zed_camera.py#L174-L189
- DROID sequential/randomized multi-camera reads and separate SVO recording:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/camera_utils/wrappers/multi_camera_wrapper.py#L61-L92
- DROID strict SVO frame timestamp check:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/camera_utils/recording_readers/svo_reader.py#L94-L113
- DROID camera startup gate and success/failure quarantine:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/user_interface/data_collector.py#L72-L114
- DROID post-processing policy for missing/corrupt recordings:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/scripts/README.md#L161-L188
- DROID fail-fast validator and exactly-three-SVO check:
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/postprocessing/util/validate.py#L1-L8
  https://github.com/droid-dataset/droid/blob/33ae6a67274f36d2e29525b86f23a56616ef43a7/droid/postprocessing/util/validate.py#L41-L55

## Open X-Embodiment and RT-X

### What is standardized

- Open X-Embodiment converts each source dataset into episodes and steps using
  RLDS. The RT-X paper explicitly describes heterogeneous robot setups with different
  RGB/depth/point-cloud modalities and robot execution rates ranging from 3 to 10 Hz.
- RLDS defines episode/step semantics (`is_first`, `is_last`, optional observation,
  action, reward, etc.) and provides an optional episode-level `invalid` flag for
  incomplete data such as machine preemption.
- A timestamp is **not** a mandatory RLDS step field. Extra metadata is allowed, but
  the format itself does not define clock synchronization, maximum skew, interpolation,
  or dropped-frame recovery.

Consequently, “OXE-compatible” means structurally consumable and standardized at the
semantic/action-mapping layer. It does not certify that cameras and robot state were
hardware synchronized, or that the physical sampling interval is uniform. Those
properties remain responsibilities of each source dataset and its conversion.

### Model-side consequence

The official Open X README describes RT-1-X inference as feeding the latest RGB image
at the robot's model rate (example: 3 Hz), and the released checkpoint uses one
workspace RGB stream rather than trying to synchronize every optional modality.
This is a valid deployment simplification, not a general repair mechanism for a
multicamera training trajectory.

Primary sources:

- RT-X paper: https://arxiv.org/abs/2310.08864
- Open X official dataset structure and RT-1-X observation description:
  https://github.com/google-deepmind/open_x_embodiment/blob/9eeb68b989efbcf474e8fb9019e01d02b962a604/README.md#L5-L35
- RLDS official format; optional `invalid` episode flag and mandatory step boundary
  fields:
  https://github.com/google-research/rlds/blob/b35dac3a6b73396b0cb8773095999c4b5d70947c/README.md#L57-L116

## Octo

### What the loader handles

- Octo operates on whole trajectories after an OXE-specific standardization function.
  It supports filters for unlabeled trajectories, excessive action values, excessive
  proprioceptive values, custom dataset filters, and optionally ignoring malformed
  dataset elements.
- A camera modality absent from an entire source dataset can be represented by an
  empty-string padding stream and a corresponding validity mask. This is schema
  padding for heterogeneous datasets; it is not intermittent dropped-frame repair.
- Octo constructs its own integer timestep as `tf.range(traj_len)`. It does not use
  acquisition timestamps in this standard loader. Therefore, once a converter removes
  physical samples and closes the index gaps, Octo cannot infer that time was missing.

### Episode boundaries and action windows

- History and future-action chunks are created independently inside each trajectory,
  so they do not cross from one RLDS episode into another.
- At the beginning, the first observation is repeated and a
  `timestep_pad_mask` marks the artificial history invalid.
- At the end, the last action is repeated to fill the requested horizon, while an
  `action_pad_mask` excludes actions after task completion/final timestep from loss.
- Padded action dimensions also have an `action_pad_mask`.

This is strong support for splitting a trajectory at a real timing discontinuity
before producing training chunks. Once split, Octo's normal per-episode chunking and
masks prevent a future-action horizon from bridging the gap. Keeping one renumbered
episode does not.

Primary sources:

- Octo trajectory-level filters and transform order:
  https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/dataset.py#L40-L143
- Octo loader standardization, optional modality padding, custom filtering, and
  `ignore_errors`:
  https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/dataset.py#L240-L370
- Octo creates integer timesteps from trajectory length:
  https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/dataset.py#L338-L358
- Octo within-trajectory history/action chunking and masks:
  https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/traj_transforms.py#L11-L99
- Octo validity masks and padded action dimensions:
  https://github.com/octo-models/octo/blob/241fb3514b7c40957a86d869fecb7c7fc353f540/octo/data/traj_transforms.py#L111-L145

## Implications for this project's 15 rejected candidate frames

The presence of rejected samples is not, by itself, evidence that the acquisition
system is unusable. DROID explicitly records timing, expects camera/data failures,
and quarantines invalid trajectories. The important distinction is:

- **acceptable:** retain source timestamps and quality reasons; only emit samples
  whose sensor skew/age satisfies the contract; split or mask training sequences at
  discontinuities;
- **not acceptable:** delete failed candidates, renumber the survivors to a uniform
  nominal clock, and let action horizons span the hidden gap.

For the examined episode, the five 133--200 ms discontinuities should therefore be
treated as explicit sequence boundaries (or the affected windows should be masked).
The raw LeRobot v3 episode can remain immutable and auditable; a PI0.5 training view
should derive multiple subepisodes/segments. Future acquisition should also distinguish
individual sensor-drop events from a scheduler stall and store both nominal index and
monotonic source time.

There is no primary-source basis for claiming that DROID, OXE, or Octo would consider
a nominally 30 Hz, gap-compressed sequence physically uniform merely because its
container validates.
