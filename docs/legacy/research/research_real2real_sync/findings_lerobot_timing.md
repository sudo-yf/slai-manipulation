# LeRobot Dataset v3 timing semantics

## Scope and evidence version

This note uses only Hugging Face's official LeRobot documentation, source code,
and repository discussions. The locally installed LeRobot `0.6.1` package was
built from official commit
[`9c82c39c7b541e9c5bd8340abb7c9d8803c98744`](https://github.com/huggingface/lerobot/commit/9c82c39c7b541e9c5bd8340abb7c9d8803c98744),
so all source links below are pinned to that exact revision.

## Findings

### 1. Standard v3 timestamps are a fixed-FPS logical timeline

`DatasetWriter.add_frame()` forbids the caller from supplying `timestamp` or
`frame_index`. For every accepted row it assigns:

```python
frame_index = episode_buffer["size"]
timestamp = frame_index / fps
```

Source:
[`dataset_writer.py`, lines 185-210](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/datasets/dataset_writer.py#L185-L210).

Consequences:

- `timestamp` is an idealized sample-grid coordinate, not a hardware capture
  timestamp and not elapsed wall-clock time.
- The writer has no representation for a missing sampling slot inside an
  episode. If a custom recorder considers 805 candidate slots but calls
  `add_frame()` only 790 times, the saved rows are numbered `0..789` and occupy
  `0..789/fps`, regardless of where the 15 rejected slots occurred.
- A dataset can therefore be structurally valid LeRobot v3 while its logical
  timeline is shorter than the physical acquisition interval.

The official v3 documentation describes timestamps as tabular time-series data
and FPS as canonical metadata, but does not claim that the standard `timestamp`
field preserves sensor-native time:
[`lerobot-dataset-v3.mdx`, lines 57-63](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/docs/source/lerobot-dataset-v3.mdx#L57-L63).

### 2. The official recorder also assumes the loop keeps up with target FPS

The official recording loop sets `control_interval = 1 / fps`, performs one
observation/action/write per loop, then sleeps only for the remaining interval.
Episode termination uses actual `perf_counter()` elapsed time. When work takes
longer than the interval, it does not insert a placeholder or advance an
explicit acquisition-slot index; it immediately starts the next iteration.

Source:
[`lerobot_record.py`, lines 277-358](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/scripts/lerobot_record.py#L277-L358).

LeRobot itself emits this warning when the loop misses its budget: dataset
frames may be dropped and robot control may be unstable; listed causes are a
camera not keeping up, slow policy inference, and CPU starvation (same source,
lines 348-354). This is direct evidence that missed-frame behavior is a known,
general recording concern, not something unique to this project.

Because `add_frame()` still creates the fixed grid described above, a slow loop
causes the saved LeRobot duration (`number_of_rows / fps`) to be shorter than
wall-clock duration. The compression is the combined result of (a) upstream
missed/rejected samples and (b) the standard fixed-grid writer; it is not video
codec compression.

### 3. Videos are also encoded as constant-frame-rate streams

For offline encoding, LeRobot sorts accepted image files and creates an output
stream at the dataset FPS. No source capture timestamps are passed to the
encoder:
[`video_utils.py`, lines 436-525](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/datasets/video_utils.py#L436-L525).

For streaming encoding, each accepted frame receives consecutive PTS
`frame_count` with `time_base = 1/fps`:
[`video_utils.py`, lines 790-840](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/datasets/video_utils.py#L790-L840).

Thus Parquet rows and MP4 frames remain mutually aligned after a rejected
candidate frame, but both hide the corresponding physical-time gap. This
explains why a dataset can decode perfectly and still contain locally
time-compressed motion.

### 4. `tolerance_s` is not a multi-device acquisition synchronizer

During video loading, LeRobot finds the decoded frame closest to each requested
timestamp and raises `FrameTimestampError` when the difference is not below
`tolerance_s`. The official error advises ignoring the item during training and
mentions timestamp synchronization during collection as a possible cause:
[`video_utils.py`, lines 180-207](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/datasets/video_utils.py#L180-L207).

This tolerance checks alignment between LeRobot query timestamps and encoded
video PTS. It does **not** compare RealSense, robot, hand, and input-device
hardware clocks, and it cannot detect an acquisition slot that was never passed
to `add_frame()`.

The project has also merged a training-config option to adjust this decoding
tolerance, showing that timestamp/PTS tolerance failures occur in other LeRobot
deployments:
[`PR #2653`](https://github.com/huggingface/lerobot/pull/2653). Increasing the
tolerance can accommodate small video-PTS differences, but it cannot restore a
missing physical interval or repair incorrect action timing.

An official issue reports the same `FrameTimestampError` class on v3 datasets,
and another contributor reports encountering it as well:
[`issue #2814`](https://github.com/huggingface/lerobot/issues/2814). This issue is
useful as evidence of prevalence, but it should not be treated as proof of the
cause of this project's 15 rejected candidate frames.

### 5. `delta_timestamps` and action horizons operate on frame indices

LeRobot requires each `delta_timestamp` to lie on the fixed FPS grid, then
converts it to an integer offset with `round(delta * fps)`:
[`feature_utils.py`, lines 160-218](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/datasets/feature_utils.py#L160-L218).

Training configuration performs the inverse mapping from requested delta
indices to seconds, `index / dataset_fps`:
[`factory.py`, lines 34-66](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/datasets/factory.py#L34-L66).

The reader then retrieves `abs_idx + delta`, clipping only at episode
boundaries and emitting padding masks there:
[`dataset_reader.py`, lines 215-246](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/src/lerobot/datasets/dataset_reader.py#L215-L246).

Therefore a future action horizon means "the next N saved rows," not "the next
N hardware-time samples." If rejected frames are removed inside an episode, a
horizon crosses the hidden gap without a padding mask. Splitting at a detected
gap converts it into a real episode boundary, after which the standard reader
clips and marks the horizon padding correctly.

The official documentation presents temporal windows as seconds relative to
the current sample, but those seconds are implemented through this FPS-grid
index conversion:
[`lerobot-dataset-v3.mdx`, lines 111-119](https://github.com/huggingface/lerobot/blob/9c82c39c7b541e9c5bd8340abb7c9d8803c98744/docs/source/lerobot-dataset-v3.mdx#L111-L119).

## Conclusions for the local recorder

1. The 15 synchronization rejections are produced by the project's acquisition
   gate, not by LeRobot. Their exact causes must be diagnosed from local reject
   reason counters and source timestamps.
2. Once those candidates are omitted, LeRobot's standard writer deliberately
   closes the row numbering. That is the direct cause of the five hidden gaps
   appearing as time compression.
3. This behavior is compatible with the standard v3 format. Format compliance
   proves schema/readability, not physical-time fidelity.
4. The correct remedies are upstream synchronization and load control first;
   when a gap still occurs, preserve native timestamps in telemetry and split
   the derived training view at the gap. Merely increasing `tolerance_s` or
   retaining a continuous episode does not repair action-horizon timing.
5. Repeating the prior frame would preserve duration but fabricate observations
   and actions. It should only be considered with explicit validity masks and a
   training policy that understands them; standard LeRobot horizon loading does
   not infer that validity automatically.

## Evidence limitations

- LeRobot's official sources establish the fixed-FPS semantics and known
  missed-loop behavior, but they cannot identify which local device caused each
  of the 15 rejected candidate frames.
- GitHub issue reports establish that timing/PTS failures occur for other users,
  but individual issue diagnoses are not equivalent to a maintainer-confirmed
  root cause. The source code is the authoritative evidence used above.
