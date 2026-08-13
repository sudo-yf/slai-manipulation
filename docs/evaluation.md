# Evaluation

The evaluation package keeps offline measurement independent from robot control and model
serving. Reports are JSON, so experiments can archive and compare them without importing a
training framework.

## Action predictions

Create an NPZ file with matching `predicted` and `target` arrays and a one-dimensional `ranges`
array. The last dimension is the action dimension.

```bash
uv run slai-evaluate actions predictions.npz --output reports/actions.json
```

## Visual domain gap

Each NPZ input contains an `images` array with shape `N x H x W x 3`, uint8 RGB. Images should
already be decoded and resized exactly as the policy sees them. The report compares channel and
luminance histograms; it is a diagnostic, not a task-success metric.

```bash
uv run slai-evaluate domain-gap real.npz sim.npz --output reports/domain.json
```

## Camera landmarks

Camera input JSON contains `intrinsics` (`fx`, `fy`, `cx`, `cy`), `world_points` (`N x 3`),
`image_points` (`N x 2`), and optional positive `weights`. At least four correspondences are
required. This command requires OpenCV in the runtime environment.

```bash
uv run slai-evaluate camera landmarks.json --output reports/camera.json
```

## LeRobot v3 validation

Validation uses the project's canonical UR5 + Wujihand contract. PyArrow is imported only when
this command runs.

```bash
uv run slai-evaluate dataset data/lerobot/my_dataset
```
