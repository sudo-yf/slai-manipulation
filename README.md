# SLaI Manipulation Intelligence

机器人操作智能项目，面向 Wujihand + UR5 的数据采集、模型训练与推理。

## 项目结构

```text
.
├── README.md
├── pyproject.toml
├── uv.lock
├── assets/
│   └── robots/
│       └── ur5_wrist_wujihand/    # 28-DoF URDF and meshes
├── configs/
│   ├── hardware.yaml              # Physical-device setup
│   ├── dataset.yaml               # LeRobot v3 contract
│   ├── tasks/                     # Task definitions
│   └── poses/                     # Named robot poses
├── data/
│   ├── raw/               # 原始采集数据
│   ├── training/pi05/     # PI0.5 LeRobot v2.1 数据 (not committed)
│   └── normalization/pi05/ # PI0.5 norm stats (not committed)
├── src/slai_mi/
│   ├── apps/              # Real/sim teleoperation and collection CLIs
│   ├── devices/           # SpaceMouse, cameras, UR5, Wujihand, iPhone, wrist
│   ├── collection/        # Synchronization, episodes, telemetry, recording
│   ├── datasets/          # LeRobot v3 schema, writer, merge, validation
│   ├── simulation/        # Isaac Lab scene and deterministic episode runtime
│   ├── retargeting/       # Hand geometry, calibration, MANO process boundary
│   ├── policies/          # Policy adapters and action chunk execution
│   ├── training/          # Dataset filtering and training runtime
│   ├── evaluation/        # Policy and sim-real metrics
│   ├── observability/     # Session manifests and incident classification
│   └── ui/                # Collection dashboard and public Pose Hub
├── clients/ios/           # iPhone pose-streaming client
├── third_party/           # Pinned Wuji retargeting subset and license
└── tests/
```

## 快速开始

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Pipeline

```text
teleop_real.py / teleop_sim.py
              ↓
collect_real.py / collect_sim.py -> data/raw -> data/processed
                                              ↓
                                         train.py -> inference.py
```

所有命令默认只输出 dry-run 计划，不控制真实机器人：

```bash
uv run slai-teleop-real
uv run slai-teleop-sim
uv run slai-collect-real
uv run slai-collect-sim
uv run slai-train
uv run slai-infer
uv run slai-pi05 train
```

启动只读采集前端：

```bash
uv run slai-collection-ui --host 127.0.0.1 --port 8080
```

iPhone Pose Hub and the 4090 handoff bridge use package entry points and default
to dry-run plans:

```bash
uv run --extra pose-hub slai-pose-hub
uv run --extra pose-hub slai-pose-hub-bridge --session SESSION_ID
```

Production services pass `--run`; credentials remain in machine-local systemd
environment files. The local robot contract is `5005` for pose output and
`5006` for measured joint-state input. See [`docs/iphone_pose.md`](docs/iphone_pose.md).

真机执行需要完整硬件配置、生产 adapter、`--execute-real` 和命令提示的确认短语。
仿真执行需要在 `robot_teaching_isaaclab` 环境中使用 `--run`。

PI0.5 的 LeRobot v3→v2.1 转换、norm stats 和 LoRA 训练流程见
[`docs/pi05.md`](docs/pi05.md)。命令默认同样只输出计划，不启动训练。

## Simulation asset

The 28-DoF UR5 + wrist + Wujihand model is located at:

```text
assets/robots/ur5_wrist_wujihand/ur5_wrist_wuji_right.urdf
```

## Configuration safety

Physical device addresses and robot poses are intentionally disabled by default.
Set `enabled: true` only after filling and validating the corresponding values in
`configs/hardware.yaml` and `configs/poses/`.

Real collection uses the 26-DoF `real_v1` state schema. Isaac articulation uses
the 28-DoF `simulation_v1` schema. Cross-schema padding or truncation is disabled.

## 数据与模型

原始数据、处理后数据、模型权重和日志默认被 `.gitignore` 忽略。

迁移覆盖范围和剩余发布门槛见 `docs/migration-status.md`。

## 归属

SLaI Academy · EACV Center
