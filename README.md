# SLaI Manipulation Intelligence

机器人操作智能项目，面向 Wujihand + UR5 的数据采集、模型训练与推理部署。

## 项目结构

```text
.
├── configs/       # 实验与设备配置
├── data/          # 数据目录（默认不提交原始数据）
├── docs/          # 项目文档
├── notebooks/     # 探索性分析
├── scripts/       # 训练、评估和数据处理脚本
├── src/slai_mi/   # Python 源码
└── tests/         # 自动化测试
```

## 快速开始

需要 Python 3.11+ 和 [uv](https://docs.astral.sh/uv/)。

```bash
uv sync
uv run pytest
uv run ruff check .
```

## Pipeline

1. 将采集数据放入 `data/raw/`。
2. 使用 `scripts/` 中的数据处理脚本生成 `data/processed/` 数据集。
3. 在 `configs/` 中固定实验配置并运行训练。
4. 使用相同配置进行评估和 UR5 推理部署。

## 数据与模型

原始数据、处理后数据、模型权重和日志默认被 `.gitignore` 忽略。请在 `docs/` 中记录数据版本、实验配置和硬件环境。

## 归属

SLaI Academy · EACV Center
