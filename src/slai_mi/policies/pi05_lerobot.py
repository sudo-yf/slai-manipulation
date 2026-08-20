"""Offline LeRobot PI0.5 checkpoint inference backend."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.input_schema import load_input_schema, split_capture_vector


class PI05OfflineBackend:
    def __init__(self, *, config: dict[str, Any], checkpoint: Path, target: str) -> None:
        if target != "offline":
            raise ValueError("PI0.5 dataset inference backend is offline-only")
        self.config = config
        self.checkpoint = checkpoint

    def run(self) -> dict[str, object]:
        import torch
        from lerobot.configs import PreTrainedConfig
        from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
        from lerobot.policies.factory import make_policy, make_pre_post_processors

        dataset_config = self.config.get("dataset")
        if not isinstance(dataset_config, dict):
            raise TypeError("inference.dataset must be a mapping")
        root = Path(dataset_config["root"]).expanduser().resolve()
        repo_id = str(dataset_config["repo_id"])
        frame_index = int(dataset_config.get("frame_index", 0))
        metadata = LeRobotDatasetMetadata(repo_id, root=root)
        dataset = LeRobotDataset(repo_id, root=root, video_backend=dataset_config.get("video_backend"))
        if not 0 <= frame_index < len(dataset):
            raise ValueError(f"frame_index {frame_index} is outside dataset length {len(dataset)}")

        policy_config = PreTrainedConfig.from_pretrained(str(self.checkpoint))
        policy_config.pretrained_path = self.checkpoint
        policy_config.device = str(self.config.get("device", "cuda"))
        started = time.perf_counter()
        policy = make_policy(policy_config, ds_meta=metadata)
        preprocessor, postprocessor = make_pre_post_processors(
            policy_config,
            pretrained_path=str(self.checkpoint),
            dataset_stats=metadata.stats,
            dataset_meta=metadata,
        )
        load_s = time.perf_counter() - started

        item = dataset[frame_index]
        batch = {
            key: value.unsqueeze(0) if isinstance(value, torch.Tensor) else [value]
            for key, value in item.items()
        }
        processed = preprocessor(batch)
        policy.eval()
        torch.cuda.synchronize() if policy_config.device.startswith("cuda") else None
        started = time.perf_counter()
        with torch.inference_mode():
            action = policy.select_action(processed)
            action = postprocessor(action)
        torch.cuda.synchronize() if policy_config.device.startswith("cuda") else None
        inference_s = time.perf_counter() - started
        action_array = np.asarray(action.detach().cpu(), dtype=np.float32)

        physical_root = Path(dataset_config["physical_v21_root"]).expanduser().resolve()
        physical_info = json.loads((physical_root / "meta" / "info.json").read_text())
        physical_dim = int(physical_info["features"]["actions"]["shape"][0])
        physical_action = action_array[..., :physical_dim]
        if not np.isfinite(physical_action).all():
            raise RuntimeError("PI0.5 produced a non-finite action")
        schema = load_input_schema(self.config.get("input_schema"))
        components = split_capture_vector(schema, "action", physical_action[0])
        return {
            "checkpoint_type": "peft_lora",
            "dataset_frames": len(dataset),
            "frame_index": frame_index,
            "input_shapes": {
                key: list(value.shape)
                for key, value in batch.items()
                if isinstance(value, torch.Tensor)
                and (key == "observation.state" or key.startswith("observation.images."))
            },
            "model_action_shape": list(action_array.shape),
            "physical_action_shape": list(physical_action.shape),
            "physical_action": physical_action.tolist(),
            "real_policy_components": {
                channel: {attribute: values.tolist() for attribute, values in fields.items()}
                for channel, fields in components.items()
            },
            "load_s": load_s,
            "inference_s": inference_s,
        }


def factory(**context: Any) -> PI05OfflineBackend:
    return PI05OfflineBackend(**context)
