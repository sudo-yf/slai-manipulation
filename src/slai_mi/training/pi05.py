"""Lazy OpenPI PI0.5 LoRA configuration and launcher."""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path
from typing import Any

import numpy as np


def make_train_config(settings: dict[str, Any], *, smoke: bool = False) -> object:
    """Build the tested UR5/Wujihand PI0.5 config without importing OpenPI at startup."""
    try:
        from openpi import transforms
        from openpi.models import pi0_config
        from openpi.training import config, optimizer, weight_loaders
    except ImportError as exc:
        raise RuntimeError("PI0.5 training requires an OpenPI environment") from exc

    state_dim = int(settings.get("state_dim", 32))
    action_dim = int(settings.get("action_dim", 26))
    prompt = str(settings["task_prompt"])

    @dataclasses.dataclass(frozen=True)
    class Inputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            from slai_mi.policies.openpi import hand_position_to_delta, make_pi05_observation

            state = np.asarray(data["state"], dtype=np.float32)
            if state.shape != (state_dim,):
                raise ValueError(f"expected PI0.5 state[{state_dim}], got {state.shape}")
            result = make_pi05_observation(
                primary_rgb=data["primary_rgb"],
                secondary_rgb=data["secondary_rgb"],
                state=state,
                prompt=data.get("prompt", prompt),
            )
            if "actions" in data:
                result["actions"] = hand_position_to_delta(data["actions"], state)
            return result

    @dataclasses.dataclass(frozen=True)
    class Outputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            from slai_mi.policies.openpi import hand_delta_to_position

            return {"actions": hand_delta_to_position(data["actions"], data["state"], action_dim=action_dim)}

    @dataclasses.dataclass(frozen=True)
    class DataFactory(config.DataConfigFactory):
        def create(self, assets_dirs: Path, model_config: object) -> object:
            return dataclasses.replace(
                self.create_base_config(assets_dirs, model_config),
                data_transforms=transforms.Group(inputs=[Inputs()], outputs=[Outputs()]),
                model_transforms=config.ModelTransformFactory(default_prompt=prompt)(model_config),
                action_sequence_keys=("actions",),
            )

    model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=int(settings.get("action_horizon", 15)),
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    return config.TrainConfig(
        name="slai_pi05_real_lora",
        exp_name=str(settings["experiment"]),
        project_name="slai-pi05-real-vla",
        model=model,
        data=DataFactory(
            repo_id=str(settings["repo_id"]),
            base_config=config.DataConfig(prompt_from_task=True),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(str(settings["base_checkpoint"])),
        lr_schedule=optimizer.CosineDecaySchedule(
            warmup_steps=1000, peak_lr=5e-5, decay_steps=30_000, decay_lr=5e-6
        ),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        seed=int(settings.get("seed", 42)),
        batch_size=1 if smoke else int(settings.get("batch_size", 16)),
        num_train_steps=1 if smoke else int(settings.get("steps", 30_000)),
        save_interval=1 if smoke else int(settings.get("save_interval", 5000)),
        keep_period=1 if smoke else int(settings.get("save_interval", 5000)),
        overwrite=smoke,
        wandb_enabled=not smoke,
        # The transform classes are defined per configuration and are not
        # pickleable under OpenPI's spawn multiprocessing context. Keep the
        # loader single-process; this also makes norm-stat generation reliable
        # on workstations and containers.
        num_workers=0,
        assets_base_dir=str(settings["assets_dir"]),
        checkpoint_base_dir=str(settings["checkpoints_dir"]),
    )


def run_openpi(command: str, settings: dict[str, Any], *, smoke: bool, max_frames: int | None) -> None:
    root = Path(settings["openpi_root"]).expanduser().resolve()
    script_name = "compute_norm_stats" if command == "norm" else "train"
    path = root / "scripts" / f"{script_name}.py"
    if not path.is_file():
        raise FileNotFoundError(f"OpenPI script is missing: {path}")
    train_config = make_train_config(settings, smoke=smoke)
    import openpi.training.config as registry

    registry._CONFIGS_DICT[train_config.name] = train_config
    spec = importlib.util.spec_from_file_location(f"openpi_project_{script_name}", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load OpenPI script: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if command == "norm":
        module.main(train_config.name, max_frames=max_frames)
    else:
        module.main(train_config)
