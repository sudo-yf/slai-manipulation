# LeRobot PI0-Fast

The active training backend is LeRobot's PyTorch PI0 policy. The base model is
the official Hugging Face release `lerobot/pi0fast-base`, which is a
`model.safetensors` checkpoint (about 11.7 GiB). It is intentionally separate
from the legacy OpenPI JAX/Orbax PI0.5 checkout.

Download the model into the ignored model directory:

```bash
cd /home/user/shiyi/workspace/active/slai-manipulation
.venv-lerobot-v3/bin/huggingface-cli download lerobot/pi0fast-base \
  --local-dir data/models/pi0fast-base
```

The converted dataset remains at
`data/training/pi05/block_into_box_v21`. LeRobot's PI0 policy uses 224x224
images, a padded 32-dimensional state, and a padded 32-dimensional action
vector; the project adapter removes the six action padding dimensions when
commands are sent to the UR5/Wujihand contract.

The checkpoint cannot be loaded by OpenPI's JAX `CheckpointWeightLoader`.
Use the LeRobot `lerobot-train` entrypoint with the PI0 policy configuration
after the download completes.

The current LeRobot environment is pinned to `transformers>=5.4,<5.6`.
Using a newer Transformers release changes the PaliGemma module names and
causes false-looking missing-key warnings while loading this checkpoint.

Start training with:

```bash
.venv-lerobot-v3/bin/lerobot-train \
  --config_path=configs/pi0fast_train.yaml
```

For a one-step wiring check, override `--steps=1 --batch_size=1` before a
full run. The first invocation downloads the tokenizer and model assets from
Hugging Face if they are not already cached.
