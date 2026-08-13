# Block-into-box sim-to-real runbook

Updated: 2026-07-27

## Current decision

- Default policy: PI0.5 LoRA checkpoint `160000` from
  `block_into_box_36ep_effbs8_4090_20260725_094031`.
- The server auto-detects dense versus LoRA Orbax parameters. A mismatch is a startup error.
- The robot client requires server metadata `model_variant=lora` and
  `checkpoint_step=160000`; a base or stale checkpoint fails before motion is armed.
- No physical motion was used for this comparison. The next gate is dry-run inference on the
  live cameras and robot state.

The previous server forced a dense PI0.5 model for every checkpoint. OpenPI then intersected the
parameter trees and silently discarded `lora_a` / `lora_b`, so changing only the checkpoint path
still produced base behavior. The current loader preserves the LoRA model structure.

## Held-out policy evidence

Dataset: `Datasets/record/block_into_box/0726-2343`, five trajectories captured after training.
Targets are aligned from the 30 Hz source to the 15 Hz, 15-action policy horizon. Errors are
normalized by the training action `q01..q99` range.

| Checkpoint | Samples/repeats | Normalized MAE | Arm | Hand |
| --- | ---: | ---: | ---: | ---: |
| `pi05_base` | 5 / 1 | 0.3227 | 0.3770 | 0.3064 |
| LoRA `80000` | 5 / 1 | 0.0290 | 0.0500 | 0.0227 |
| LoRA `120000` | 15 / 3 | 0.0350 | **0.0339** | 0.0353 |
| LoRA `160000` | 15 / 3 | **0.0323** | 0.0374 | **0.0308** |

The 160k checkpoint wins overall and on hand actions; 120k remains the conservative arm-error
fallback. Offline imitation error does not prove task success, so do not skip live dry-run or the
existing motion confirmation gate.

Re-run one checkpoint comparison while its server is active:

```bash
/home/user/shiyi/openpi/.venv/bin/python tools/evaluate_pi05_heldout.py \
  --dataset Datasets/record/block_into_box/0726-2343 \
  --norm-stats data/openpi_assets/pi05_real_vla_lora/shiyi/block-into-box-pi05-v21-36ep/norm_stats.json \
  --samples 15 \
  --output outputs/simreal_eval/checkpoint_heldout.json
```

## Visual-domain evidence

Both sides are converted with OpenPI's `224x224 resize_with_pad` before measurement. Comparing the
training videos with the 2026-07-26 real capture gives:

| Camera | RGB histogram JSD | Luminance JSD | Mean luminance delta | Edge-density delta |
| --- | ---: | ---: | ---: | ---: |
| Primary | 0.1095 | 0.1359 | 0.1016 | 0.0059 |
| Secondary | 0.0630 | 0.1439 | 0.1050 | 0.0025 |

The contact sheets show stable workspace composition and object placement. The dominant shift is
brighter exposure and stronger orange/green color, not a large geometry or camera-view change.
For the next simulator pass, keep the measured intrinsics and framing fixed, then randomize light
intensity, exposure, white balance, table roughness, and backdrop shade around the real reference.
OpenPI already applies crop/rotation and color jitter during training; simulator randomization
should cover residual rendering differences rather than replace camera calibration.

Generate the same report for an Isaac render by replacing `--candidate` with its video or image
directory:

```bash
/home/user/shiyi/openpi/.venv/bin/python tools/report_visual_domain_gap.py \
  --reference data/lerobot/shiyi/block-into-box-pi05-v21-36ep/videos/chunk-000/primary_rgb \
  --candidate /path/to/isaac/primary \
  --max-frames 64 \
  --output outputs/simreal_eval/primary_train_vs_sim.json \
  --contact-sheet outputs/simreal_eval/primary_train_vs_sim.png
```

## Next hardware gate

1. Start the configured server and verify metadata reports LoRA step 160000.
2. Run `python inference/main.py --config inference/config.yaml` without hardware flags.
3. Review action magnitudes, signs, camera freshness, and repeated replans from several object
   starts.
4. Compare the same dry-run with checkpoint 120000 if arm direction or approach stability is worse.
5. Only after operator review, use the existing double-arm command and conservative safety limits.

The Record3D probe and the 2DGS-SLAM checkout are present, but a five-frame probe is not sufficient
for a scene reconstruction. Capture a slow 300-frame RGB-D orbit before treating 2DGS as a source of
simulator geometry.
