#!/usr/bin/env bash
set -euo pipefail

readonly LOCAL_PROJECT="${LOCAL_PROJECT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
readonly H100_HOST="${H100_HOST:-root@slai-h100.tail57cdd3.ts.net}"
readonly REMOTE_UID="${REMOTE_UID:-10100}"
readonly REMOTE_GID="${REMOTE_GID:-10100}"
readonly REMOTE_BASE="${REMOTE_BASE:-/mnt/afs/250010074/yifan/slai-manipulation}"
readonly REMOTE_PROJECT="${REMOTE_PROJECT:-/mnt/afs/250010074/yifan/slai-manipulation-h100-wrist8d}"
readonly REMOTE_HOME="${REMOTE_HOME:-/mnt/afs/250010074/yifan/.h100-home-wrist8d}"
readonly DATASET_NAME="${DATASET_NAME:-block_into_box-wrist8d-20260823T213054}"
readonly LOCAL_DATASET="$LOCAL_PROJECT/data/lerobot/$DATASET_NAME"
readonly REMOTE_DATASET="$REMOTE_PROJECT/data/lerobot/$DATASET_NAME"
readonly CONFIG_REL="${CONFIG_REL:-configs/pi05_wrist_8dof_h100_jax.yaml}"
readonly NUM_GPUS="${NUM_GPUS:-1}"
readonly STEPS="${STEPS:-10}"
readonly BATCH_SIZE="${BATCH_SIZE:-8}"
readonly RUN_TAG="${RUN_TAG:-$(date -u +%Y%m%dT%H%M%SZ)}"
readonly LOG_REL="logs/h100-wrist8d-$RUN_TAG.log"
readonly RSYNC_SSH="ssh -T -o Compression=no -o IPQoS=throughput -o ServerAliveInterval=30"

remote_user() {
    ssh "$H100_HOST" setpriv \
        "--reuid=$REMOTE_UID" "--regid=$REMOTE_GID" --clear-groups \
        env "HOME=$REMOTE_HOME" bash -s
}

[[ -f "$LOCAL_DATASET/meta/info.json" ]] || {
    echo "Missing local dataset: $LOCAL_DATASET" >&2
    exit 1
}

git -C "$LOCAL_PROJECT" diff --quiet -- "$CONFIG_REL" deploy/h100_train_jax.sh deploy/h100_wrist8d_pipeline.sh
git -C "$LOCAL_PROJECT" diff --cached --quiet -- "$CONFIG_REL" deploy/h100_train_jax.sh deploy/h100_wrist8d_pipeline.sh
[[ -z "$(git -C "$LOCAL_PROJECT" ls-files --others --exclude-standard -- "$CONFIG_REL" deploy/h100_train_jax.sh deploy/h100_wrist8d_pipeline.sh)" ]] || {
    echo "Deployment files must be committed before H100 deployment" >&2
    exit 1
}

readonly BRANCH="$(git -C "$LOCAL_PROJECT" branch --show-current)"
readonly LOCAL_HEAD="$(git -C "$LOCAL_PROJECT" rev-parse HEAD)"
git -C "$LOCAL_PROJECT" push origin "$BRANCH"

ssh "$H100_HOST" mkdir -p "$REMOTE_HOME"
ssh "$H100_HOST" chown "$REMOTE_UID:$REMOTE_GID" "$REMOTE_HOME"

remote_user <<EOF
set -euo pipefail
git -C "$REMOTE_BASE" fetch origin "$BRANCH"
if [[ ! -e "$REMOTE_PROJECT/.git" ]]; then
    git -C "$REMOTE_BASE" worktree add --detach "$REMOTE_PROJECT" "origin/$BRANCH"
else
    git -C "$REMOTE_PROJECT" fetch origin "$BRANCH"
    git -C "$REMOTE_PROJECT" checkout --detach "origin/$BRANCH"
fi
[[ "\$(git -C "$REMOTE_PROJECT" rev-parse HEAD)" == "$LOCAL_HEAD" ]]
for path in third_party .venv-openpi .venv-lerobot-v3; do
    if [[ ! -e "$REMOTE_PROJECT/\$path" ]]; then
        ln -s "$REMOTE_BASE/\$path" "$REMOTE_PROJECT/\$path"
    fi
done
mkdir -p "$REMOTE_PROJECT/data/lerobot" "$REMOTE_PROJECT/logs"
mkdir -p "$REMOTE_PROJECT/.h100-home"
EOF

rsync_options=(
    --archive
    --whole-file
    --partial
    --info=progress2
    -e "$RSYNC_SSH"
    --rsync-path="setpriv --reuid=$REMOTE_UID --regid=$REMOTE_GID --clear-groups rsync"
)
rsync "${rsync_options[@]}" --exclude=/videos/ \
    "$LOCAL_DATASET/" "$H100_HOST:$REMOTE_DATASET/"

remote_user <<EOF
set -euo pipefail
mkdir -p "$REMOTE_DATASET/videos"
EOF

transfer_pids=()
for camera_dir in "$LOCAL_DATASET"/videos/*; do
    [[ -d "$camera_dir" ]] || continue
    camera_name="$(basename "$camera_dir")"
    rsync "${rsync_options[@]}" \
        "$camera_dir/" "$H100_HOST:$REMOTE_DATASET/videos/$camera_name/" &
    transfer_pids+=("$!")
done
transfer_failed=0
for transfer_pid in "${transfer_pids[@]}"; do
    wait "$transfer_pid" || transfer_failed=1
done
[[ "$transfer_failed" == 0 ]] || {
    echo "One or more parallel video transfers failed" >&2
    exit 1
}

if [[ -n "$(rsync --archive --whole-file --checksum --dry-run --itemize-changes -e "$RSYNC_SSH" \
    --rsync-path="setpriv --reuid=$REMOTE_UID --regid=$REMOTE_GID --clear-groups rsync" \
    "$LOCAL_DATASET/" "$H100_HOST:$REMOTE_DATASET/")" ]]; then
    echo "Remote dataset verification failed: $REMOTE_DATASET" >&2
    exit 1
fi

remote_user <<EOF
set -euo pipefail
cd "$REMOTE_PROJECT"
[[ -x .venv-openpi/bin/python ]]
[[ -x .venv-lerobot-v3/bin/python ]]
[[ -f "$CONFIG_REL" ]]
[[ -f "data/lerobot/$DATASET_NAME/meta/info.json" ]]
nohup bash -lc '
    set -euo pipefail
    cd "$REMOTE_PROJECT"
    export HOME="$REMOTE_PROJECT/.h100-home"
    export OPENPI_DATA_HOME="$REMOTE_BASE/.cache/openpi"
    export HF_HOME="$REMOTE_PROJECT/.cache/huggingface"
    export XDG_CACHE_HOME="$REMOTE_PROJECT/.cache"
    export JAX_COMPILATION_CACHE_DIR="$REMOTE_PROJECT/.cache/jax"
    export PYTHONPATH="$REMOTE_PROJECT/src:$REMOTE_PROJECT/third_party/openpi/src:$REMOTE_PROJECT/third_party/openpi/packages/openpi-client/src"
    .venv-openpi/bin/python -m slai_mi.apps.pi05 convert --config "$CONFIG_REL" --execute
    .venv-openpi/bin/python -m slai_mi.apps.pi05 norm --config "$CONFIG_REL" --execute
    PROJECT="$REMOTE_PROJECT" TRAIN_HOME="$REMOTE_BASE/.h100-home" CONFIG="$REMOTE_PROJECT/$CONFIG_REL" NUM_GPUS="$NUM_GPUS" STEPS="$STEPS" BATCH_SIZE="$BATCH_SIZE" bash deploy/h100_train_jax.sh
' </dev/null >"$LOG_REL" 2>&1 &
pipeline_pid=\$!
echo "pipeline_pid=\$pipeline_pid"
echo "log=$REMOTE_PROJECT/$LOG_REL"
EOF
