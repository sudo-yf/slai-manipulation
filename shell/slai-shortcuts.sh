# SLaI commands. Every runner works from any current directory.
export SLAI_ROOT=/home/user/shiyi/slai-manipulation

_srun() (
    cd "$SLAI_ROOT" || exit
    uv run "$@"
)

_scollect() (
    cd "$SLAI_ROOT" || exit
    collect_python="$SLAI_ROOT/.venv-lerobot-v3/bin/python"
    if [ ! -x "$collect_python" ]; then
        printf '采集环境不存在: %s\n' "$collect_python" >&2
        exit 1
    fi
    if ! "$collect_python" -c 'import lerobot, slai_mi' >/dev/null 2>&1; then
        printf '采集环境缺少 lerobot/slai_mi: %s\n' "$collect_python" >&2
        exit 1
    fi
    exec "$collect_python" -m slai_mi.apps.collect_real "$@"
)

s() { cd "$SLAI_ROOT" || return; }

sr()   { _srun slai-teleop-real "$@"; }
srx()  { _srun slai-teleop-real "$@" --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION; }
sim()  { _srun slai-teleop-sim "$@"; }
simx() { _srun slai-teleop-sim "$@" --run; }

sc()   { _scollect "$@"; }
scw()  {
    _scollect \
        --strategy ur5e_wrist_8dof_collection \
        --task configs/tasks/block_into_box.yaml \
        "$@"
}
scx()  {
    local arg episode_mode=
    for arg in "$@"; do
        case "$arg" in
            --episodes|--episodes=*|--continuous) episode_mode=1 ;;
        esac
    done
    if [ -n "$episode_mode" ]; then
        _scollect "$@" --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
    else
        _scollect "$@" --continuous --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
    fi
}
scc()  { _scollect "$@" --continuous --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION; }
scwx() (
    restore_ui=
    if systemctl --user is-active --quiet slai-collection-ui.service; then
        systemctl --user stop slai-collection-ui.service || exit
        restore_ui=1
    fi
    trap 'if [ -n "$restore_ui" ]; then systemctl --user start slai-collection-ui.service; fi' EXIT
    episode_args=--continuous
    for arg in "$@"; do
        case "$arg" in
            --episodes|--episodes=*|--continuous) episode_args= ;;
        esac
    done
    _scollect \
        --strategy ur5e_wrist_8dof_collection \
        --task configs/tasks/block_into_box.yaml \
        ${episode_args:+$episode_args} \
        "$@" \
        --execute-real \
        --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
)
scs()  { _srun slai-collect-sim "$@"; }
scsx() { _srun slai-collect-sim "$@" --run; }

su()   { _srun slai-collection-ui "$@"; }
sul()  { _srun slai-collection-ui "$@" --live; }
st()   { _srun slai-train "$@"; }
si()   { _srun slai-mi-infer "$@"; }
se()   { _srun slai-evaluate "$@"; }
sp()   { _srun slai-pi05 "$@"; }
sd()   { _srun slai-deploy-real "$@"; }
sdx()  { _srun slai-deploy-real "$@" --execute-real --confirm I_UNDERSTAND_REAL_ROBOT_MOTION; }

sph()  { _srun --extra pose-hub slai-pose-hub "$@"; }
sphx() { _srun --extra pose-hub slai-pose-hub "$@" --run; }
spb()  { _srun --extra pose-hub slai-pose-hub-bridge "$@"; }
spbx() { _srun --extra pose-hub slai-pose-hub-bridge "$@" --run; }

sv()   { _srun pytest "$@"; }
sl()   { _srun ruff check . "$@"; }
rec()  (
    cd "$SLAI_ROOT" || exit
    ./.venv/bin/slai-record-pose "$@" \
        --execute-real \
        --confirm I_UNDERSTAND_REAL_ROBOT_MOTION
)
wuji-check() (
    cd "$SLAI_ROOT" || exit
    ./.venv/bin/python -m slai_mi.apps.check_wuji_retarget "$@" --run
)
wrist-park() (
    cd "$SLAI_ROOT" || exit
    uv run slai-park-wrist "$@"
)

# Backwards-compatible shortcut from the previous setup.
sm() {
    case "${1:-}" in
        collect) shift; scc "$@" ;;
        *) printf '用法: sm collect\n' >&2; return 2 ;;
    esac
}
