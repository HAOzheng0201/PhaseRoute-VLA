#!/usr/bin/env bash

set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

REPO_ROOT="${REPO_ROOT:-${repo_default}}"
CHECKPOINT="${REPO_ROOT}/model/libero_exit"
OUTPUT_ROOT="${1:-${REPO_ROOT}/reports/m1_phase_collection_20260801}"
PYTHON_BIN="${PYTHON_BIN:-python}"

mkdir -p "${OUTPUT_ROOT}"

declare -a pids=()
for gpu_id in 0 1 2 3; do
    task_id="${gpu_id}"
    job_dir="${OUTPUT_ROOT}/gpu${gpu_id}_task${task_id}"
    mkdir -p "${job_dir}"

    env \
        CUDA_VISIBLE_DEVICES="${gpu_id}" \
        MUJOCO_EGL_DEVICE_ID="${gpu_id}" \
        DATA_DIR="${REPO_ROOT}" \
        HF_HOME="${HF_HOME:-${REPO_ROOT}/.cache/huggingface}" \
        VLA_CONFIG_YAML="libero_simulation.yaml" \
        MUJOCO_GL="egl" \
        PYOPENGL_PLATFORM="egl" \
        HF_HUB_OFFLINE="1" \
        TF_CPP_MIN_LOG_LEVEL="3" \
        PYTHONUNBUFFERED="1" \
        "${PYTHON_BIN}" "${REPO_ROOT}/scripts/dynamic_compute/collect_m1_task.py" \
            --checkpoint "${CHECKPOINT}" \
            --task-suite libero_spatial \
            --task-id "${task_id}" \
            --num-episodes 5 \
            --seed 20260801 \
            --fm-steps 10 \
            --output-dir "${job_dir}" \
            >"${job_dir}/console.log" 2>&1 &
    job_pid="$!"
    pids+=("${job_pid}")
    echo "started gpu=${gpu_id} task=${task_id} pid=${job_pid} log=${job_dir}/console.log"
done

failed=0
for index in "${!pids[@]}"; do
    if wait "${pids[${index}]}"; then
        echo "completed gpu=${index} pid=${pids[${index}]}"
    else
        echo "failed gpu=${index} pid=${pids[${index}]}" >&2
        failed=1
    fi
done

exit "${failed}"
