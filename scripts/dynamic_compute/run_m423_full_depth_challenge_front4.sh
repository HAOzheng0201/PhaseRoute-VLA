#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT]}"
repo_root="${2:-${repo_default}}"
python_bin="${PYTHON_BIN:-python}"
checkpoint_sha=dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f

if [[ -e "${output_root}" ]]; then
  echo "Refusing to reuse existing output root: ${output_root}" >&2
  exit 1
fi

declare -a gpus=(0 1 2 3)
declare -a gpu_uuids=()
declare -a orders=(
  "early_exit rp_pep full_depth"
  "rp_pep full_depth early_exit"
  "full_depth early_exit rp_pep"
  "full_depth rp_pep early_exit"
)

for gpu in "${gpus[@]}"; do
  expected_uuid="$(
    nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader
  )"
  read -r visible_count actual_uuid < <(
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONNOUSERSITE=1 \
      "${python_bin}" \
      -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)'
  )
  if [[ "${visible_count}" != "1" || "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for physical GPU${gpu}: host=${expected_uuid} visible_count=${visible_count} visible_uuid=${actual_uuid}" >&2
    exit 3
  fi
  gpu_uuids+=("${expected_uuid}")
  echo "verified physical_gpu=${gpu} host_uuid=${expected_uuid} visible_uuid=${actual_uuid} visible_count=${visible_count}"
done

cache_args=(
  --cache-dir "${repo_root}/reports/m48_teacher_cache_v3_spatial_task0_1ep_20260802_v1/teacher_calls"
  --cache-dir "${repo_root}/reports/m48_teacher_cache_v3_spatial_task1_1ep_20260802_v1/teacher_calls"
  --cache-dir "${repo_root}/reports/m48_teacher_cache_v3_spatial_task2_1ep_20260802_v1/teacher_calls"
  --cache-dir "${repo_root}/reports/m48_teacher_cache_v3_spatial_task3_1ep_20260802_v1/teacher_calls"
  --cache-dir "${repo_root}/reports/m416_teacher_cache_v3_spatial_tasks4_9_1ep_20260803_v1/task4/teacher_calls"
  --cache-dir "${repo_root}/reports/m416_teacher_cache_v3_spatial_tasks4_9_1ep_20260803_v1/task5/teacher_calls"
  --cache-dir "${repo_root}/reports/m416_teacher_cache_v3_spatial_tasks4_9_1ep_20260803_v1/task6/teacher_calls"
  --cache-dir "${repo_root}/reports/m416_teacher_cache_v3_spatial_tasks4_9_1ep_20260803_v1/task7/teacher_calls"
  --cache-dir "${repo_root}/reports/m416_teacher_cache_v3_spatial_tasks4_9_1ep_20260803_v1/task8/teacher_calls"
  --cache-dir "${repo_root}/reports/m416_teacher_cache_v3_spatial_tasks4_9_1ep_20260803_v1/task9/teacher_calls"
)

mkdir -p "${output_root}"
declare -a worker_pids=()

for gpu in "${gpus[@]}"; do
  (
    position=0
    for policy in ${orders[$gpu]}; do
      position=$((position + 1))
      run_dir="${output_root}/gpu${gpu}/${position}_${policy}"
      mkdir -p "${run_dir}"
      echo "starting physical_gpu=${gpu} position=${position} policy=${policy} dir=${run_dir}"
      env \
        CUDA_VISIBLE_DEVICES="${gpu}" \
        DATA_DIR="${repo_root}" \
        HF_HOME="${HF_HOME:-${repo_root}/.cache/huggingface}" \
        HF_HUB_OFFLINE=1 \
        TRANSFORMERS_OFFLINE=1 \
        VLA_CONFIG_YAML=libero_simulation.yaml \
        TF_CPP_MIN_LOG_LEVEL=3 \
        MUJOCO_GL=egl \
        PYOPENGL_PLATFORM=egl \
        CUBLAS_WORKSPACE_CONFIG=:4096:8 \
        PYTHONNOUSERSITE=1 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        "${python_bin}" \
        "${repo_root}/scripts/dynamic_compute/profile_m423_fixed_observations.py" \
        --policy "${policy}" \
        --checkpoint "${repo_root}/model/libero_exit" \
        --checkpoint-sha256 "${checkpoint_sha}" \
        "${cache_args[@]}" \
        --output "${run_dir}/result.json" \
        --expected-gpu-uuid "${gpu_uuids[$gpu]}" \
        --physical-gpu-index "${gpu}" \
        --order-position "${position}" \
        --fm-steps 10 \
        --seed 20260823 \
        --exit-layer 11 \
        --exit-layer 13 \
        --exit-layer 27 \
        --records-per-exit 4 \
        --repeats 2 \
        --warmup-calls 1 \
        >"${run_dir}/stdout.log" 2>&1
      echo "completed physical_gpu=${gpu} position=${position} policy=${policy}"
    done
  ) &
  worker_pids+=("$!")
  echo "started worker physical_gpu=${gpu} pid=${worker_pids[-1]} order=${orders[$gpu]}"
done

while true; do
  alive=0
  for pid in "${worker_pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  completed="$(find "${output_root}" -type f -name result.json | wc -l)"
  echo "progress completed_sessions=${completed}/12 alive_workers=${alive}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 30
done

status=0
for gpu in "${gpus[@]}"; do
  if wait "${worker_pids[$gpu]}"; then
    echo "worker complete physical_gpu=${gpu}"
  else
    echo "worker failed physical_gpu=${gpu}; inspect ${output_root}/gpu${gpu}/*/stdout.log" >&2
    status=1
  fi
done
exit "${status}"
