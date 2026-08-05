#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT]}"
repo_root="${2:-${repo_default}}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"

declare -a gpus=(0 1 2 3)
declare -a seeds=(20264804 20264804 20265804 20265804)
declare -a shards=(shard0 shard1 shard0 shard1)
declare -a state_groups=(states27_28 states27_28 states29_31 states29_31)
declare -a expected_counts=(10 10 15 15)
declare -a gpu_uuids=()

for gpu in 0 1 2 3; do
  expected_uuid="$(
    nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader
  )"
  read -r visible_count actual_uuid < <(
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONNOUSERSITE=1 \
      "${PYTHON_BIN:-python}" \
      -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)'
  )
  if [[ "${visible_count}" != "1" || "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for physical GPU${gpu}: host=${expected_uuid} visible_count=${visible_count} visible_uuid=${actual_uuid}" >&2
    exit 3
  fi
  gpu_uuids+=("${expected_uuid}")
  echo "verified physical_gpu=${gpu} host_uuid=${expected_uuid} visible_uuid=${actual_uuid} visible_count=${visible_count}"
done

declare -a run_dirs=()
for worker in 0 1 2 3; do
  run_dir="${output_root}/seed${seeds[$worker]}_${state_groups[$worker]}/${shards[$worker]}"
  if [[ -e "${run_dir}/result.json" || -e "${run_dir}/episodes.jsonl" || -e "${run_dir}/eval_logs" ]]; then
    echo "Refusing to overwrite ${run_dir}" >&2
    exit 1
  fi
  run_dirs+=("${run_dir}")
done

declare -a pids=()
for worker in 0 1 2 3; do
  gpu="${gpus[$worker]}"
  shard="${shards[$worker]}"
  run_dir="${run_dirs[$worker]}"
  mkdir -p "${run_dir}"
  if [[ "${shard}" == "shard0" ]]; then
    tasks=(0 1 2 3 4)
  else
    tasks=(5 6 7 8 9)
  fi
  task_args=()
  for task_id in "${tasks[@]}"; do
    task_args+=(--task-id "${task_id}")
  done
  if [[ "${state_groups[$worker]}" == "states27_28" ]]; then
    episode_args=(--episode-index 27 --episode-index 28 --num-episodes 2)
  else
    episode_args=(--episode-index 29 --episode-index 30 --episode-index 31 --num-episodes 3)
  fi
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
    "${PYTHON_BIN:-python}" \
    "${repo_root}/scripts/dynamic_compute/collect_m418_persistent_tasks.py" \
    --policy full_depth \
    --checkpoint "${repo_root}/model/libero_exit" \
    --checkpoint-sha256 "${checkpoint_sha}" \
    --task-suite libero_spatial \
    "${task_args[@]}" \
    "${episode_args[@]}" \
    --seed "${seeds[$worker]}" \
    --fm-steps 10 \
    --expected-gpu-uuid "${gpu_uuids[$worker]}" \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  pids+=("$!")
  echo "started worker=${worker} policy=full_depth tasks=${tasks[*]} states=${state_groups[$worker]} seed=${seeds[$worker]} physical_gpu=${gpu} pid=${pids[-1]}"
done

while true; do
  alive=0
  status_line="progress"
  for worker in 0 1 2 3; do
    if kill -0 "${pids[$worker]}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
    episodes=0
    if [[ -f "${run_dirs[$worker]}/episodes.jsonl" ]]; then
      episodes="$(wc -l < "${run_dirs[$worker]}/episodes.jsonl")"
    fi
    status_line+=" w${worker}=${episodes}/${expected_counts[$worker]}"
  done
  echo "${status_line} alive=${alive}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 30
done

status=0
for worker in 0 1 2 3; do
  if wait "${pids[$worker]}"; then
    echo "complete worker=${worker} dir=${run_dirs[$worker]}"
  else
    echo "failed worker=${worker}; inspect ${run_dirs[$worker]}/stdout.log" >&2
    status=1
  fi
done
exit "${status}"
