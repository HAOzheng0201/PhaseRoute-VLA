#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT]}"
repo_root="${2:-${repo_default}}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
declare -a seeds=(20266804 20267804 20268804)
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

for seed in "${seeds[@]}"; do
  for policy in early_exit full_depth; do
    run_dir="${output_root}/seed${seed}/${policy}"
    if [[ -e "${run_dir}/result.json" || -e "${run_dir}/episodes.jsonl" || -e "${run_dir}/eval_logs" ]]; then
      echo "Refusing to overwrite ${run_dir}" >&2
      exit 1
    fi
  done
done

launch_run() {
  local policy="$1"
  local seed="$2"
  local gpu="$3"
  local run_dir="${output_root}/seed${seed}/${policy}"
  mkdir -p "${run_dir}"
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
    --policy "${policy}" \
    --checkpoint "${repo_root}/model/libero_exit" \
    --checkpoint-sha256 "${checkpoint_sha}" \
    --task-suite libero_spatial \
    --task-id 4 \
    --episode-index 29 \
    --num-episodes 1 \
    --seed "${seed}" \
    --fm-steps 10 \
    --expected-gpu-uuid "${gpu_uuids[$gpu]}" \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  launched_pid="$!"
  launched_dir="${run_dir}"
  echo "started policy=${policy} seed=${seed} task=4 state=29 physical_gpu=${gpu} pid=${launched_pid}"
}

wait_wave() {
  local wave="$1"
  shift
  local pids=("$@")
  while true; do
    local alive=0
    local status_line="progress wave=${wave}"
    for index in "${!pids[@]}"; do
      if kill -0 "${pids[$index]}" 2>/dev/null; then
        alive=$((alive + 1))
      fi
      local episodes=0
      if [[ -f "${wave_dirs[$index]}/episodes.jsonl" ]]; then
        episodes="$(wc -l < "${wave_dirs[$index]}/episodes.jsonl")"
      fi
      status_line+=" w${index}=${episodes}/1"
    done
    echo "${status_line} alive=${alive}"
    if [[ "${alive}" == "0" ]]; then
      break
    fi
    sleep 30
  done
  local status=0
  for index in "${!pids[@]}"; do
    if wait "${pids[$index]}"; then
      echo "complete wave=${wave} dir=${wave_dirs[$index]}"
    else
      echo "failed wave=${wave} dir=${wave_dirs[$index]}" >&2
      status=1
    fi
  done
  return "${status}"
}

declare -a wave1_policies=(early_exit early_exit full_depth full_depth)
declare -a wave1_seeds=(20266804 20267804 20266804 20267804)
declare -a wave1_gpus=(0 1 2 3)
declare -a wave1_pids=()
declare -a wave_dirs=()
for index in 0 1 2 3; do
  launch_run "${wave1_policies[$index]}" "${wave1_seeds[$index]}" "${wave1_gpus[$index]}"
  wave1_pids+=("${launched_pid}")
  wave_dirs+=("${launched_dir}")
done
wait_wave 1 "${wave1_pids[@]}"

declare -a wave2_policies=(early_exit full_depth)
declare -a wave2_gpus=(0 2)
declare -a wave2_pids=()
wave_dirs=()
for index in 0 1; do
  launch_run "${wave2_policies[$index]}" 20268804 "${wave2_gpus[$index]}"
  wave2_pids+=("${launched_pid}")
  wave_dirs+=("${launched_dir}")
done
wait_wave 2 "${wave2_pids[@]}"
