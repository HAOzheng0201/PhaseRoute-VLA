#!/usr/bin/env bash
set -euo pipefail
repo_default="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

output_root="${1:?usage: $0 OUTPUT_ROOT [REPO_ROOT]}"
repo_root="${2:-${repo_default}}"
checkpoint_sha="dcafd9ee8a3d3a4ced8840e59c90b0c4b20d41a7900adc9ff469c1a57e631b7f"
declare -a seeds=(20261804 20262804 20263804)

for gpu in 0 1 2 3; do
  expected_uuid="$(
    nvidia-smi -i "${gpu}" --query-gpu=uuid --format=csv,noheader
  )"
  actual_uuid="$(
    env \
      CUDA_VISIBLE_DEVICES="${gpu}" \
      PYTHONNOUSERSITE=1 \
      "${PYTHON_BIN:-python}" \
      -c 'import torch; print(torch.cuda.get_device_properties(0).uuid)'
  )"
  if [[ "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for requested physical GPU${gpu}: expected ${expected_uuid}, got ${actual_uuid}" >&2
    exit 3
  fi
  echo "verified physical_gpu=${gpu} uuid=${expected_uuid} visible_device_count=1"
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
    --task-id 5 \
    --episode-index 2 \
    --episode-index 14 \
    --episode-index 22 \
    --num-episodes 3 \
    --seed "${seed}" \
    --fm-steps 10 \
    --output-dir "${run_dir}" \
    >"${run_dir}/stdout.log" 2>&1 &
  launched_pid="$!"
  echo "started policy=${policy} seed=${seed} task=5 episodes=2,14,22 physical_gpu=${gpu} pid=${launched_pid}"
}

declare -a wave1_policies=(early_exit full_depth early_exit full_depth)
declare -a wave1_seeds=(20261804 20261804 20262804 20262804)
declare -a wave1_gpus=(0 2 1 3)
wave1_pids=()
for index in 0 1 2 3; do
  launch_run "${wave1_policies[$index]}" "${wave1_seeds[$index]}" "${wave1_gpus[$index]}"
  wave1_pids+=("${launched_pid}")
done
wave1_status=0
for index in 0 1 2 3; do
  if wait "${wave1_pids[$index]}"; then
    echo "complete wave=1 policy=${wave1_policies[$index]} seed=${wave1_seeds[$index]}"
  else
    echo "failed wave=1 policy=${wave1_policies[$index]} seed=${wave1_seeds[$index]}" >&2
    wave1_status=1
  fi
done
if (( wave1_status != 0 )); then
  exit "${wave1_status}"
fi

declare -a wave2_policies=(early_exit full_depth)
declare -a wave2_gpus=(0 2)
wave2_pids=()
for index in 0 1; do
  launch_run "${wave2_policies[$index]}" 20263804 "${wave2_gpus[$index]}"
  wave2_pids+=("${launched_pid}")
done
wave2_status=0
for index in 0 1; do
  if wait "${wave2_pids[$index]}"; then
    echo "complete wave=2 policy=${wave2_policies[$index]} seed=20263804"
  else
    echo "failed wave=2 policy=${wave2_policies[$index]} seed=20263804" >&2
    wave2_status=1
  fi
done
exit "${wave2_status}"
