#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin=/home/haozheng/.conda/envs/a1/bin/python
output_root="${repo_root}/reports/v3_d9d_same_noise_replay"
log_root="${repo_root}/reports/v3_d9d_same_noise_logs"
checkpoint="${repo_root}/model/v3_d2/libero_exit"
attestation=/data3/haozheng/A1/source/reports/model_libero_exit_sha256_attestation_20260809_v1.json
data_dir=/data3/haozheng/A1/source
declare -a gpus=(0 1 2 3)

if [[ -n "$(git -C "${repo_root}" status --porcelain=v1)" ]]; then
  echo "D9D replay requires a clean frozen-runner worktree" >&2
  exit 2
fi
"${python_bin}" -c 'import sys; from a1.vla.dynamic_compute.v3.same_noise_replay import validate_d9c_collection,validate_d9d_runner_readiness; validate_d9c_collection(sys.argv[1]); validate_d9d_runner_readiness(sys.argv[1]); print("D9D prerequisites: PASS")' "${repo_root}"
if [[ -e "${output_root}" || -e "${log_root}" ]]; then
  echo "D9D refuses to reuse replay output or log roots" >&2
  exit 3
fi

nvidia_field() {
  local gpu="$1"
  local query="$2"
  local attempt value
  for attempt in 1 2 3 4 5; do
    if value="$(nvidia-smi -i "${gpu}" --query-gpu="${query}" --format=csv,noheader,nounits 2>/dev/null)"; then
      printf '%s\n' "${value}"
      return 0
    fi
    sleep 2
  done
  echo "nvidia-smi failed for GPU${gpu} field ${query}" >&2
  return 1
}

declare -a pids=()
declare -a gpu_uuids=()
for shard in 0 1 2 3; do
  gpu="${gpus[$shard]}"
  expected_uuid="$(nvidia_field "${gpu}" uuid)"
  memory_used="$(nvidia_field "${gpu}" memory.used)"
  utilization="$(nvidia_field "${gpu}" utilization.gpu)"
  if (( memory_used > 100 || utilization > 5 )); then
    echo "Refusing busy physical GPU${gpu}: memory=${memory_used}MiB utilization=${utilization}%" >&2
    exit 4
  fi
  read -r visible_count actual_uuid < <(
    env CUDA_VISIBLE_DEVICES="${gpu}" PYTHONNOUSERSITE=1 \
      "${python_bin}" -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)'
  )
  if [[ "${visible_count}" != "1" || "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for physical GPU${gpu}" >&2
    exit 5
  fi
  gpu_uuids+=("${expected_uuid}")
done

mkdir -p "${output_root}" "${log_root}"
git -C "${repo_root}" rev-parse HEAD >"${log_root}/source_git_commit.txt"
nvidia-smi -i 0,1,2,3 --query-gpu=index,uuid,name,memory.used,utilization.gpu --format=csv,noheader \
  >"${log_root}/gpu_preflight.csv"
for shard in 0 1 2 3; do
  gpu="${gpus[$shard]}"
  console_log="${log_root}/shard${shard}.log"
  env \
    CUDA_VISIBLE_DEVICES="${gpu}" \
    DATA_DIR="${data_dir}" \
    HF_HOME=/data3/haozheng/A1/hf_cache \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    VLA_CONFIG_YAML=libero_simulation.yaml \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    PYTHONPATH="${data_dir}/robot_experiments/libero/LIBERO:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" \
    "${repo_root}/scripts/dynamic_compute/v3/replay_v3_d9d_shard.py" \
    --shard-index "${shard}" \
    --physical-gpu-index "${gpu}" \
    --expected-gpu-uuid "${gpu_uuids[$shard]}" \
    --checkpoint "${checkpoint}" \
    --model-attestation "${attestation}" \
    --output-dir "${output_root}/shard${shard}" \
    >"${console_log}" 2>&1 &
  pids+=("$!")
  echo "started D9D shard=${shard} physical_gpu=${gpu} pid=${pids[-1]} log=${console_log}"
done

while true; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  echo "D9D same-noise replay alive_workers=${alive}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 30
done

failed=0
for shard in 0 1 2 3; do
  if wait "${pids[$shard]}"; then
    echo "completed D9D shard=${shard} physical_gpu=${gpus[$shard]}"
  else
    echo "failed D9D shard=${shard}; inspect ${log_root}/shard${shard}.log" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 6
fi
echo "PASS_V3_D9D_FRONT4_SAME_NOISE_REPLAY"
