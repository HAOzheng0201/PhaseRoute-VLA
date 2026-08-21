#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin=/home/haozheng/.conda/envs/a1/bin/python
raw_root="${repo_root}/reports/v3_d8_fresh_raw"
context_result="${repo_root}/reports/v3_d8_fresh_context/result.json"
output_root="${repo_root}/reports/v3_d8_fresh_candidates"
log_root="${repo_root}/reports/v3_d8_fresh_candidate_logs"
checkpoint="${repo_root}/model/v3_d2/libero_exit"
attestation=/data3/haozheng/A1/source/reports/model_libero_exit_sha256_attestation_20260809_v1.json
data_dir=/data3/haozheng/A1/source

if [[ -n "$(git -C "${repo_root}" status --porcelain=v1)" ]]; then
  echo "D8C candidate replay requires a clean worktree" >&2
  exit 2
fi
"${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "PASS_V3_D8C_CONTEXT"' "${context_result}"
if [[ -e "${output_root}" || -e "${log_root}" ]]; then
  echo "D8C refuses to reuse candidate output or log roots" >&2
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
  expected_uuid="$(nvidia_field "${shard}" uuid)"
  memory_used="$(nvidia_field "${shard}" memory.used)"
  utilization="$(nvidia_field "${shard}" utilization.gpu)"
  if (( memory_used > 100 || utilization > 5 )); then
    echo "Refusing busy physical GPU${shard}: memory=${memory_used}MiB utilization=${utilization}%" >&2
    exit 4
  fi
  read -r visible_count actual_uuid < <(
    env CUDA_VISIBLE_DEVICES="${shard}" PYTHONNOUSERSITE=1 \
      "${python_bin}" -c 'import torch; print(torch.cuda.device_count(), torch.cuda.get_device_properties(0).uuid)'
  )
  if [[ "${visible_count}" != "1" || "${actual_uuid}" != "${expected_uuid#GPU-}" ]]; then
    echo "GPU mapping mismatch for physical GPU${shard}" >&2
    exit 5
  fi
  gpu_uuids+=("${expected_uuid}")
done

mkdir -p "${output_root}" "${log_root}"
git -C "${repo_root}" rev-parse HEAD >"${log_root}/source_git_commit.txt"
for shard in 0 1 2 3; do
  console_log="${log_root}/shard${shard}.log"
  env \
    CUDA_VISIBLE_DEVICES="${shard}" \
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
    "${repo_root}/scripts/dynamic_compute/v3/replay_v3_d8c_candidates.py" \
    --raw-root "${raw_root}" \
    --context-result "${context_result}" \
    --shard-index "${shard}" \
    --physical-gpu-index "${shard}" \
    --expected-gpu-uuid "${gpu_uuids[$shard]}" \
    --checkpoint "${checkpoint}" \
    --model-attestation "${attestation}" \
    --output-dir "${output_root}/shard${shard}" \
    >"${console_log}" 2>&1 &
  pids+=("$!")
  echo "started candidate shard=${shard} physical_gpu=${shard} pid=${pids[-1]} log=${console_log}"
done

while true; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  echo "candidate_replay alive_workers=${alive}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 30
done

failed=0
for shard in 0 1 2 3; do
  if wait "${pids[$shard]}"; then
    echo "completed candidate shard=${shard}"
  else
    echo "failed candidate shard=${shard}; inspect ${log_root}/shard${shard}.log" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 6
fi
echo "PASS_V3_D8C_CANDIDATE_FRONT4"
