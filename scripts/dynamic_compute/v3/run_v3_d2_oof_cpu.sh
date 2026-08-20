#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
python_bin=/home/haozheng/.conda/envs/a1/bin/python
dataset_result="${repo_root}/reports/v3_d2_development_dataset/result.json"
output_root="${repo_root}/reports/v3_d2_development_oof_folds"
log_root="${repo_root}/reports/v3_d2_development_oof_logs"

if [[ -n "$(git -C "${repo_root}" status --porcelain=v1)" ]]; then
  echo "V3-D2 nested OOF requires a clean worktree" >&2
  exit 2
fi
if [[ -e "${output_root}" || -e "${log_root}" ]]; then
  echo "V3-D2 refuses to reuse OOF output or log roots" >&2
  exit 3
fi
"${python_bin}" -c 'import json,sys; d=json.load(open(sys.argv[1])); assert d["status"] == "PASS_V3_D2_DATASET"' "${dataset_result}"
mkdir -p "${output_root}" "${log_root}"
git -C "${repo_root}" rev-parse HEAD >"${log_root}/source_git_commit.txt"

declare -a pids=()
declare -a episodes=()
for episode in {12..29}; do
  console_log="${log_root}/episode${episode}.log"
  env \
    CUDA_VISIBLE_DEVICES=-1 \
    OMP_NUM_THREADS=2 \
    MKL_NUM_THREADS=2 \
    OPENBLAS_NUM_THREADS=2 \
    NUMEXPR_NUM_THREADS=2 \
    TF_CPP_MIN_LOG_LEVEL=3 \
    PYTHONNOUSERSITE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH="${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${python_bin}" \
    "${repo_root}/scripts/dynamic_compute/v3/run_v3_d2_oof_outer_fold.py" \
    --dataset-result "${dataset_result}" \
    --outer-episode "${episode}" \
    --output-dir "${output_root}/episode${episode}" \
    >"${console_log}" 2>&1 &
  pids+=("$!")
  episodes+=("${episode}")
  echo "started OOF outer_episode=${episode} pid=${pids[-1]} log=${console_log}"
done

while true; do
  alive=0
  for pid in "${pids[@]}"; do
    if kill -0 "${pid}" 2>/dev/null; then
      alive=$((alive + 1))
    fi
  done
  echo "nested_oof alive_workers=${alive}"
  if [[ "${alive}" == "0" ]]; then
    break
  fi
  sleep 30
done

failed=0
for index in "${!pids[@]}"; do
  episode="${episodes[$index]}"
  if wait "${pids[$index]}"; then
    echo "completed OOF outer_episode=${episode}"
  else
    echo "failed OOF outer_episode=${episode}; inspect ${log_root}/episode${episode}.log" >&2
    failed=1
  fi
done
if (( failed != 0 )); then
  exit 4
fi
echo "PASS_V3_D2_OOF_ALL_OUTER_FOLDS"
