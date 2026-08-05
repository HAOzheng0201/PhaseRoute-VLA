#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
libero_root="${repo_root}/robot_experiments/libero/LIBERO"
patch_file="${repo_root}/patches/libero-pytorch-2.6.patch"
python_bin="${PYTHON_BIN:-python}"
expected_commit="8f1084e3132a39270c3a13ebe37270a43ece2a01"
config_root="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"
config_file="${config_root}/config.yaml"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi

git -C "${repo_root}" submodule update --init robot_experiments/libero/LIBERO

actual_commit="$(git -C "${libero_root}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "Unexpected LIBERO commit: ${actual_commit}; expected ${expected_commit}" >&2
  exit 1
fi

if git -C "${libero_root}" apply --check "${patch_file}" >/dev/null 2>&1; then
  git -C "${libero_root}" apply "${patch_file}"
  echo "Applied the PyTorch 2.6 LIBERO compatibility patch."
elif git -C "${libero_root}" apply --reverse --check "${patch_file}" >/dev/null 2>&1; then
  echo "LIBERO compatibility patch is already applied."
else
  echo "LIBERO is not at the expected commit or the patch conflicts." >&2
  exit 1
fi

"${python_bin}" -m pip install --no-deps --no-build-isolation -e "${libero_root}"

mkdir -p "${config_root}"
if [[ ! -f "${config_file}" ]]; then
  benchmark_root="${libero_root}/libero/libero"
  {
    printf 'assets: %s/assets\n' "${benchmark_root}"
    printf 'bddl_files: %s/bddl_files\n' "${benchmark_root}"
    printf 'benchmark_root: %s\n' "${benchmark_root}"
    printf 'datasets: %s/libero/datasets\n' "${libero_root}"
    printf 'init_states: %s/init_files\n' "${benchmark_root}"
  } >"${config_file}"
  echo "Created non-interactive LIBERO config: ${config_file}"
else
  echo "Keeping existing LIBERO config: ${config_file}"
fi

echo "LIBERO setup complete: ${actual_commit}"
echo "Use LIBERO_CONFIG_PATH=${config_root} when running PhaseRoute-VLA."
