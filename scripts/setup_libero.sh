#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
libero_source="${repo_root}/robot_experiments/libero/LIBERO"
libero_root="${LIBERO_PATCHED_ROOT:-${repo_root}/.cache/libero-build}"
patch_files=(
  "${repo_root}/patches/libero-pytorch-2.6.patch"
  "${repo_root}/patches/libero-setuptools-editable.patch"
)
python_bin="${PYTHON_BIN:-python}"
expected_commit="8f1084e3132a39270c3a13ebe37270a43ece2a01"
config_root="${LIBERO_CONFIG_PATH:-${repo_root}/.cache/libero}"
config_file="${config_root}/config.yaml"
source_stamp="${libero_root}/.phase_route_source_commit"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
  echo "Python executable not found: ${python_bin}" >&2
  exit 2
fi

git -C "${repo_root}" submodule update --init robot_experiments/libero/LIBERO

actual_commit="$(git -C "${libero_source}" rev-parse HEAD)"
if [[ "${actual_commit}" != "${expected_commit}" ]]; then
  echo "Unexpected LIBERO commit: ${actual_commit}; expected ${expected_commit}" >&2
  exit 1
fi

if [[ ! -e "${libero_root}" ]]; then
  mkdir -p "${libero_root}"
  cp -a "${libero_source}/." "${libero_root}/"
  rm -f "${libero_root}/.git"
  printf '%s\n' "${expected_commit}" >"${source_stamp}"
  echo "Created isolated LIBERO build copy: ${libero_root}"
elif [[ ! -f "${source_stamp}" ]] || [[ "$(<"${source_stamp}")" != "${expected_commit}" ]]; then
  echo "Refusing unverified LIBERO build copy: ${libero_root}" >&2
  echo "Use a new LIBERO_PATCHED_ROOT or remove the stale ignored cache." >&2
  exit 1
fi

for patch_file in "${patch_files[@]}"; do
  patch_name="$(basename "${patch_file}")"
  if patch --dry-run --silent -d "${libero_root}" -p1 <"${patch_file}"; then
    patch --batch --forward -d "${libero_root}" -p1 <"${patch_file}"
    echo "Applied LIBERO compatibility patch: ${patch_name}"
  elif patch --dry-run --silent --reverse -d "${libero_root}" -p1 <"${patch_file}"; then
    echo "LIBERO compatibility patch already applied: ${patch_name}"
  else
    echo "LIBERO build copy conflicts with compatibility patch: ${patch_name}" >&2
    exit 1
  fi
done

"${python_bin}" -m pip install --no-deps --no-build-isolation -e "${libero_root}"

mkdir -p "${config_root}"
if [[ ! -f "${config_file}" ]]; then
  benchmark_root="${libero_source}/libero/libero"
  {
    printf 'assets: %s/assets\n' "${benchmark_root}"
    printf 'bddl_files: %s/bddl_files\n' "${benchmark_root}"
    printf 'benchmark_root: %s\n' "${benchmark_root}"
    printf 'datasets: %s/libero/datasets\n' "${libero_source}"
    printf 'init_states: %s/init_files\n' "${benchmark_root}"
  } >"${config_file}"
  echo "Created non-interactive LIBERO config: ${config_file}"
else
  echo "Keeping existing LIBERO config: ${config_file}"
fi

echo "LIBERO setup complete: ${actual_commit}"
echo "Use LIBERO_CONFIG_PATH=${config_root} when running PhaseRoute-VLA."
