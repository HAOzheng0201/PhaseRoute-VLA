from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_runner_directory_is_gpu_scoped_and_subsecond_unique() -> None:
    runner = (REPO_ROOT / "scripts/run_libero_phase_route_v3.sh").read_text(
        encoding="utf-8"
    )

    assert 'timestamp="$(date +%Y%m%d_%H%M%S_%N)"' in runner
    assert 'run_dir="${output_root}/libero_10_gpu${gpu_index}_${timestamp}"' in runner
