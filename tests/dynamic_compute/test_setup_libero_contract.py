from pathlib import Path
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_make_targets_honor_the_config_path_override() -> None:
    makefile = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    assert "LIBERO_CONFIG_PATH ?= $(CURDIR)/.cache/libero" in makefile
    assert makefile.count('LIBERO_CONFIG_PATH="$(LIBERO_CONFIG_PATH)"') == 3


def test_setup_patches_an_ignored_copy_instead_of_the_submodule() -> None:
    setup_script = (REPO_ROOT / "scripts/setup_libero.sh").read_text(
        encoding="utf-8"
    )

    assert 'libero_source="${repo_root}/robot_experiments/libero/LIBERO"' in setup_script
    assert 'libero_root="${LIBERO_PATCHED_ROOT:-${repo_root}/.cache/libero-build}"' in setup_script
    assert 'cp -a "${libero_source}/." "${libero_root}/"' in setup_script
    assert 'rm -f "${libero_root}/.git"' in setup_script
    assert 'pip install --no-deps --no-build-isolation -e "${libero_root}"' in setup_script
    assert 'pip install --no-deps --no-build-isolation -e "${libero_source}"' not in setup_script


def test_setuptools_editable_patch_applies_to_the_pinned_setup(tmp_path: Path) -> None:
    source = REPO_ROOT / "robot_experiments/libero/LIBERO/setup.py"
    target = tmp_path / "setup.py"
    target.write_bytes(source.read_bytes())
    patch = (REPO_ROOT / "patches/libero-setuptools-editable.patch").read_text(
        encoding="utf-8"
    )

    completed = subprocess.run(
        ["patch", "--batch", "--forward", "-p1"],
        cwd=tmp_path,
        input=patch,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    patched = target.read_text(encoding="utf-8")
    assert 'packages=["libero"]' in patched
    assert 'find_packages(where="libero")' in patched
    assert 'package_dir={"libero": "libero"}' in patched
    compile(patched, str(target), "exec")
