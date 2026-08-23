PYTHON ?= python

.PHONY: help install setup-libero download-checkpoint preflight preflight-v3 test test-release test-v3-release check run-rp-pep run-v3 smoke-front4

help:
	@echo "make preflight     CPU environment + artifact audit"
	@echo "make preflight-v3  CPU V3 bundle + five-head payload audit"
	@echo "make install       Install PhaseRoute-VLA and pinned LIBERO dependencies"
	@echo "make setup-libero  Initialize, patch and install the pinned LIBERO submodule"
	@echo "make download-checkpoint  Download and verify the frozen A1 early-exit checkpoint"
	@echo "make test-release  Fast release-gate unit tests"
	@echo "make test-v3-release  Fast V3 artifact/launcher tests"
	@echo "make test          Full dynamic-compute regression suite"
	@echo "make check         pip check + shell/Python syntax + git diff check"
	@echo "make run-rp-pep    Validated LIBERO Spatial runtime on GPU_INDEX=0..3"
	@echo "make run-v3        Frozen V3 LIBERO-10 research runtime on GPU_INDEX=0..3"
	@echo "make smoke-front4  Ten-task state-30 release smoke on physical GPUs 0..3"

install:
	$(PYTHON) -m pip install -e ".[libero]" -c requirements/constraints-cu124.txt

setup-libero:
	PYTHON_BIN="$(PYTHON)" LIBERO_CONFIG_PATH="$(CURDIR)/.cache/libero" bash scripts/setup_libero.sh

download-checkpoint:
	bash scripts/download_checkpoint.sh

preflight:
	PYTHONNOUSERSITE=1 LIBERO_CONFIG_PATH="$(CURDIR)/.cache/libero" VLA_CONFIG_YAML=libero_simulation.yaml MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $(PYTHON) scripts/validate_phase_route_release.py

preflight-v3:
	PYTHONNOUSERSITE=1 LIBERO_CONFIG_PATH="$(CURDIR)/.cache/libero" VLA_CONFIG_YAML=libero_simulation.yaml MUJOCO_GL=egl PYOPENGL_PLATFORM=egl $(PYTHON) scripts/validate_phase_route_v3_release.py

test-release:
	PYTHONNOUSERSITE=1 $(PYTHON) -m pytest -q tests/dynamic_compute/test_release_gate.py tests/dynamic_compute/test_release_smoke_summary.py

test-v3-release:
	PYTHONNOUSERSITE=1 $(PYTHON) -m pytest -q tests/dynamic_compute/v3/test_release.py

test:
	PYTHONNOUSERSITE=1 $(PYTHON) -m pytest -q tests/dynamic_compute

check:
	PYTHONNOUSERSITE=1 $(PYTHON) -m pip check
	$(PYTHON) -m py_compile scripts/validate_phase_route_release.py scripts/validate_phase_route_v3_release.py scripts/validate_phase_route_v3_run.py scripts/run_phase_route_v3.py a1/vla/dynamic_compute/release.py a1/vla/dynamic_compute/v3/release.py scripts/dynamic_compute/summarize_release_smoke.py
	bash -n eval_libero.sh eval_libero_exit.sh train_libero.sh
	bash -n scripts/*.sh
	bash -n scripts/dynamic_compute/*.sh
	git diff --check

run-rp-pep:
	bash scripts/run_libero_rp_pep.sh

run-v3:
	bash scripts/run_libero_phase_route_v3.sh

smoke-front4:
	bash scripts/dynamic_compute/run_release_smoke_front4.sh "reports/release_smoke_$$(date +%Y%m%d_%H%M%S)"
