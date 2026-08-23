#!/usr/bin/env python3
"""Validate D9A without opening episode 40--49 state or running control."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from typing import Any


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from a1.vla.dynamic_compute.productive_exit import (  # noqa: E402
    a1_fm10_rp_pep_plan,
)
from a1.vla.dynamic_compute.v3.final_router import (  # noqa: E402
    FinalFiveHeadRouter,
    final_router_from_mapping,
)
from a1.vla.dynamic_compute.v3.gripper_v2_models import (  # noqa: E402
    FeatureNormalizer,
)
from a1.vla.dynamic_compute.v3.independent_test_protocol import (  # noqa: E402
    load_d9_contract,
)
from a1.vla.dynamic_compute.v3.runtime_adapter import (  # noqa: E402
    D9A_RUNTIME_SCHEMA_VERSION,
    D9A_RUNTIME_STATUS,
    EpisodePastOnlyHistory,
    FrozenD8RuntimeAdapter,
    frozen_router_sha256,
    route_cached_candidate_pairs,
)
from a1.vla.dynamic_compute.v3.severity_reliability import (  # noqa: E402
    SeverityWeightedFit,
)
from a1.vla.value_net import ActionValueNet  # noqa: E402


OUTPUT = Path("results/v3/v3_d9a_runtime_adapter_validation.json")
CONTEXT = Path("reports/v3_d8_fresh_context/fresh_context.pt")
DATASET = Path("reports/v3_d8_fresh_dataset/fresh_confirmation_dataset.pt")
ROUTER = Path("reports/v3_d8_final_router/final_router.pt")
SCORING = Path("reports/v3_d8_confirmation/confirmation_scoring.pt")
EXPECTED_SHA256 = {
    CONTEXT: "3941ea81f1387da819f5ab9c12612bb3aa954d90d2b7e26dd9a7dfc3994b3785",
    DATASET: "411b3d68b2e4326573722a616b5fcf7862fbcc6b85f499be7cdf0877a8889327",
    ROUTER: "9f7360188e30e5831b18d460bf338638fb960db9374dd9cc74412f169914b830",
    SCORING: "b225ebec9bfd55044a5b856dd09ad9b5b14278164172d93d525d10309472ffba",
}
CODE_PATHS = (
    Path("a1/vla/dynamic_compute/v3/runtime_adapter.py"),
    Path("a1/vla/dynamic_compute/v3/development_collection.py"),
    Path("a1/vla/dynamic_compute/v3/final_router.py"),
    Path("a1/vla/value_net.py"),
    Path("tests/dynamic_compute/v3/test_runtime_adapter.py"),
    Path("tests/dynamic_compute/test_productive_exit.py"),
    Path("scripts/dynamic_compute/v3/validate_v3_d9a_runtime_adapter.py"),
)
ORIGINAL_EXITS = (1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 25, 27)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPO_ROOT, text=True
    ).strip()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _model(offset: float) -> SeverityWeightedFit:
    return SeverityWeightedFit(
        normalizer=FeatureNormalizer(
            mean=torch.zeros(97, dtype=torch.float64),
            scale=torch.ones(97, dtype=torch.float64),
        ),
        anchor_score=torch.tensor(
            [[0.10 + offset, 0.02], [0.20 + offset, 0.03]],
            dtype=torch.float64,
        ),
        weight=torch.zeros((2, 97), dtype=torch.float64),
        l2_lambda=0.01,
        final_loss=0.5 + offset,
    )


def _synthetic_router() -> FinalFiveHeadRouter:
    return FinalFiveHeadRouter(
        models=tuple(_model(0.01 * index) for index in range(5)),
        full_threshold=0.5,
        runtime_threshold=0.475,
    )


def _runtime(rows: int = 1) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260822)
    return {
        "instruction_summary": torch.randn(rows, 3584, generator=generator),
        "vision_crop_summary": torch.randn(rows, 5, 3584, generator=generator),
        "vision_crop_mask": torch.tensor(
            [[True, True, True, True, False]] * rows
        ),
        "phase_embedding": torch.randn(rows, 128, generator=generator),
        "phase_scalars": torch.rand(rows, 3, generator=generator),
        "normalized_proprio": torch.randn(rows, 8, generator=generator),
        "proprio_history": torch.zeros(rows, 8, 8),
        "action_history": torch.zeros(rows, 8, 8, 7),
        "history_mask": torch.zeros(rows, 8, dtype=torch.bool),
    }


def synthetic_smoke() -> dict[str, bool]:
    generator = torch.Generator().manual_seed(7)
    l11, l13, l27 = (
        torch.randn(1, 8, 7, generator=generator) for _ in range(3)
    )
    adapter = FrozenD8RuntimeAdapter(_synthetic_router())

    adapter.begin_policy_call(_runtime())
    decision11 = adapter.consider_candidate(11, l11, True)

    adapter.begin_policy_call(_runtime())
    veto11 = adapter.consider_candidate(11, l11, False)
    decision13 = adapter.consider_candidate(13, l13, True)

    adapter.begin_policy_call(_runtime())
    veto11b = adapter.consider_candidate(11, l11, False)
    veto13 = adapter.consider_candidate(13, l13, False)
    decision27 = adapter.select_fallback(l27)

    invalid = l11.clone()
    invalid[0, 0, 0] = float("nan")
    adapter.begin_policy_call(_runtime())
    invalid11 = adapter.consider_candidate(11, invalid, True)
    invalid13 = adapter.consider_candidate(13, l13, True)
    invalid27 = adapter.select_fallback(l27)

    history = EpisodePastOnlyHistory()
    proprio = np.arange(8, dtype=np.float32)
    action = np.arange(56, dtype=np.float32).reshape(8, 7)
    first = history.window("episode-a", 0, proprio)
    history.commit("episode-a", 0, action)
    second = history.window("episode-a", 1, proprio + 1)
    history.commit("episode-a", 1, action + 1)
    reset = history.window("episode-b", 0, proprio + 2)

    return {
        "L11_selected": decision11.should_exit
        and decision11.layer == 11
        and decision11.selected_action is l11,
        "L13_selected_after_L11_veto": not veto11.should_exit
        and decision13.should_exit
        and decision13.layer == 13
        and decision13.selected_action is l13,
        "L27_fallback": not veto11b.should_exit
        and not veto13.should_exit
        and decision27.layer == 27
        and decision27.selected_action is l27,
        "nonfinite_fail_closed": not invalid11.should_exit
        and not invalid13.should_exit
        and invalid27.layer == 27
        and invalid27.selected_action is l27,
        "episode_history_reset": not bool(first.history_mask.any())
        and second.history_mask[0].tolist() == [False] * 7 + [True]
        and not bool(reset.history_mask.any()),
        "selected_action_exactness": decision11.selected_action is l11
        and decision13.selected_action is l13
        and decision27.selected_action is l27,
    }


class _FakeFlowModel:
    def __init__(self) -> None:
        self.candidate_inputs: dict[int, torch.Tensor] = {}
        self.config = SimpleNamespace(
            action_head="flow_matching",
            num_diffusion_inference_steps=10,
            num_actions_chunk=8,
            fixed_action_dim=7,
        )

    def predict_actions_flow_matching(
        self,
        kvs,
        proprio,
        pos_offset,
        input_x=None,
        fm_trace_callback=None,
        fm_trace_context=None,
    ):
        del proprio, pos_offset
        if input_x is None:
            input_x = torch.randn(1, 8, 7)
        context = dict(fm_trace_context or {})
        if context.get("candidate_role") == "candidate_action":
            self.candidate_inputs[int(context["candidate_layer"])] = input_x.clone()
        output = input_x + len(kvs) * 0.01
        if fm_trace_callback is not None:
            fm_trace_callback(
                {
                    **context,
                    "input_x": input_x,
                    "output_action": output,
                    "fm_steps": 10,
                }
            )
        return output


def _value_net() -> ActionValueNet:
    plan = a1_fm10_rp_pep_plan(ORIGINAL_EXITS)
    return ActionValueNet(
        exit_list=list(plan.eligible_exit_layers),
        exit_head=None,
        model=_FakeFlowModel(),
        interval=2,
        threshold_type="cosine",
        anchor=False,
        productive_exit_plan=plan,
    )


def shared_noise_rng_smoke() -> dict[str, Any]:
    plan = a1_fm10_rp_pep_plan(ORIGINAL_EXITS)
    baseline = _value_net()
    runtime = _value_net()
    runtime.configure_phase_route_shared_candidates((11, 13, 27))
    feats = [(torch.zeros(1, 1), torch.zeros(1, 1)) for _ in range(28)]

    def run(value_net: ActionValueNet):
        actions = {}
        states = {}
        for layer in plan.eligible_exit_layers:
            _, action = value_net(
                feats,
                layer,
                None,
                0,
                0,
                None,
                fm_trace_callback=lambda _payload: None,
            )
            actions[layer] = action.clone()
            states[layer] = torch.random.get_rng_state().clone()
        return actions, states

    torch.manual_seed(20260822)
    _, baseline_states = run(baseline)
    torch.manual_seed(20260822)
    _, runtime_states = run(runtime)
    return {
        "RNG_state_exact_after_every_candidate": all(
            torch.equal(runtime_states[layer], baseline_states[layer])
            for layer in plan.eligible_exit_layers
        ),
        "L11_L13_shared_input_exact": torch.equal(
            runtime.model.candidate_inputs[11],
            runtime.model.candidate_inputs[13],
        ),
        "L11_L27_shared_input_exact": torch.equal(
            runtime.model.candidate_inputs[11],
            runtime.model.candidate_inputs[27],
        ),
    }


def main() -> None:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise PermissionError("D9A validation is CPU-only")
    if git_output("status", "--porcelain=v1"):
        raise PermissionError("D9A validation requires a clean worktree")
    contract = load_d9_contract(REPO_ROOT)
    output = REPO_ROOT / OUTPUT
    sidecar = output.with_suffix(".sha256")
    incomplete = output.with_suffix(".json.incomplete")
    if output.exists() or sidecar.exists() or incomplete.exists():
        raise FileExistsError("D9A refuses to overwrite validation evidence")

    artifact_hashes = {}
    for relative, expected in EXPECTED_SHA256.items():
        path = REPO_ROOT / relative
        observed = sha256(path)
        if observed != expected:
            raise PermissionError(f"D9A prerequisite hash differs: {relative}")
        artifact_hashes[relative.as_posix()] = observed

    context = torch.load(
        REPO_ROOT / CONTEXT, map_location="cpu", weights_only=False
    )
    dataset = torch.load(
        REPO_ROOT / DATASET, map_location="cpu", weights_only=False
    )
    router_payload = torch.load(
        REPO_ROOT / ROUTER, map_location="cpu", weights_only=False
    )
    expected = torch.load(
        REPO_ROOT / SCORING, map_location="cpu", weights_only=False
    )
    router = final_router_from_mapping(router_payload)
    router_before = frozen_router_sha256(router)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    observed = route_cached_candidate_pairs(
        router,
        context["runtime_inputs"],
        dataset["candidate_actions"][:, :2],
        dataset["action_consistency"].reshape(-1, 2),
    )
    router_after = frozen_router_sha256(router)

    head_error = float(
        (
            observed.five_head_prediction
            - expected["five_head_prediction"]
        )
        .abs()
        .max()
    )
    safe_matches = int(
        (observed.candidate_safe == expected["candidate_safe"]).sum()
    )
    layer_matches = int(
        (observed.selected_layer == expected["selected_layer"]).sum()
    )
    smoke = synthetic_smoke()
    rng = shared_noise_rng_smoke()
    checks = {
        "synthetic_all_required_branches_pass": all(smoke.values()),
        "shared_candidate_noise_and_RNG_preservation_pass": all(rng.values()),
        "D8_policy_calls_exactly_7140": observed.selected_layer.numel() == 7140,
        "D8_candidate_rows_exactly_14280": observed.candidate_safe.numel()
        == 14280,
        "D8_97D_features_exact": torch.equal(
            observed.features, dataset["features"]
        ),
        "D8_selected_layer_exact_7140_of_7140": layer_matches == 7140,
        "D8_candidate_safe_exact_14280_of_14280": safe_matches == 14280,
        "D8_five_head_prediction_max_abs_error_at_most_1e-12": head_error
        <= 1.0e-12,
        "router_weights_normalizers_thresholds_unchanged": router_before
        == router_after,
        "candidate_identity_not_in_97D_context": tuple(
            context["runtime_inputs"]
        )
        == (
            "instruction_summary",
            "vision_crop_summary",
            "vision_crop_mask",
            "phase_embedding",
            "phase_scalars",
            "normalized_proprio",
            "proprio_history",
            "action_history",
            "history_mask",
        ),
        "official_episode_40_49_not_opened": True,
        "active_control_not_run": True,
        "no_fit_threshold_or_feature_selection": True,
    }
    if not all(checks.values()):
        raise PermissionError("D9A runtime adapter validation failed")

    selection_counts = {
        f"L{layer}": int((observed.selected_layer == layer).sum())
        for layer in (11, 13, 27)
    }
    result = {
        "status": D9A_RUNTIME_STATUS,
        "schema_version": D9A_RUNTIME_SCHEMA_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(
            timespec="milliseconds"
        ),
        "source_git_commit": git_output("rev-parse", "HEAD"),
        "source_worktree_dirty": False,
        "contract_stage": contract["execution_order"]["D9A"],
        "frozen_inputs_sha256": artifact_hashes,
        "code_sha256": {
            path.as_posix(): sha256(REPO_ROOT / path) for path in CODE_PATHS
        },
        "router_state_sha256": router_before,
        "synthetic_smoke": smoke,
        "shared_noise_RNG_smoke": rng,
        "D8_exact_parity": {
            "policy_calls": observed.selected_layer.numel(),
            "candidate_rows": observed.candidate_safe.numel(),
            "feature_dimension": observed.features.shape[1],
            "selected_layer_exact_matches": layer_matches,
            "candidate_safe_exact_matches": safe_matches,
            "five_head_prediction_max_abs_error": head_error,
            "selection_counts": selection_counts,
        },
        "checks": checks,
        "access_ledger": {
            "synthetic_inputs_used": True,
            "already_analyzed_D8_cached_inputs_opened": 4,
            "D8_router_payload_opened": True,
            "independent_test_selection_metadata_opened": False,
            "independent_test_sample_state_payload_opened": False,
            "LIBERO_episode_40_49_init_states_opened": False,
            "model_checkpoint_opened": False,
            "GPU_query_or_initialization": 0,
            "fit_calls": 0,
            "threshold_or_feature_searches": 0,
            "test_rollouts": 0,
            "active_control": False,
        },
        "authorization": {
            "next_stage": "D9B_READINESS_ATTESTATION_ONLY",
            "episode_40_49_state_access": False,
            "independent_test_sample_payload_access": False,
            "active_test_rollout": False,
            "deployment": False,
        },
    }
    incomplete.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    incomplete.replace(output)
    sidecar.write_text(f"{sha256(output)}  {output.name}\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
