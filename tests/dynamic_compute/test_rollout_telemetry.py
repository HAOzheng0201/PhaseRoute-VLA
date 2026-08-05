import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from a1.vla.dynamic_compute.budget_profiles import BudgetProfile, ResolvedBudget
from a1.vla.dynamic_compute.phase_estimator import PhaseState
from a1.vla.dynamic_compute.telemetry import (
    NullTelemetryLogger,
    SafeJSONLTelemetryLogger,
)
from a1.vla.dynamic_compute.phase_cache import (
    NullPhaseCacheWriter,
    SafePhaseCacheWriter,
)


class FakeModelConfig:
    vision_backbone = SimpleNamespace(image_default_input_size=(336, 336))
    num_actions_chunk = 8
    action_dim = 7
    proprio_dim = 8
    action_head = "flow_matching"
    n_layers = 28

    @staticmethod
    def get_max_crops():
        return 4


class FakeModel:
    def __init__(self):
        self.config = FakeModelConfig()
        self.calls = []
        self.action = torch.arange(56, dtype=torch.float32).reshape(1, 8, 7) / 100
        self.tokenizer = SimpleNamespace(encode=lambda text: [1, 2, 3, 4])
        self.transformer = SimpleNamespace(
            wte=lambda input_ids: input_ids.to(torch.float32).unsqueeze(-1).repeat(1, 1, 8)
        )
        self.phase_plan_active_during_predict = []

    @staticmethod
    def get_all_exit_idx(exit_interval):
        assert exit_interval == 2
        return [1, 3, 5, 27]

    def predict_actions(self, **kwargs):
        self.calls.append(kwargs)
        callback = kwargs.get("telemetry_callback")
        if callback is not None:
            callback(
                "exit_candidate",
                {
                    "layer_idx": 3,
                    "evaluated": True,
                    "should_exit": True,
                    "action_delta": 0.125,
                    "fm_calls": 1,
                    "fm_steps": 10,
                },
            )
        phase_callback = kwargs.get("phase_signal_callback")
        if phase_callback is not None:
            phase_callback(
                {
                    "visual_summary": torch.arange(6, dtype=torch.float32).reshape(1, 6),
                    "instruction_summary": torch.arange(8, dtype=torch.float32).reshape(1, 8),
                    "visual_token_count": torch.tensor([2]),
                    "instruction_token_count": torch.tensor([3]),
                }
            )
        controller = kwargs.get("exit_controller")
        self.phase_plan_active_during_predict.append(
            bool(getattr(controller, "phase_plan_active", False))
            if controller is not None
            else False
        )
        return self.action.clone()


class FakeExitController:
    def __init__(self):
        self.clear_calls = 0
        self.set_calls = []
        self.phase_plan_active = False

    def clear_phase_plan(self):
        self.clear_calls += 1
        self.phase_plan_active = False

    def set_phase_plan(self, **kwargs):
        self.set_calls.append(kwargs)
        self.phase_plan_active = True


class FakeSparseExitController(FakeExitController):
    exit_id_list = [3, 11, 13, 27]


class FakePhaseDepthRuntime:
    enabled = True
    exit_policy = object()

    def __init__(self):
        self.prepare_calls = []
        self.update_calls = []

    def prepare_plan(self, **kwargs):
        self.prepare_calls.append(kwargs)
        state = PhaseState(
            stage_embedding=torch.zeros(1, 4),
            progress=torch.tensor([[0.4]]),
            boundary_prob=torch.tensor([[0.2]]),
            uncertainty=torch.tensor([[0.1]]),
            next_hidden=torch.zeros(1, 1, 5),
        )
        budget = ResolvedBudget(
            profile_id=2,
            profile=BudgetProfile("B2", None, 1.0, 0.5, 0.75, 10),
            min_exit_rank=2,
            min_exit_layer=5,
            eligible_exit_layers=(1, 3, 5, 27),
        )
        return SimpleNamespace(
            phase_state=state,
            routing_phase_state=state,
            selection=SimpleNamespace(reasons=("test_phase",)),
            budget=budget,
            boundary_rise=0.1,
            boundary_crossed=False,
            latency_ms=1.25,
            fallback=False,
            error=None,
        )

    def update_after_action(self, **kwargs):
        self.update_calls.append(kwargs)
        return True


class FailingPhaseDepthRuntime(FakePhaseDepthRuntime):
    def prepare_plan(self, **kwargs):
        self.prepare_calls.append(kwargs)
        raise RuntimeError("synthetic phase failure")


class FakeCollator:
    def __init__(self, *args, **kwargs):
        del args, kwargs

    def __call__(self, records):
        assert len(records) == 1
        return {
            "input_ids": torch.tensor([[10, 11, 12, -1, -1]]),
            "images": None,
            "image_masks": None,
            "attention_mask": torch.tensor([[1, 1, 1, 0, 0]]),
            "attention_bias": None,
            "loss_masks": torch.zeros(1, 5),
            "image_input_idx": torch.tensor([[0, 1, -1]]),
            "subsegment_ids": None,
            "position_ids": torch.tensor([[0, 1, 2, 0, 0]]),
            "proprio": None,
            "proprio_token_idx": None,
        }


def _patch_policy_dependencies(monkeypatch, module):
    blank_image = np.zeros((336, 336, 3), dtype=np.uint8)
    monkeypatch.setattr(
        module,
        "prepare_images_for_vla",
        lambda images, cfg, image_size: [blank_image.copy(), blank_image.copy()],
    )
    monkeypatch.setattr(module, "_load_dataset_stats", lambda checkpoint: {})
    monkeypatch.setattr(module, "build_mm_preprocessor", lambda **kwargs: lambda record: record)
    monkeypatch.setattr(module, "MMCollatorForAction", FakeCollator)
    monkeypatch.setattr(
        module,
        "_unnormalize_actions",
        lambda actions, norm_stats, normalization_type, unnorm_key: actions,
    )
    monkeypatch.setattr(module.torch.cuda, "is_available", lambda: False)


def _config():
    return SimpleNamespace(
        num_images_in_input=2,
        use_wrist_image=True,
        use_proprio=False,
        center_crop=False,
        pretrained_checkpoint="unused",
        normalization_type="unused",
        unnorm_key="libero_spatial_no_noops",
        sequence_length=680,
        num_open_loop_steps=8,
        exit_interval=2,
        exit_layer_id=None,
    )


def _observation():
    return {
        "full_image": np.zeros((336, 336, 3), dtype=np.uint8),
        "wrist_image": np.zeros((336, 336, 3), dtype=np.uint8),
        "state": np.array([0.0, 0.1, 0.2, 0.0, 0.0, 0.0, -0.03, 0.03]),
    }


def _set_import_environment(monkeypatch):
    monkeypatch.setenv("DATA_DIR", "/tmp/phase-route-vla")
    monkeypatch.setenv("HF_HOME", "/tmp/a1-test-hf-cache")
    monkeypatch.setenv("VLA_CONFIG_YAML", "libero_simulation.yaml")


@pytest.mark.parametrize(
    "module_name",
    ["robot_experiments.vla_utils", "robot_experiments.libero.exit_vla_utils"],
)
def test_disabled_rollout_path_does_not_pass_callback_or_change_action(monkeypatch, module_name):
    _set_import_environment(monkeypatch)
    module = importlib.import_module(module_name)
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()

    baseline = module.get_vla_action(
        _config(), model, torch.device("cpu"), _observation(), "pick up the bowl"
    )
    disabled = module.get_vla_action(
        _config(),
        model,
        torch.device("cpu"),
        _observation(),
        "pick up the bowl",
        telemetry_logger=NullTelemetryLogger(),
    )

    assert len(baseline) == len(disabled) == 8
    np.testing.assert_array_equal(np.stack(baseline), np.stack(disabled))
    assert "telemetry_callback" not in model.calls[0]
    assert "telemetry_callback" not in model.calls[1]
    assert "phase_signal_callback" not in model.calls[0]
    assert "phase_signal_callback" not in model.calls[1]


@pytest.mark.parametrize(
    "module_name",
    ["robot_experiments.vla_utils", "robot_experiments.libero.exit_vla_utils"],
)
def test_enabled_rollout_writes_one_complete_policy_call(monkeypatch, tmp_path: Path, module_name):
    _set_import_environment(monkeypatch)
    module = importlib.import_module(module_name)
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()
    output_path = tmp_path / (module_name.replace(".", "_") + ".jsonl")

    with SafeJSONLTelemetryLogger(output_path, flush_every=1) as logger:
        actions = module.get_vla_action(
            _config(),
            model,
            torch.device("cpu"),
            _observation(),
            "pick up the bowl",
            telemetry_logger=logger,
            telemetry_context={
                "episode_id": "libero_spatial:task0:episode0",
                "step_id": 10,
                "task_id": 0,
                "previous_action": [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0],
            },
        )

    assert len(actions) == 8
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["episode_id"] == "libero_spatial:task0:episode0"
    assert payload["step_id"] == 10
    assert payload["candidate_exit_layers"] == [1, 3, 5, 27]
    assert payload["action_delta_by_exit"] == [None, 0.125, None, None]
    assert payload["exit_layer"] == 3
    assert payload["fm_calls"] == 1
    assert payload["fm_steps_total"] == 10
    assert payload["action_shape"] == [1, 8, 7]
    assert payload["extra"]["visual_tokens"] == 2
    assert len(payload["active_tokens_by_layer"]) == 28


@pytest.mark.parametrize(
    "module_name",
    ["robot_experiments.vla_utils", "robot_experiments.libero.exit_vla_utils"],
)
def test_telemetry_prefers_runtime_controller_exit_layers(
    monkeypatch, tmp_path: Path, module_name
):
    _set_import_environment(monkeypatch)
    module = importlib.import_module(module_name)
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()
    output_path = tmp_path / (module_name.replace(".", "_") + "_sparse.jsonl")

    with SafeJSONLTelemetryLogger(output_path, flush_every=1) as logger:
        module.get_vla_action(
            _config(),
            model,
            torch.device("cpu"),
            _observation(),
            "pick up the bowl",
            exit_controller=FakeSparseExitController(),
            telemetry_logger=logger,
            telemetry_context={"episode_id": "sparse", "step_id": 10},
        )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["candidate_exit_layers"] == [3, 11, 13, 27]


def test_disabled_phase_cache_does_not_pass_callback_or_create_files(monkeypatch, tmp_path: Path):
    _set_import_environment(monkeypatch)
    module = importlib.import_module("robot_experiments.libero.exit_vla_utils")
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()

    actions = module.get_vla_action(
        _config(),
        model,
        torch.device("cpu"),
        _observation(),
        "pick up the bowl",
        phase_cache_writer=NullPhaseCacheWriter(),
    )

    assert len(actions) == 8
    assert "phase_signal_callback" not in model.calls[0]
    assert list(tmp_path.iterdir()) == []


def test_enabled_phase_cache_writes_one_model_summary_call(monkeypatch, tmp_path: Path):
    _set_import_environment(monkeypatch)
    module = importlib.import_module("robot_experiments.libero.exit_vla_utils")
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()
    writer = SafePhaseCacheWriter(tmp_path / "phase_cache")

    actions = module.get_vla_action(
        _config(),
        model,
        torch.device("cpu"),
        _observation(),
        "pick up the bowl",
        phase_cache_writer=writer,
        phase_cache_context={
            "episode_id": "libero_spatial:task0:episode0",
            "step_id": 10,
            "task_id": 0,
            "previous_action": None,
        },
    )
    writer.close()

    assert len(actions) == 8
    assert "phase_signal_callback" in model.calls[0]
    assert writer.records_written == 1
    manifest = json.loads((tmp_path / "phase_cache" / "manifest.jsonl").read_text())
    shard = np.load(tmp_path / "phase_cache" / manifest["array_path"])
    assert manifest["summary_counts"] == {
        "visual_tokens": 2,
        "instruction_tokens": 4,
    }
    np.testing.assert_array_equal(
        shard["raw_proprio"], _observation()["state"].astype(np.float32)
    )
    np.testing.assert_array_equal(shard["action_chunk"], model.action.numpy()[0])


def test_disabled_phase_depth_is_bitwise_identical_and_passes_no_callback(monkeypatch):
    _set_import_environment(monkeypatch)
    module = importlib.import_module("robot_experiments.libero.exit_vla_utils")
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()
    controller = FakeExitController()

    baseline = module.get_vla_action(
        _config(), model, torch.device("cpu"), _observation(), "pick up the bowl"
    )
    disabled = module.get_vla_action(
        _config(),
        model,
        torch.device("cpu"),
        _observation(),
        "pick up the bowl",
        exit_controller=controller,
        phase_depth_runtime=SimpleNamespace(enabled=False),
    )

    np.testing.assert_array_equal(np.stack(baseline), np.stack(disabled))
    assert "phase_signal_callback" not in model.calls[0]
    assert "phase_signal_callback" not in model.calls[1]
    assert controller.clear_calls == 1
    assert controller.set_calls == []


def test_enabled_phase_depth_installs_plan_before_exit_and_updates_after_action(
    monkeypatch, tmp_path: Path
):
    _set_import_environment(monkeypatch)
    module = importlib.import_module("robot_experiments.libero.exit_vla_utils")
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()
    controller = FakeExitController()
    runtime = FakePhaseDepthRuntime()
    context = {
        "episode_id": "libero_spatial:task0:episode0",
        "step_id": 10,
        "task_id": 0,
        "previous_action": [0.1, 0.0, 0.0, 0.0, 0.2, 0.0, 1.0],
    }
    output_path = tmp_path / "phase_depth.jsonl"

    with SafeJSONLTelemetryLogger(output_path, flush_every=1) as logger:
        actions = module.get_vla_action(
            _config(),
            model,
            torch.device("cpu"),
            _observation(),
            "pick up the bowl",
            exit_controller=controller,
            telemetry_logger=logger,
            telemetry_context=context,
            phase_depth_runtime=runtime,
            phase_depth_context=context,
        )

    assert len(actions) == 8
    assert controller.clear_calls == 1
    assert len(controller.set_calls) == 1
    assert model.phase_plan_active_during_predict == [True]
    assert len(runtime.prepare_calls) == 1
    assert len(runtime.update_calls) == 1
    # The only history update happens after prepare_plan has returned.
    assert runtime.prepare_calls[0]["context"] == context
    np.testing.assert_array_equal(
        runtime.prepare_calls[0]["normalized_proprio"], _observation()["state"]
    )
    torch.testing.assert_close(
        runtime.prepare_calls[0]["instruction_summary"],
        torch.full((1, 8), 2.5),
    )
    np.testing.assert_array_equal(
        runtime.update_calls[0]["normalized_action_chunk"], model.action.numpy()[0]
    )

    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["profile_id"] == 2
    assert record["progress"] == pytest.approx(0.4)
    assert record["boundary_prob"] == pytest.approx(0.2)
    assert record["uncertainty"] == pytest.approx(0.1)
    assert record["extra"]["phase_plan"]["min_exit_layer"] == 5
    assert record["extra"]["phase_plan"]["profile_reason"] == "test_phase"


def test_width_only_phase_plan_does_not_change_exit_depth(monkeypatch, tmp_path: Path):
    _set_import_environment(monkeypatch)
    module = importlib.import_module("robot_experiments.libero.exit_vla_utils")
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()
    controller = FakeExitController()
    runtime = FakePhaseDepthRuntime()
    context = {
        "episode_id": "libero_spatial:task0:episode0",
        "step_id": 10,
        "task_id": 0,
    }
    output_path = tmp_path / "phase_width_only.jsonl"

    with SafeJSONLTelemetryLogger(output_path, flush_every=1) as logger:
        actions = module.get_vla_action(
            _config(),
            model,
            torch.device("cpu"),
            _observation(),
            "pick up the bowl",
            exit_controller=controller,
            telemetry_logger=logger,
            telemetry_context=context,
            phase_depth_runtime=runtime,
            phase_depth_context=context,
            apply_phase_depth_plan=False,
        )

    np.testing.assert_array_equal(np.stack(actions), model.action.numpy()[0])
    assert len(runtime.prepare_calls) == 1
    assert len(runtime.update_calls) == 1
    assert controller.set_calls == []
    assert model.phase_plan_active_during_predict == [False]
    record = json.loads(output_path.read_text(encoding="utf-8"))
    assert record["extra"]["phase_plan"]["profile_name"] == "B2"


def test_phase_depth_failure_is_isolated_from_action_generation(monkeypatch):
    _set_import_environment(monkeypatch)
    module = importlib.import_module("robot_experiments.libero.exit_vla_utils")
    _patch_policy_dependencies(monkeypatch, module)
    model = FakeModel()
    controller = FakeExitController()
    runtime = FailingPhaseDepthRuntime()
    context = {"episode_id": "episode-0", "step_id": 0}

    actions = module.get_vla_action(
        _config(),
        model,
        torch.device("cpu"),
        _observation(),
        "pick up the bowl",
        exit_controller=controller,
        phase_depth_runtime=runtime,
        phase_depth_context=context,
    )

    np.testing.assert_array_equal(np.stack(actions), model.action.numpy()[0])
    assert model.phase_plan_active_during_predict == [False]
    assert controller.clear_calls == 2
    assert controller.set_calls == []
    assert len(runtime.update_calls) == 1
