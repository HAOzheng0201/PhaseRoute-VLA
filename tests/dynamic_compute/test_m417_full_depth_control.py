import numpy as np

from scripts.dynamic_compute.collect_m417_full_depth_task import (
    control_status_ok,
    sha256_array,
)


def test_initial_state_hash_includes_shape_and_dtype():
    values = np.arange(8, dtype=np.float32)

    assert sha256_array(values) == sha256_array(values.copy())
    assert sha256_array(values) != sha256_array(values.reshape(2, 4))
    assert sha256_array(values) != sha256_array(values.astype(np.float64))


def test_success_is_not_required_for_engineering_pass():
    assert control_status_ok(
        requested_episodes=1,
        completed_episodes=1,
        successes=0,
        policy_calls=28,
        action_chunk_lengths=[8] * 28,
        model_class="a1.vla.affordvla.AffordVLA",
    )


def test_early_exit_model_cannot_masquerade_as_full_depth_control():
    assert not control_status_ok(
        requested_episodes=1,
        completed_episodes=1,
        successes=1,
        policy_calls=10,
        action_chunk_lengths=[8] * 10,
        model_class="a1.vla.affordvla_early_exit.AffordVLAEarlyExit",
    )
