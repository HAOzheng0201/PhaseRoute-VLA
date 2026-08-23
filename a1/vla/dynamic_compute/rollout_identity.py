"""Small identity helper shared by rollout observers and cache writers."""

from __future__ import annotations


def resolve_policy_episode_id(
    task_suite_name: object,
    task_id: object,
    episode_idx: object,
    episode_id_override: str | None = None,
) -> str:
    """Preserve legacy IDs unless an explicit nonempty identity is supplied."""

    if episode_id_override is None:
        return f"{task_suite_name}:task{task_id}:episode{episode_idx}"
    if not isinstance(episode_id_override, str) or not episode_id_override:
        raise ValueError("episode_id_override must be a nonempty string or None")
    return episode_id_override


__all__ = ["resolve_policy_episode_id"]
