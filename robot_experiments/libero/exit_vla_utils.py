import os
import json
import random
import time
from typing import Any, Dict, List, Mapping, Optional, Tuple, Union, Callable

import torch
import numpy as np
import requests
import tensorflow as tf

from PIL import Image

from huggingface_hub import HfApi, hf_hub_download

from launch_scripts.utils import DEFAULT_VISION_BACKBONE

from a1.data.vla.rlds_datasets import RLDSBatchTransform
from a1.data import build_mm_preprocessor
from a1.data.collator import MMCollatorForAction


from a1.data.vla.rlds.utils.data_utils import NormalizationType
from a1.vla.dynamic_compute.telemetry import (
    TelemetryLogger,
    build_policy_call_telemetry,
)
from a1.vla.dynamic_compute.phase_cache import PhaseCacheWriter
from a1.vla.dynamic_compute.phase_depth_runtime import SafePhaseDepthRuntime
from a1.vla.dynamic_compute.vision_aggregation import StaticVisionAggregationConfig
from a1.vla.dynamic_compute.vision_teacher_cache import VisionTeacherCacheWriter
from a1.vla.dynamic_compute.v3.active_runtime import ActivePhaseRouteRuntime

# 尝试导入torch.profiler用于FLOPs统计

from torch.profiler import profile, ProfilerActivity


def model_is_on_hf_hub(model_path: str) -> bool:
    """Checks whether a model path points to a model on Hugging Face Hub."""
    # If the API call below runs without error, the model is on the hub
    try:
        HfApi().model_info(model_path)
        return True
    except Exception:
        return False
    

def _load_dataset_stats( checkpoint_path: str) -> None:
    """
    Load dataset statistics used during training for action normalization.

    Args:
        vla: The VLA model
        checkpoint_path: Path to the checkpoint directory
    """
    if model_is_on_hf_hub(checkpoint_path):
        # Download dataset stats directly from HF Hub
        dataset_statistics_path = hf_hub_download(
            repo_id=checkpoint_path,
            filename="dataset_statistics.json",
        )
    else:
        # 如果传入的是文件（如 ckpt_stepXXXX.pt），则使用其所在目录
        base_path = os.path.dirname(checkpoint_path) if os.path.isfile(checkpoint_path) else checkpoint_path
        dataset_statistics_path = os.path.join(base_path, "dataset_statistics.json")
    
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        # vla.norm_stats = norm_stats
        return norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

def check_unnorm_key(cfg, norm_stats) -> None:
    """Check that the model contains the action un-normalization key."""
    # Initialize unnorm_key
    unnorm_key = cfg.task_suite_name

    # In some cases, the key must be manually modified (e.g. after training on a modified version of the dataset
    # with the suffix "_no_noops" in the dataset name)
    if unnorm_key not in norm_stats and f"{unnorm_key}_no_noops" in norm_stats:
        unnorm_key = f"{unnorm_key}_no_noops"

    assert unnorm_key in norm_stats, f"Action un-norm key {unnorm_key} not found in VLA `norm_stats`!"

    # Set the unnorm_key in cfg
    cfg.unnorm_key = unnorm_key


def normalize_proprio(proprio: np.ndarray, norm_stats: Dict[str, Any], normalization_type: NormalizationType) -> np.ndarray:
    """
    Normalize proprioception data to match training distribution.

    Args:
        proprio: Raw proprioception data
        norm_stats: Normalization statistics

    Returns:
        np.ndarray: Normalized proprioception data
    """
    if normalization_type == NormalizationType.BOUNDS:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["min"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["max"]), np.array(norm_stats["min"])
    elif normalization_type == NormalizationType.BOUNDS_Q99:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["q01"], dtype=bool))
        proprio_high, proprio_low = np.array(norm_stats["q99"]), np.array(norm_stats["q01"])
    elif normalization_type == NormalizationType.NORMAL:
        mask = norm_stats.get("mask", np.ones_like(norm_stats["mean"], dtype=bool))
        mean = np.array(norm_stats["mean"])  # E[x]
        std = np.array(norm_stats["std"])    # sqrt(Var[x])
        normalized_proprio = np.where(mask, (proprio - mean) / (std + 1e-8), proprio)
        return normalized_proprio
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

    normalized_proprio = np.clip(
        np.where(
            mask,
            2 * (proprio - proprio_low) / (proprio_high - proprio_low + 1e-8) - 1,
            proprio,
        ),
        a_min=-1.0,
        a_max=1.0,
    )

    return normalized_proprio


def resize_image_for_policy(img: np.ndarray, resize_size: Union[int, Tuple[int, int]]) -> np.ndarray:
    """
    Resize an image to match the policy's expected input size.

    Uses the same resizing scheme as in the training data pipeline for distribution matching.

    Args:
        img: Numpy array containing the image
        resize_size: Target size as int (square) or (height, width) tuple

    Returns:
        np.ndarray: The resized image
    """
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)

    # Resize using the same pipeline as in RLDS dataset builder
    img = tf.image.encode_jpeg(img)  # Encode as JPEG
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)

    return img.numpy()

def crop_and_resize(image: tf.Tensor, crop_scale: float, batch_size: int) -> tf.Tensor:
    """
    Center-crop an image and resize it back to original dimensions.

    Uses the same logic as in the training data pipeline for distribution matching.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) with values in [0,1]
        crop_scale: Area of center crop relative to original image
        batch_size: Batch size

    Returns:
        tf.Tensor: The cropped and resized image
    """
    # Handle 3D inputs by adding batch dimension if needed
    assert image.shape.ndims in (3, 4), "Image must be 3D or 4D tensor"
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Calculate crop dimensions (note: we use sqrt(crop_scale) for h/w)
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Create bounding box for the crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Apply crop and resize
    image = tf.image.crop_and_resize(
        # image, bounding_boxes, tf.range(batch_size), (OPENVLA_IMAGE_SIZE, OPENVLA_IMAGE_SIZE)
        image, bounding_boxes, tf.range(batch_size), DEFAULT_VISION_BACKBONE.image_default_input_size
    )

    # Remove batch dimension if it was added
    if expanded_dims:
        image = image[0]

    return image


def center_crop_image(image: Union[np.ndarray, Image.Image]) -> Image.Image:
    """
    Center crop an image to match training data distribution.

    Args:
        image: Input image (PIL or numpy array)

    Returns:
        Image.Image: Cropped PIL Image
    """
    batch_size = 1
    crop_scale = 0.9

    # Convert to TF Tensor if needed
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(np.array(image))

    orig_dtype = image.dtype

    # Convert to float32 in range [0,1]
    image = tf.image.convert_image_dtype(image, tf.float32)

    # Apply center crop and resize
    image = crop_and_resize(image, crop_scale, batch_size)

    # Convert back to original data type
    image = tf.clip_by_value(image, 0, 1)
    image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)

    # Convert to PIL Image
    return Image.fromarray(image.numpy()).convert("RGB")


def check_image_format(image: Any) -> None:
    """
    Validate input image format.

    Args:
        image: Image to check

    Raises:
        AssertionError: If image format is invalid
    """
    is_numpy_array = isinstance(image, np.ndarray)
    has_correct_shape = len(image.shape) == 3 and image.shape[-1] == 3
    has_correct_dtype = image.dtype == np.uint8

    assert is_numpy_array and has_correct_shape and has_correct_dtype, (
        "Incorrect image format detected! Make sure that the input image is a "
        "numpy array with shape (H, W, 3) and dtype np.uint8!"
    )

def prepare_images_for_vla(images: List[np.ndarray], cfg: Any, image_size: Tuple[int, int]) -> List[Image.Image]:
    """
    Prepare images for VLA input by resizing and cropping as needed.

    Args:
        images: List of input images as numpy arrays
        cfg: Configuration object with parameters

    Returns:
        List[Image.Image]: Processed images ready for the model
    """
    processed_images = []

    for image in images:
        # Validate format
        check_image_format(image)

        # Resize if needed
        if image.shape != (image_size[0], image_size[1], 3):
            image = resize_image_for_policy(image, image_size)

        # Convert to PIL image
        pil_image = Image.fromarray(image).convert("RGB")

        # Apply center crop if configured
        if cfg.center_crop:
            pil_image = center_crop_image(pil_image)

        processed_images.append(pil_image)

    return processed_images

def get_action_stats(norm_stats,unnorm_key: Optional[str] = None) -> Dict[str, Any]:
    """Get all the logged statistics for the given dataset."""
    def _check_unnorm_key(norm_stats: Dict[str, Dict[str, Any]], unnorm_key: Optional[str]) -> str:
        """Validate and resolve the unnormalization key for action statistics"""
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, "
                f"please pass a `unnorm_key` from the following options to choose the statistics "
                f"used for un-normalizing actions: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))

        assert unnorm_key in norm_stats, (
            f"The `unnorm_key` you chose is not in the set of available dataset statistics, "
            f"please choose from: {norm_stats.keys()}"
        )
        return unnorm_key
    unnorm_key = _check_unnorm_key(norm_stats, unnorm_key)
    return norm_stats[unnorm_key]["action"] if "action" in norm_stats[unnorm_key] else norm_stats[unnorm_key]["actions"]


def _unnormalize_actions(normalized_actions, norm_stats, normalization_type, unnorm_key=None):
    """Unnormalize actions using dataset statistics"""
    action_norm_stats = get_action_stats(norm_stats, unnorm_key)

    if normalization_type == NormalizationType.BOUNDS:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["max"]), np.array(action_norm_stats["min"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions
    elif normalization_type == NormalizationType.BOUNDS_Q99:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
        action_high, action_low = np.array(action_norm_stats["q99"]), np.array(action_norm_stats["q01"])
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low + 1e-8) + action_low,
            normalized_actions,
        )
        return actions
    elif normalization_type == NormalizationType.NORMAL:
        mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["mean"], dtype=bool))
        mean = np.array(action_norm_stats["mean"])  # E[x]
        std = np.array(action_norm_stats["std"])    # sqrt(Var[x])
        actions = np.where(mask, normalized_actions * (std + 1e-8) + mean, normalized_actions)
        return actions
    else:
        raise ValueError("Unsupported action/proprio normalization type detected!")

def convert_gripper_qpos_to_1d(gripper_qpos_2d):
    """将2维夹爪关节位置转换为夹爪开合距离"""
    return abs(gripper_qpos_2d[:,0] - gripper_qpos_2d[:,1])

def get_vla_action(
    cfg: Any,
    model: torch.nn.Module,
    device,
    obs: Dict[str, Any],
    task_label: str,
    exit_controller=None,
    output_hidden_states = False,
    log_fn: Optional[Callable[[str], None]] = None,
    force_profile: bool = False,
    telemetry_logger: Optional[TelemetryLogger] = None,
    telemetry_context: Optional[Mapping[str, Any]] = None,
    phase_cache_writer: Optional[PhaseCacheWriter] = None,
    phase_cache_context: Optional[Mapping[str, Any]] = None,
    phase_depth_runtime: Optional[SafePhaseDepthRuntime] = None,
    phase_depth_context: Optional[Mapping[str, Any]] = None,
    vision_aggregation_config: Optional[StaticVisionAggregationConfig] = None,
    learnable_vision_aggregator: Optional[torch.nn.Module] = None,
    vision_teacher_cache_writer: Optional[VisionTeacherCacheWriter] = None,
    vision_teacher_cache_context: Optional[Mapping[str, Any]] = None,
    apply_phase_depth_plan: bool = True,
    phase_route_runtime: Optional[ActivePhaseRouteRuntime] = None,
    phase_route_context: Optional[Mapping[str, Any]] = None,
) -> Tuple[List[np.ndarray], float, float]:
    """
    Generate action predictions with the VLA policy.

    Args:
        cfg: Configuration object with parameters
        vla: The VLA model
        processor: Model processor for inputs
        obs: Observation dictionary
        task_label: Text description of the task

    Returns:
        List[np.ndarray]: Predicted actions
        float: TFLOPs
        float: Inference time (seconds)
    """
    with torch.inference_mode():
        telemetry_enabled = bool(
            telemetry_logger is not None and getattr(telemetry_logger, "enabled", False)
        )
        phase_cache_enabled = bool(
            phase_cache_writer is not None
            and getattr(phase_cache_writer, "enabled", False)
        )
        vision_teacher_cache_enabled = bool(
            vision_teacher_cache_writer is not None
            and getattr(vision_teacher_cache_writer, "enabled", False)
        )
        phase_depth_enabled = bool(
            phase_depth_runtime is not None
            and getattr(phase_depth_runtime, "enabled", False)
            and exit_controller is not None
            and hasattr(exit_controller, "set_phase_plan")
        )
        phase_route_enabled = bool(
            phase_route_runtime is not None
            and getattr(phase_route_runtime, "enabled", False)
            and exit_controller is not None
            and getattr(exit_controller, "phase_route_runtime_adapter", None)
            is getattr(phase_route_runtime, "adapter", None)
        )
        learned_vision_aggregation_enabled = learnable_vision_aggregator is not None
        if learned_vision_aggregation_enabled and vision_aggregation_config is not None:
            raise ValueError(
                "static and learned vision aggregation cannot be enabled together"
            )
        # A plan is valid for exactly one policy call.  Clear any stale state
        # even when this call has dynamic depth disabled.
        if exit_controller is not None and hasattr(exit_controller, "clear_phase_plan"):
            try:
                exit_controller.clear_phase_plan()
            except Exception:
                phase_depth_enabled = False
        if phase_depth_enabled and hasattr(phase_depth_runtime, "clear_current_plan"):
            try:
                phase_depth_runtime.clear_current_plan()
            except Exception:
                phase_depth_enabled = False
        telemetry_events: List[Dict[str, Any]] = []
        raw_proprio = (
            np.asarray(obs["state"]).copy()
            if (
                telemetry_enabled
                or phase_cache_enabled
                or phase_depth_enabled
                or vision_teacher_cache_enabled
                or phase_route_enabled
            )
            and "state" in obs
            else None
        )
        phase_signal_payload: Dict[str, Any] = {}
        vision_teacher_payload: Dict[str, Any] = {}
        fm_trace_payloads: List[Dict[str, Any]] = []
        phase_instruction_summary: Optional[torch.Tensor] = None
        normalized_proprio_for_phase = None

        def vision_teacher_feature_callback(payload: Mapping[str, Any]) -> None:
            if phase_route_enabled:
                phase_route_runtime.capture_visual_features(payload)
            if not vision_teacher_cache_enabled:
                return
            projected = payload.get("projected_features")
            positions = payload.get("image_input_idx")
            if not isinstance(projected, torch.Tensor) or projected.ndim != 4:
                raise ValueError("projected_features must have shape [B, C, M, D]")
            if not isinstance(positions, torch.Tensor) or positions.shape != projected.shape[:3]:
                raise ValueError("image_input_idx must align with projected_features")
            if projected.shape[0] != 1:
                raise ValueError("LIBERO teacher cache requires batch size 1")
            vision_teacher_payload["projected_features"] = (
                projected[0].detach().to(device="cpu", dtype=torch.float16).numpy()
            )
            vision_teacher_payload["image_input_idx"] = (
                positions[0].detach().to(device="cpu", dtype=torch.int32).numpy()
            )

        def fm_trace_callback(payload: Mapping[str, Any]) -> None:
            input_x = payload.get("input_x")
            output_action = payload.get("output_action")
            if not isinstance(input_x, torch.Tensor) or input_x.ndim != 3:
                raise ValueError("FM input_x must have shape [B, H, A]")
            if not isinstance(output_action, torch.Tensor) or output_action.shape != input_x.shape:
                raise ValueError("FM output_action must align with input_x")
            if input_x.shape[0] != 1:
                raise ValueError("LIBERO FM trace collection requires batch size 1")
            fm_trace_payloads.append(
                {
                    "candidate_layer": int(payload["candidate_layer"]),
                    "candidate_role": str(payload["candidate_role"]),
                    "fm_steps": int(payload["fm_steps"]),
                    "input_x": input_x[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy(),
                    "output_action": output_action[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy(),
                }
            )

        def telemetry_event_callback(event_name: str, payload: Mapping[str, Any]) -> None:
            telemetry_events.append({"event": event_name, **dict(payload)})
            if phase_route_enabled:
                phase_route_runtime.record_route_event(event_name, payload)

        def phase_signal_callback(payload: Mapping[str, Any]) -> None:
            if phase_route_enabled:
                phase_route_runtime.prepare_policy_call()
            summaries = {}
            for name in ("visual_summary",):
                value = payload.get(name)
                if not isinstance(value, torch.Tensor):
                    raise ValueError(f"{name} was not emitted as a tensor")
                array = value.detach().to(device="cpu", dtype=torch.float32).numpy()
                if array.shape[0] != 1:
                    raise ValueError("LIBERO phase-cache collection expects batch size 1")
                summaries[name] = array[0]
            for name in ("visual_token_count",):
                value = payload.get(name)
                if not isinstance(value, torch.Tensor) or value.numel() != 1:
                    raise ValueError(f"{name} was not emitted as one tensor scalar")
                summaries[name] = int(value.detach().cpu().item())
            # Test doubles and future backends may emit a precomputed raw-task
            # summary.  The real A1 path below overwrites it using task-label
            # token IDs so multimodal template tokens cannot leak into e_l.
            instruction_summary = payload.get("instruction_summary")
            instruction_count = payload.get("instruction_token_count")
            if (
                phase_instruction_summary is None
                and isinstance(instruction_summary, torch.Tensor)
            ):
                array = instruction_summary.detach().to(
                    device="cpu", dtype=torch.float32
                ).numpy()
                if array.shape[0] == 1:
                    summaries["instruction_summary"] = array[0]
            if (
                phase_instruction_summary is None
                and isinstance(instruction_count, torch.Tensor)
                and instruction_count.numel() == 1
            ):
                summaries["instruction_token_count"] = int(
                    instruction_count.detach().cpu().item()
                )
            phase_signal_payload.update(summaries)

            if phase_depth_enabled:
                try:
                    visual_summary = payload.get("visual_summary")
                    if not isinstance(visual_summary, torch.Tensor):
                        raise ValueError("visual_summary was not emitted as a tensor")
                    instruction_summary = phase_instruction_summary
                    if not isinstance(instruction_summary, torch.Tensor):
                        # Deliberately feed an invalid shape through the safe
                        # runtime so it installs its conservative B3 fallback.
                        instruction_summary = torch.empty(
                            (1, 0), device=visual_summary.device
                        )
                    runtime_context = dict(phase_depth_context or {})
                    plan = phase_depth_runtime.prepare_plan(
                        context=runtime_context,
                        visual_summary=visual_summary,
                        instruction_summary=instruction_summary,
                        normalized_proprio=normalized_proprio_for_phase,
                        previous_action=runtime_context.get("previous_action"),
                    )
                    if apply_phase_depth_plan:
                        exit_controller.set_phase_plan(
                            policy=phase_depth_runtime.exit_policy,
                            phase_state=plan.routing_phase_state,
                            budget=plan.budget,
                            profile_reason=plan.selection.reasons[0],
                        )
                    phase_signal_payload.update(
                        {
                            "phase_profile_id": plan.budget.profile_id,
                            "phase_profile_name": plan.budget.profile.name,
                            "phase_min_exit_layer": plan.budget.min_exit_layer,
                            "phase_progress": float(plan.phase_state.progress[0, 0]),
                            "phase_boundary_prob": float(
                                plan.phase_state.boundary_prob[0, 0]
                            ),
                            "phase_uncertainty": float(
                                plan.phase_state.uncertainty[0, 0]
                            ),
                            "phase_routing_boundary": float(
                                plan.routing_phase_state.boundary_prob[0, 0]
                            ),
                            "phase_boundary_rise": plan.boundary_rise,
                            "phase_boundary_crossed": plan.boundary_crossed,
                            "phase_latency_ms": plan.latency_ms,
                            "phase_fallback": plan.fallback,
                            "phase_error": plan.error,
                        }
                    )
                    if telemetry_enabled:
                        telemetry_event_callback(
                            "phase_plan",
                            {
                                "profile_id": plan.budget.profile_id,
                                "profile_name": plan.budget.profile.name,
                                "profile_reason": plan.selection.reasons[0],
                                "progress": float(plan.phase_state.progress[0, 0]),
                                "boundary_prob": float(
                                    plan.phase_state.boundary_prob[0, 0]
                                ),
                                "uncertainty": float(
                                    plan.phase_state.uncertainty[0, 0]
                                ),
                                "routing_boundary_prob": float(
                                    plan.routing_phase_state.boundary_prob[0, 0]
                                ),
                                "boundary_rise": plan.boundary_rise,
                                "boundary_crossed": plan.boundary_crossed,
                                "min_exit_layer": plan.budget.min_exit_layer,
                                "min_exit_rank": plan.budget.min_exit_rank,
                                "exit_threshold_scale": (
                                    plan.budget.profile.exit_threshold_scale
                                ),
                                "phase_latency_ms": plan.latency_ms,
                                "fallback": plan.fallback,
                                "error": plan.error,
                            },
                        )
                except Exception:
                    # Dynamic routing must never be able to interrupt robot
                    # control.  The cleared controller falls back to A1.
                    try:
                        exit_controller.clear_phase_plan()
                    except Exception:
                        pass

        # Collect all input images
        all_images = [obs["full_image"]]
        if cfg.num_images_in_input > 1:
            all_images.extend([obs[k] for k in obs.keys() if "wrist" in k])

        # Process images
        all_images = prepare_images_for_vla(all_images, cfg,image_size=model.config.vision_backbone.image_default_input_size)

        # Extract primary image and additional images
        image_primary = all_images.pop(0)
        image_wrist = all_images.pop(0)

        image_primary = np.array(image_primary)
        image_wrist = np.array(image_wrist) if image_wrist is not None else None

        prompt = f"{task_label.lower()}"
        
        norm_stats = _load_dataset_stats(cfg.pretrained_checkpoint)
        if norm_stats is None:
            raise FileNotFoundError("dataset_statistics.json 未找到，请确认检查点目录包含该文件或修改配置关闭归一化依赖。")
        # Process proprioception data if used
        proprio = None
        if cfg.use_proprio:
            proprio = obs["state"]
            check_unnorm_key(cfg, norm_stats)
            proprio_norm_stats = norm_stats[cfg.unnorm_key]["proprio"] ## 
            obs["state"] = normalize_proprio(proprio, proprio_norm_stats, cfg.normalization_type)
            proprio = obs["state"]
            proprio = torch.tensor(proprio, dtype=torch.float32).to(device).unsqueeze(0)  # 添加batch维度
            # 这里要和训练集对应，是否吧libero的proprio信息从8维改成了7维
            if model.config.proprio_dim == 7 :
                proprio_lastone = convert_gripper_qpos_to_1d(proprio[:,-2:])
                proprio = proprio[:,:-1] # 去掉最后一个关节位置
                proprio[:,-1] = proprio_lastone  # 保留最后一个夹爪关节位置作为夹爪开合距离
            proprio = proprio.unsqueeze(1)  # 添加时间步维度，变为 (batch_size, 1, proprio_dim)
            proprio = proprio.cpu().numpy()  # 转换为numpy数组

        normalized_proprio_for_phase = (
            np.asarray(proprio).reshape(-1)
            if proprio is not None
            else raw_proprio
        )

        # 使用与训练时相同的预处理器
        preprocessor = build_mm_preprocessor(
            model_config=model.config,
            for_inference=True,  # 设置为推理模式
            shuffle_messages=True,
            require_image_features=True,
            is_training=False,
            # is_training=True,  # 这里设置为True是因为我们需要使用训练时的预处理方式
        )
        # 构建输入数据 - 模拟训练时的数据格式
        dummy_action = np.zeros((model.config.num_actions_chunk, model.config.action_dim), dtype=np.float32)  # dummy action for inference
        input_data = {
            # "image": np.array(primary_image),
            "question": prompt,
            "proprio": proprio,  
            "action": dummy_action,
            "action_pad_mask": np.zeros_like(dummy_action, dtype=bool),
            "answer": "Action",  # 不起作用
            "style": "action",
            "metadata": {},
            
        }
        if cfg.use_wrist_image:
            # input_data["images"] = [image_primary, image_wrist]
            input_data["images"] = [ image_primary,image_wrist]
        else:
            input_data["image"] = image_primary

       # 通过预处理器处理
        processed_input = preprocessor(input_data)
        
        # 创建collator进行批处理
        collator = MMCollatorForAction(
                model_config=model.config,
                use_proprio=cfg.use_proprio,
                max_sequence_length=cfg.sequence_length, 
                include_metadata=False,
            pad="to_max", max_crops=model.config.get_max_crops()
        )
        # 批处理数据
        batch_data = collator([processed_input])
        if telemetry_enabled:
            active_token_count = int((batch_data["input_ids"][0] != -1).sum().item())
            image_input_idx = batch_data.get("image_input_idx")
            visual_token_count = (
                int((image_input_idx[0] >= 0).sum().item())
                if image_input_idx is not None
                else 0
            )
        else:
            active_token_count = 0
            visual_token_count = 0
        # print(f"******Batch data keys: {batch_data.keys()}")  # 调试输出
        # print(f"batch_data[input_ids].shape,{batch_data['input_ids'].shape}")  # 调试输出
        # print(f"batch_data[images].shape,{batch_data['images'].shape}")  # 调试输出
        
        # 移动到设备
        for key in batch_data:
            if isinstance(batch_data[key], torch.Tensor):
                batch_data[key] = batch_data[key].to(device)

        # 在推理前打印末尾 30 个位置的 position_ids 与对应的 input_ids，确认 action 段 position_ids 单调递增且长度匹配：
        # 在推理前打印动作段对应的 position_ids 与 input_ids，避免尾部 padding 的干扰
        # try:
        #     proprio_idx = int(batch_data["proprio_token_idx"][0].item()) if batch_data.get("proprio_token_idx") is not None else None
        # except Exception:
        #     proprio_idx = None
        # valid_mask = (batch_data["input_ids"][0] != -1)
        # valid_indices = valid_mask.nonzero(as_tuple=False)
        # if valid_indices.numel() > 0:
        #     last_valid_idx = int(valid_indices[-1].item())
        # else:
        #     last_valid_idx = 0

        # start_idx = proprio_idx + 1 if proprio_idx is not None else max(0, last_valid_idx - 30)
        # end_idx = last_valid_idx + 1

        # print("pos ids action:", batch_data["position_ids"][0, start_idx:end_idx])
        # print("ids action:", batch_data["input_ids"][0, start_idx:end_idx])
        ##

        # 准备模型输入
        model_inputs = {
            "input_ids": batch_data["input_ids"],
            "images": batch_data.get("images"),
            "image_masks": batch_data.get("image_masks"),
            "attention_mask":batch_data.get("attention_mask"),
            "attention_bias":batch_data.get("attention_bias"),
            "response_mask":(batch_data["loss_masks"] > 0) if "loss_masks" in batch_data else None,
            "image_input_idx": batch_data.get("image_input_idx"),
            "subsegment_ids": batch_data.get("subsegment_ids"),
            "position_ids": batch_data.get("position_ids"),
            # "target_actions": torch.zeros((1, NUM_ACTIONS_CHUNK, ACTION_DIM), device=device),
            "action_proprio":batch_data.get("proprio"),  
            "proprio_token_idx":  batch_data.get("proprio_token_idx"), 
            "output_hidden_states": output_hidden_states,
            "use_cache": True if model.config.action_head == 'flow_matching' else False,
        }
        # 准备捕获 exit id 的机制，用于 FLOPs 缓存键
        captured_exit_id = [None]
        def capturing_log_fn(msg):
            # 尝试解析 exit id，格式参考 eval_libero_early_exit.py
            key = "Exit by exit_controller, block_idx:"
            if key in msg:
                idx_str = msg.split(key)[-1].strip()
                idx = int(idx_str.split()[0].strip().strip(','))
                captured_exit_id[0] = idx

            if log_fn:
                log_fn(msg)

        if exit_controller is not None:
            model_inputs["exit_controller"] = exit_controller
            # 拦截 log_fn 以获取 exit_id
            model_inputs["log_fn"] = capturing_log_fn
        if telemetry_enabled or vision_teacher_cache_enabled or phase_route_enabled:
            model_inputs["telemetry_callback"] = telemetry_event_callback
        if phase_cache_enabled or phase_depth_enabled or phase_route_enabled:
            model_inputs["phase_signal_callback"] = phase_signal_callback
        if vision_aggregation_config is not None:
            model_inputs["vision_aggregation_config"] = vision_aggregation_config
        if learned_vision_aggregation_enabled:
            model_inputs["learnable_vision_aggregator"] = learnable_vision_aggregator
        if vision_teacher_cache_enabled or phase_route_enabled:
            model_inputs["vision_teacher_feature_callback"] = (
                vision_teacher_feature_callback
            )
        if vision_teacher_cache_enabled:
            model_inputs["fm_trace_callback"] = fm_trace_callback

        if (
            phase_cache_enabled
            or phase_depth_enabled
            or vision_teacher_cache_enabled
            or learned_vision_aggregation_enabled
            or phase_route_enabled
        ):
            try:
                # Use the raw task label rather than the multimodal template:
                # e_l must not contain image/action/proprio placeholder tokens.
                instruction_ids = model.tokenizer.encode(prompt)
                if not instruction_ids:
                    raise ValueError("task instruction tokenization returned no tokens")
                instruction_input_ids = torch.tensor(
                    instruction_ids,
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(0)
                phase_instruction_summary = model.transformer.wte(
                    instruction_input_ids
                ).mean(dim=1)
                phase_signal_payload["instruction_summary"] = (
                    phase_instruction_summary[0]
                    .detach()
                    .to(device="cpu", dtype=torch.float32)
                    .numpy()
                )
                phase_signal_payload["instruction_token_count"] = len(instruction_ids)
            except Exception:
                # Cache integrity checks or the safe runtime fallback handle a
                # missing instruction summary without changing A1's action.
                phase_instruction_summary = None
        if phase_route_enabled:
            phase_route_runtime.begin_policy_call(
                context=dict(phase_route_context or {}),
                instruction_summary=phase_instruction_summary,
                normalized_proprio=normalized_proprio_for_phase,
            )
        if learned_vision_aggregation_enabled:
            if phase_instruction_summary is None:
                raise RuntimeError(
                    "learned vision aggregation could not build instruction summary"
                )
            model_inputs["vision_instruction_summary"] = phase_instruction_summary
        # elif log_fn is not None:
        #      model_inputs["log_fn"] = log_fn
        # 检查是否有 FLOPs 缓存
        if not hasattr(get_vla_action, "_flops_cache"):
            get_vla_action._flops_cache = {}

        cpu_rng_state = (
            torch.get_rng_state().cpu().numpy().copy()
            if vision_teacher_cache_enabled
            else None
        )
        cuda_rng_state = (
            torch.cuda.get_rng_state(device).cpu().numpy().copy()
            if vision_teacher_cache_enabled and torch.cuda.is_available()
            else None
        )

        if force_profile:
            assert profile is not None, "profile is not found"
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            
            # 重置 capture id
            captured_exit_id[0] = None
            
            # Run with profiler
            # Record time for consistency, though profiling overhead affects it
            start_time = time.time()
            with profile(activities=activities, record_shapes=False, with_stack=False, profile_memory=False, with_flops=True) as prof:
                normalized_actions = model.predict_actions(**model_inputs)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            
            total_flops = 0
            for evt in prof.key_averages():
                fl = getattr(evt, "flops", None)
                if isinstance(fl, (int, float)):
                    total_flops += fl
            
            total_tflops = total_flops / 1e12 if total_flops > 0 else 0
            
            # Cache the result
            exit_id = captured_exit_id[0]
            cache_key = exit_id if exit_id is not None else "default"
            if total_tflops > 0:
                get_vla_action._flops_cache[cache_key] = total_tflops
                
                # 记录日志
                tflops_msg = f"TFLOPs per inference (Exit Layer {cache_key}): {total_tflops:.4f}"
                if log_fn:
                    log_fn(tflops_msg)

        else:
            # 进行一次正常推理（不开启 Profiler），获取结果和可能的 Exit ID
            start_time = time.time()
            normalized_actions = model.predict_actions(**model_inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            end_time = time.time()
            inference_time = end_time - start_time
            
            total_tflops = 0

        # if predicted_actions is None:
        #     raise ValueError("Model did not return predicted actions")
        if phase_route_enabled:
            phase_route_runtime.commit_selected_action(normalized_actions)
        
        if telemetry_enabled:
            action_shape = list(normalized_actions.shape)
            action_dtype = str(normalized_actions.dtype)
        normalized_actions = normalized_actions.to(torch.float32)  # 确保是float32格式
        normalized_actions = normalized_actions.cpu().numpy()

        if phase_depth_enabled:
            try:
                runtime_context = dict(phase_depth_context or {})
                phase_depth_runtime.update_after_action(
                    context=runtime_context,
                    normalized_proprio=normalized_proprio_for_phase,
                    normalized_action_chunk=(
                        normalized_actions[0]
                        if normalized_actions.ndim == 3
                        else normalized_actions
                    ),
                )
            except Exception:
                # History is auxiliary state; a failed update cannot affect
                # the action that has already been generated.
                pass
        
        # 反归一化
        actions = _unnormalize_actions(normalized_actions, norm_stats, cfg.normalization_type, cfg.unnorm_key)

        # 确保输出格式正确
        if actions.ndim == 3:  # (batch, seq, action_dim)
            actions = actions[0]  # 取第一个batch

        if phase_cache_enabled:
            try:
                cache_context = dict(phase_cache_context or {})
                phase_cache_writer.log_call(
                    context=cache_context,
                    instruction=task_label,
                    raw_proprio=raw_proprio,
                    normalized_proprio=(
                        np.asarray(proprio).reshape(-1)
                        if proprio is not None
                        else raw_proprio
                    ),
                    previous_action=cache_context.get("previous_action"),
                    normalized_action_chunk=(
                        normalized_actions[0]
                        if normalized_actions.ndim == 3
                        else normalized_actions
                    ),
                    action_chunk=actions,
                    visual_summary=phase_signal_payload.get("visual_summary"),
                    instruction_summary=phase_signal_payload.get("instruction_summary"),
                    visual_token_count=int(
                        phase_signal_payload.get("visual_token_count", 0)
                    ),
                    instruction_token_count=int(
                        phase_signal_payload.get("instruction_token_count", 0)
                    ),
                )
            except Exception:
                # A third-party writer must not be able to alter robot control.
                pass

        if vision_teacher_cache_enabled:
            try:
                cache_context = dict(vision_teacher_cache_context or {})
                exit_layer = captured_exit_id[0]
                matching_traces = [
                    trace
                    for trace in fm_trace_payloads
                    if trace["candidate_role"] == "candidate_action"
                    and trace["candidate_layer"] == exit_layer
                ]
                if len(matching_traces) != 1:
                    raise ValueError(
                        "expected exactly one candidate-action FM trace for "
                        f"exit layer {exit_layer}, found {len(matching_traces)}"
                    )
                teacher_trace = matching_traces[0]

                def cache_sequence_tensor(value, dtype):
                    if value is None:
                        return np.empty((0,), dtype=dtype)
                    if not isinstance(value, torch.Tensor):
                        raise TypeError("cached sequence field must be a tensor or None")
                    array = value.detach().to(device="cpu").numpy()
                    if array.ndim > 0 and array.shape[0] == 1:
                        array = array[0]
                    return np.asarray(array, dtype=dtype)

                vision_teacher_cache_writer.log_call(
                    context=cache_context,
                    instruction=task_label,
                    projected_features=vision_teacher_payload.get(
                        "projected_features"
                    ),
                    image_input_idx=vision_teacher_payload.get("image_input_idx"),
                    instruction_summary=(
                        phase_instruction_summary[0]
                        .detach()
                        .to(device="cpu", dtype=torch.float32)
                        .numpy()
                        if isinstance(phase_instruction_summary, torch.Tensor)
                        else None
                    ),
                    normalized_proprio=normalized_proprio_for_phase,
                    teacher_normalized_action=(
                        normalized_actions[0]
                        if normalized_actions.ndim == 3
                        else normalized_actions
                    ),
                    teacher_action=actions,
                    cpu_rng_state=cpu_rng_state,
                    cuda_rng_state=cuda_rng_state,
                    teacher_exit_layer=exit_layer,
                    fm_calls=sum(
                        int(event.get("fm_calls", 0)) for event in telemetry_events
                    ),
                    fm_steps_total=sum(
                        int(event.get("fm_steps", 0)) for event in telemetry_events
                    ),
                    input_ids=cache_sequence_tensor(
                        batch_data.get("input_ids"), np.int64
                    ),
                    attention_mask=cache_sequence_tensor(
                        batch_data.get("attention_mask"), np.bool_
                    ),
                    attention_bias=cache_sequence_tensor(
                        batch_data.get("attention_bias"), np.float32
                    ),
                    response_mask=cache_sequence_tensor(
                        (batch_data["loss_masks"] > 0)
                        if "loss_masks" in batch_data
                        else None,
                        np.bool_,
                    ),
                    subsegment_ids=cache_sequence_tensor(
                        batch_data.get("subsegment_ids"), np.int64
                    ),
                    position_ids=cache_sequence_tensor(
                        batch_data.get("position_ids"), np.int64
                    ),
                    action_proprio=cache_sequence_tensor(
                        batch_data.get("proprio"), np.float32
                    ),
                    proprio_token_idx=np.asarray(
                        cache_sequence_tensor(
                            batch_data.get("proprio_token_idx"), np.int64
                        )
                    ).reshape(-1),
                    teacher_exit_input_x=teacher_trace["input_x"],
                    teacher_exit_trace_action=teacher_trace["output_action"],
                    fm_trace_count=len(fm_trace_payloads),
                    fm_trace_layers=np.asarray(
                        [trace["candidate_layer"] for trace in fm_trace_payloads],
                        dtype=np.int16,
                    ),
                    fm_trace_roles=np.asarray(
                        [
                            1 if trace["candidate_role"] == "candidate_action" else 0
                            for trace in fm_trace_payloads
                        ],
                        dtype=np.uint8,
                    ),
                    fm_trace_steps=np.asarray(
                        [trace["fm_steps"] for trace in fm_trace_payloads],
                        dtype=np.int16,
                    ),
                    fm_trace_input_x=np.stack(
                        [trace["input_x"] for trace in fm_trace_payloads]
                    ),
                    fm_trace_output_action=np.stack(
                        [trace["output_action"] for trace in fm_trace_payloads]
                    ),
                )
            except Exception:
                # Teacher collection is a side channel and cannot alter action.
                pass

        if telemetry_enabled:
            candidate_exit_layers = (
                list(exit_controller.exit_id_list)
                if exit_controller is not None
                and hasattr(exit_controller, "exit_id_list")
                else (
                    list(model.get_all_exit_idx(getattr(cfg, "exit_interval", 2)))
                    if hasattr(model, "get_all_exit_idx")
                    else []
                )
            )
            record = build_policy_call_telemetry(
                context=telemetry_context,
                instruction=task_label,
                raw_proprio=raw_proprio,
                active_token_count=active_token_count,
                n_layers=int(model.config.n_layers),
                visual_token_count=visual_token_count,
                candidate_exit_layers=candidate_exit_layers,
                telemetry_events=telemetry_events,
                latency_ms=float(inference_time * 1000.0),
                action_shape=action_shape,
                action_dtype=action_dtype,
                normalization_key=cfg.unnorm_key,
                fallback_exit_layer=captured_exit_id[0],
            )
            try:
                telemetry_logger.log(record)
            except Exception:
                # A third-party logger must not be able to alter robot control.
                pass

        return [actions[i] for i in range(min(len(actions), cfg.num_open_loop_steps))]
