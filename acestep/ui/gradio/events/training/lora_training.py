"""LoRA training handlers for the training UI.

Contains functions for starting LoRA training, stopping training,
opening output folders, and exporting trained LoRA weights.
"""

import json
import os
import re
import time
from typing import Dict, Tuple

from loguru import logger

from acestep.gpu_config import get_global_gpu_config
from acestep.training.lora_naming import validate_lora_name
from acestep.training.lora_output_paths import (
    resolve_lora_export_root,
    resolve_lora_training_output_dir,
)
from acestep.training.path_inputs import normalize_user_path
from acestep.training.path_safety import safe_path
from acestep.ui.gradio.events.local_path_dialogs import open_folder_path
from acestep.ui.gradio.i18n import t
from .service_auto_init import ensure_dit_ready
from .subprocess_control import request_training_subprocess_stop
from .step_estimate import estimate_lora_total_steps
from .training_progress_stats import TrainingProgressTimer, build_training_progress_text
from .training_utils import (
    _format_duration,
    _training_loss_figure,
)


def _as_bool(value) -> bool:
    """Coerce Gradio checkbox values to bool."""

    return bool(value)


def _as_positive_int(value, default: int) -> int:
    """Coerce a Gradio numeric value to a positive int."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, parsed)


def _as_nonnegative_int(value, default: int) -> int:
    """Coerce a Gradio numeric value to a non-negative int."""

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(0, parsed)


def _uses_fp8_scaled(base_quantization: str) -> bool:
    """Return whether the selected frozen-base quantization is scaled FP8."""

    return str(base_quantization or "").strip().casefold() == "fp8 scaled"


def _model_config_token(value: object) -> str:
    """Return a comparable model identifier from a UI model path/name."""

    text = str(value or "").strip().replace("\\", "/").rstrip("/")
    if not text:
        return ""
    return text.rsplit("/", 1)[-1].casefold()


def _sample_model_mismatch(
    training_model_config: object,
    sample_model_config: object,
) -> str:
    """Return an error message when sample and training base models differ."""

    training_token = _model_config_token(training_model_config)
    sample_token = _model_config_token(sample_model_config)
    if not training_token or not sample_token or training_token == sample_token:
        return ""
    return (
        "Sample generation uses the Advanced tab base model, but it does not match "
        f"the LoRA training base model. Advanced model: {sample_model_config}; "
        f"training model: {training_model_config}. Select the same base model before "
        "starting training with checkpoint samples enabled."
    )


def _save_training_config_snapshot(lora_config, training_config) -> None:
    """Persist the LoRA and training config used for the run."""

    output_dir = safe_path(training_config.output_dir)
    training_config.output_dir = output_dir
    os.makedirs(output_dir, exist_ok=True)
    payload = {
        "lora": lora_config.to_dict(),
        "training": training_config.to_dict(),
    }
    path = os.path.join(output_dir, "training_run_config.json")
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    training_config.save_json(os.path.join(output_dir, "training_config.json"))


def open_lora_output_folder(lora_output_dir: str, lora_name: str = "") -> str:
    """Open the selected LoRA output folder or named LoRA run folder.

    Args:
        lora_output_dir: User-selected parent directory for LoRA runs.
        lora_name: Optional LoRA training name used for the per-run folder.

    Returns:
        User-facing status from the platform folder opener.
    """

    normalized_output_dir = normalize_user_path(lora_output_dir)
    if not normalized_output_dir:
        return "Please enter a LoRA output directory path."

    normalized_name = str(lora_name or "").strip()
    try:
        if normalized_name:
            target_dir = resolve_lora_training_output_dir(
                normalized_output_dir,
                normalized_name,
            )
        else:
            target_dir = safe_path(normalized_output_dir)
    except ValueError as exc:
        return f"Invalid LoRA output folder: {exc}"

    return open_folder_path(target_dir)


def start_training(
    tensor_dir: str,
    dit_handler,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    learning_rate: float,
    train_epochs: int,
    train_batch_size: int,
    gradient_accumulation: int,
    save_every_n_epochs: int,
    training_shift: float,
    training_seed: int,
    lora_output_dir: str,
    resume_checkpoint_dir: str,
    training_state: Dict,
    training_num_inference_steps: int = 8,
    lora_name: str = "",
    gradient_checkpointing: bool = True,
    activation_cpu_offload: bool = False,
    offload_non_decoder: bool = True,
    keep_frozen_base_in_compute_dtype: bool = True,
    use_8bit_adam: bool = True,
    base_quantization: str = "Disabled",
    empty_cache_every_n_steps: int = 10,
    sample_generation_enabled: bool = False,
    sample_every_n_epochs: int = 10,
    sample_prompt: str = "",
    sample_lyrics: str = "",
    sample_duration: float = 30.0,
    sample_inference_steps: int = 8,
    sample_seed: int = 42,
    sample_offload_training_model: bool = False,
    sample_offload_generation: bool = True,
    model_config: str | None = None,
    vram_preset: str = "",
    sample_generation_model_config: str | None = None,
    sample_generation_settings: dict | None = None,
    progress=None,
):
    """Start LoRA training from preprocessed tensors.

    This is a generator function that yields progress updates as
    (status, log_text, plot_figure, training_state) tuples.
    """
    tensor_dir = normalize_user_path(tensor_dir)
    if not tensor_dir:
        yield "❌ Please enter a tensor directory path", "", None, training_state
        return

    try:
        tensor_dir = safe_path(tensor_dir)
    except ValueError:
        yield f"❌ Rejected unsafe tensor directory path: {tensor_dir}", "", None, training_state
        return
    if not os.path.isdir(tensor_dir):
        yield f"❌ Tensor directory not found: {tensor_dir}", "", None, training_state
        return

    normalized_lora_name, name_error = validate_lora_name(lora_name)
    if name_error is not None:
        yield f"\u274c Invalid LoRA training name: {name_error}", "", None, training_state
        return

    lora_output_dir = normalize_user_path(lora_output_dir)
    if not lora_output_dir:
        yield "❌ Please enter a LoRA output directory path", "", None, training_state
        return
    try:
        lora_output_dir = resolve_lora_training_output_dir(
            lora_output_dir,
            normalized_lora_name,
        )
    except ValueError:
        yield (
            f"❌ Rejected unsafe LoRA output directory path: {lora_output_dir}",
            "",
            None,
            training_state,
        )
        return

    # The preset dropdown only updates Gradio controls; training uses the submitted values.
    _ = vram_preset

    if _as_bool(sample_generation_enabled):
        mismatch = _sample_model_mismatch(model_config, sample_generation_model_config)
        if mismatch:
            yield f"\u274c {mismatch}", "", None, training_state
            return

    dit_ready, dit_status = ensure_dit_ready(dit_handler, config_path=model_config)
    if not dit_ready:
        status = dit_status or "Model not initialized. Please initialize the service first."
        yield f"\u274c {status}", "", None, training_state
        return
    if dit_status:
        yield dit_status, "", None, training_state

    if dit_handler is None or dit_handler.model is None:
        yield "❌ Model not initialized. Please initialize the service first.", "", None, training_state
        return

    # Training preset: LoRA training must run on non-quantized DiT.
    if getattr(dit_handler, "quantization", None) is not None:
        gpu_config = get_global_gpu_config()
        if gpu_config.gpu_memory_gb <= 0:
            yield (
                "WARNING: CPU-only training detected. Using best-effort training path "
                "(non-quantized DiT). Performance will be sub-optimal.",
                "", None, training_state,
            )
        elif gpu_config.tier in {"tier1", "tier2", "tier3", "tier4"}:
            yield (
                f"WARNING: Low VRAM tier detected ({gpu_config.gpu_memory_gb:.1f} GB, "
                f"{gpu_config.tier}). Using best-effort training path (non-quantized DiT). "
                "Performance may be sub-optimal.",
                "", None, training_state,
            )

        yield "Switching model to training preset (disable quantization)...", "", None, training_state
        if hasattr(dit_handler, "switch_to_training_preset"):
            switch_status, switched = dit_handler.switch_to_training_preset()
            if not switched:
                yield f"❌ {switch_status}", "", None, training_state
                return
            yield f"✅ {switch_status}", "", None, training_state
        else:
            yield (
                "❌ Training requires non-quantized DiT, and auto-switch is unavailable in this build.",
                "", None, training_state,
            )
            return

    # Check for required training dependencies
    try:
        from lightning.fabric import Fabric  # noqa: F401
        from peft import get_peft_model, LoraConfig  # noqa: F401
    except ImportError as e:
        yield (
            f"❌ Missing required packages: {e}\nPlease install: pip install peft lightning",
            "", None, training_state,
        )
        return

    training_state["is_training"] = True
    training_state["should_stop"] = False

    try:
        from acestep.training.trainer import LoRATrainer
        from acestep.training.configs import LoRAConfig as LoRAConfigClass, TrainingConfig

        lora_config = LoRAConfigClass(r=lora_rank, alpha=lora_alpha, dropout=lora_dropout)

        device_attr = getattr(dit_handler, "device", "")
        if hasattr(device_attr, "type"):
            device_type = str(device_attr.type).lower()
        else:
            device_type = str(device_attr).split(":", 1)[0].lower()

        # Device-tuned dataloader defaults
        if device_type == "cuda":
            num_workers, pin_memory, prefetch_factor = 4, True, 2
            persistent_workers, pin_memory_device, mixed_precision = True, "cuda", "bf16"
        elif device_type == "xpu":
            num_workers, pin_memory, prefetch_factor = 4, True, 2
            persistent_workers, pin_memory_device, mixed_precision = True, "", "bf16"
        elif device_type == "mps":
            num_workers, pin_memory, prefetch_factor = 0, False, 2
            persistent_workers, pin_memory_device, mixed_precision = False, "", "fp16"
        else:
            cpu_count = os.cpu_count() or 2
            num_workers = min(4, max(1, cpu_count // 2))
            pin_memory, prefetch_factor = False, 2
            persistent_workers = num_workers > 0
            pin_memory_device, mixed_precision = "", "fp32"

        logger.info(
            f"Training loader config: device={device_type}, workers={num_workers}, "
            f"pin_memory={pin_memory}, pin_memory_device={pin_memory_device}, "
            f"persistent_workers={persistent_workers}"
        )
        training_config = TrainingConfig(
            shift=training_shift, learning_rate=learning_rate,
            num_inference_steps=_as_positive_int(training_num_inference_steps, 8),
            batch_size=train_batch_size, gradient_accumulation_steps=gradient_accumulation,
            max_epochs=train_epochs,
            save_every_n_epochs=_as_nonnegative_int(save_every_n_epochs, 10),
            seed=training_seed, output_dir=lora_output_dir,
            lora_name=normalized_lora_name,
            use_fp8=_uses_fp8_scaled(base_quantization),
            gradient_checkpointing=_as_bool(gradient_checkpointing),
            activation_cpu_offload=_as_bool(activation_cpu_offload),
            offload_non_decoder=_as_bool(offload_non_decoder),
            keep_frozen_base_in_compute_dtype=_as_bool(keep_frozen_base_in_compute_dtype),
            use_8bit_adam=_as_bool(use_8bit_adam),
            empty_cache_every_n_steps=_as_nonnegative_int(
                empty_cache_every_n_steps, 10
            ),
            num_workers=num_workers, pin_memory=pin_memory,
            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
            pin_memory_device=pin_memory_device, mixed_precision=mixed_precision,
            sample_every_n_epochs=(
                _as_positive_int(sample_every_n_epochs, 10)
                if _as_bool(sample_generation_enabled)
                else 0
            ),
            sample_prompt=str(sample_prompt or ""),
            sample_lyrics=str(sample_lyrics or ""),
            sample_duration=float(sample_duration or 30.0),
            sample_inference_steps=_as_positive_int(sample_inference_steps, 8),
            sample_seed=int(sample_seed or 42),
            sample_output_dir="",
            sample_offload_training_model=_as_bool(sample_offload_training_model),
            sample_offload_generation=_as_bool(sample_offload_generation),
            sample_generation_settings=dict(sample_generation_settings or {}),
        )
        _save_training_config_snapshot(lora_config, training_config)

        log_lines: list = []
        step_list: list = []
        loss_list: list = []
        total_steps = estimate_lora_total_steps(
            tensor_dir,
            train_batch_size,
            gradient_accumulation,
            train_epochs,
        )
        initial_plot = _training_loss_figure(training_state, step_list, loss_list)
        start_time = time.time()
        progress_timer = TrainingProgressTimer(start_time)

        yield f"🚀 Starting training from {tensor_dir}...", "", initial_plot, training_state

        trainer = LoRATrainer(
            dit_handler=dit_handler, lora_config=lora_config, training_config=training_config,
        )

        training_failed = False
        failure_message = ""

        resume_from = None
        resume_checkpoint_dir = normalize_user_path(resume_checkpoint_dir)
        if resume_checkpoint_dir:
            try:
                normalized_resume = safe_path(resume_checkpoint_dir)
                if os.path.exists(normalized_resume):
                    resume_from = normalized_resume
            except ValueError:
                logger.warning(f"Rejected unsafe resume path: {resume_checkpoint_dir}")
                resume_from = None

        for step, loss, status in trainer.train_from_preprocessed(
            tensor_dir, training_state, resume_from=resume_from,
        ):
            status_text = str(status)
            status_lower = status_text.lower()
            if (
                status_text.startswith("❌")
                or "training failed" in status_lower
                or "error:" in status_lower
                or "module not found" in status_lower
            ):
                training_failed = True
                failure_message = status_text

            now = time.time()
            timing = progress_timer.update(
                status,
                step=step,
                total_steps=total_steps,
                now=now,
            )
            display_status = build_training_progress_text(
                status,
                step=step,
                total_epochs=int(train_epochs or 0),
                elapsed_seconds=timing.elapsed_seconds,
                loss=loss,
                total_steps=total_steps,
                elapsed_label=timing.elapsed_label,
                speed=timing.speed,
                eta_seconds=timing.eta_seconds,
            )
            log_lines.append(display_status)
            if len(log_lines) > 15:
                log_lines = log_lines[-15:]
            log_text = "\n".join(log_lines)

            if step > 0 and loss is not None and loss == loss:  # NaN check
                step_list.append(step)
                loss_list.append(float(loss))

            plot_figure = _training_loss_figure(training_state, step_list, loss_list)
            yield display_status, log_text, plot_figure, training_state

            if training_state.get("should_stop", False):
                logger.info("ℹ️ Training stopped by user")
                log_lines.append("ℹ️ Training stopped by user")
                yield display_status, "\n".join(log_lines[-15:]), plot_figure, training_state
                break

        total_time = time.time() - start_time
        training_state["is_training"] = False
        final_plot = _training_loss_figure(training_state, step_list, loss_list)
        if training_failed:
            final_msg = f"{failure_message}\nElapsed: {_format_duration(total_time)}"
            logger.warning(final_msg)
            log_lines.append(failure_message)
            yield final_msg, "\n".join(log_lines[-15:]), final_plot, training_state
            return
        completion_msg = f"✅ Training completed! Total time: {_format_duration(total_time)}"
        logger.info(completion_msg)
        log_lines.append(completion_msg)
        yield completion_msg, "\n".join(log_lines[-15:]), final_plot, training_state

    except Exception as e:
        logger.exception("Training error")
        training_state["is_training"] = False
        yield f"❌ Error: {str(e)}", str(e), _training_loss_figure({}, [], []), training_state


def stop_training(training_state: Dict) -> Tuple[str, Dict]:
    """Stop the current training process.

    Returns:
        Tuple of (status, training_state).
    """
    stopped_subprocess = request_training_subprocess_stop()
    if not training_state.get("is_training", False):
        if stopped_subprocess:
            training_state["should_stop"] = True
            return "Stopping isolated training subprocess...", training_state
        return t("training.stop_no_training"), training_state

    training_state["should_stop"] = True
    if stopped_subprocess:
        return "Stopping isolated training subprocess...", training_state
    return t("training.stop_stopping"), training_state


def _checkpoint_epoch_from_name(name: str) -> int | None:
    """Return an epoch number from old or named LoRA checkpoint folders."""

    old_match = re.match(r"^epoch_(\d+)(?:_|$)", name)
    if old_match:
        return int(old_match.group(1))

    flat_match = re.search(r"-epoch-(\d+)(?:-|\.|$)", name)
    if flat_match:
        return int(flat_match.group(1))

    named_match = re.search(r"-(\d+)(?:$|-sample$)", name)
    if named_match:
        return int(named_match.group(1))
    return None


def _latest_flat_lora_artifact(output_dir: str) -> str | None:
    """Return the preferred flat LoRA safetensors artifact from a run folder."""

    if not os.path.isdir(output_dir):
        return None
    artifacts = [
        name for name in os.listdir(output_dir) if name.endswith(".safetensors")
    ]
    if not artifacts:
        return None

    def artifact_key(name: str) -> tuple[int, int, str]:
        epoch = _checkpoint_epoch_from_name(name) or 0
        final_rank = 1 if name.endswith("-final.safetensors") else 0
        return epoch, final_rank, name

    artifacts.sort(key=artifact_key)
    return os.path.join(output_dir, artifacts[-1])


def _copy_lora_export_source(source_path: str, export_path: str) -> None:
    """Copy a LoRA export source file or legacy directory to the destination."""

    import shutil

    parent_dir = os.path.dirname(export_path) or "."
    os.makedirs(parent_dir, exist_ok=True)

    if os.path.isdir(source_path):
        if os.path.exists(export_path):
            shutil.rmtree(export_path)
        shutil.copytree(source_path, export_path)
        return

    if os.path.isdir(export_path):
        destination = os.path.join(export_path, os.path.basename(source_path))
    elif export_path.endswith(".safetensors"):
        destination = export_path
    else:
        os.makedirs(export_path, exist_ok=True)
        destination = os.path.join(export_path, os.path.basename(source_path))
    shutil.copy2(source_path, destination)


def export_lora(export_path: str, lora_output_dir: str, lora_name: str = "") -> str:
    """Export the trained LoRA weights.

    Returns:
        Status message.
    """
    export_path = normalize_user_path(export_path)
    if not export_path:
        return t("training.export_path_required")

    lora_output_dir = normalize_user_path(lora_output_dir)
    if not lora_output_dir:
        return t("training.invalid_lora_output_dir")

    try:
        safe_lora_dir = resolve_lora_export_root(lora_output_dir, lora_name)
    except ValueError:
        return t("training.invalid_lora_output_dir")

    flat_artifact = _latest_flat_lora_artifact(safe_lora_dir)
    final_dir = os.path.join(safe_lora_dir, "final")
    checkpoint_dir = os.path.join(safe_lora_dir, "checkpoints")

    if flat_artifact:
        source_path = flat_artifact
    elif os.path.exists(final_dir):
        source_path = final_dir
    elif os.path.exists(checkpoint_dir):
        checkpoints = [
            d
            for d in os.listdir(checkpoint_dir)
            if os.path.isdir(os.path.join(checkpoint_dir, d))
            and _checkpoint_epoch_from_name(d) is not None
        ]
        if not checkpoints:
            return t("training.no_checkpoints_found")
        checkpoints.sort(key=lambda x: _checkpoint_epoch_from_name(x) or 0)
        latest = checkpoints[-1]
        source_path = os.path.join(checkpoint_dir, latest)
    else:
        return t("training.no_trained_model_found", path=lora_output_dir)

    try:
        safe_export = safe_path(export_path)
    except ValueError:
        return t("training.invalid_export_path")

    try:
        _copy_lora_export_source(source_path, safe_export)
        return t("training.lora_exported", path=safe_export)

    except Exception as e:
        logger.exception("Export error")
        return t("training.export_failed", error=str(e))
