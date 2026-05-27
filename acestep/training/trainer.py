"""
LoRA Trainer for ACE-Step

Lightning Fabric-based trainer for LoRA fine-tuning of ACE-Step DiT decoder.
Supports training from preprocessed tensor files for optimal performance.
"""

import os
import time
import random
import math
import shutil
from typing import Optional, List, Dict, Any, Tuple, Generator
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from contextlib import nullcontext
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

try:
    from lightning.fabric import Fabric
    from lightning.fabric.loggers import TensorBoardLogger

    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning(
        "Lightning Fabric not installed. Training will use basic training loop."
    )

# OPTIMIZATION: Use 8-bit Adam to save some VRAM
try:
    import bitsandbytes as bnb

    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    logger.warning("bitsandbytes not installed. Using standard AdamW.")

from acestep.training.configs import LoRAConfig, LoKRConfig, TrainingConfig
from acestep.training.lora_injection import inject_lora_into_dit
from acestep.training.lora_naming import lora_epoch_name
from acestep.training.lora_utils import check_peft_available
from acestep.training.lora_checkpoint import (
    save_training_checkpoint,
    load_training_checkpoint,
)
from acestep.training.lokr_utils import (
    inject_lokr_into_dit,
    save_lokr_weights,
    save_lokr_training_checkpoint,
    check_lycoris_available,
)
from acestep.training.data_module import PreprocessedDataModule
from acestep.training.adaptive_timestep import AdaptiveTimestepSampler
from acestep.training.optim import build_optimizer, build_scheduler
from acestep.training.path_safety import safe_path
from acestep.training.save_best import BestMetricTracker
from acestep.training.sample_generation_inprocess import run_training_sample_inprocess
from acestep.training.timestep_schedule import build_shifted_timestep_schedule
from acestep.training_v2.timestep_sampling import apply_cfg_dropout, sample_timesteps
from acestep.training.vram_optimizations import (
    apply_training_fp8_scaled,
    cast_training_parameter_dtypes,
    cuda_peak_gb,
    offload_handler_training_modules,
    offload_non_decoder_modules,
    reset_cuda_peak,
    sample_generation_vram_guard,
)


def _named_lora_epoch(training_config: TrainingConfig, epoch: int) -> str:
    """Return the configured LoRA artifact basename for an epoch."""

    return lora_epoch_name(getattr(training_config, "lora_name", "lora"), epoch)


def _adapter_display_name(training_config: TrainingConfig) -> str:
    """Return the user-facing adapter type name for training messages."""

    if str(getattr(training_config, "adapter_type", "lora")).lower() == "dora":
        return "DoRA"
    return "LoRA"


def _checkpoint_save_interval(training_config: TrainingConfig) -> int:
    """Return the non-negative periodic checkpoint interval."""

    try:
        return max(0, int(getattr(training_config, "save_every_n_epochs", 0)))
    except (TypeError, ValueError):
        return 0


def _should_save_epoch_checkpoint(training_config: TrainingConfig, epoch: int) -> bool:
    """Return whether a periodic checkpoint should be saved for an epoch."""

    interval = _checkpoint_save_interval(training_config)
    return interval > 0 and int(epoch) % interval == 0


def _save_final_lora_artifacts(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    training_config: TrainingConfig,
    global_step: int,
) -> str:
    """Save final LoRA weights and a resumable final state file."""

    final_epoch = int(training_config.max_epochs)
    final_artifact_name = f"{_named_lora_epoch(training_config, final_epoch)}-final"
    save_training_checkpoint(
        model,
        optimizer,
        scheduler,
        final_epoch,
        global_step,
        training_config.output_dir,
        artifact_name=final_artifact_name,
        state_suffix="final",
    )

    return training_config.output_dir


def _save_lora_epoch_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    training_config: TrainingConfig,
    epoch: int,
    global_step: int,
    *,
    state_suffix: str = "",
) -> str:
    """Save one flat LoRA epoch checkpoint and return its state path."""

    artifact_name = _named_lora_epoch(training_config, epoch)
    if state_suffix:
        artifact_name = f"{artifact_name}-{state_suffix}"
    return save_training_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        training_config.output_dir,
        artifact_name=artifact_name,
        state_suffix=state_suffix,
    )


def _save_best_lora_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    training_config: TrainingConfig,
    epoch: int,
    global_step: int,
) -> str:
    """Save the current best adapter, replacing the previous best directory."""

    best_dir = safe_path(os.path.join(training_config.output_dir, "best"))
    if os.path.isdir(best_dir):
        shutil.rmtree(best_dir)
    os.makedirs(best_dir, exist_ok=True)
    artifact_name = f"{getattr(training_config, 'lora_name', 'lora')}-best"
    return save_training_checkpoint(
        model,
        optimizer,
        scheduler,
        epoch,
        global_step,
        best_dir,
        artifact_name=artifact_name,
        state_suffix="best",
    )


def _load_lora_resume_state_dict(
    adapter_path: str,
    device: torch.device,
) -> Dict[str, Any]:
    """Load LoRA adapter tensors from a flat file or legacy adapter directory."""

    if os.path.isfile(adapter_path):
        if adapter_path.endswith(".safetensors"):
            from safetensors.torch import load_file

            return load_file(adapter_path)
        return torch.load(adapter_path, map_location=device, weights_only=True)

    adapter_weights_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(adapter_weights_path):
        adapter_weights_path = os.path.join(adapter_path, "adapter_model.bin")
    if not os.path.exists(adapter_weights_path):
        raise FileNotFoundError(f"Adapter weights not found in {adapter_path}")
    if adapter_weights_path.endswith(".safetensors"):
        from safetensors.torch import load_file

        return load_file(adapter_weights_path)
    return torch.load(adapter_weights_path, map_location=device, weights_only=True)


def _normalize_device_type(device: Any) -> str:
    """Normalize torch device or string to canonical device type."""
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)


def _select_compute_dtype(device_type: str) -> torch.dtype:
    """Pick the compute dtype for each accelerator."""
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    """Pick Fabric precision plugin setting for each accelerator."""
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        # Use AMP on MPS for better throughput. Trainable LoRA parameters are
        # explicitly forced to fp32 before optimizer/Fabric setup.
        return "16-mixed"
    return "32-true"


def _ensure_trainable_params_fp32(module: nn.Module) -> Tuple[int, int]:
    """Force trainable floating-point parameters to fp32."""
    casted = 0
    total = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.is_floating_point() and p.dtype != torch.float32:
            with torch.no_grad():
                p.data = p.data.float()
            casted += 1
    return casted, total


def _count_nonfinite_grads(params: List[torch.nn.Parameter]) -> Tuple[int, int]:
    """Count non-finite gradient tensors among params with gradients."""
    nonfinite = 0
    total_with_grad = 0
    for p in params:
        g = p.grad
        if g is None:
            continue
        total_with_grad += 1
        if not torch.isfinite(g).all():
            nonfinite += 1
    return nonfinite, total_with_grad


def _ensure_optimizer_params_fp32(optimizer: torch.optim.Optimizer) -> Tuple[int, int]:
    """Force optimizer parameter tensors to fp32 when trainable."""
    casted = 0
    total = 0
    for group in optimizer.param_groups:
        for p in group.get("params", []):
            if p is None:
                continue
            total += 1
            if p.is_floating_point() and p.dtype != torch.float32:
                with torch.no_grad():
                    p.data = p.data.float()
                casted += 1
    return casted, total


def _build_param_name_lookup(
    module: nn.Module, extra_module: Optional[nn.Module] = None
) -> Dict[int, str]:
    """Build a best-effort id(param) -> name lookup for debug logging."""
    lookup: Dict[int, str] = {}
    for name, p in module.named_parameters():
        lookup[id(p)] = name
    if extra_module is not None:
        for name, p in extra_module.named_parameters():
            lookup.setdefault(id(p), f"lycoris_net.{name}")
    return lookup


def _count_nonfinite_grads_detailed(
    params: List[torch.nn.Parameter],
    param_name_lookup: Dict[int, str],
    detail_limit: int = 8,
) -> Tuple[int, int, List[str]]:
    """Count non-finite grads and return up to `detail_limit` offending tensor details."""
    nonfinite = 0
    total_with_grad = 0
    details: List[str] = []

    for p in params:
        g = p.grad
        if g is None:
            continue
        total_with_grad += 1
        if torch.isfinite(g).all():
            continue

        nonfinite += 1
        if len(details) >= detail_limit:
            continue

        pname = param_name_lookup.get(id(p), f"<unnamed:{id(p)}>")
        g32 = g.detach().float()
        nan_count = int(torch.isnan(g32).sum().item())
        inf_count = int(torch.isinf(g32).sum().item())
        finite_vals = g32[torch.isfinite(g32)]
        max_abs_finite = (
            float(finite_vals.abs().max().item())
            if finite_vals.numel()
            else float("nan")
        )

        p32 = p.detach().float()
        param_nonfinite = int((~torch.isfinite(p32)).sum().item())
        details.append(
            f"{pname} | shape={tuple(p.shape)} grad_dtype={g.dtype} "
            f"nan={nan_count} inf={inf_count} max_abs_finite={max_abs_finite:.3e} "
            f"param_nonfinite={param_nonfinite}"
        )

    return nonfinite, total_with_grad, details


def _collect_lokr_trainable_params(
    module: nn.Module, lycoris_net: Optional[nn.Module]
) -> List[torch.nn.Parameter]:
    """
    Collect LoKr trainable params robustly.

    Primary path is model parameter traversal. If that returns empty due to
    wrapper/registration quirks, fall back to LyCORIS module parameters.
    """
    params = [p for p in module.parameters() if p.requires_grad]
    if params:
        return list({id(p): p for p in params}.values())

    fallback: List[torch.nn.Parameter] = []
    if lycoris_net is None:
        return fallback

    for m in getattr(lycoris_net, "loras", []) or []:
        for p in m.parameters():
            if p.requires_grad:
                fallback.append(p)
    if not fallback:
        for p in lycoris_net.parameters():
            if p.requires_grad:
                fallback.append(p)
    return list({id(p): p for p in fallback}.values())


def _unwrap_stale_fabric_decoder(model: nn.Module) -> bool:
    """
    Unwrap stale Lightning Fabric wrappers from decoder left by previous runs.

    Returns:
        True if decoder was unwrapped, else False.
    """
    if model is None or not hasattr(model, "decoder"):
        return False
    decoder = model.decoder
    unwrapped = False
    while hasattr(decoder, "_forward_module") and isinstance(
        getattr(decoder, "_forward_module"), nn.Module
    ):
        decoder = decoder._forward_module
        unwrapped = True
    if unwrapped:
        model.decoder = decoder
    return unwrapped


def _iter_module_wrappers(module: nn.Module) -> List[nn.Module]:
    """Collect wrapper chain modules (Fabric/PEFT/compile/base-model wrappers)."""
    modules: List[nn.Module] = []
    stack = [module]
    visited = set()

    while stack:
        current = stack.pop()
        if not isinstance(current, nn.Module):
            continue
        module_id = id(current)
        if module_id in visited:
            continue
        visited.add(module_id)
        modules.append(current)

        for attr_name in (
            "_forward_module",
            "_orig_mod",
            "base_model",
            "model",
            "module",
        ):
            child = getattr(current, attr_name, None)
            if isinstance(child, nn.Module):
                stack.append(child)

    return modules


def _configure_training_memory_features(decoder: nn.Module) -> Tuple[bool, bool, bool]:
    """
    Enable gradient checkpointing and disable use_cache across wrapped decoder modules.

    Returns:
        Tuple[checkpointing_enabled, cache_disabled, input_grads_enabled]
    """
    checkpointing_enabled = False
    cache_disabled = False
    input_grads_enabled = False

    for mod in _iter_module_wrappers(decoder):
        if hasattr(mod, "gradient_checkpointing_enable"):
            try:
                mod.gradient_checkpointing_enable()
                checkpointing_enabled = True
            except Exception:
                pass
        elif hasattr(mod, "gradient_checkpointing"):
            try:
                mod.gradient_checkpointing = True
                checkpointing_enabled = True
            except Exception:
                pass

        # PEFT + gradient checkpointing can require input embeddings to have
        # gradients enabled, otherwise loss may be detached (no grad_fn).
        if hasattr(mod, "enable_input_require_grads"):
            try:
                mod.enable_input_require_grads()
                hook_enabled = bool(
                    getattr(mod, "_acestep_input_grads_hook_enabled", False)
                )
                has_require_hook = getattr(mod, "_require_grads_hook", None) is not None
                if hook_enabled or has_require_hook:
                    input_grads_enabled = True
            except Exception:
                pass

        cfg = getattr(mod, "config", None)
        if cfg is not None and hasattr(cfg, "use_cache"):
            try:
                if getattr(cfg, "use_cache", None) is not False:
                    cfg.use_cache = False
                    cache_disabled = True
            except Exception:
                pass

    return checkpointing_enabled, cache_disabled, input_grads_enabled


def sample_discrete_timestep(bsz, timesteps_tensor):
    """Sample timesteps from the configured discrete training schedule.

    For each sample in the batch, randomly select one timestep derived from
    the submitted training ``shift`` and ``num_inference_steps`` values.

    Args:
        bsz: Batch size
        timesteps_tensor: Configured schedule tensor.

    Returns:
        Tuple of (t, r) where both are the same sampled timestep
    """
    # Randomly select indices for each sample in batch
    indices = torch.randint(
        0, timesteps_tensor.shape[0], (bsz,), device=timesteps_tensor.device
    )
    t = timesteps_tensor[indices]

    # r = t for this training setup
    r = t

    return t, r


class PreprocessedLoRAModule(nn.Module):
    """LoRA Training Module using preprocessed tensors.

    This module trains only the DiT decoder with LoRA adapters.
    All inputs are pre-computed tensors - no VAE or text encoder needed!

    Training flow:
    1. Load pre-computed tensors (target_latents, encoder_hidden_states, context_latents)
    2. Sample noise and timestep
    3. Forward through decoder (with LoRA)
    4. Compute flow matching loss
    """

    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize the training module.

        Args:
            model: The AceStepConditionGenerationModel
            lora_config: LoRA configuration
            training_config: Training configuration
            device: Device to use
            dtype: Data type to use
        """
        super().__init__()

        self.lora_config = lora_config
        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(self.device_type)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")
        self._timestep_mode = str(
            getattr(training_config, "timestep_mode", "continuous") or "continuous"
        ).lower()
        self._cfg_ratio = float(getattr(training_config, "cfg_ratio", 0.15) or 0.0)
        self._timestep_mu = float(getattr(training_config, "timestep_mu", -0.4))
        self._timestep_sigma = float(getattr(training_config, "timestep_sigma", 1.0))
        self._data_proportion = float(getattr(training_config, "data_proportion", 0.5))
        adaptive_ratio = float(
            getattr(training_config, "adaptive_timestep_ratio", 0.0) or 0.0
        )
        self._adaptive_sampler = (
            AdaptiveTimestepSampler(ratio=adaptive_ratio)
            if self._timestep_mode == "continuous" and adaptive_ratio > 0.0
            else None
        )
        timestep_schedule = build_shifted_timestep_schedule(
            training_config.num_inference_steps,
            training_config.shift,
        )
        self.timesteps_tensor = torch.tensor(
            timestep_schedule, device=self.device, dtype=self.dtype
        )
        logger.info(
            f"LoRA training timestep mode: {self._timestep_mode}, "
            f"steps={training_config.num_inference_steps}, "
            f"shift={training_config.shift}, cfg_ratio={self._cfg_ratio:.3f}, "
            f"adaptive_ratio={adaptive_ratio:.3f}"
        )
        # When gradient checkpointing is enabled via wrapper layers that don't expose
        # enable_input_require_grads(), force at least one forward input to require grad
        # so checkpointed segments keep a valid autograd graph.
        self.force_input_grads_for_checkpointing = False

        # Inject LoRA into the decoder only
        if check_peft_available():
            # Fix: Force tensors out of inference mode before injection
            for param in model.parameters():
                param.data = param.data.clone()
                if param.is_inference():
                    with torch.no_grad():
                        param.data = param.data.clone()

            self.model, self.lora_info = inject_lora_into_dit(model, lora_config)
            logger.info(
                f"{_adapter_display_name(training_config)} injected: "
                f"{self.lora_info['trainable_params']:,} trainable params"
            )
        else:
            self.model = model
            self.lora_info = {}
            logger.warning("PEFT not available, training without LoRA adapters")

        # torch.compile: optional perf optimization.
        # PEFT LoRA wraps the decoder in PeftModelForFeatureExtraction which is
        # incompatible with torch.compile/inductor on PyTorch 2.7.x
        # (AssertionError at first forward pass, not at compile time).
        # Only compile when NOT using PEFT adapters.
        has_peft = bool(self.lora_info)
        if hasattr(torch, "compile") and self.device_type == "cuda" and not has_peft:
            try:
                logger.info("Compiling DiT decoder...")
                self.model.decoder = torch.compile(self.model.decoder, mode="default")
                logger.info("torch.compile successful")
            except Exception as e:
                logger.warning(
                    f"torch.compile failed ({e}), continuing without compilation"
                )
        else:
            if has_peft:
                logger.info(
                    "Skipping torch.compile (incompatible with PEFT LoRA adapters)"
                )
            else:
                logger.info(
                    "torch.compile not available on this device/PyTorch version, skipping"
                )

        # Model config for flow matching
        self.config = model.config
        self._null_cond_emb = getattr(model, "null_condition_emb", None)
        if self._null_cond_emb is None and self._cfg_ratio > 0.0:
            logger.warning("model.null_condition_emb not found; CFG dropout disabled.")

        # Store training losses
        self.training_losses = []

    def training_step(
        self,
        batch: Dict[str, torch.Tensor],
        record_loss: bool = True,
    ) -> torch.Tensor:
        """Single training step using preprocessed tensors.

        Timestep sampling is controlled by ``training_config.timestep_mode``.
        Continuous mode uses logit-normal training timesteps and CFG dropout;
        discrete mode keeps the legacy shifted inference schedule.

        Args:
            batch: Dictionary containing pre-computed tensors:
                - target_latents: [B, T, 64] - VAE encoded audio
                - attention_mask: [B, T] - Valid audio mask
                - encoder_hidden_states: [B, L, D] - Condition encoder output
                - encoder_attention_mask: [B, L] - Condition mask
                - context_latents: [B, T, 128] - Source context
            record_loss: If True, append loss to training_losses (set False for validation).

        Returns:
            Loss tensor (float32 for stable backward)
        """
        # Use autocast for mixed precision training (bf16 on CUDA/XPU, fp16 on MPS)
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(
                device_type=self.device_type, dtype=self.dtype
            )
        else:
            autocast_ctx = nullcontext()

        if (
            getattr(self.training_config, "activation_cpu_offload", False)
            and self.device_type == "cuda"
            and hasattr(torch.autograd.graph, "save_on_cpu")
        ):
            saved_tensor_ctx = torch.autograd.graph.save_on_cpu(
                pin_memory=bool(getattr(self.training_config, "pin_memory", True))
            )
        else:
            saved_tensor_ctx = nullcontext()

        with saved_tensor_ctx, autocast_ctx:
            # Get tensors from batch (already on device from Fabric dataloader)
            target_latents = batch["target_latents"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )  # x0
            attention_mask = batch["attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_hidden_states = batch["encoder_hidden_states"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_attention_mask = batch["encoder_attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            context_latents = batch["context_latents"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )

            bsz = target_latents.shape[0]

            # Flow matching: sample noise x1 and interpolate with data x0
            x1 = torch.randn_like(target_latents)  # Noise
            x0 = target_latents  # Data

            if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
                null_cond_emb = self._null_cond_emb.to(
                    device=encoder_hidden_states.device,
                    dtype=encoder_hidden_states.dtype,
                )
                encoder_hidden_states = apply_cfg_dropout(
                    encoder_hidden_states,
                    null_cond_emb,
                    cfg_ratio=self._cfg_ratio,
                )

            if self._timestep_mode == "discrete":
                t, r = sample_discrete_timestep(bsz, self.timesteps_tensor)
            elif self._adaptive_sampler is not None:
                t, r = self._adaptive_sampler.sample(
                    batch_size=bsz,
                    base_sampler=sample_timesteps,
                    device=self.device,
                    dtype=self.dtype,
                    data_proportion=self._data_proportion,
                    timestep_mu=self._timestep_mu,
                    timestep_sigma=self._timestep_sigma,
                    use_meanflow=False,
                )
            else:
                t, r = sample_timesteps(
                    batch_size=bsz,
                    device=self.device,
                    dtype=self.dtype,
                    data_proportion=self._data_proportion,
                    timestep_mu=self._timestep_mu,
                    timestep_sigma=self._timestep_sigma,
                    use_meanflow=False,
                )
            t_ = t.unsqueeze(-1).unsqueeze(-1)

            # Interpolate: x_t = t * x1 + (1 - t) * x0
            xt = t_ * x1 + (1.0 - t_) * x0
            if self.force_input_grads_for_checkpointing:
                xt = xt.requires_grad_(True)

            # Forward through decoder
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=r,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            # Flow matching loss: predict the flow field v = x1 - x0
            flow = x1 - x0
            per_sample_loss = F.mse_loss(
                decoder_outputs[0],
                flow,
                reduction="none",
            ).mean(dim=tuple(range(1, flow.ndim)))
            diffusion_loss = per_sample_loss.mean()

            if self._adaptive_sampler is not None:
                self._adaptive_sampler.update(t, per_sample_loss)

        # Convert loss to float32 for stable backward pass
        diffusion_loss = diffusion_loss.float()

        if record_loss:
            self.training_losses.append(diffusion_loss.item())

        return diffusion_loss


class LoRATrainer:
    """High-level trainer for ACE-Step LoRA fine-tuning.

    Uses Lightning Fabric for distributed training and mixed precision.
    Supports training from preprocessed tensor directories.
    """

    def __init__(
        self,
        dit_handler,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
    ):
        """Initialize the trainer.

        Args:
            dit_handler: Initialized DiT handler (for model access)
            lora_config: LoRA configuration
            training_config: Training configuration
        """
        self.dit_handler = dit_handler
        self.lora_config = lora_config
        # Validate output_dir early so all downstream path operations are safe
        training_config.output_dir = safe_path(training_config.output_dir)
        self.training_config = training_config

        self.module = None
        self.fabric = None
        self.is_training = False

    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Train LoRA adapters from preprocessed tensor files.

        This is the recommended training method for best performance.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            training_state: Optional state dict for stopping control
            resume_from: Optional path to checkpoint directory to resume from

        Yields:
            Tuples of (step, loss, status_message)
        """
        self.is_training = True

        try:
            # LoRA injection via PEFT is incompatible with torchao-quantized
            # decoder modules in this runtime. Fail fast with actionable guidance.
            quantization_mode = getattr(self.dit_handler, "quantization", None)
            if quantization_mode is not None:
                yield (
                    0,
                    0.0,
                    (
                        "❌ LoRA training requires a non-quantized DiT model. "
                        f"Current quantization: {quantization_mode}. "
                        "Re-initialize service with INT8 Quantization disabled, then retry training."
                    ),
                )
                return

            # Validate tensor directory
            try:
                tensor_dir = safe_path(tensor_dir)
            except ValueError:
                yield 0, 0.0, f"❌ Rejected unsafe tensor directory: {tensor_dir}"
                return
            if not os.path.isdir(tensor_dir):
                yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
                return

            # Create training module
            torch.manual_seed(self.training_config.seed)
            random.seed(self.training_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.training_config.seed)
            try:
                import numpy as np

                np.random.seed(self.training_config.seed)
            except Exception:
                pass

            self.module = PreprocessedLoRAModule(
                model=self.dit_handler.model,
                lora_config=self.lora_config,
                training_config=self.training_config,
                device=self.dit_handler.device,
                dtype=self.dit_handler.dtype,
            )
            if getattr(self.training_config, "offload_non_decoder", True):
                moved = (
                    offload_non_decoder_modules(self.module.model)
                    + offload_handler_training_modules(self.dit_handler)
                )
                if moved:
                    yield 0, 0.0, f"Offloaded unused modules to CPU: {', '.join(moved)}"

            if getattr(self.training_config, "use_fp8", False):
                try:
                    checkpoint_path = None
                    last_init = getattr(self.dit_handler, "last_init_params", None) or {}
                    if last_init.get("project_root") and last_init.get("resolved_config_path"):
                        checkpoint_path = os.path.join(
                            str(last_init["project_root"]),
                            "models",
                            str(last_init["resolved_config_path"]),
                        )
                    msg = apply_training_fp8_scaled(
                        self.module.model,
                        checkpoint_path=checkpoint_path,
                        device=self.module.device,
                    )
                    yield 0, 0.0, f"FP8 scaled base weights enabled: {msg}"
                except Exception as exc:
                    logger.exception("Failed to apply FP8 scaled training quantization")
                    yield 0, 0.0, f"FP8 scaled base weights unavailable: {exc}"

            if getattr(self.training_config, "gradient_checkpointing", True):
                ckpt_enabled, cache_disabled, input_grads_enabled = (
                    _configure_training_memory_features(self.module.model.decoder)
                )
                # DiT decoder does not expose token embeddings like causal LMs.
                # Force grad-carrying inputs for checkpointed segments to avoid
                # detached losses regardless of wrapper hook availability.
                self.module.force_input_grads_for_checkpointing = ckpt_enabled
                logger.info(
                    f"Training memory features: gradient_checkpointing={ckpt_enabled}, "
                    f"use_cache_disabled={cache_disabled}, "
                    f"input_grads_enabled={input_grads_enabled}"
                )
            else:
                ckpt_enabled = False
                input_grads_enabled = True
                self.module.force_input_grads_for_checkpointing = False
                yield 0, 0.0, "Gradient checkpointing disabled by user"

            # Create data module
            data_module = PreprocessedDataModule(
                tensor_dir=tensor_dir,
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                prefetch_factor=self.training_config.prefetch_factor,
                persistent_workers=self.training_config.persistent_workers,
                pin_memory_device=self.training_config.pin_memory_device,
                val_split=getattr(self.training_config, "val_split", 0.0),
            )

            # Setup data
            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                yield 0, 0.0, "❌ No valid samples found in tensor directory"
                return

            yield (
                0,
                0.0,
                f"📂 Loaded {len(data_module.train_dataset)} preprocessed samples",
            )
            if ckpt_enabled:
                yield 0, 0.0, "🧠 Gradient checkpointing enabled for decoder"
            elif getattr(self.training_config, "gradient_checkpointing", True):
                yield (
                    0,
                    0.0,
                    "⚠️ Gradient checkpointing not enabled (model wrapper did not expose it)",
                )
            if not input_grads_enabled:
                yield (
                    0,
                    0.0,
                    "ℹ️ Input-grad hook not available on this DiT; using explicit checkpointing fallback",
                )

            reset_cuda_peak()
            if LIGHTNING_AVAILABLE:
                yield from self._train_with_fabric(
                    data_module, training_state, resume_from
                )
            else:
                yield from self._train_basic(data_module, training_state)

        except Exception as e:
            logger.exception("Training failed")
            yield 0, 0.0, f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False

    def _train_with_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Train using Lightning Fabric."""
        # Create output directory
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        device_type = self.module.device_type
        precision = _select_fabric_precision(device_type)
        accelerator = (
            device_type if device_type in ("cuda", "xpu", "mps", "cpu") else "auto"
        )

        # Initialize Fabric
        fabric_kwargs = {
            "accelerator": accelerator,
            "devices": 1,
            "precision": precision,
        }
        self.fabric = Fabric(**fabric_kwargs)
        self.fabric.launch()

        yield (
            0,
            0.0,
            f"🚀 Starting training (device: {device_type}, precision: {precision})...",
        )

        # Keep trainable adapter tensors in fp32, but avoid promoting the
        # frozen base decoder to fp32 unless the user disables the VRAM saver.
        if (
            getattr(self.training_config, "keep_frozen_base_in_compute_dtype", True)
            or getattr(self.training_config, "use_fp8", False)
        ):
            casted_trainable, casted_frozen = cast_training_parameter_dtypes(
                self.module.model.decoder,
                frozen_dtype=self.module.dtype,
                keep_frozen_in_compute_dtype=True,
            )
            logger.info(
                "Training dtype fixup: trainable_fp32={}, frozen_compute_dtype={}",
                casted_trainable,
                casted_frozen,
            )
        else:
            if device_type == "mps" or precision.endswith("-mixed"):
                self.module.model.decoder = self.module.model.decoder.to(
                    dtype=torch.float32
                )
            else:
                self.module.model.decoder = self.module.model.decoder.to(
                    dtype=self.module.dtype
                )
            casted_trainable, total_trainable_tensors = _ensure_trainable_params_fp32(
                self.module.model.decoder
            )
            logger.info(
                f"Trainable tensor dtype fixup: "
                f"casted {casted_trainable}/{total_trainable_tensors} to fp32"
            )

        # Get dataloader
        train_loader = data_module.train_dataloader()
        val_loader = (
            data_module.val_dataloader()
            if hasattr(data_module, "val_dataloader")
            else None
        )

        if training_state is not None:
            training_state["plot_steps"] = []
            training_state["plot_loss"] = []
            training_state["plot_ema"] = []
            training_state["plot_val_steps"] = []
            training_state["plot_val_loss"] = []
            training_state["plot_best_step"] = None
        ema_loss = None
        ema_alpha = 0.1
        best_tracker = BestMetricTracker(
            smoothing_window=getattr(
                self.training_config,
                "save_best_smoothing_window",
                5,
            ),
            min_delta=getattr(self.training_config, "save_best_min_delta", 0.001),
        )

        # Setup optimizer - only LoRA parameters
        trainable_params = [
            p for p in self.module.model.parameters() if p.requires_grad
        ]

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        yield (
            0,
            0.0,
            f"🎯 Training {sum(p.numel() for p in trainable_params):,} parameters",
        )

        optimizer_type = str(
            getattr(self.training_config, "optimizer_type", "") or ""
        ).lower()
        if not optimizer_type:
            optimizer_type = (
                "adamw8bit"
                if getattr(self.training_config, "use_8bit_adam", True)
                else "adamw"
            )
        optimizer = build_optimizer(
            trainable_params,
            optimizer_type=optimizer_type,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=device_type,
        )
        yield 0, 0.0, f"Optimizer: {optimizer_type}"

        # Calculate total steps
        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_loader) / self.training_config.gradient_accumulation_steps
            ),
        )
        total_steps = steps_per_epoch * self.training_config.max_epochs
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))
        scheduler_type = str(
            getattr(self.training_config, "scheduler_type", "constant") or "constant"
        ).lower()
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=self.training_config.learning_rate,
        )
        yield 0, 0.0, f"Scheduler: {scheduler_type}"

        # Setup with Fabric - only the decoder (which has LoRA)
        self.module.model.decoder, optimizer = self.fabric.setup(
            self.module.model.decoder, optimizer
        )
        casted_opt_params, total_opt_params = _ensure_optimizer_params_fp32(optimizer)
        logger.info(
            f"Optimizer param dtype fixup: casted {casted_opt_params}/{total_opt_params} to fp32"
        )
        train_loader = self.fabric.setup_dataloaders(train_loader)

        # Handle resume from checkpoint (load AFTER Fabric setup)
        start_epoch = 0
        global_step = 0
        checkpoint_info = None

        if resume_from:
            try:
                resume_from = safe_path(resume_from)
            except ValueError:
                yield (
                    0,
                    0.0,
                    f"⚠️ Rejected unsafe checkpoint path: {resume_from}, starting fresh",
                )
                resume_from = None
        if resume_from and os.path.exists(resume_from):
            try:
                yield 0, 0.0, f"🔄 Loading checkpoint from {resume_from}..."

                # Load checkpoint using utility function
                checkpoint_info = load_training_checkpoint(
                    resume_from,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.module.device,
                )

                if checkpoint_info["adapter_path"]:
                    adapter_path = checkpoint_info["adapter_path"]
                    state_dict = _load_lora_resume_state_dict(
                        adapter_path,
                        self.module.device,
                    )

                    # Get the decoder (might be wrapped by Fabric)
                    decoder = self.module.model.decoder
                    if hasattr(decoder, "_forward_module"):
                        decoder = decoder._forward_module

                    decoder.load_state_dict(state_dict, strict=False)

                    start_epoch = checkpoint_info["epoch"]
                    global_step = checkpoint_info["global_step"]

                    status_parts = [
                        f"✅ Resumed from epoch {start_epoch}, step {global_step}"
                    ]
                    if checkpoint_info["loaded_optimizer"]:
                        status_parts.append("optimizer ✓")
                    if checkpoint_info["loaded_scheduler"]:
                        status_parts.append("scheduler ✓")
                    yield 0, 0.0, ", ".join(status_parts)
                else:
                    yield 0, 0.0, f"⚠️ No valid checkpoint found in {resume_from}"

            except Exception as e:
                logger.exception("Failed to load checkpoint")
                yield 0, 0.0, f"⚠️ Failed to load checkpoint: {e}, starting fresh"
                start_epoch = 0
                global_step = 0
        elif resume_from:
            yield 0, 0.0, f"⚠️ Checkpoint path not found: {resume_from}, starting fresh"

        # Training loop
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        self.module.model.decoder.train()

        for epoch in range(start_epoch, self.training_config.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start_time = time.time()

            for _batch_idx, batch in enumerate(train_loader):
                # Check for stop signal
                if training_state and training_state.get("should_stop", False):
                    yield (
                        global_step,
                        accumulated_loss / max(accumulation_step, 1),
                        "⏹️ Training stopped by user",
                    )
                    return

                # Forward pass
                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps

                # Backward pass
                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1

                # Optimizer step
                if (
                    accumulation_step
                    >= self.training_config.gradient_accumulation_steps
                ):
                    nonfinite_grads, grad_tensors = _count_nonfinite_grads(
                        trainable_params
                    )
                    if nonfinite_grads > 0:
                        optimizer.zero_grad(set_to_none=True)
                        yield (
                            global_step,
                            float("nan"),
                            (
                                f"⚠️ Non-finite gradients ({nonfinite_grads}/{grad_tensors}); "
                                "skipping optimizer step"
                            ),
                        )
                        accumulated_loss = 0.0
                        accumulation_step = 0
                        continue

                    self.fabric.clip_gradients(
                        self.module.model.decoder,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                        error_if_nonfinite=False,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                    global_step += 1
                    cache_every = getattr(
                        self.training_config, "empty_cache_every_n_steps", 0
                    )
                    if torch.cuda.is_available() and cache_every > 0:
                        if global_step % cache_every == 0:
                            torch.cuda.empty_cache()

                    # Log
                    avg_loss = accumulated_loss / accumulation_step
                    if global_step % self.training_config.log_every_n_steps == 0:
                        if training_state is not None:
                            if ema_loss is None:
                                ema_loss = avg_loss
                            else:
                                ema_loss = (
                                    ema_alpha * avg_loss + (1 - ema_alpha) * ema_loss
                                )
                            training_state["plot_steps"].append(global_step)
                            training_state["plot_loss"].append(avg_loss)
                            training_state["plot_ema"].append(ema_loss)
                        self.fabric.log("train/loss", avg_loss, step=global_step)
                        self.fabric.log(
                            "train/lr", scheduler.get_last_lr()[0], step=global_step
                        )
                        yield (
                            global_step,
                            avg_loss,
                            f"Epoch {epoch + 1}/{self.training_config.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                        )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            # Flush remainder to avoid dropping gradients when epoch length is not
            # divisible by gradient_accumulation_steps.
            if accumulation_step > 0:
                remainder_skipped = False
                nonfinite_grads, grad_tensors = _count_nonfinite_grads(trainable_params)
                if nonfinite_grads > 0:
                    optimizer.zero_grad(set_to_none=True)
                    yield (
                        global_step,
                        float("nan"),
                        (
                            f"⚠️ Non-finite gradients ({nonfinite_grads}/{grad_tensors}); "
                            "skipping optimizer remainder step"
                        ),
                    )
                    accumulated_loss = 0.0
                    accumulation_step = 0
                    remainder_skipped = True
                else:
                    self.fabric.clip_gradients(
                        self.module.model.decoder,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                        error_if_nonfinite=False,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

                if not remainder_skipped:
                    global_step += 1
                    cache_every = getattr(
                        self.training_config, "empty_cache_every_n_steps", 0
                    )
                    if torch.cuda.is_available() and cache_every > 0:
                        if global_step % cache_every == 0:
                            torch.cuda.empty_cache()
                    avg_loss = accumulated_loss / accumulation_step
                    if global_step % self.training_config.log_every_n_steps == 0:
                        if training_state is not None:
                            if ema_loss is None:
                                ema_loss = avg_loss
                            else:
                                ema_loss = ema_alpha * avg_loss + (
                                    1 - ema_alpha
                                ) * ema_loss
                            training_state["plot_steps"].append(global_step)
                            training_state["plot_loss"].append(avg_loss)
                            training_state["plot_ema"].append(ema_loss)
                        self.fabric.log("train/loss", avg_loss, step=global_step)
                        self.fabric.log(
                            "train/lr", scheduler.get_last_lr()[0], step=global_step
                        )
                        yield (
                            global_step,
                            avg_loss,
                            f"Epoch {epoch + 1}/{self.training_config.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                        )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            if training_state is not None:
                if ema_loss is None:
                    ema_loss = avg_epoch_loss
                else:
                    ema_loss = ema_alpha * avg_epoch_loss + (1 - ema_alpha) * ema_loss
                # Avoid duplicating the last step if it was already logged in the batch loop
                plot_steps = training_state["plot_steps"]
                if not plot_steps or plot_steps[-1] != global_step:
                    training_state["plot_steps"].append(global_step)
                    training_state["plot_loss"].append(avg_epoch_loss)
                    training_state["plot_ema"].append(ema_loss)
            self.fabric.log("train/epoch_loss", avg_epoch_loss, step=epoch + 1)

            best_candidate = avg_epoch_loss
            best_candidate_label = "training loss"

            # Validation and best-checkpoint tracking
            if val_loader is not None:
                self.module.model.decoder.eval()
                total_val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        v_loss = self.module.training_step(val_batch, record_loss=False)
                        total_val_loss += v_loss.item()
                        n_val += 1
                self.module.model.decoder.train()
                val_loss = total_val_loss / max(n_val, 1)
                best_candidate = val_loss
                best_candidate_label = "validation loss"
                if training_state is not None:
                    training_state["plot_val_steps"].append(global_step)
                    training_state["plot_val_loss"].append(val_loss)
                yield (
                    global_step,
                    avg_epoch_loss,
                    f"Validation loss: {val_loss:.6f}",
                )

            if (
                getattr(self.training_config, "save_best", False)
                and epoch + 1 >= int(getattr(self.training_config, "save_best_after", 1))
            ):
                is_new_best, smoothed_metric = best_tracker.observe(best_candidate)
                if is_new_best:
                    if training_state is not None:
                        training_state["plot_best_step"] = global_step
                    self.module.model.decoder.eval()
                    best_path = _save_best_lora_checkpoint(
                        self.module.model,
                        optimizer,
                        scheduler,
                        self.training_config,
                        epoch + 1,
                        global_step,
                    )
                    self.module.model.decoder.train()
                    yield (
                        global_step,
                        avg_epoch_loss,
                        (
                            f"🏆 New best {_adapter_display_name(self.training_config)} "
                            f"saved ({best_candidate_label} MA"
                            f"{best_tracker.smoothing_window}: "
                            f"{smoothed_metric:.6f}, raw: {best_candidate:.6f}) "
                            f"to {best_path}"
                        ),
                    )

            # Save checkpoint
            if _should_save_epoch_checkpoint(self.training_config, epoch + 1):
                _save_lora_epoch_checkpoint(
                    self.module.model,
                    optimizer,
                    scheduler,
                    self.training_config,
                    epoch + 1,
                    global_step,
                )
                yield (
                    global_step,
                    avg_epoch_loss,
                    f"💾 Checkpoint saved at epoch {epoch + 1}",
                )

            sample_every = int(
                getattr(self.training_config, "sample_every_n_epochs", 0) or 0
            )
            if sample_every > 0 and (epoch + 1) % sample_every == 0:
                yield (
                    global_step,
                    avg_epoch_loss,
                    (
                        f"Generating {_adapter_display_name(self.training_config)} "
                        f"sample for epoch {epoch + 1}..."
                    ),
                )
                sample_msg = self._generate_checkpoint_sample(
                    "",
                    epoch + 1,
                )
                if sample_msg:
                    yield global_step, avg_epoch_loss, sample_msg

        # Save final model and a resumable final state regardless of
        # periodic checkpoint interval.
        final_path = _save_final_lora_artifacts(
            self.module.model,
            optimizer,
            scheduler,
            self.training_config,
            global_step,
        )

        final_loss = (
            self.module.training_losses[-1] if self.module.training_losses else 0.0
        )
        peak_vram = cuda_peak_gb()
        peak_suffix = (
            f"\nPeak training VRAM: {peak_vram:.2f} GiB" if peak_vram > 0 else ""
        )
        yield (
            global_step,
            final_loss,
            (
                f"✅ Training complete! {_adapter_display_name(self.training_config)} "
                f"saved to {final_path}{peak_suffix}"
            ),
        )

    def _generate_checkpoint_sample(
        self,
        checkpoint_dir: str,
        epoch: int,
    ) -> str:
        """Generate a low-spike audio sample during LoRA training."""

        sample_every = int(getattr(self.training_config, "sample_every_n_epochs", 0) or 0)
        if sample_every <= 0:
            return ""

        prompt = str(getattr(self.training_config, "sample_prompt", "") or "").strip()
        lyrics = str(getattr(self.training_config, "sample_lyrics", "") or "").strip()
        if not prompt or not lyrics:
            return "Sample generation skipped: prompt or lyrics is empty"

        sample_root = os.path.join(self.training_config.output_dir, "samples")
        artifact_basename = f"{getattr(self.training_config, 'lora_name', 'lora')}_{epoch}"

        with sample_generation_vram_guard(
            self.module,
            enabled=bool(getattr(
                self.training_config, "sample_offload_training_model", True
            )),
            target_device=self.module.device,
        ):
            result = run_training_sample_inprocess(
                handler=self.dit_handler,
                output_dir=sample_root,
                artifact_basename=artifact_basename,
                prompt=prompt,
                lyrics=lyrics,
                generation_settings=dict(
                    getattr(self.training_config, "sample_generation_settings", {})
                    or {}
                ),
                fallback_duration=float(
                    getattr(self.training_config, "sample_duration", 30.0)
                ),
                fallback_inference_steps=int(
                    getattr(self.training_config, "sample_inference_steps", 8)
                ),
                fallback_seed=int(getattr(self.training_config, "sample_seed", 42)),
                offload_generation=bool(
                    getattr(self.training_config, "sample_offload_generation", True)
                ),
            )

        peak = float(result.get("peak_vram_gb") or 0.0)
        if result.get("success"):
            audios = result.get("audios") or []
            first_path = audios[0].get("path") if audios else sample_root
            return f"Sample generated for epoch {epoch}: {first_path} (peak {peak:.2f} GiB)"
        error = result.get("error") or result.get("stderr_tail") or "unknown error"
        return f"Sample generation failed for epoch {epoch}: {error} (peak {peak:.2f} GiB)"

    def _train_basic(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Basic training loop without Fabric."""
        yield 0, 0.0, "🚀 Starting basic training loop..."

        os.makedirs(self.training_config.output_dir, exist_ok=True)

        train_loader = data_module.train_dataloader()
        val_loader = (
            data_module.val_dataloader()
            if hasattr(data_module, "val_dataloader")
            else None
        )

        trainable_params = [
            p for p in self.module.model.parameters() if p.requires_grad
        ]

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        cast_training_parameter_dtypes(
            self.module.model.decoder,
            frozen_dtype=self.module.dtype,
            keep_frozen_in_compute_dtype=bool(getattr(
                self.training_config, "keep_frozen_base_in_compute_dtype", True
            )),
        )

        optimizer_type = str(
            getattr(self.training_config, "optimizer_type", "") or ""
        ).lower()
        if not optimizer_type:
            optimizer_type = (
                "adamw8bit"
                if getattr(self.training_config, "use_8bit_adam", True)
                else "adamw"
            )
        optimizer = build_optimizer(
            trainable_params,
            optimizer_type=optimizer_type,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=self.module.device_type,
        )
        yield 0, 0.0, f"Optimizer: {optimizer_type}"

        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_loader) / self.training_config.gradient_accumulation_steps
            ),
        )
        total_steps = steps_per_epoch * self.training_config.max_epochs
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        scheduler_type = str(
            getattr(self.training_config, "scheduler_type", "constant") or "constant"
        ).lower()
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            lr=self.training_config.learning_rate,
        )
        yield 0, 0.0, f"Scheduler: {scheduler_type}"

        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0
        best_tracker = BestMetricTracker(
            smoothing_window=getattr(
                self.training_config,
                "save_best_smoothing_window",
                5,
            ),
            min_delta=getattr(self.training_config, "save_best_min_delta", 0.001),
        )
        optimizer.zero_grad(set_to_none=True)

        if training_state is not None:
            training_state.setdefault("plot_val_steps", [])
            training_state.setdefault("plot_val_loss", [])
            training_state.setdefault("plot_best_step", None)

        self.module.model.decoder.train()

        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    yield (
                        global_step,
                        accumulated_loss / max(accumulation_step, 1),
                        "⏹️ Training stopped",
                    )
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1

                if (
                    accumulation_step
                    >= self.training_config.gradient_accumulation_steps
                ):
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.training_config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1
                    cache_every = getattr(
                        self.training_config, "empty_cache_every_n_steps", 0
                    )
                    if torch.cuda.is_available() and cache_every > 0:
                        if global_step % cache_every == 0:
                            torch.cuda.empty_cache()

                    avg_loss = accumulated_loss / accumulation_step
                    if global_step % self.training_config.log_every_n_steps == 0:
                        yield (
                            global_step,
                            avg_loss,
                            f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                        )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            if accumulation_step > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, self.training_config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                cache_every = getattr(
                    self.training_config, "empty_cache_every_n_steps", 0
                )
                if torch.cuda.is_available() and cache_every > 0:
                    if global_step % cache_every == 0:
                        torch.cuda.empty_cache()

                avg_loss = accumulated_loss / accumulation_step
                if global_step % self.training_config.log_every_n_steps == 0:
                    yield (
                        global_step,
                        avg_loss,
                        f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                    )

                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            best_candidate = avg_epoch_loss
            best_candidate_label = "training loss"

            if val_loader is not None:
                self.module.model.decoder.eval()
                total_val_loss = 0.0
                n_val = 0
                with torch.no_grad():
                    for val_batch in val_loader:
                        v_loss = self.module.training_step(val_batch, record_loss=False)
                        total_val_loss += v_loss.item()
                        n_val += 1
                self.module.model.decoder.train()
                val_loss = total_val_loss / max(n_val, 1)
                best_candidate = val_loss
                best_candidate_label = "validation loss"
                if training_state is not None:
                    training_state["plot_val_steps"].append(global_step)
                    training_state["plot_val_loss"].append(val_loss)
                yield (
                    global_step,
                    avg_epoch_loss,
                    f"Validation loss: {val_loss:.6f}",
                )

            yield (
                global_step,
                avg_epoch_loss,
                f"✅ Epoch {epoch + 1}/{self.training_config.max_epochs} in {epoch_time:.1f}s",
            )

            if (
                getattr(self.training_config, "save_best", False)
                and epoch + 1 >= int(getattr(self.training_config, "save_best_after", 1))
            ):
                is_new_best, smoothed_metric = best_tracker.observe(best_candidate)
                if is_new_best:
                    if training_state is not None:
                        training_state["plot_best_step"] = global_step
                    self.module.model.decoder.eval()
                    best_path = _save_best_lora_checkpoint(
                        self.module.model,
                        optimizer,
                        scheduler,
                        self.training_config,
                        epoch + 1,
                        global_step,
                    )
                    self.module.model.decoder.train()
                    yield (
                        global_step,
                        avg_epoch_loss,
                        (
                            f"🏆 New best {_adapter_display_name(self.training_config)} "
                            f"saved ({best_candidate_label} MA"
                            f"{best_tracker.smoothing_window}: "
                            f"{smoothed_metric:.6f}, raw: {best_candidate:.6f}) "
                            f"to {best_path}"
                        ),
                    )

            if _should_save_epoch_checkpoint(self.training_config, epoch + 1):
                _save_lora_epoch_checkpoint(
                    self.module.model,
                    optimizer,
                    scheduler,
                    self.training_config,
                    epoch + 1,
                    global_step,
                )
                yield global_step, avg_epoch_loss, "💾 Checkpoint saved"

            sample_every = int(
                getattr(self.training_config, "sample_every_n_epochs", 0) or 0
            )
            if sample_every > 0 and (epoch + 1) % sample_every == 0:
                yield (
                    global_step,
                    avg_epoch_loss,
                    (
                        f"Generating {_adapter_display_name(self.training_config)} "
                        f"sample for epoch {epoch + 1}..."
                    ),
                )
                sample_msg = self._generate_checkpoint_sample(
                    "",
                    epoch + 1,
                )
                if sample_msg:
                    yield global_step, avg_epoch_loss, sample_msg

        final_path = _save_final_lora_artifacts(
            self.module.model,
            optimizer,
            scheduler,
            self.training_config,
            global_step,
        )
        final_loss = (
            self.module.training_losses[-1] if self.module.training_losses else 0.0
        )
        peak_vram = cuda_peak_gb()
        peak_suffix = (
            f"\nPeak training VRAM: {peak_vram:.2f} GiB" if peak_vram > 0 else ""
        )
        yield (
            global_step,
            final_loss,
            (
                f"✅ Training complete! {_adapter_display_name(self.training_config)} "
                f"saved to {final_path}{peak_suffix}"
            ),
        )

    def stop(self):
        """Stop training."""
        self.is_training = False


class PreprocessedLoKRModule(nn.Module):
    """LoKr training module using preprocessed tensors."""

    def __init__(
        self,
        model: nn.Module,
        lokr_config: LoKRConfig,
        training_config: TrainingConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.lokr_config = lokr_config
        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(self.device_type)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")
        timestep_schedule = build_shifted_timestep_schedule(
            training_config.num_inference_steps,
            training_config.shift,
        )
        self.timesteps_tensor = torch.tensor(
            timestep_schedule, device=self.device, dtype=self.dtype
        )
        logger.info(
            f"LoKr training timestep schedule: steps={training_config.num_inference_steps}, "
            f"shift={training_config.shift}"
        )
        self.force_input_grads_for_checkpointing = False
        self.lycoris_net = None

        if check_lycoris_available():
            self.model, self.lycoris_net, self.lokr_info = inject_lokr_into_dit(
                model, lokr_config
            )
            logger.info(
                f"LoKr injected: {self.lokr_info['trainable_params']:,} trainable params"
            )
        else:
            self.model = model
            self.lokr_info = {}
            logger.warning("LyCORIS not available, training without LoKr adapters")

        self.config = model.config
        self.training_losses = []

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single LoKr training step."""
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(
                device_type=self.device_type, dtype=self.dtype
            )
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            target_latents = batch["target_latents"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            attention_mask = batch["attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_hidden_states = batch["encoder_hidden_states"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            encoder_attention_mask = batch["encoder_attention_mask"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )
            context_latents = batch["context_latents"].to(
                self.device, dtype=self.dtype, non_blocking=self.transfer_non_blocking
            )

            bsz = target_latents.shape[0]
            x1 = torch.randn_like(target_latents)
            x0 = target_latents

            t, _ = sample_discrete_timestep(bsz, self.timesteps_tensor)
            t_ = t.unsqueeze(-1).unsqueeze(-1)
            xt = t_ * x1 + (1.0 - t_) * x0
            if self.force_input_grads_for_checkpointing:
                xt = xt.requires_grad_(True)

            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

        diffusion_loss = diffusion_loss.float()
        self.training_losses.append(diffusion_loss.item())
        return diffusion_loss


class LoKRTrainer:
    """High-level trainer for ACE-Step LoKr fine-tuning."""

    def __init__(
        self,
        dit_handler,
        lokr_config: LoKRConfig,
        training_config: TrainingConfig,
    ):
        self.dit_handler = dit_handler
        self.lokr_config = lokr_config
        # Validate output_dir early so all downstream path operations are safe
        training_config.output_dir = safe_path(training_config.output_dir)
        self.training_config = training_config

        self.module = None
        self.fabric = None
        self.is_training = False
        self.run_metadata: Dict[str, Any] = {}

    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Train LoKr adapters from preprocessed tensors."""
        self.is_training = True
        try:
            if _unwrap_stale_fabric_decoder(self.dit_handler.model):
                logger.info(
                    "Unwrapped stale Fabric decoder wrapper before LoKr training"
                )

            quantization_mode = getattr(self.dit_handler, "quantization", None)
            if quantization_mode is not None:
                yield (
                    0,
                    0.0,
                    (
                        "❌ LoKr training requires a non-quantized DiT model. "
                        f"Current quantization: {quantization_mode}. "
                        "Re-initialize service with INT8 Quantization disabled, then retry training."
                    ),
                )
                return

            try:
                tensor_dir = safe_path(tensor_dir)
            except ValueError:
                yield 0, 0.0, f"❌ Rejected unsafe tensor directory: {tensor_dir}"
                return
            if not os.path.isdir(tensor_dir):
                yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
                return

            if not check_lycoris_available():
                yield (
                    0,
                    0.0,
                    "❌ LyCORIS not installed. Install lycoris-lora to train LoKr.",
                )
                return

            torch.manual_seed(self.training_config.seed)
            random.seed(self.training_config.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.training_config.seed)
            try:
                import numpy as np

                np.random.seed(self.training_config.seed)
            except Exception:
                pass

            self.module = PreprocessedLoKRModule(
                model=self.dit_handler.model,
                lokr_config=self.lokr_config,
                training_config=self.training_config,
                device=self.dit_handler.device,
                dtype=self.dit_handler.dtype,
            )
            ckpt_enabled, cache_disabled, input_grads_enabled = (
                _configure_training_memory_features(self.module.model.decoder)
            )
            self.module.force_input_grads_for_checkpointing = ckpt_enabled
            logger.info(
                f"Training memory features: gradient_checkpointing={ckpt_enabled}, "
                f"use_cache_disabled={cache_disabled}, input_grads_enabled={input_grads_enabled}"
            )

            data_module = PreprocessedDataModule(
                tensor_dir=tensor_dir,
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                prefetch_factor=self.training_config.prefetch_factor,
                persistent_workers=self.training_config.persistent_workers,
                pin_memory_device=self.training_config.pin_memory_device,
                val_split=self.training_config.val_split,
            )
            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                yield 0, 0.0, "❌ No valid samples found in tensor directory"
                return

            self.run_metadata = {
                "tensor_dir": tensor_dir,
                "num_samples": int(len(data_module.train_dataset)),
                "training_config": self.training_config.to_dict(),
            }

            yield (
                0,
                0.0,
                f"📂 Loaded {len(data_module.train_dataset)} preprocessed samples",
            )
            if ckpt_enabled:
                yield 0, 0.0, "🧠 Gradient checkpointing enabled for decoder"
            else:
                yield (
                    0,
                    0.0,
                    "⚠️ Gradient checkpointing not enabled (model wrapper did not expose it)",
                )
            if not input_grads_enabled:
                yield (
                    0,
                    0.0,
                    "ℹ️ Input-grad hook not available on this DiT; using explicit checkpointing fallback",
                )

            if LIGHTNING_AVAILABLE:
                yield from self._train_with_fabric(data_module, training_state)
            else:
                yield from self._train_basic(data_module, training_state)

        except Exception as e:
            logger.exception("LoKr training failed")
            yield 0, 0.0, f"❌ Training failed: {str(e)}"
        finally:
            if self.module is not None and hasattr(self.module, "model"):
                _unwrap_stale_fabric_decoder(self.module.model)
            if (
                getattr(self, "dit_handler", None) is not None
                and getattr(self.dit_handler, "model", None) is not None
            ):
                _unwrap_stale_fabric_decoder(self.dit_handler.model)
            self.is_training = False

    def _train_with_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
    ) -> Generator[Tuple[int, float, str], None, None]:
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        device_type = self.module.device_type
        precision = _select_fabric_precision(device_type)
        accelerator = (
            device_type if device_type in ("cuda", "xpu", "mps", "cpu") else "auto"
        )
        manual_nonfinite_check = not precision.endswith("-mixed")

        tb_logger = None
        try:
            tb_logger = TensorBoardLogger(
                root_dir=self.training_config.output_dir,
                name="logs",
            )
        except ModuleNotFoundError as exc:
            logger.warning(
                f"TensorBoard logger unavailable, continuing without logger: {exc}"
            )

        fabric_kwargs = {
            "accelerator": accelerator,
            "devices": 1,
            "precision": precision,
        }
        if tb_logger is not None:
            fabric_kwargs["loggers"] = [tb_logger]
        self.fabric = Fabric(**fabric_kwargs)
        self.fabric.launch()

        yield (
            0,
            0.0,
            f"🚀 Starting training (device: {device_type}, precision: {precision})...",
        )
        if not manual_nonfinite_check:
            logger.info(
                "LoKr mixed precision detected: disabling pre-unscale non-finite grad checks; "
                "relying on AMP/GradScaler handling."
            )

        if device_type == "mps" or precision.endswith("-mixed"):
            self.module.model.decoder = self.module.model.decoder.to(
                dtype=torch.float32
            )
        else:
            self.module.model.decoder = self.module.model.decoder.to(
                dtype=self.module.dtype
            )
        casted_trainable, total_trainable_tensors = _ensure_trainable_params_fp32(
            self.module.model.decoder
        )
        if (
            total_trainable_tensors == 0
            and getattr(self.module, "lycoris_net", None) is not None
        ):
            casted_fallback, total_fallback = _ensure_trainable_params_fp32(
                self.module.lycoris_net
            )
            casted_trainable += casted_fallback
            total_trainable_tensors += total_fallback
        logger.info(
            f"Trainable tensor dtype fixup: casted {casted_trainable}/{total_trainable_tensors} to fp32"
        )

        train_loader = data_module.train_dataloader()
        trainable_params = _collect_lokr_trainable_params(
            self.module.model,
            getattr(self.module, "lycoris_net", None),
        )
        param_name_lookup = _build_param_name_lookup(
            self.module.model,
            getattr(self.module, "lycoris_net", None),
        )

        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return
        if total_trainable_tensors == 0:
            logger.warning(
                "LoKr trainable params discovered via LyCORIS fallback traversal; "
                "decoder parameter traversal returned 0 trainables."
            )

        yield (
            0,
            0.0,
            f"🎯 Training {sum(p.numel() for p in trainable_params):,} parameters",
        )

        optimizer_kwargs = {
            "lr": self.training_config.learning_rate,
            "weight_decay": self.training_config.weight_decay,
        }
        if self.module.device.type == "cuda":
            optimizer_kwargs["fused"] = True
        optimizer = AdamW(trainable_params, **optimizer_kwargs)

        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_loader) / self.training_config.gradient_accumulation_steps
            ),
        )
        total_steps = steps_per_epoch * self.training_config.max_epochs
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            T_mult=1,
            eta_min=self.training_config.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        self.module.model.decoder, optimizer = self.fabric.setup(
            self.module.model.decoder, optimizer
        )
        casted_opt_params, total_opt_params = _ensure_optimizer_params_fp32(optimizer)
        logger.info(
            f"Optimizer param dtype fixup: casted {casted_opt_params}/{total_opt_params} to fp32"
        )
        train_loader = self.fabric.setup_dataloaders(train_loader)

        accumulation_step = 0
        accumulated_loss = 0.0
        global_step = 0
        optimizer.zero_grad(set_to_none=True)
        self.module.model.decoder.train()

        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    yield (
                        global_step,
                        accumulated_loss / max(accumulation_step, 1),
                        "⏹️ Training stopped by user",
                    )
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps

                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1

                if (
                    accumulation_step
                    >= self.training_config.gradient_accumulation_steps
                ):
                    if manual_nonfinite_check:
                        nonfinite_grads, grad_tensors, nonfinite_details = (
                            _count_nonfinite_grads_detailed(
                                trainable_params,
                                param_name_lookup,
                                detail_limit=10,
                            )
                        )
                        if nonfinite_grads > 0:
                            if nonfinite_details:
                                logger.warning(
                                    f"LoKr non-finite gradients ({nonfinite_grads}/{grad_tensors}) at epoch "
                                    f"{epoch + 1}, step {global_step}. Top offending tensors:\n"
                                    + "\n".join(f"  - {d}" for d in nonfinite_details)
                                )
                            optimizer.zero_grad(set_to_none=True)
                            yield (
                                global_step,
                                float("nan"),
                                (
                                    f"⚠️ Non-finite gradients ({nonfinite_grads}/{grad_tensors}); "
                                    "skipping optimizer step (see logs for tensor names)"
                                ),
                            )
                            accumulated_loss = 0.0
                            accumulation_step = 0
                            continue

                    self.fabric.clip_gradients(
                        self.module.model.decoder,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                        error_if_nonfinite=False,
                    )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    avg_loss = accumulated_loss / accumulation_step
                    if global_step % self.training_config.log_every_n_steps == 0:
                        self.fabric.log("train/loss", avg_loss, step=global_step)
                        self.fabric.log(
                            "train/lr", scheduler.get_last_lr()[0], step=global_step
                        )
                        yield (
                            global_step,
                            avg_loss,
                            (
                                f"Epoch {epoch + 1}/{self.training_config.max_epochs}, "
                                f"Step {global_step}, Loss: {avg_loss:.4f}"
                            ),
                        )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            if accumulation_step > 0:
                if manual_nonfinite_check:
                    nonfinite_grads, grad_tensors, nonfinite_details = (
                        _count_nonfinite_grads_detailed(
                            trainable_params,
                            param_name_lookup,
                            detail_limit=10,
                        )
                    )
                    if nonfinite_grads > 0:
                        if nonfinite_details:
                            logger.warning(
                                f"LoKr non-finite remainder gradients ({nonfinite_grads}/{grad_tensors}) at epoch "
                                f"{epoch + 1}, step {global_step}. Top offending tensors:\n"
                                + "\n".join(f"  - {d}" for d in nonfinite_details)
                            )
                        optimizer.zero_grad(set_to_none=True)
                        yield (
                            global_step,
                            float("nan"),
                            (
                                f"⚠️ Non-finite gradients ({nonfinite_grads}/{grad_tensors}); "
                                "skipping optimizer remainder step (see logs for tensor names)"
                            ),
                        )
                        accumulated_loss = 0.0
                        accumulation_step = 0
                        continue

                self.fabric.clip_gradients(
                    self.module.model.decoder,
                    optimizer,
                    max_norm=self.training_config.max_grad_norm,
                    error_if_nonfinite=False,
                )

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                avg_loss = accumulated_loss / accumulation_step
                if global_step % self.training_config.log_every_n_steps == 0:
                    self.fabric.log("train/loss", avg_loss, step=global_step)
                    self.fabric.log(
                        "train/lr", scheduler.get_last_lr()[0], step=global_step
                    )
                    yield (
                        global_step,
                        avg_loss,
                        (
                            f"Epoch {epoch + 1}/{self.training_config.max_epochs}, "
                            f"Step {global_step}, Loss: {avg_loss:.4f}"
                        ),
                    )

                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_updates, 1)

            self.fabric.log("train/epoch_loss", avg_epoch_loss, step=epoch + 1)
            yield (
                global_step,
                avg_epoch_loss,
                (
                    f"✅ Epoch {epoch + 1}/{self.training_config.max_epochs} "
                    f"in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}"
                ),
            )

            if _should_save_epoch_checkpoint(self.training_config, epoch + 1):
                checkpoint_dir = os.path.join(
                    self.training_config.output_dir, "checkpoints", f"epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}"
                )
                save_lokr_training_checkpoint(
                    self.module.lycoris_net,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    checkpoint_dir,
                    lokr_config=self.lokr_config,
                    run_metadata=self.run_metadata,
                )
                yield (
                    global_step,
                    avg_epoch_loss,
                    f"💾 Checkpoint saved at epoch {epoch + 1}",
                )

        final_path = os.path.join(self.training_config.output_dir, "final")
        final_metadata: Dict[str, Any] = {"lokr_config": self.lokr_config.to_dict()}
        if self.run_metadata:
            final_metadata["run_metadata"] = self.run_metadata
        save_lokr_weights(
            self.module.lycoris_net,
            final_path,
            metadata=final_metadata,
        )
        final_loss = (
            self.module.training_losses[-1] if self.module.training_losses else 0.0
        )
        yield (
            global_step,
            final_loss,
            f"✅ Training complete! LoKr saved to {final_path}",
        )

    def _train_basic(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
    ) -> Generator[Tuple[int, float, str], None, None]:
        yield 0, 0.0, "🚀 Starting basic training loop..."
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        train_loader = data_module.train_dataloader()
        trainable_params = _collect_lokr_trainable_params(
            self.module.model,
            getattr(self.module, "lycoris_net", None),
        )
        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        optimizer = AdamW(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        steps_per_epoch = max(
            1,
            math.ceil(
                len(train_loader) / self.training_config.gradient_accumulation_steps
            ),
        )
        total_steps = steps_per_epoch * self.training_config.max_epochs
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        warmup_scheduler = LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            T_mult=1,
            eta_min=self.training_config.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        self.module.model.decoder.train()

        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    yield (
                        global_step,
                        accumulated_loss / max(accumulation_step, 1),
                        "⏹️ Training stopped",
                    )
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1

                if (
                    accumulation_step
                    >= self.training_config.gradient_accumulation_steps
                ):
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, self.training_config.max_grad_norm
                    )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    avg_loss = accumulated_loss / accumulation_step
                    if global_step % self.training_config.log_every_n_steps == 0:
                        yield (
                            global_step,
                            avg_loss,
                            f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                        )

                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            if accumulation_step > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, self.training_config.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                avg_loss = accumulated_loss / accumulation_step
                if global_step % self.training_config.log_every_n_steps == 0:
                    yield (
                        global_step,
                        avg_loss,
                        f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
                    )

                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            yield (
                global_step,
                avg_epoch_loss,
                f"✅ Epoch {epoch + 1}/{self.training_config.max_epochs} in {epoch_time:.1f}s",
            )

            if _should_save_epoch_checkpoint(self.training_config, epoch + 1):
                checkpoint_dir = os.path.join(
                    self.training_config.output_dir, "checkpoints", f"epoch_{epoch + 1}_loss_{avg_epoch_loss:.4f}"
                )
                save_lokr_training_checkpoint(
                    self.module.lycoris_net,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    checkpoint_dir,
                    lokr_config=self.lokr_config,
                    run_metadata=self.run_metadata,
                )
                yield global_step, avg_epoch_loss, "💾 Checkpoint saved"

        final_path = os.path.join(self.training_config.output_dir, "final")
        final_metadata: Dict[str, Any] = {"lokr_config": self.lokr_config.to_dict()}
        if self.run_metadata:
            final_metadata["run_metadata"] = self.run_metadata
        save_lokr_weights(
            self.module.lycoris_net,
            final_path,
            metadata=final_metadata,
        )
        final_loss = (
            self.module.training_losses[-1] if self.module.training_losses else 0.0
        )
        yield (
            global_step,
            final_loss,
            f"✅ Training complete! LoKr saved to {final_path}",
        )

    def stop(self):
        """Stop training."""
        self.is_training = False
