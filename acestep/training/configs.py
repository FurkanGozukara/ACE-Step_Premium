"""
Training Configuration Classes

Contains dataclasses for LoRA and training configurations.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.
    
    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor (alpha/r determines the scaling)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA/DoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
        use_dora: Whether to enable PEFT DoRA weight decomposition.
        target_mlp: Whether the target list includes MLP/FFN layers.
    """
    r: int = 8
    alpha: int = 128
    dropout: float = 0.0
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    use_dora: bool = False
    target_mlp: bool = False
    
    def to_dict(self):
        """Convert to dictionary for PEFT config."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
            "use_dora": self.use_dora,
            "target_mlp": self.target_mlp,
        }


@dataclass
class LoKRConfig:
    """Configuration for LoKr (Low-Rank Kronecker) training."""

    linear_dim: int = 64
    linear_alpha: int = 128
    factor: int = -1
    decompose_both: bool = False
    use_tucker: bool = False
    use_scalar: bool = False
    weight_decompose: bool = False
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    full_matrix: bool = False
    bypass_mode: bool = False
    rs_lora: bool = False
    unbalanced_factorization: bool = False

    def to_dict(self):
        """Convert to dictionary for LyCORIS config."""
        return {
            "linear_dim": self.linear_dim,
            "linear_alpha": self.linear_alpha,
            "factor": self.factor,
            "decompose_both": self.decompose_both,
            "use_tucker": self.use_tucker,
            "use_scalar": self.use_scalar,
            "weight_decompose": self.weight_decompose,
            "target_modules": self.target_modules,
            "full_matrix": self.full_matrix,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
            "unbalanced_factorization": self.unbalanced_factorization,
        }


@dataclass
class TrainingConfig:
    """Configuration for LoRA training process.
    
    Training uses:
    - Device-aware mixed precision (bf16 on CUDA/XPU, fp16 on MPS, fp32 on CPU)
    - Continuous logit-normal timesteps by default, with legacy discrete mode available
    - Randomly samples one configured timestep per training step
    
    Attributes:
        shift: Timestep shift factor used to build the training schedule
        num_inference_steps: Number of timestep points to sample from
        learning_rate: Initial learning rate
        batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_epochs: Maximum number of training epochs
        save_every_n_epochs: Save checkpoint every N epochs; 0 disables
            periodic checkpoints while final artifacts still save
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        mixed_precision: Preferred precision mode for logging/config tracking
        seed: Random seed for reproducibility
        output_dir: Directory to save LoRA safetensors, resume states, and samples
    """
    shift: float = 3.0
    num_inference_steps: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_epochs: int = 100
    save_every_n_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    use_fp8: bool = False
    gradient_checkpointing: bool = True
    activation_cpu_offload: bool = False
    offload_non_decoder: bool = True
    keep_frozen_base_in_compute_dtype: bool = True
    compile_model: bool = False
    use_8bit_adam: bool = True
    optimizer_type: str = "adamw8bit"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_epsilon: float = 1e-8
    adamw8bit_min_8bit_size: int = 4096
    adamw8bit_percentile_clipping: int = 100
    adamw8bit_block_wise: bool = True
    adamw8bit_paged: bool = False
    adafactor_epsilon1: float = 1e-30
    adafactor_epsilon2: float = 1e-3
    adafactor_clip_threshold: float = 1.0
    adafactor_decay_rate: float = -0.8
    adafactor_beta1: float = 0.0
    adafactor_scale_parameter: bool = False
    adafactor_relative_step: bool = False
    adafactor_warmup_init: bool = False
    scheduler_type: str = "constant"
    empty_cache_every_n_steps: int = 10
    seed: int = 42
    output_dir: str = "./lora_output"
    lora_name: str = "lora"
    adapter_type: str = "lora"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = True
    pin_memory_device: str = ""
    
    # Logging
    log_every_n_steps: int = 10

    # Validation (for loss curve and best-checkpoint tracking)
    val_split: float = 0.0
    save_best: bool = True
    save_best_after: int = 10
    save_best_smoothing_window: int = 5
    save_best_min_delta: float = 0.001

    # Corrected timestep controls
    timestep_mode: str = "continuous"
    cfg_ratio: float = 0.15
    timestep_mu: float = -0.4
    timestep_sigma: float = 1.0
    data_proportion: float = 0.5
    adaptive_timestep_ratio: float = 0.0

    # Optional checkpoint sample generation
    sample_every_n_epochs: int = 0
    sample_prompt: str = ""
    sample_lyrics: str = ""
    sample_duration: float = 30.0
    sample_inference_steps: int = 8
    sample_seed: int = 42
    sample_output_dir: str = ""
    sample_offload_training_model: bool = False
    sample_offload_generation: bool = True
    sample_generation_settings: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not 0.0 <= self.val_split < 1.0:
            raise ValueError("val_split must be in [0.0, 1.0).")
        if str(self.optimizer_type).lower() not in {"adamw", "adamw8bit", "adafactor"}:
            raise ValueError("optimizer_type must be adamw, adamw8bit, or adafactor.")
        if str(self.scheduler_type).lower() not in {
            "cosine",
            "cosine_restarts",
            "linear",
            "constant",
            "constant_with_warmup",
        }:
            raise ValueError("scheduler_type is not supported.")
        if str(self.timestep_mode).lower() not in {"continuous", "discrete"}:
            raise ValueError("timestep_mode must be continuous or discrete.")
        if not 0.0 <= float(self.cfg_ratio) <= 1.0:
            raise ValueError("cfg_ratio must be in [0.0, 1.0].")
        if not 0.0 <= float(self.adaptive_timestep_ratio) <= 1.0:
            raise ValueError("adaptive_timestep_ratio must be in [0.0, 1.0].")
        if int(self.save_best_after) < 1:
            raise ValueError("save_best_after must be >= 1.")
        if int(self.save_best_smoothing_window) < 1:
            raise ValueError("save_best_smoothing_window must be >= 1.")
        if float(self.save_best_min_delta) < 0.0:
            raise ValueError("save_best_min_delta must be >= 0.")

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "shift": self.shift,
            "num_inference_steps": self.num_inference_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "save_every_n_epochs": self.save_every_n_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "use_fp8": self.use_fp8,
            "gradient_checkpointing": self.gradient_checkpointing,
            "activation_cpu_offload": self.activation_cpu_offload,
            "offload_non_decoder": self.offload_non_decoder,
            "keep_frozen_base_in_compute_dtype": self.keep_frozen_base_in_compute_dtype,
            "compile_model": self.compile_model,
            "use_8bit_adam": self.use_8bit_adam,
            "optimizer_type": self.optimizer_type,
            "adam_beta1": self.adam_beta1,
            "adam_beta2": self.adam_beta2,
            "adam_epsilon": self.adam_epsilon,
            "adamw8bit_min_8bit_size": self.adamw8bit_min_8bit_size,
            "adamw8bit_percentile_clipping": self.adamw8bit_percentile_clipping,
            "adamw8bit_block_wise": self.adamw8bit_block_wise,
            "adamw8bit_paged": self.adamw8bit_paged,
            "adafactor_epsilon1": self.adafactor_epsilon1,
            "adafactor_epsilon2": self.adafactor_epsilon2,
            "adafactor_clip_threshold": self.adafactor_clip_threshold,
            "adafactor_decay_rate": self.adafactor_decay_rate,
            "adafactor_beta1": self.adafactor_beta1,
            "adafactor_scale_parameter": self.adafactor_scale_parameter,
            "adafactor_relative_step": self.adafactor_relative_step,
            "adafactor_warmup_init": self.adafactor_warmup_init,
            "scheduler_type": self.scheduler_type,
            "empty_cache_every_n_steps": self.empty_cache_every_n_steps,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "lora_name": self.lora_name,
            "adapter_type": self.adapter_type,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "pin_memory_device": self.pin_memory_device,
            "log_every_n_steps": self.log_every_n_steps,
            "val_split": self.val_split,
            "save_best": self.save_best,
            "save_best_after": self.save_best_after,
            "save_best_smoothing_window": self.save_best_smoothing_window,
            "save_best_min_delta": self.save_best_min_delta,
            "timestep_mode": self.timestep_mode,
            "cfg_ratio": self.cfg_ratio,
            "timestep_mu": self.timestep_mu,
            "timestep_sigma": self.timestep_sigma,
            "data_proportion": self.data_proportion,
            "adaptive_timestep_ratio": self.adaptive_timestep_ratio,
            "sample_every_n_epochs": self.sample_every_n_epochs,
            "sample_prompt": self.sample_prompt,
            "sample_lyrics": self.sample_lyrics,
            "sample_duration": self.sample_duration,
            "sample_inference_steps": self.sample_inference_steps,
            "sample_seed": self.sample_seed,
            "sample_output_dir": self.sample_output_dir,
            "sample_offload_training_model": self.sample_offload_training_model,
            "sample_offload_generation": self.sample_offload_generation,
            "sample_generation_settings": dict(self.sample_generation_settings),
        }

    def save_json(self, path: str | Path) -> None:
        """Save this training config as JSON."""

        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: str | Path) -> "TrainingConfig":
        """Load a training config JSON file."""

        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        fields = cls.__dataclass_fields__
        return cls(**{key: value for key, value in payload.items() if key in fields})
