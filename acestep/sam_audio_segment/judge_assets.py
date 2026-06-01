"""Local SAM-Audio Judge asset preparation helpers."""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger

from .paths import DEFAULT_MODELS_DIR

JUDGE_MODEL_NAME = "Sam-Audio-Judge.safetensors"
JUDGE_BF16_MODEL_NAME = "Sam-Audio-Judge-BF16.safetensors"
JUDGE_RUNTIME_CONFIG_NAME = "config.json"
JUDGE_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "preprocessor_config.json",
)
JUDGE_ASSETS_DIR = Path(__file__).resolve().parent / "assets" / "sam_audio_judge"
JUDGE_ASSET_FILES = (JUDGE_RUNTIME_CONFIG_NAME, *JUDGE_TOKENIZER_FILES)


def local_judge_assets_available(
    models_dir: Path = DEFAULT_MODELS_DIR,
    assets_dir: Path = JUDGE_ASSETS_DIR,
) -> bool:
    """Return whether the Judge checkpoint and bundled metadata are present."""

    root = Path(models_dir)
    checkpoint_available = (
        (root / JUDGE_BF16_MODEL_NAME).is_file()
        or (root / JUDGE_MODEL_NAME).is_file()
    )
    return checkpoint_available and _bundled_judge_assets_available(Path(assets_dir))


def prepare_local_judge_model_dir(
    models_dir: Path = DEFAULT_MODELS_DIR,
    assets_dir: Path = JUDGE_ASSETS_DIR,
) -> Path:
    """Return the bundled folder layout expected by the Judge processor."""

    root = Path(models_dir)
    source = Path(assets_dir)
    _require_judge_checkpoint(root)
    _require_bundled_judge_assets(source)
    return source


def resolve_local_judge_checkpoint(models_dir: Path = DEFAULT_MODELS_DIR) -> Path:
    """Return the preferred local Judge checkpoint, creating BF16 if needed."""

    root = Path(models_dir)
    _require_judge_checkpoint(root)
    bf16_checkpoint = root / JUDGE_BF16_MODEL_NAME
    fp32_checkpoint = root / JUDGE_MODEL_NAME
    if bf16_checkpoint.is_file() and not fp32_checkpoint.is_file():
        return bf16_checkpoint
    if (
        bf16_checkpoint.is_file()
        and bf16_checkpoint.stat().st_mtime >= fp32_checkpoint.stat().st_mtime
    ):
        return bf16_checkpoint

    if "F32" in inspect_safetensors_dtypes(fp32_checkpoint):
        logger.info(
            "[sam_audio] Converting SAM-Audio Judge checkpoint to BF16: {}",
            bf16_checkpoint,
        )
        convert_safetensors_to_bf16(fp32_checkpoint, bf16_checkpoint)
        return bf16_checkpoint
    return fp32_checkpoint


def inspect_safetensors_dtypes(checkpoint: Path) -> set[str]:
    """Return dtype names from a safetensors checkpoint without loading tensors."""

    from safetensors import safe_open

    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        return {handle.get_slice(key).get_dtype() for key in handle.keys()}


def convert_safetensors_to_bf16(source: Path, target: Path) -> None:
    """Convert float32 tensors in a safetensors checkpoint to bfloat16."""

    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    if temporary.exists():
        temporary.unlink()

    tensors = {}
    with safe_open(str(source), framework="pt", device="cpu") as handle:
        metadata = handle.metadata()
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if tensor.dtype == torch.float32:
                tensor = tensor.to(torch.bfloat16)
            tensors[key] = tensor.contiguous()

    try:
        save_file(tensors, str(temporary), metadata=metadata)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _bundled_judge_assets_available(assets_dir: Path) -> bool:
    return all((assets_dir / name).is_file() for name in JUDGE_ASSET_FILES)


def _require_bundled_judge_assets(assets_dir: Path) -> None:
    missing = [
        str(assets_dir / name)
        for name in JUDGE_ASSET_FILES
        if not (assets_dir / name).is_file()
    ]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"Missing bundled SAM-Audio Judge metadata: {joined}")


def _require_judge_checkpoint(models_dir: Path) -> None:
    has_bf16 = (models_dir / JUDGE_BF16_MODEL_NAME).is_file()
    has_fp32 = (models_dir / JUDGE_MODEL_NAME).is_file()
    if has_bf16 or has_fp32:
        return
    raise FileNotFoundError(
        "Missing SAM-Audio Judge checkpoint: "
        f"{models_dir / JUDGE_BF16_MODEL_NAME} or {models_dir / JUDGE_MODEL_NAME}"
    )
