"""Download and precheck helpers for service initialization."""

from pathlib import Path
from typing import Optional, Tuple

from loguru import logger

from acestep.model_downloader import (
    DEFAULT_VAE_VARIANT,
    check_model_exists,
    check_shared_main_components_exist,
    check_vae_exists,
    download_shared_main_components,
    ensure_dit_model,
    ensure_vae_model,
    get_legacy_fallback_model_name,
    resolve_existing_model_name,
)


class InitServiceDownloadsMixin:
    """Helpers that validate and fetch required model checkpoints."""

    @staticmethod
    def _resolve_model_name_for_loading(config_path: str, checkpoint_path: Path) -> str:
        """Return the installed model folder that should be loaded.

        Args:
            config_path: Requested DiT model directory name.
            checkpoint_path: Root directory containing model subdirectories.

        Returns:
            The requested model name when present, otherwise a supported older
            source-checkpoint folder for generated BF16 names.
        """
        resolved_name = resolve_existing_model_name(config_path, checkpoint_path)
        if resolved_name is None:
            return config_path
        if resolved_name != config_path:
            logger.warning(
                "[initialize_service] DiT model '{}' not found; using older "
                "compatible model folder '{}' instead.",
                config_path,
                resolved_name,
            )
        return resolved_name

    def _ensure_models_present(
        self,
        *,
        checkpoint_path: Path,
        config_path: str,
        prefer_source: Optional[str],
        vae_variant: Optional[str] = None,
    ) -> Optional[Tuple[str, bool]]:
        """Ensure required checkpoint assets exist locally, downloading when missing."""
        if config_path == "":
            logger.warning(
                "[initialize_service] Empty config_path; pass None to use the default model."
            )

        model_exists = check_model_exists(config_path, checkpoint_path)
        shared_exists = check_shared_main_components_exist(checkpoint_path)

        if not model_exists:
            fallback_name = get_legacy_fallback_model_name(config_path)
            if fallback_name:
                return (
                    f"ERROR: DiT model '{config_path}' was not found. "
                    f"Also checked older compatible folder '{fallback_name}', "
                    "but it is missing or has no model weights.",
                    False,
                )
            logger.info(
                f"[initialize_service] DiT model '{config_path}' not found, "
                "starting auto-download..."
            )
            success, msg = ensure_dit_model(
                config_path,
                checkpoint_path,
                prefer_source=prefer_source,
            )
            if not success:
                return f"ERROR: Failed to download DiT model '{config_path}': {msg}", False
            logger.info(f"[initialize_service] {msg}")
        elif not shared_exists:
            logger.info(
                "[initialize_service] Shared runtime components not found, "
                "starting auto-download..."
            )
            success, msg = download_shared_main_components(
                checkpoint_path, prefer_source=prefer_source
            )
            if not success:
                return f"ERROR: Failed to download shared runtime components: {msg}", False
            logger.info(f"[initialize_service] {msg}")

        if vae_variant and vae_variant != DEFAULT_VAE_VARIANT:
            if not check_vae_exists(vae_variant, checkpoint_path):
                logger.info(
                    f"[initialize_service] VAE variant '{vae_variant}' not found, "
                    "starting auto-download..."
                )
                success, msg = ensure_vae_model(
                    vae_variant, checkpoint_path, prefer_source=prefer_source
                )
                if not success:
                    return f"ERROR: Failed to download VAE variant '{vae_variant}': {msg}", False
                logger.info(f"[initialize_service] {msg}")

        return None

    @staticmethod
    def _sync_model_code_if_needed(config_path: str, checkpoint_path: Path) -> None:
        """Sync model-side python files when checkpoint code metadata diverges."""
        from acestep.model_downloader import _check_code_mismatch, _sync_model_code_files

        mismatched = _check_code_mismatch(config_path, checkpoint_path)
        if mismatched:
            logger.info(
                f"[initialize_service] Model code mismatch detected for '{config_path}': "
                f"{mismatched}. Auto-syncing from acestep/models/..."
            )
            _sync_model_code_files(config_path, checkpoint_path)
            logger.info("[initialize_service] Model code files synced successfully.")
