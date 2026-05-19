"""
ACE-Step V1.5 Pipeline
Handler wrapper connecting model and UI
"""

import os
import sys


def _configure_stdio_for_windows() -> None:
    """Prefer UTF-8 console streams when the runtime allows reconfiguration."""

    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            continue


_configure_stdio_for_windows()

# Load environment variables from .env file at most once per process to avoid
# epoch-boundary stalls (e.g. on Windows when Gradio yields during training)
_env_loaded = False  # module-level so we never reload .env in the same process
try:
    from dotenv import load_dotenv

    if not _env_loaded:
        _current_file = os.path.abspath(__file__)
        _project_root = os.path.dirname(os.path.dirname(_current_file))
        _env_path = os.path.join(_project_root, ".env")
        _env_example_path = os.path.join(_project_root, ".env.example")
        if os.path.exists(_env_path):
            load_dotenv(_env_path)
            print(f"Loaded configuration from {_env_path}")
        elif os.path.exists(_env_example_path):
            load_dotenv(_env_example_path)
            print(f"Loaded configuration from {_env_example_path} (fallback)")
        _env_loaded = True
except ImportError:
    # python-dotenv not installed, skip loading .env
    pass

# Clear proxy settings that may affect Gradio
for proxy_var in [
    "http_proxy",
    "https_proxy",
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
]:
    os.environ.pop(proxy_var, None)

# Force torchaudio to use ffmpeg backend (torchcodec not available on XPU/Windows)
os.environ["TORCHAUDIO_USE_BACKEND"] = "ffmpeg"

try:
    # When executed as a module: `python -m acestep.acestep_v15_pipeline`
    from .cli_args import parse_quantization_arg
    from .dataset_handler import DatasetHandler
    from acestep.ui.gradio.i18n import get_i18n, available_languages_info
    from .lazy_runtime import (
        LazyAceStepHandler,
        LazyLLMHandler,
        build_startup_gpu_state,
    )
    from .gpu_config import (
        find_best_lm_model_on_disk,
        is_lm_model_size_allowed,
        resolve_lm_backend,
        set_global_gpu_config,
        VRAM_AUTO_OFFLOAD_THRESHOLD_GB,
    )
    from .model_downloader import (
        DEFAULT_BASE_DIT_MODEL,
        DEFAULT_LM_MODEL,
        DEFAULT_PREMIUM_DIT_MODEL,
        DEFAULT_TURBO_DIT_MODEL,
        SOURCE_BASE_DIT_MODEL,
        SOURCE_PREMIUM_DIT_MODEL,
        SOURCE_TURBO_DIT_MODEL,
        ensure_lm_model,
        get_models_dir,
    )
except ImportError:
    # When executed as a script: `python acestep/acestep_v15_pipeline.py`
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.insert(0, project_root)
    from acestep.cli_args import parse_quantization_arg
    from acestep.dataset_handler import DatasetHandler
    from acestep.ui.gradio.i18n import get_i18n, available_languages_info
    from acestep.lazy_runtime import (
        LazyAceStepHandler,
        LazyLLMHandler,
        build_startup_gpu_state,
    )
    from acestep.gpu_config import (
        find_best_lm_model_on_disk,
        is_lm_model_size_allowed,
        resolve_lm_backend,
        set_global_gpu_config,
        VRAM_AUTO_OFFLOAD_THRESHOLD_GB,
    )
    from acestep.model_downloader import (
        DEFAULT_BASE_DIT_MODEL,
        DEFAULT_LM_MODEL,
        DEFAULT_PREMIUM_DIT_MODEL,
        DEFAULT_TURBO_DIT_MODEL,
        SOURCE_BASE_DIT_MODEL,
        SOURCE_PREMIUM_DIT_MODEL,
        SOURCE_TURBO_DIT_MODEL,
        ensure_lm_model,
        get_models_dir,
    )


def _import_real_handlers():
    """Import the heavyweight runtime handler classes only when needed."""

    try:
        from .handler import AceStepHandler
        from .llm_inference import LLMHandler
    except ImportError:
        from acestep.handler import AceStepHandler
        from acestep.llm_inference import LLMHandler
    return AceStepHandler, LLMHandler


def _select_preferred_dit_model(available_models: list[str]) -> str:
    """Return the preferred startup DiT model from discovered models."""

    preferred_order = (
        DEFAULT_TURBO_DIT_MODEL,
        DEFAULT_PREMIUM_DIT_MODEL,
        DEFAULT_BASE_DIT_MODEL,
        SOURCE_TURBO_DIT_MODEL,
        SOURCE_PREMIUM_DIT_MODEL,
        SOURCE_BASE_DIT_MODEL,
        "acestep-v15-turbo",
        "acestep-v15-sft",
        "acestep-v15-base",
    )
    for preferred in preferred_order:
        if preferred in available_models:
            return preferred
    return available_models[0]


def _import_gradio_factory():
    """Import the Gradio interface factory lazily."""

    try:
        from .ui.gradio import create_gradio_interface
    except ImportError:
        from acestep.ui.gradio import create_gradio_interface
    return create_gradio_interface


def _import_generation_cancel_route():
    """Import the Gradio generation cancel route registrar lazily."""

    try:
        from .ui.gradio.events.generation.cancel_api import register_generation_cancel_route
    except ImportError:
        from acestep.ui.gradio.events.generation.cancel_api import (
            register_generation_cancel_route,
        )
    return register_generation_cancel_route


def create_demo(init_params=None, language="en"):
    """
    Create Gradio demo interface

    Args:
        init_params: Dictionary containing initialization parameters and state.
                    If None, service will not be pre-initialized.
                    Keys: 'pre_initialized' (bool), 'checkpoint', 'config_path', 'device',
                          'init_llm', 'lm_model_path', 'backend', 'use_flash_attention',
                          'offload_to_cpu', 'offload_dit_to_cpu', 'init_status',
                          'dit_handler', 'llm_handler' (initialized handlers if pre-initialized),
                          'language' (UI language code)
        language: UI language code ('en', 'zh', 'ja', default: 'en')

    Returns:
        Gradio Blocks instance
    """
    startup_gpu_config = (init_params or {}).get("gpu_config")
    if startup_gpu_config is not None:
        set_global_gpu_config(startup_gpu_config)

    # Use pre-initialized handlers if available, otherwise create new ones
    if (
        init_params
        and init_params.get("pre_initialized")
        and "dit_handler" in init_params
    ):
        dit_handler = init_params["dit_handler"]
        llm_handler = init_params["llm_handler"]
    else:
        startup_gpu_state = (init_params or {}).get("startup_gpu_state")
        dit_handler = LazyAceStepHandler(startup_gpu_state=startup_gpu_state)
        llm_handler = LazyLLMHandler()

    dataset_handler = DatasetHandler()  # Dataset handler

    # Create Gradio interface with all handlers and initialization parameters
    create_gradio_interface = _import_gradio_factory()
    demo = create_gradio_interface(
        dit_handler,
        llm_handler,
        dataset_handler,
        init_params=init_params,
        language=language,
    )

    return demo


def _resolve_startup_lm_backend(requested_backend: str | None, gpu_config) -> str:
    """Resolve the startup LM backend against hardware compatibility restrictions."""
    resolved_backend = resolve_lm_backend(requested_backend, gpu_config)
    normalized_backend = (requested_backend or "").strip().lower()

    if normalized_backend and normalized_backend != resolved_backend:
        print(
            f"Requested LM backend '{normalized_backend}' is not supported on this hardware. "
            f"Using '{resolved_backend}' instead."
        )

    return resolved_backend


def _select_startup_lm_model(args, llm_handler, gpu_config) -> bool:
    """Select the tier-recommended LM model for startup service initialization."""
    if args.lm_model_path is not None:
        return True

    available_lm_models = llm_handler.get_available_5hz_lm_models()
    if not available_lm_models:
        print(
            "Warning: No LM models available, skipping LM initialization",
            file=sys.stderr,
        )
        args.init_llm = False
        return False

    recommended_lm = getattr(gpu_config, "recommended_lm_model", "")
    args.lm_model_path = find_best_lm_model_on_disk(
        recommended_lm,
        available_lm_models,
    )
    print(f"Using default LM model: {args.lm_model_path}")
    return bool(args.lm_model_path)


def _apply_4b_lm_memory_safety(args, gpu_config, gpu_memory_gb: float) -> None:
    """Apply startup offload and fallback rules for 4B LM selections."""
    if not args.lm_model_path or "4B" not in args.lm_model_path:
        return

    if 0 < gpu_memory_gb <= 24 and not args.offload_to_cpu:
        args.offload_to_cpu = True
        print(
            "Auto-enabling CPU offload "
            f"(4B LM model requires offloading on {gpu_memory_gb:.0f}GB GPU)"
        )

    if 0 < gpu_memory_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB:
        available_lm_models = getattr(gpu_config, "available_lm_models", [])
        if not is_lm_model_size_allowed(args.lm_model_path, available_lm_models):
            fallback = args.lm_model_path.replace("4B", "1.7B")
            print(
                f"WARNING: 4B LM model is too large for {gpu_memory_gb:.0f}GB GPU. "
                f"Downgrading to 1.7B variant: {fallback}"
            )
            args.lm_model_path = fallback


def main():
    """Main entry function"""
    import argparse

    # Detect GPU info without importing torch into the parent Gradio process.
    startup_gpu_state = build_startup_gpu_state()
    gpu_config = startup_gpu_state.gpu_config
    set_global_gpu_config(gpu_config)  # Set global config for use across modules

    gpu_memory_gb = gpu_config.gpu_memory_gb
    _is_mac = startup_gpu_state.device_kind == "mps"
    # Enable auto-offload for GPUs below 20 GB.  16 GB GPUs cannot hold all
    # models simultaneously (DiT ~4.7 + VAE ~0.3 + text_enc ~1.2 + LM â‰¥1.2 +
    # activations) so they *must* offload.  The old threshold of 16 GB caused
    # 16 GB GPUs to never offload, leading to OOM.
    # Mac (Apple Silicon) uses unified memory â€” offloading provides no benefit.
    auto_offload = (
        (not _is_mac)
        and gpu_memory_gb > 0
        and gpu_memory_gb < VRAM_AUTO_OFFLOAD_THRESHOLD_GB
    )
    _default_backend = gpu_config.recommended_backend

    # Print GPU configuration info
    print(f"\n{'=' * 60}")
    print("GPU Configuration Detected:")
    print(f"{'=' * 60}")
    print(f"  Device: {startup_gpu_state.device_name}")
    print(f"  GPU Memory: {gpu_memory_gb:.2f} GB")
    print(f"  Configuration Tier: {gpu_config.tier}")
    print(
        f"  Max Duration (with LM): {gpu_config.max_duration_with_lm}s ({gpu_config.max_duration_with_lm // 60} min)"
    )
    print(
        f"  Max Duration (without LM): {gpu_config.max_duration_without_lm}s ({gpu_config.max_duration_without_lm // 60} min)"
    )
    print(f"  Max Batch Size (with LM): {gpu_config.max_batch_size_with_lm}")
    print(f"  Max Batch Size (without LM): {gpu_config.max_batch_size_without_lm}")
    print(f"  Default LM Init: {gpu_config.init_lm_default}")
    print(f"  Available LM Models: {gpu_config.available_lm_models or 'None'}")
    print(f"{'=' * 60}\n")

    if _is_mac:
        print(
            f"Apple Silicon (MPS) detected â€” unified memory {gpu_memory_gb:.1f}GB, no CPU offload needed, backend={_default_backend}"
        )
    elif auto_offload:
        print(
            f"Auto-enabling CPU offload (GPU {gpu_memory_gb:.1f}GB < {VRAM_AUTO_OFFLOAD_THRESHOLD_GB}GB threshold)"
        )
    elif gpu_memory_gb > 0:
        print(
            f"CPU offload disabled by default (GPU {gpu_memory_gb:.1f}GB >= {VRAM_AUTO_OFFLOAD_THRESHOLD_GB}GB threshold)"
        )
    else:
        print("No GPU detected, running on CPU")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.environ.setdefault("ACESTEP_PROJECT_ROOT", project_root)
    os.environ.setdefault(
        "ACESTEP_CHECKPOINTS_DIR",
        str(get_models_dir(project_root=project_root)),
    )

    # Define local outputs directory
    configured_output_dir = os.environ.get("ACESTEP_OUTPUT_DIR", "").strip()
    output_dir = configured_output_dir or os.path.join(project_root, "outputs")
    # Normalize path to use forward slashes for Gradio 6 compatibility on Windows
    output_dir = output_dir.replace("\\", "/")
    os.environ.setdefault("ACESTEP_OUTPUT_DIR", output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Initialize i18n with default language (en)
    get_i18n()

    parser = argparse.ArgumentParser(
        description="Gradio Demo for ACE-Step V1.5",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--port", type=int, default=None, help="Port to run the gradio server on"
    )
    parser.add_argument("--share", action="store_true", help="Create a public link")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--server",
        "--server-name",
        dest="server_name",
        type=str,
        default=None,
        help="Server name / host IP (optional, use 0.0.0.0 for all interfaces)",
    )

    # language argument
    available_languages = available_languages_info()
    parser.add_argument(
        "--language",
        type=str,
        default="en",
        choices=[language[0] for language in available_languages],
        help="UI language:\n  "
        + "\n  ".join(
            (
                code
                + f" ({native_name}"
                + (f"/{name})" if name != native_name else ")")
                for code, name, native_name in available_languages
            )
        ),
    )
    del available_languages

    parser.add_argument(
        "--allowed-path",
        action="append",
        default=[],
        help="Additional allowed file paths for Gradio (repeatable).",
    )

    # Service mode argument
    parser.add_argument(
        "--service_mode",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
        help="Enable service mode (default: False). When enabled, uses preset models and restricts UI options.",
    )

    # Service initialization arguments
    parser.add_argument(
        "--init_service",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=False,
        help="Initialize service on startup (default: False)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Checkpoint file path (optional, for display purposes)",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help=f"Main model path (e.g., '{DEFAULT_TURBO_DIT_MODEL}')",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "xpu", "cpu"],
        help="Processing device (default: auto)",
    )
    parser.add_argument(
        "--init_llm",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=None,
        help="Initialize 5Hz LM (default: auto based on GPU memory)",
    )
    parser.add_argument(
        "--lm_model_path",
        type=str,
        default=None,
        help="5Hz LM model path (e.g., 'acestep-5Hz-lm-0.6B')",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=_default_backend,
        choices=["vllm", "pt", "mlx"],
        help=f"5Hz LM backend (default: {_default_backend}, use 'mlx' for native Apple Silicon acceleration)",
    )
    parser.add_argument(
        "--use_flash_attention",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=None,
        help="Use flash attention (default: auto-detect)",
    )
    parser.add_argument(
        "--offload_to_cpu",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=auto_offload,
        help=f"Offload models to CPU (default: {'True' if auto_offload else 'False'}, auto-detected based on GPU VRAM)",
    )
    _default_offload_dit = (
        gpu_config.offload_dit_to_cpu_default if not _is_mac else False
    )
    parser.add_argument(
        "--offload_dit_to_cpu",
        type=lambda x: x.lower() in ["true", "1", "yes"],
        default=_default_offload_dit,
        help=f"Offload DiT to CPU after diffusion (default: {_default_offload_dit}, auto-detected based on GPU tier)",
    )
    _default_quantization = None
    if gpu_config.quantization_default and not _is_mac:
        _default_quantization = "int8_weight_only"
        if startup_gpu_state.cuda_compute_major is not None and startup_gpu_state.cuda_compute_major < 7:
            _default_quantization = "w8a8_dynamic"
    parser.add_argument(
        "--quantization",
        type=parse_quantization_arg,
        default=_default_quantization,
        help=(
            "DiT quantization method: int8_weight_only, fp8_weight_only, "
            "fp8_scaled, w8a8_dynamic, or none "
            f"(default: {_default_quantization}, auto-detected based on GPU tier)"
        ),
    )
    parser.add_argument(
        "--download-source",
        type=str,
        default=None,
        choices=["huggingface", "modelscope", "auto"],
        help="Preferred model download source (default: auto-detect based on network)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Default batch size for generation. Defaults to 1 from VRAM presets if not specified",
    )

    # API mode argument
    parser.add_argument(
        "--enable-api",
        action="store_true",
        help="Enable API endpoints (default: False)",
    )

    # Authentication arguments
    parser.add_argument(
        "--auth-username",
        type=str,
        default=None,
        help="Username for Gradio authentication",
    )
    parser.add_argument(
        "--auth-password",
        type=str,
        default=None,
        help="Password for Gradio authentication",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="API key for API endpoints authentication",
    )

    args = parser.parse_args()

    # Enable API requires init_service
    if args.enable_api:
        args.init_service = True
        # Load config from .env if not specified
        if args.config_path is None:
            args.config_path = os.environ.get("ACESTEP_CONFIG_PATH")
        if args.lm_model_path is None:
            args.lm_model_path = os.environ.get("ACESTEP_LM_MODEL_PATH")
        if os.environ.get("ACESTEP_LM_BACKEND"):
            args.backend = os.environ.get("ACESTEP_LM_BACKEND")

    # Service mode defaults (can be configured via .env file)
    if args.service_mode:
        print("Service mode enabled - applying preset configurations...")
        # Default DiT model for service mode (from env or fallback)
        if args.config_path is None:
            args.config_path = os.environ.get(
                "SERVICE_MODE_DIT_MODEL", DEFAULT_TURBO_DIT_MODEL
            )
        # Default LM model for service mode (from env or fallback)
        if args.lm_model_path is None:
            args.lm_model_path = os.environ.get(
                "SERVICE_MODE_LM_MODEL", DEFAULT_LM_MODEL
            )
        # Backend for service mode (from env or fallback to vllm)
        args.backend = os.environ.get("SERVICE_MODE_BACKEND", "vllm")
        print(f"  DiT model: {args.config_path}")
        print(f"  LM model: {args.lm_model_path}")

    args.backend = _resolve_startup_lm_backend(args.backend, gpu_config)

    if args.service_mode:
        print(f"  Backend: {args.backend}")

    _apply_4b_lm_memory_safety(args, gpu_config, gpu_memory_gb)

    try:
        init_params = None
        dit_handler = None
        llm_handler = None

        # If init_service is True, perform initialization before creating UI
        if args.init_service:
            print("Initializing service from command line...")

            # Create handler instances for initialization
            AceStepHandler, LLMHandler = _import_real_handlers()
            dit_handler = AceStepHandler()
            llm_handler = LLMHandler()

            # Auto-select config_path if not provided
            if args.config_path is None:
                available_models = dit_handler.get_available_acestep_v15_models()
                if available_models:
                    args.config_path = _select_preferred_dit_model(available_models)
                    print(f"Auto-selected config_path: {args.config_path}")
                else:
                    print(
                        "Error: No available models found. Please specify --config_path",
                        file=sys.stderr,
                    )
                    sys.exit(1)

            if args.init_llm is None:
                args.init_llm = gpu_config.init_lm_default
                print(
                    f"Auto-setting init_llm to {args.init_llm} based on GPU configuration"
                )

            if args.init_llm and _select_startup_lm_model(
                args, llm_handler, gpu_config
            ):
                _apply_4b_lm_memory_safety(args, gpu_config, gpu_memory_gb)

            # Get project root (same logic as in handler)
            current_file = os.path.abspath(__file__)
            project_root = os.path.dirname(os.path.dirname(current_file))

            # Determine flash attention setting
            use_flash_attention = args.use_flash_attention
            if use_flash_attention is None:
                use_flash_attention = dit_handler.is_flash_attention_available(
                    args.device
                )

            # Determine download source preference
            prefer_source = None
            if args.download_source and args.download_source != "auto":
                prefer_source = args.download_source
                print(f"Using preferred download source: {prefer_source}")

            # Initialize DiT handler
            print(f"Initializing DiT model: {args.config_path} on {args.device}...")
            compile_model = os.environ.get(
                "ACESTEP_COMPILE_MODEL", ""
            ).strip().lower() in {"1", "true", "yes", "y", "on"}

            init_status, enable_generate = dit_handler.initialize_service(
                project_root=project_root,
                config_path=args.config_path,
                device=args.device,
                use_flash_attention=use_flash_attention,
                compile_model=compile_model,
                offload_to_cpu=args.offload_to_cpu,
                offload_dit_to_cpu=args.offload_dit_to_cpu,
                quantization=args.quantization,
                prefer_source=prefer_source,
            )

            if not enable_generate:
                print(f"Error initializing DiT model: {init_status}", file=sys.stderr)
                sys.exit(1)

            print(f"DiT model initialized successfully")

            # Initialize LM handler if requested
            lm_status = ""
            if args.init_llm:
                if args.init_llm and args.lm_model_path:
                    checkpoint_dir = str(get_models_dir(project_root=project_root))

                    # Ensure LM model is downloaded before initialization
                    prefer_source = None
                    if args.download_source and args.download_source != "auto":
                        prefer_source = args.download_source
                    try:
                        dl_ok, dl_msg = ensure_lm_model(
                            model_name=args.lm_model_path,
                            checkpoints_dir=checkpoint_dir,
                            prefer_source=prefer_source,
                        )
                        if not dl_ok:
                            print(
                                f"Warning: LM model download failed: {dl_msg}",
                                file=sys.stderr,
                            )
                    except Exception as e:
                        print(
                            f"Warning: Failed to download LM model: {e}",
                            file=sys.stderr,
                        )

                    print(
                        f"Initializing 5Hz LM: {args.lm_model_path} on {args.device}..."
                    )
                    lm_status, lm_success = llm_handler.initialize(
                        checkpoint_dir=checkpoint_dir,
                        lm_model_path=args.lm_model_path,
                        backend=args.backend,
                        device=args.device,
                        offload_to_cpu=args.offload_to_cpu,
                        dtype=None,
                    )

                    if lm_success:
                        print(f"5Hz LM initialized successfully")
                        llm_handler.last_init_params = {
                            "lm_model_path": args.lm_model_path,
                            "backend": args.backend,
                            "device": args.device,
                            "offload_to_cpu": args.offload_to_cpu,
                        }
                        init_status += f"\n{lm_status}"
                    else:
                        print(
                            f"Warning: 5Hz LM initialization failed: {lm_status}",
                            file=sys.stderr,
                        )
                        init_status += f"\n{lm_status}"

            # Prepare initialization parameters for UI
            init_params = {
                "pre_initialized": True,
                "service_mode": args.service_mode,
                "checkpoint": args.checkpoint,
                "config_path": args.config_path,
                "device": args.device,
                "init_llm": args.init_llm,
                "lm_model_path": args.lm_model_path,
                "backend": args.backend,
                "use_flash_attention": use_flash_attention,
                "offload_to_cpu": args.offload_to_cpu,
                "offload_dit_to_cpu": args.offload_dit_to_cpu,
                "quantization": args.quantization,
                "init_status": init_status,
                "enable_generate": enable_generate,
                "dit_handler": dit_handler,
                "llm_handler": llm_handler,
                "language": args.language,
                "gpu_config": gpu_config,  # Pass GPU config to UI
                "gpu_device_name": startup_gpu_state.device_name,
                "startup_gpu_state": startup_gpu_state,
                "output_dir": output_dir,  # Pass output dir to UI
                "default_batch_size": args.batch_size,  # Pass user-specified default batch size
            }

            print("Service initialization completed successfully!")

        # Create and launch demo
        print(f"Creating Gradio interface with language: {args.language}...")

        # If not using init_service, still pass gpu_config to init_params
        if init_params is None:
            init_params = {
                "gpu_config": gpu_config,
                "gpu_device_name": startup_gpu_state.device_name,
                "startup_gpu_state": startup_gpu_state,
                "language": args.language,
                "output_dir": output_dir,  # Pass output dir to UI
                "default_batch_size": args.batch_size,  # Pass user-specified default batch size
            }

        demo = create_demo(init_params=init_params, language=args.language)

        # Enable queue for multi-user support
        # This ensures proper request queuing and prevents concurrent generation conflicts
        print("Enabling queue for multi-user support...")
        demo.queue(
            max_size=20,  # Maximum queue size (adjust based on your needs)
            status_update_rate="auto",  # Update rate for queue status
            default_concurrency_limit=1,  # Prevents VRAM saturation
        )
        _import_generation_cancel_route()(demo)

        launch_host = args.server_name or "127.0.0.1"
        launch_port = args.port if args.port is not None else 7860
        print(f"Launching server on {launch_host}:{launch_port}...")

        # Setup authentication if provided
        auth = None
        if args.auth_username and args.auth_password:
            auth = (args.auth_username, args.auth_password)
            print("Authentication enabled")

        from acestep.training.scan_permissions import configure_scan_permissions

        scan_allowed_paths = configure_scan_permissions([output_dir, *args.allowed_path])
        print(
            "Dataset scan roots allowed for Gradio file serving: "
            f"{', '.join(scan_allowed_paths) if scan_allowed_paths else '(none)'}"
        )

        allowed_paths = []
        allowed_path_keys = set()

        def add_allowed_path(path: str) -> None:
            """Add a Gradio allowed path after canonicalising it."""
            if not path:
                return
            try:
                normalised = os.path.normpath(os.path.realpath(os.path.abspath(path)))
            except (OSError, ValueError):
                normalised = path
            key = os.path.normcase(normalised)
            if key not in allowed_path_keys:
                allowed_path_keys.add(key)
                allowed_paths.append(normalised)

        add_allowed_path(output_dir)
        for p in scan_allowed_paths:
            add_allowed_path(p)
        for p in args.allowed_path:
            add_allowed_path(p)

        # Enable API endpoints if requested
        launch_kwargs = {
            "share": args.share,
            "debug": args.debug,
            "show_error": True,
            "inbrowser": True,
            "auth": auth,
            "allowed_paths": allowed_paths,
            "theme": getattr(demo, "_ace_launch_theme", None),
            "css": getattr(demo, "_ace_launch_css", None),
            "head": getattr(demo, "_ace_launch_head", None),
            "_app": demo.app,
        }
        if args.server_name:
            launch_kwargs["server_name"] = args.server_name
        if args.port is not None:
            launch_kwargs["server_port"] = args.port

        if args.enable_api:
            print("Enabling API endpoints...")
            from acestep.ui.gradio.api.api_routes import setup_api_routes

            # Launch Gradio first with prevent_thread_lock=True
            demo.launch(
                prevent_thread_lock=True,  # Don't block, so we can add routes
                **launch_kwargs,
            )

            # Now add API routes to Gradio's FastAPI app (app is available after launch)
            setup_api_routes(demo, dit_handler, llm_handler, api_key=args.api_key)

            if args.api_key:
                print("API authentication enabled")
            print(
                "API endpoints enabled: /health, /v1/models, /release_task, /query_result, /create_random_sample, /format_lyrics"
            )

            # Keep the main thread alive
            try:
                while True:
                    import time

                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...")
        else:
            demo.launch(
                prevent_thread_lock=False,
                **launch_kwargs,
            )
    except Exception as e:
        print(f"Error launching Gradio: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
