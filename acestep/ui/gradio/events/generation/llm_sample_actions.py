"""LLM sample-creation action handlers for generation UI."""

import gradio as gr

from acestep.inference import create_sample
from acestep.ui.gradio.i18n import t

from .llm_auto_init import ensure_llm_ready
from .llm_action_params import convert_lm_params
from .validation import clamp_duration_to_gpu_limit


def handle_create_sample(
    llm_handler,
    query: str,
    instrumental: bool,
    vocal_language: str,
    lm_temperature: float,
    lm_top_k: int,
    lm_top_p: float,
    constrained_decoding_debug: bool = False,
    lm_model_path: str | None = None,
    backend: str | None = None,
    device: str | None = None,
    offload_to_cpu: bool = False,
):
    """Handle Simple-mode sample creation.

    Args:
        llm_handler: LLM handler instance.
        query: User natural-language description.
        instrumental: Whether the sample should be instrumental.
        vocal_language: Preferred vocal language.
        lm_temperature: LLM temperature value.
        lm_top_k: LLM top-k value.
        lm_top_p: LLM top-p value.
        constrained_decoding_debug: Whether constrained-decoding debug logs are enabled.

    Returns:
        Tuple of 15 UI updates for wired Gradio outputs.
    """
    auto_init_ok, auto_init_status = ensure_llm_ready(
        llm_handler,
        lm_model_path=lm_model_path,
        backend=backend,
        device=device,
        offload_to_cpu=offload_to_cpu,
    )
    if not auto_init_ok:
        status_message = auto_init_status or t("messages.lm_not_initialized")
        gr.Warning(status_message)
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            False,
            gr.update(),
            gr.update(),
            status_message,
            gr.update(),
        )

    top_k_value, top_p_value = convert_lm_params(lm_top_k, lm_top_p)
    result = create_sample(
        llm_handler=llm_handler,
        query=query,
        instrumental=instrumental,
        vocal_language=vocal_language,
        temperature=lm_temperature,
        top_k=top_k_value,
        top_p=top_p_value,
        use_constrained_decoding=True,
        constrained_decoding_debug=constrained_decoding_debug,
    )

    if not result.success:
        gr.Warning(result.status_message or t("messages.sample_creation_failed"))
        return (
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(interactive=False),
            False,
            gr.update(),
            gr.update(),
            result.status_message or t("messages.sample_creation_failed"),
            gr.update(),
        )

    gr.Info(t("messages.sample_created"))
    clamped_duration = clamp_duration_to_gpu_limit(result.duration, llm_handler)
    audio_duration_value = clamped_duration if clamped_duration and clamped_duration > 0 else -1
    status_message = str(result.status_message or "").strip()
    if auto_init_status:
        status_message = f"{auto_init_status}\n{status_message}" if status_message else auto_init_status

    return (
        result.caption,
        result.lyrics,
        result.bpm,
        audio_duration_value,
        result.keyscale,
        result.language,
        result.language,
        result.timesignature,
        result.instrumental,
        gr.update(interactive=True),
        True,
        True,
        True,
        status_message,
        gr.update(value="Custom"),
    )
