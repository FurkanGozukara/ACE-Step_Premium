"""Trim-focused audio preview presentation constants for the Gradio UI."""

TRIM_AUDIO_PREVIEW_CLASS = "ace-trim-audio-preview"
TRIM_AUDIO_PREVIEW_ELEM_CLASSES = [TRIM_AUDIO_PREVIEW_CLASS]
SOURCE_AUDIO_PREVIEW_ELEM_ID = "acestep-source-audio-preview"
SOURCE_AUDIO_PREVIEW_ELEM_CLASSES = [
    "has-info-container",
    TRIM_AUDIO_PREVIEW_CLASS,
    "ace-source-audio-preview",
]
TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS = {
    "waveform_color": "#9ca3af",
    "waveform_progress_color": "#38bdf8",
    "trim_region_color": "rgba(249, 115, 22, 0.5)",
    "show_recording_waveform": True,
    "skip_length": 5,
    "sample_rate": 44100,
}
SOURCE_AUDIO_PREVIEW_WAVEFORM_OPTIONS = TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS

AUDIO_PROCESSING_PREVIEW_ELEM_CLASSES = [
    TRIM_AUDIO_PREVIEW_CLASS,
    "ace-audio-processing-preview",
]
AUDIO_PROCESSING_UPLOAD_PREVIEW_ELEM_ID = "acestep-audio-processing-upload-preview"
AUDIO_PROCESSING_BEFORE_PREVIEW_ELEM_ID = "acestep-audio-processing-before-preview"
AUDIO_PROCESSING_AFTER_PREVIEW_ELEM_ID = "acestep-audio-processing-after-preview"
AUDIO_PROCESSING_OUTPUT_PREVIEW_ELEM_ID = "acestep-audio-processing-output-preview"
AUDIO_PROCESSING_PREVIEW_WAVEFORM_OPTIONS = TRIM_AUDIO_PREVIEW_WAVEFORM_OPTIONS

GENERATION_REFERENCE_PREVIEW_ELEM_ID = "acestep-generation-reference-audio-preview"
GENERATION_LM_CODES_PREVIEW_ELEM_ID = "acestep-generation-lm-codes-audio-preview"
GENERATION_UPLOAD_PREVIEW_ELEM_CLASSES = [
    TRIM_AUDIO_PREVIEW_CLASS,
    "ace-generation-upload-audio-preview",
]

SAM_UPLOAD_AUDIO_PREVIEW_ELEM_ID = "acestep-sam-upload-audio-preview"
SAM_TARGET_AUDIO_PREVIEW_ELEM_ID = "acestep-sam-target-audio-preview"
SAM_RESIDUAL_AUDIO_PREVIEW_ELEM_ID = "acestep-sam-residual-audio-preview"
SAM_AUDIO_PREVIEW_ELEM_CLASSES = [
    TRIM_AUDIO_PREVIEW_CLASS,
    "ace-sam-audio-preview",
]


def _trim_preview_css(selector: str) -> str:
    """Return scoped CSS for an editable Gradio waveform audio preview."""

    return f"""
{selector} .component-wrapper {{
    padding: 14px 16px 12px !important;
}}
{selector} .waveform-container {{
    padding-inline: 12px !important;
}}
{selector} ::part(region) {{
    background: rgba(249, 115, 22, 0.32) !important;
    border-radius: 5px !important;
    box-sizing: border-box !important;
    box-shadow:
        inset 0 0 0 2px #fb923c,
        inset 0 0 0 4px rgba(255, 255, 255, 0.18),
        0 0 0 1px rgba(15, 23, 42, 0.65),
        0 0 22px rgba(249, 115, 22, 0.35) !important;
    height: 98% !important;
    top: 1% !important;
}}
{selector} ::part(region)::after {{
    background:
        linear-gradient(90deg, rgba(251, 146, 60, 0.3), rgba(251, 146, 60, 0.1)),
        repeating-linear-gradient(
            90deg,
            rgba(255, 255, 255, 0.18) 0,
            rgba(255, 255, 255, 0.18) 1px,
            transparent 1px,
            transparent 12px
        ) !important;
    border-radius: 5px !important;
    content: "" !important;
    inset: 0 !important;
    opacity: 1 !important;
    pointer-events: none !important;
    position: absolute !important;
}}
{selector} ::part(region-handle) {{
    background: #f97316 !important;
    border: 2px solid #fff7ed !important;
    border-radius: 5px !important;
    box-shadow:
        0 0 0 1px rgba(15, 23, 42, 0.8),
        0 0 18px rgba(249, 115, 22, 0.58) !important;
    cursor: ew-resize !important;
    opacity: 1 !important;
    width: 14px !important;
}}
{selector} ::part(region-handle)::before {{
    border-left: 2px solid rgba(255, 247, 237, 0.9) !important;
    border-right: 2px solid rgba(255, 247, 237, 0.9) !important;
    content: "" !important;
    inset: 5px 4px !important;
    opacity: 0.95 !important;
    pointer-events: none !important;
    position: absolute !important;
}}
{selector} ::part(region-handle-left) {{
    border-left-width: 4px !important;
}}
{selector} ::part(region-handle-right) {{
    border-right-width: 4px !important;
}}
{selector} .timestamps {{
    box-sizing: border-box !important;
    min-height: 34px !important;
    padding: 7px clamp(24px, 5vw, 76px) 8px clamp(24px, 4vw, 56px) !important;
}}
{selector} .timestamps > div {{
    align-items: center !important;
    display: inline-flex !important;
    gap: 8px !important;
}}
{selector} .timestamps time {{
    background: rgba(15, 23, 42, 0.86) !important;
    border: 1px solid rgba(148, 163, 184, 0.35) !important;
    border-radius: 5px !important;
    color: #ffffff !important;
    font-size: 12px !important;
    font-weight: 700 !important;
    line-height: 1.25 !important;
    min-width: 46px !important;
    padding: 3px 7px !important;
    text-align: center !important;
}}
{selector} .timestamps #trim-duration,
{selector} .timestamps .trim-duration {{
    background: rgba(249, 115, 22, 0.94) !important;
    border-color: rgba(255, 237, 213, 0.72) !important;
    min-width: 54px !important;
}}
{selector} .settings-wrapper {{
    gap: 6px !important;
    min-width: max-content !important;
}}
{selector} .settings-wrapper button[aria-label="Trim audio to selection"] {{
    align-items: center !important;
    background: linear-gradient(135deg, #ea580c 0%, #f97316 100%) !important;
    border: 1px solid rgba(255, 237, 213, 0.65) !important;
    border-radius: 6px !important;
    box-shadow: 0 8px 18px rgba(249, 115, 22, 0.28) !important;
    color: #ffffff !important;
    display: inline-flex !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    gap: 6px !important;
    height: 32px !important;
    justify-content: center !important;
    line-height: 1 !important;
    margin-left: 8px !important;
    min-width: 88px !important;
    padding: 0 12px !important;
    width: auto !important;
}}
{selector} button[aria-label="Trim audio to selection"]::after {{
    content: "Trim";
}}
{selector} .settings-wrapper button[aria-label="Trim audio to selection"] svg {{
    height: 17px !important;
    width: 17px !important;
}}
{selector} .settings-wrapper .text-button {{
    border-radius: 6px !important;
    font-size: 13px !important;
    font-weight: 800 !important;
    height: 32px !important;
    padding: 0 12px !important;
}}
@media (max-width: 600px) {{
    {selector} .timestamps {{
        padding-inline: 14px !important;
    }}
    {selector} .settings-wrapper button[aria-label="Trim audio to selection"] {{
        min-width: 78px !important;
        padding-inline: 10px !important;
    }}
}}
"""


SOURCE_AUDIO_PREVIEW_CSS = _trim_preview_css(f".{TRIM_AUDIO_PREVIEW_CLASS}")
AUDIO_PROCESSING_PREVIEW_CSS = ""
