"""Dataset operations for the training UI.

Contains handlers for scanning directories, auto-labeling samples,
previewing/editing individual samples, updating settings, and saving
datasets to JSON.
"""

import re
from typing import Any, List, Optional, Tuple

import gradio as gr
from loguru import logger

from acestep.training.dataset_builder import DatasetBuilder
from acestep.training.dataset_builder_modules.label_hydration import (
    has_unlabeled_samples,
    hydrate_samples_from_label_dir,
)
from acestep.training.path_inputs import normalize_user_path
from acestep.training.path_safety import safe_path
from .auto_label_control import (
    clear_auto_label_cancel_request,
    is_auto_label_cancel_requested,
    mark_inline_auto_label_finished,
    mark_inline_auto_label_started,
)
from .raw_lyrics_preview import raw_lyrics_preview_update
from .service_auto_init import ensure_training_services_ready
from .training_utils import _safe_slider


_SUCCESS = "\u2705"
_LABEL_POSITION_RE = re.compile(r"^Labeling (\d+)/(\d+)")


def _apply_current_settings(
    builder_state: Optional[DatasetBuilder],
    custom_tag: Optional[str] = None,
    tag_position: Optional[str] = None,
    all_instrumental: Optional[bool] = None,
    genre_ratio: Optional[int] = None,
) -> Optional[DatasetBuilder]:
    """Apply current dataset settings to the builder state."""

    if builder_state is None:
        return None

    if custom_tag is not None or tag_position is not None:
        current_tag = (
            builder_state.metadata.custom_tag
            if custom_tag is None
            else custom_tag
        )
        current_position = (
            builder_state.metadata.tag_position
            if tag_position is None
            else tag_position
        )
        builder_state.set_custom_tag(current_tag, current_position)

    if all_instrumental is not None:
        builder_state.set_all_instrumental(all_instrumental)

    if genre_ratio is not None:
        builder_state.metadata.genre_ratio = int(genre_ratio)

    return builder_state


def scan_directory(
    audio_dir: str,
    dataset_name: str,
    custom_tag: str,
    tag_position: str,
    all_instrumental: bool,
    builder_state: Optional[DatasetBuilder],
) -> Tuple[Any, str, Any, DatasetBuilder]:
    """Scan a directory for audio files.

    Returns:
        Tuple of (table_data, status, slider_update, builder_state).
    """
    audio_dir = normalize_user_path(audio_dir)
    if not audio_dir:
        return (
            [],
            "❌ Please enter a directory path",
            _safe_slider(0, value=0, visible=False),
            builder_state,
        )

    builder = builder_state if builder_state else DatasetBuilder()

    builder.metadata.name = dataset_name
    builder.metadata.all_instrumental = all_instrumental
    builder.set_custom_tag(custom_tag, tag_position)

    samples, status = builder.scan_directory(audio_dir)

    if not samples:
        return [], status, _safe_slider(0, value=0, visible=False), builder

    table_data = builder.get_samples_dataframe_data()
    slider_max = max(0, len(samples) - 1)

    return table_data, status, _safe_slider(slider_max, value=0, visible=len(samples) > 1), builder


def auto_label_all(
    dit_handler,
    llm_handler,
    builder_state: Optional[DatasetBuilder],
    skip_metas: bool = False,
    format_lyrics: bool = False,
    transcribe_lyrics: bool = False,
    lm_lyrics_language: str = "unknown",
    only_unlabeled: bool = True,
    batch_size: int = 1,
    progress=None,
    model_config: str | None = None,
    save_path: str | None = None,
    dataset_name: str | None = None,
    label_output_dir: str | None = None,
    label_source_root: str | None = None,
) -> Tuple[List[List[Any]], str, DatasetBuilder]:
    """Auto-label all samples in the dataset.

    Args:
        dit_handler: DiT handler for audio processing.
        llm_handler: LLM handler for caption generation.
        builder_state: Dataset builder state.
        skip_metas: Skip generating BPM/Key/TimeSig but still generate caption/genre.
        format_lyrics: Use LLM to format user-provided lyrics from .txt files.
        transcribe_lyrics: Use LLM to transcribe lyrics from audio.
        lm_lyrics_language: Optional language hint for LM lyric generation.
        only_unlabeled: Only label samples without caption.
        batch_size: Number of samples per auto-label LM metadata batch.
        progress: Progress callback.
        model_config: Optional DiT model name selected for dataset actions.
        save_path: Optional dataset JSON path for per-sample checkpoint saves.
        dataset_name: Optional dataset name to persist with checkpoint saves.
        label_output_dir: Folder for processed per-song label JSON files.
        label_source_root: Optional source root for mirroring nested audio folders.

    Returns:
        Tuple of (table_data, status, builder_state).
    """
    if builder_state is None:
        return [], "❌ Please scan a directory first", builder_state

    if not builder_state.samples:
        return [], "❌ No samples to label. Please scan a directory first.", builder_state

    resolved_label_output_dir = normalize_user_path(label_output_dir)
    if not resolved_label_output_dir:
        return (
            builder_state.get_samples_dataframe_data(),
            "\u274c Please choose a processed labels folder before auto-labeling.",
            builder_state,
        )
    try:
        resolved_label_output_dir = safe_path(resolved_label_output_dir)
    except ValueError:
        return (
            builder_state.get_samples_dataframe_data(),
            f"\u274c Rejected unsafe processed labels folder: {resolved_label_output_dir}",
            builder_state,
        )

    status_prefixes: list[str] = []
    if only_unlabeled:
        hydrated_count = hydrate_samples_from_label_dir(
            builder_state.samples,
            resolved_label_output_dir,
        )
        if hydrated_count:
            hydration_status = (
                f"{_SUCCESS} Loaded {hydrated_count} existing labels from processed label folder."
            )
            status_prefixes.append(hydration_status)
            if progress:
                try:
                    progress(None, desc=hydration_status)
                except Exception:
                    pass

        if not has_unlabeled_samples(builder_state.samples):
            clear_auto_label_cancel_request()
            mark_inline_auto_label_started()
            try:
                _samples, status = builder_state.label_all_samples(
                    dit_handler=dit_handler,
                    llm_handler=llm_handler,
                    format_lyrics=format_lyrics,
                    transcribe_lyrics=transcribe_lyrics,
                    lm_lyrics_language=lm_lyrics_language,
                    skip_metas=skip_metas,
                    only_unlabeled=only_unlabeled,
                    batch_size=batch_size,
                    label_output_dir=resolved_label_output_dir,
                    label_source_root=label_source_root,
                    cancel_callback=is_auto_label_cancel_requested,
                )
            finally:
                mark_inline_auto_label_finished()
            if status_prefixes:
                status = "\n".join([*status_prefixes, status])
            table_data = builder_state.get_samples_dataframe_data()
            return gr.update(value=table_data), gr.update(value=status), builder_state

    services_ready, auto_init_status = ensure_training_services_ready(
        dit_handler,
        llm_handler,
        require_llm=True,
        config_path=model_config,
    )
    if not services_ready:
        status = auto_init_status or (
            "Model not initialized. Please initialize the service first."
        )
        error_status = status if status.startswith("❌") else f"❌ {status}"
        if status_prefixes:
            error_status = "\n".join([*status_prefixes, error_status])
        return (
            builder_state.get_samples_dataframe_data(),
            error_status,
            builder_state,
        )

    def progress_callback(msg):
        if progress:
            try:
                match = _LABEL_POSITION_RE.match(str(msg))
                if match:
                    position = int(match.group(1))
                    total = int(match.group(2))
                    current = position if "complete" in str(msg) else max(position - 1, 0)
                    progress((current, total), desc=msg)
                else:
                    progress(None, desc=msg)
            except Exception:
                pass

    resolved_save_path = normalize_user_path(save_path)
    if resolved_save_path and not resolved_save_path.lower().endswith(".json"):
        resolved_save_path = f"{resolved_save_path}.json"

    def sample_labeled_callback(_sample_idx: int, _sample: Any, _status: str) -> None:
        if not resolved_save_path:
            return
        save_status = builder_state.save_dataset(resolved_save_path, dataset_name)
        if not save_status.startswith(_SUCCESS):
            logger.warning(f"Auto-label dataset checkpoint save failed: {save_status}")

    clear_auto_label_cancel_request()
    mark_inline_auto_label_started()
    try:
        _samples, status = builder_state.label_all_samples(
            dit_handler=dit_handler,
            llm_handler=llm_handler,
            format_lyrics=format_lyrics,
            transcribe_lyrics=transcribe_lyrics,
            lm_lyrics_language=lm_lyrics_language,
            skip_metas=skip_metas,
            only_unlabeled=only_unlabeled,
            batch_size=batch_size,
            progress_callback=progress_callback,
            sample_labeled_callback=sample_labeled_callback,
            label_output_dir=resolved_label_output_dir,
            label_source_root=label_source_root,
            cancel_callback=is_auto_label_cancel_requested,
        )
    finally:
        mark_inline_auto_label_finished()
    if auto_init_status:
        status = f"{auto_init_status}\n{status}" if status else auto_init_status
    if status_prefixes:
        status = "\n".join([*status_prefixes, status])
    if progress:
        try:
            progress(1.0, desc=status)
        except Exception:
            pass

    table_data = builder_state.get_samples_dataframe_data()
    return gr.update(value=table_data), gr.update(value=status), builder_state


def get_sample_preview(
    sample_idx: int,
    builder_state: Optional[DatasetBuilder],
):
    """Get preview data for a specific sample.

    Returns:
        Tuple of (audio_path, filename, caption, genre, prompt_override, lyrics,
                  bpm, keyscale, timesig, duration, language, instrumental,
                  raw_lyrics_update, raw_lyrics_visible).
    """
    empty = (
        None,
        "",
        "",
        "",
        "Use Global Ratio",
        "",
        None,
        "",
        "",
        0.0,
        "instrumental",
        True,
        raw_lyrics_preview_update("", False),
        False,
    )

    if builder_state is None or not builder_state.samples:
        return empty

    if sample_idx is None:
        return empty

    idx = int(sample_idx)
    if idx < 0 or idx >= len(builder_state.samples):
        return empty

    sample = builder_state.samples[idx]
    has_raw = sample.has_raw_lyrics()

    if sample.prompt_override == "genre":
        override_choice = "Genre"
    elif sample.prompt_override == "caption":
        override_choice = "Caption"
    else:
        override_choice = "Use Global Ratio"

    display_lyrics = sample.lyrics if sample.lyrics else sample.formatted_lyrics

    return (
        sample.audio_path,
        sample.filename,
        sample.caption,
        sample.genre,
        override_choice,
        display_lyrics,
        sample.bpm,
        sample.keyscale,
        sample.timesignature,
        sample.duration,
        sample.language,
        sample.is_instrumental,
        raw_lyrics_preview_update(sample.raw_lyrics, has_raw),
        has_raw,
    )


def select_sample_from_table(
    builder_state: Optional[DatasetBuilder],
    evt: gr.SelectData,
) -> Tuple[Any, ...]:
    """Load Step 3 preview fields for the selected Found Audio Files row."""

    if builder_state is None or not builder_state.samples:
        return (gr.update(), *get_sample_preview(None, builder_state))

    sample_idx = _selected_table_row_index(evt)
    if sample_idx is None or sample_idx < 0 or sample_idx >= len(builder_state.samples):
        return (gr.update(), *get_sample_preview(None, builder_state))

    slider_update = gr.update(
        value=sample_idx,
        maximum=max(1, len(builder_state.samples) - 1),
        visible=len(builder_state.samples) > 1,
    )
    return (slider_update, *get_sample_preview(sample_idx, builder_state))


def _selected_table_row_index(evt: gr.SelectData) -> int | None:
    """Return the dataframe row index from a Gradio select event."""

    index = getattr(evt, "index", None)
    if isinstance(index, (list, tuple)) and index:
        index = index[0]
    try:
        return int(index)
    except (TypeError, ValueError):
        return None


def save_sample_edit(
    sample_idx: int,
    caption: str,
    genre: str,
    prompt_override: str,
    lyrics: str,
    bpm: Optional[int],
    keyscale: str,
    timesig: str,
    language: str,
    is_instrumental: bool,
    builder_state: Optional[DatasetBuilder],
) -> Tuple[List[List[Any]], str, DatasetBuilder]:
    """Save edits to a sample.

    Returns:
        Tuple of (table_data, status, builder_state).
    """
    if builder_state is None:
        return [], "❌ No dataset loaded", builder_state

    idx = int(sample_idx)

    if prompt_override == "Genre":
        override_value = "genre"
    elif prompt_override == "Caption":
        override_value = "caption"
    else:
        override_value = None

    updated_lyrics = lyrics if not is_instrumental else "[Instrumental]"
    updated_formatted = updated_lyrics if updated_lyrics and updated_lyrics != "[Instrumental]" else ""
    sample, status = builder_state.update_sample(
        idx,
        caption=caption,
        genre=genre,
        prompt_override=override_value,
        lyrics=updated_lyrics,
        formatted_lyrics=updated_formatted,
        bpm=int(bpm) if bpm else None,
        keyscale=keyscale,
        timesignature=timesig,
        language="unknown" if is_instrumental else language,
        is_instrumental=is_instrumental,
        labeled=True,
    )

    table_data = builder_state.get_samples_dataframe_data()
    return table_data, status, builder_state


def update_settings(
    custom_tag: str,
    tag_position: str,
    all_instrumental: bool,
    genre_ratio: int,
    builder_state: Optional[DatasetBuilder],
) -> DatasetBuilder:
    """Update dataset settings.

    Returns:
        Updated builder_state.
    """
    return _apply_current_settings(
        builder_state,
        custom_tag,
        tag_position,
        all_instrumental,
        genre_ratio,
    )


def save_dataset(
    save_path: str,
    dataset_name: str,
    builder_state: Optional[DatasetBuilder],
    custom_tag: Optional[str] = None,
    tag_position: Optional[str] = None,
    all_instrumental: Optional[bool] = None,
    genre_ratio: Optional[int] = None,
) -> Tuple[str, Any]:
    """Save the dataset to a JSON file.

    Returns:
        Tuple of (status, save_path_update).
    """
    if builder_state is None:
        return "❌ No dataset to save. Please scan a directory first.", gr.update()

    if not builder_state.samples:
        return "❌ No samples in dataset.", gr.update()

    builder_state = _apply_current_settings(
        builder_state,
        custom_tag,
        tag_position,
        all_instrumental,
        genre_ratio,
    )

    save_path = normalize_user_path(save_path)
    if not save_path:
        return "❌ Please enter a save path.", gr.update()

    if not save_path.lower().endswith(".json"):
        save_path = save_path + ".json"

    labeled_count = builder_state.get_labeled_count()
    status_prefix = ""
    if labeled_count == 0:
        status_prefix = (
            "⚠️ Warning: No samples have been labeled. Consider auto-labeling first.\n"
            "Saving anyway..."
        )

    status = builder_state.save_dataset(save_path, dataset_name)
    if status_prefix:
        status = f"{status_prefix}\n{status}"
    return status, gr.update(value=save_path)
