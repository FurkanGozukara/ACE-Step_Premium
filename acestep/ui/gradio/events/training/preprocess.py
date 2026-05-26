"""Preprocessing and dataset loading for the training UI.

Contains handlers for loading existing datasets, preprocessing
audio to tensor files, and loading preprocessed tensor datasets.
"""

import json
import os
from typing import Any, Optional

import gradio as gr
from loguru import logger

from acestep.debug_utils import debug_end_for, debug_log_for, debug_start_for
from acestep.training.dataset_builder import DatasetBuilder
from acestep.training.path_inputs import normalize_user_path
from acestep.training.path_safety import safe_path
from .preprocess_control import (
    clear_preprocess_cancel_request,
    is_preprocess_cancel_requested,
    mark_inline_preprocess_finished,
    mark_inline_preprocess_started,
)
from .raw_lyrics_preview import raw_lyrics_preview_update
from .service_auto_init import ensure_training_services_ready
from .training_utils import _safe_slider


def load_existing_dataset_for_preprocess(
    dataset_path: str,
    builder_state: Optional[DatasetBuilder],
):
    """Load an existing dataset JSON file for preprocessing.

    This allows users to load a previously saved dataset and proceed to
    preprocessing without having to re-scan and re-label.

    Returns:
        Tuple of (status, table_data, slider_update, builder_state,
                  audio_path, filename, caption, genre, prompt_override,
                  lyrics, bpm, keyscale, timesig, duration, language,
                  instrumental, raw_lyrics_update, has_raw,
                  name_update, tag_update, pos_update, instr_update,
                  ratio_update, trigger_only_update).
    """
    empty_preview = (
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

    dataset_path = normalize_user_path(dataset_path)
    if not dataset_path:
        updates = tuple(gr.update() for _ in range(6))
        return (
            "❌ Please enter a dataset path",
            [],
            _safe_slider(0, value=0, visible=False),
            builder_state,
        ) + empty_preview + updates

    try:
        dataset_path = safe_path(dataset_path)
    except ValueError:
        updates = tuple(gr.update() for _ in range(6))
        return (
            f"❌ Rejected unsafe dataset path: {dataset_path}",
            [],
            _safe_slider(0, value=0, visible=False),
            builder_state,
        ) + empty_preview + updates

    debug_log_for("dataset", f"UI load_existing_dataset_for_preprocess: path='{dataset_path}'")

    if not os.path.exists(dataset_path):
        updates = tuple(gr.update() for _ in range(6))
        return (
            f"❌ Dataset not found: {dataset_path}",
            [],
            _safe_slider(0, value=0, visible=False),
            builder_state,
        ) + empty_preview + updates

    builder = DatasetBuilder()

    t0 = debug_start_for("dataset", "load_dataset")
    samples, status = builder.load_dataset(dataset_path)
    debug_end_for("dataset", "load_dataset", t0)

    if not samples:
        updates = tuple(gr.update() for _ in range(6))
        return (
            status,
            [],
            _safe_slider(0, value=0, visible=False),
            builder,
        ) + empty_preview + updates

    table_data = builder.get_samples_dataframe_data()
    slider_max = max(0, len(samples) - 1)

    labeled_count = builder.get_labeled_count()
    info = f"📂 Loaded dataset: {builder.metadata.name}\n"
    info += f"🔢 Samples: {len(samples)} ({labeled_count} labeled)\n"
    info += f"🏷️ Custom Tag: {builder.metadata.custom_tag or '(none)'}\n"
    info += "✅ Ready for preprocessing! You can also edit samples below."
    if any((s.formatted_lyrics and not s.lyrics) for s in builder.samples):
        info += "\nℹ️ Showing formatted lyrics where lyrics are empty."

    first_sample = builder.samples[0]
    has_raw = first_sample.has_raw_lyrics()

    if first_sample.prompt_override == "genre":
        override_choice = "Genre"
    elif first_sample.prompt_override == "caption":
        override_choice = "Caption"
    else:
        override_choice = "Use Global Ratio"

    display_lyrics = first_sample.lyrics if first_sample.lyrics else first_sample.formatted_lyrics

    preview = (
        first_sample.audio_path,
        first_sample.filename,
        first_sample.caption,
        first_sample.genre,
        override_choice,
        display_lyrics,
        first_sample.bpm,
        first_sample.keyscale,
        first_sample.timesignature,
        first_sample.duration,
        first_sample.language,
        first_sample.is_instrumental,
        raw_lyrics_preview_update(first_sample.raw_lyrics, has_raw),
        has_raw,
    )

    updates = (
        gr.update(value=builder.metadata.name),
        gr.update(value=builder.metadata.custom_tag),
        gr.update(value=builder.metadata.tag_position),
        gr.update(value=builder.metadata.all_instrumental),
        gr.update(value=builder.metadata.genre_ratio),
        gr.update(value=builder.metadata.use_only_custom_trigger),
    )

    return (
        info,
        table_data,
        _safe_slider(slider_max, value=0, visible=len(samples) > 1),
        builder,
    ) + preview + updates


def preprocess_dataset(
    output_dir: str,
    preprocess_mode: str,
    dit_handler,
    builder_state: Optional[DatasetBuilder],
    progress=None,
    model_config: str | None = None,
    save_debug_text: bool = False,
) -> str:
    """Preprocess dataset to tensor files for fast training.

    Converts audio files to VAE latents and text to embeddings.

    Args:
        output_dir: Directory where tensor files should be written.
        preprocess_mode: Target adapter mode, ``"lora"``, ``"dora"``, or ``"lokr"``.
        dit_handler: DiT handler used for tensor preprocessing.
        builder_state: Dataset builder state containing labeled samples.
        progress: Optional progress callback.
        model_config: Optional DiT model name selected for dataset actions.
        save_debug_text: Save final text prompts as readable files during
            tensor generation.

    Returns:
        Status message.
    """
    if builder_state is None:
        return "❌ No dataset loaded. Please scan a directory first."

    if not builder_state.samples:
        return "❌ No samples in dataset."

    labeled_count = builder_state.get_labeled_count()
    if labeled_count == 0:
        return "❌ No labeled samples. Please auto-label or manually label samples first."

    output_dir = normalize_user_path(output_dir)
    if not output_dir:
        return "❌ Please enter an output directory."

    try:
        output_dir = safe_path(output_dir)
    except ValueError:
        return f"❌ Rejected unsafe output directory path: {normalize_user_path(output_dir)}"

    services_ready, auto_init_status = ensure_training_services_ready(
        dit_handler,
        None,
        require_llm=False,
        config_path=model_config,
    )
    if not services_ready:
        status = auto_init_status or (
            "Model not initialized. Please initialize the service first."
        )
        return f"❌ {status}"

    def progress_callback(msg):
        if progress:
            try:
                progress(msg)
            except Exception:
                pass

    mode = str(preprocess_mode or "lora").strip().lower()
    if mode == "dora":
        mode = "lora"
    if mode not in {"lora", "lokr"}:
        mode = "lora"

    clear_preprocess_cancel_request()
    mark_inline_preprocess_started()
    t0 = debug_start_for("dataset", "preprocess_to_tensors")
    try:
        output_paths, status = builder_state.preprocess_to_tensors(
            dit_handler=dit_handler,
            output_dir=output_dir,
            preprocess_mode=mode,
            progress_callback=progress_callback,
            cancel_callback=is_preprocess_cancel_requested,
            save_debug_text=save_debug_text,
        )
    finally:
        mark_inline_preprocess_finished()
        debug_end_for("dataset", "preprocess_to_tensors", t0)

    if auto_init_status:
        return f"{auto_init_status}\n{status}" if status else auto_init_status
    return status


def load_training_dataset(tensor_dir: str) -> str:
    """Load a preprocessed tensor dataset for training.

    Returns:
        Info text about the dataset.
    """
    tensor_dir = normalize_user_path(tensor_dir)
    if not tensor_dir:
        return "❌ Please enter a tensor directory path"

    try:
        tensor_dir = safe_path(tensor_dir)
    except ValueError:
        return f"❌ Rejected unsafe tensor directory path: {tensor_dir}"

    if not os.path.exists(tensor_dir):
        return f"❌ Directory not found: {tensor_dir}"

    if not os.path.isdir(tensor_dir):
        return f"❌ Not a directory: {tensor_dir}"

    manifest_path = os.path.join(tensor_dir, "manifest.json")
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r") as f:
                manifest = json.load(f)

            num_samples = manifest.get("num_samples", 0)
            metadata = manifest.get("metadata", {})
            name = metadata.get("name", "Unknown")
            custom_tag = metadata.get("custom_tag", "")

            info = f"📂 Loaded preprocessed dataset: {name}\n"
            info += f"🔢 Samples: {num_samples} preprocessed tensors\n"
            info += f"🏷️ Custom Tag: {custom_tag or '(none)'}"

            return info
        except Exception as e:
            logger.warning(f"Failed to read manifest: {e}")

    pt_files = [f for f in os.listdir(tensor_dir) if f.endswith(".pt")]

    if not pt_files:
        return f"❌ No .pt tensor files found in {tensor_dir}"

    info = f"📂 Found {len(pt_files)} tensor files in {tensor_dir}\n"
    info += "ℹ️ No manifest.json found - using all .pt files"

    return info
