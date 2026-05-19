"""Visible LoRA training guide for the Gradio training tab."""

from __future__ import annotations

import gradio as gr


LORA_TRAINING_GUIDE_COLUMNS = (
    """
### What LoRA Training Does

LoRA trains a small adapter for the ACE-Step DiT decoder while leaving the base model
unchanged. The adapter learns the style, voice, arrangement habits, or trigger tag from
your dataset. After training, set the saved adapter path in generation settings or select
it from the LoRAs Folder dropdown so it influences the next generation automatically.
""",
    """
### Dataset Workflow

1. **Prepare audio and labels**  
   Use the Dataset Builder tab to scan clean audio files. Review each caption and lyrics
   field. Add a unique activation tag if you want a reliable trigger phrase for the style.

2. **Preprocess the dataset**  
   Save the dataset JSON, choose a tensor output directory, select LoRA mode, then click
   Preprocess. This tab trains from the preprocessed tensor directory, not from raw audio.

3. **Load preprocessed tensors**  
   Paste the tensor directory into Preprocessed Tensors Directory and click Load Dataset.
   Confirm the sample count before starting training.

4. **Choose LoRA adapter settings**  
   Rank controls capacity. Use 16-32 for a lighter style adapter, 64 as a balanced default,
   and 128+ only when you have enough data and VRAM. Alpha is commonly 2x rank. Dropout
   around 0.05-0.10 helps reduce overfitting.
""",
    """
### Training And Use

5. **Choose training settings**  
   Start with batch size 1 if VRAM is limited. Use gradient accumulation to increase the
   effective batch size. Keep Shift at 3.0 unless you are intentionally matching a different
   training schedule. Save checkpoints often so you can compare earlier adapters.

6. **Start training and monitor loss**  
   Click Start Training. The loss should generally trend down. If it collapses quickly or
   generated results copy the training audio too closely, reduce epochs, rank, or learning
   rate.

7. **Use the saved adapter**  
   Successful training writes checkpoints under `checkpoints/` and the final adapter under
   `final/adapter`. To use it, copy or export the adapter folder, open generation settings,
   set the LoRA path or select it from the LoRAs Folder dropdown, then generate. The LoRA
   path field accepts relative paths, full Windows paths, full Linux paths, quoted pasted
   paths, `final/adapter`, `final`, and checkpoint folders that contain an `adapter` child.

### Practical Defaults

- Learning rate: start around `1e-4` to `3e-4`.
- Batch size: `1` for most consumer GPUs.
- Gradient accumulation: `4` to `8` when you want a larger effective batch.
- Output directory: use a new folder per experiment.
- Resume checkpoint: set this only when continuing a previous interrupted run.
""",
)


LORA_TRAINING_NOTES_COLUMNS = (
    """
### Recommended Starting Values

- Learning rate: `1e-4`
- Batch size: `1`
- Gradient accumulation: `4`
- Rank / alpha: `64 / 128`
- Save every: `10-50` epochs depending on total epochs

### Epoch Guide

- `1-10 songs`: `200-500` epochs
- `10-50 songs`: `100-200` epochs
- `50+ songs`: `50-100` epochs
- Stop early if the loss plateaus or test generations start copying training audio too closely.
""",
    """
### Data And Learning

### Dataset Quality Checklist

- Use clean, full-quality audio with no clipping.
- Avoid live recordings, crowd noise, heavy room noise, and bad masters unless that is the style you want.
- Keep captions accurate and specific.
- Use accurate BPM, key, time signature, and language metadata when possible.
- Correct lyrics manually when vocals matter.

### What LoRA Can Learn

- Artist or genre style tendencies
- Vocal delivery and timbre tendencies
- Instrumentation and arrangement habits
- Vocal-only, instrumental-only, or full-song style, depending on the dataset

### What LoRA Cannot Reliably Fix

- Bad metadata or wrong labels
- Incorrect lyrics
- Low-quality source audio
- Too few inconsistent songs
- Exact BPM/key inference when metadata is missing
""",
    """
### Training Diagnostics

### Overfit / Underfit Signs

- Overfit: copies training songs, outputs become narrow, repeated patterns appear.
- If overfitting: reduce epochs, rank, or learning rate; add more varied data.
- Underfit: LoRA barely changes the base model's style.
- If underfitting: train longer, improve captions, or use a more consistent dataset.

### VRAM Notes

- Training uses a non-quantized DiT.
- Restart Gradio after LM labeling or preprocessing if VRAM is tight.
- Batch size `1` is the safest starting point.
- Increase batch size only if VRAM allows it.

### Checkpoint Advice

- The final epoch is not always the best adapter.
- Export and compare multiple checkpoints.
- Compare checkpoints with the same prompt, seed, LoRA scale, and generation settings.

### File Naming Reminder

- `song.mp3`
- `song.lyrics.txt` or `song.txt`
- `song.caption.txt`
- `song.json`
- CSV files can provide batch BPM/key metadata.
""",
)


def _render_markdown_columns(columns: tuple[str, ...]) -> None:
    """Render Markdown content in equal-width Gradio columns."""

    with gr.Row():
        for column_markdown in columns:
            with gr.Column(scale=1):
                gr.Markdown(column_markdown)


def build_lora_training_guide() -> None:
    """Render the visible LoRA workflow guide above the training controls."""

    with gr.Accordion("LoRA Training Guide", open=False):
        _render_markdown_columns(LORA_TRAINING_GUIDE_COLUMNS)
    with gr.Accordion("Important Training Notes", open=False):
        _render_markdown_columns(LORA_TRAINING_NOTES_COLUMNS)
