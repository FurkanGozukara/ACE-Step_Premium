"""Visible LoRA training guide for the Gradio training tab."""

from __future__ import annotations

import gradio as gr


LORA_TRAINING_GUIDE = """
### What LoRA Training Does

LoRA trains a small adapter for the ACE-Step DiT decoder while leaving the base model
unchanged. The adapter learns the style, voice, arrangement habits, or trigger tag from
your dataset. After training, set the saved adapter path in generation settings or select
it from the LoRAs Folder dropdown so it influences the next generation automatically.

### Step-by-Step Workflow

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
"""


def build_lora_training_guide() -> None:
    """Render the visible LoRA workflow guide above the training controls."""

    with gr.Accordion("LoRA Training Guide", open=True):
        gr.Markdown(LORA_TRAINING_GUIDE)
