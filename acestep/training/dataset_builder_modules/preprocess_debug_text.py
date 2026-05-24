"""Debug text-output helpers for tensor preprocessing."""

from __future__ import annotations

import os
import re


DEBUG_TEXT_PROMPT_DIRNAME = "debug_text_prompts"
_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def save_debug_text_prompt(
    output_dir: str,
    sample_id: str,
    text_prompt: str,
    lyrics: str = "",
) -> str:
    """Save the exact text inputs used for tensor preprocessing.

    Args:
        output_dir: Tensor output directory.
        sample_id: Stable sample identifier used for the tensor filename.
        text_prompt: Final prompt string passed to the text encoder.
        lyrics: Final lyrics string passed to the lyrics encoder.

    Returns:
        Path to the written debug text file.
    """

    debug_dir = os.path.join(output_dir, DEBUG_TEXT_PROMPT_DIRNAME)
    os.makedirs(debug_dir, exist_ok=True)
    safe_id = _SAFE_FILENAME_RE.sub("_", str(sample_id or "sample")).strip("._")
    debug_path = os.path.join(debug_dir, f"{safe_id or 'sample'}.txt")
    with open(debug_path, "w", encoding="utf-8") as file_obj:
        file_obj.write("# Text Encoder Input\n")
        file_obj.write(text_prompt)
        file_obj.write("\n\n# Lyrics Encoder Input\n")
        file_obj.write(lyrics)
    return debug_path
