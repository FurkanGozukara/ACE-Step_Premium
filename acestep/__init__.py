"""ACE-Step package."""

from __future__ import annotations

import sys
from pathlib import Path

_BUNDLED_NANOVLLM = Path(__file__).resolve().parent / "third_parts" / "nano-vllm"
if (_BUNDLED_NANOVLLM / "nanovllm").is_dir():
    _bundled_nanovllm_path = str(_BUNDLED_NANOVLLM)
    if _bundled_nanovllm_path not in sys.path:
        sys.path.insert(0, _bundled_nanovllm_path)
