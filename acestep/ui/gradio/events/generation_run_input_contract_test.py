"""Contract tests for consumers of ordered generation-run arguments."""

from __future__ import annotations

from collections import defaultdict
import unittest

from acestep.ui.gradio.events.batch_folder_args import (
    AUDIO_DURATION_ARG_INDEX,
    GENERATION_ARG_COUNT as BATCH_GENERATION_ARG_COUNT,
    LORA_SCALE_ARG_INDEX,
)
from acestep.ui.gradio.events.grid_testing_args import (
    GENERATION_ARG_COUNT as GRID_GENERATION_ARG_COUNT,
    LORA_DROPDOWN_ARG_INDEX,
    LORA_PATH_ARG_INDEX,
    USE_LORA_ARG_INDEX,
)
from acestep.ui.gradio.events.wiring.generation_run_wiring import (
    build_generation_run_inputs,
)


class GenerationRunInputContractTests(unittest.TestCase):
    """Keep positional batch/grid adapters aligned with GUI wiring."""

    def test_batch_and_grid_indices_match_live_generation_input_order(self) -> None:
        """Positional adapters must point at the intended GUI components."""

        generation = defaultdict(lambda: object())
        results = defaultdict(lambda: object())
        inputs = build_generation_run_inputs(generation, results)

        self.assertGreaterEqual(len(inputs), BATCH_GENERATION_ARG_COUNT)
        self.assertGreaterEqual(len(inputs), GRID_GENERATION_ARG_COUNT)
        self.assertIs(inputs[AUDIO_DURATION_ARG_INDEX], generation["audio_duration"])
        self.assertIs(inputs[LORA_DROPDOWN_ARG_INDEX], generation["lora_dropdown"])
        self.assertIs(inputs[LORA_PATH_ARG_INDEX], generation["lora_path"])
        self.assertIs(inputs[USE_LORA_ARG_INDEX], generation["use_lora_checkbox"])
        self.assertIs(inputs[LORA_SCALE_ARG_INDEX], generation["lora_scale_slider"])


if __name__ == "__main__":
    unittest.main()
