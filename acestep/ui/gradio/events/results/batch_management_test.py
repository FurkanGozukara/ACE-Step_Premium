"""Unit tests for ``generate_with_batch_management`` wrapper behavior."""

import inspect
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

try:
    from ._batch_management_test_support import build_progress_result
    from ._batch_management_test_support import load_batch_management_module
except ImportError:  # pragma: no cover - supports direct file execution
    from _batch_management_test_support import build_progress_result
    from _batch_management_test_support import load_batch_management_module


def _load_output_paths_module():
    """Load output path context helpers without importing the full Gradio package."""
    module_name = "acestep.ui.gradio.events.results.output_paths"
    if module_name in sys.modules:
        return sys.modules[module_name]
    module_path = Path(__file__).with_name("output_paths.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[union-attr]
    return module


_OUTPUT_PATHS = _load_output_paths_module()
use_generation_run_name = _OUTPUT_PATHS.use_generation_run_name
use_results_dir = _OUTPUT_PATHS.use_results_dir


def _build_call_kwargs(module):
    """Build complete kwargs for ``generate_with_batch_management``."""
    kwargs = {}
    for name in list(inspect.signature(module.generate_with_batch_management).parameters)[2:]:
        if name == "progress":
            continue
        if name == "batch_size_input":
            kwargs[name] = 2
        elif name in ("allow_lm_batch", "auto_lrc", "autogen_checkbox", "auto_score"):
            kwargs[name] = False
        elif name == "current_batch_index":
            kwargs[name] = 0
        elif name == "total_batches":
            kwargs[name] = 0
        elif name in ("batch_queue", "generation_params_state"):
            kwargs[name] = {}
        elif name == "complete_track_classes":
            kwargs[name] = []
        else:
            kwargs[name] = None
    return kwargs


def _write_peft_adapter(path: Path) -> Path:
    """Create a minimal PEFT adapter directory for LoRA tests."""

    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text('{"peft_type": "LORA"}', encoding="utf-8")
    (path / "adapter_model.safetensors").write_bytes(b"")
    return path


def _ensure_in_process_service_ready(module):
    """Return the foreground service auto-init helper from the loaded wrapper."""

    return module.generate_with_batch_management.__globals__["_ensure_in_process_service_ready"]


class _ForegroundDitHandler:
    """Minimal DiT handler for foreground service auto-init tests."""

    def __init__(self, events, config_path: str):
        """Initialize handler state with an active model and init params."""

        self.events = events
        self.model = object()
        self.last_init_params = {
            "config_path": config_path,
            "device": "cuda",
            "use_flash_attention": False,
            "offload_to_cpu": False,
            "offload_dit_to_cpu": False,
            "compile_model": False,
            "quantization": None,
            "use_mlx_dit": True,
            "vae_checkpoint": "official",
        }

    def initialize_service(self, **kwargs):
        """Record DiT initialization and update active init params."""

        self.events.append("dit_init")
        self.last_init_params = dict(kwargs)
        self.model = object()
        return "Initialized foreground DiT", True


class _ForegroundLlmHandler:
    """Minimal LM handler for foreground service auto-init tests."""

    def __init__(self, events, lm_model_path: str = "lm-old"):
        """Initialize handler state with active LM runtime objects."""

        self.events = events
        self.llm = object()
        self.llm_tokenizer = object()
        self.constrained_processor = object()
        self._mlx_model = None
        self.llm_initialized = True
        self.last_init_params = {
            "lm_model_path": lm_model_path,
            "backend": "pt",
            "device": "cuda",
            "offload_to_cpu": False,
            "compile_model": False,
        }

    def unload(self):
        """Record unload and clear runtime state."""

        self.events.append("lm_unload")
        self.llm = None
        self.llm_tokenizer = None
        self.constrained_processor = None
        self._mlx_model = None
        self.llm_initialized = False

    def initialize(self, **kwargs):
        """Record LM initialization and update active init params."""

        self.events.append("lm_init")
        self.llm = object()
        self.llm_tokenizer = object()
        self.constrained_processor = object()
        self.llm_initialized = True
        self.last_init_params = {
            "lm_model_path": kwargs["lm_model_path"],
            "backend": kwargs["backend"],
            "device": kwargs["device"],
            "offload_to_cpu": kwargs["offload_to_cpu"],
            "compile_model": kwargs["compile_model"],
        }
        return "Initialized foreground LM", True


class BatchManagementWrapperTests(unittest.TestCase):
    """Tests for streaming and final wrapper output mapping."""

    def test_non_windows_streams_partial_and_final_outputs(self):
        """Non-Windows path should emit partial UI updates plus final state."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield one standard progress result for wrapper streaming."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(len(outputs[0]), 63)
        self.assertEqual(len(outputs[1]), 63)
        self.assertEqual(outputs[0][0]["playback_position"], 0)
        self.assertEqual(outputs[1][0]["playback_position"], 0)
        self.assertEqual(len(state["store_calls"]), 1)

    def test_windows_emits_only_final_output(self):
        """Windows path should skip intermediate yields and emit final state only."""
        module, _state = load_batch_management_module(is_windows=True)

        def _gen(*_args, **_kwargs):
            """Yield one standard progress result for Windows final-output path."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0]), 63)
        self.assertEqual(outputs[0][0]["playback_position"], 0)

    def test_all_audio_paths_none_skips_batch_storage(self):
        """When inner result has no audio paths, wrapper should not store a batch."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield a result with no audio paths to trigger early return."""
            yield build_progress_result(length=56, all_audio_paths=None)

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[1][16], None)
        self.assertEqual(len(state["store_calls"]), 0)

    def test_prompt_wildcards_expand_before_generation(self):
        """Wildcard prompt text should be expanded before backend generation."""
        module, state = load_batch_management_module(is_windows=False)
        captured = {}

        def _gen(*args, **_kwargs):
            """Capture generation prompt args and return a standard result."""
            captured["captions"] = args[2]
            captured["lyrics"] = args[3]
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["captions"] = "{warm|bright} pop"
        kwargs["lyrics"] = "[Verse]\nI feel {alive|free}"
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertIn(captured["captions"], {"warm pop", "bright pop"})
        self.assertIn(captured["lyrics"], {"[Verse]\nI feel alive", "[Verse]\nI feel free"})
        self.assertEqual(state["store_calls"][0]["generation_params"]["captions"], captured["captions"])
        self.assertEqual(state["store_calls"][0]["generation_params"]["lyrics"], captured["lyrics"])

    def test_prompt_wildcard_syntax_error_warns_and_skips_generation(self):
        """Invalid wildcard syntax should warn and not call backend generation."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            raise AssertionError("generate_with_progress should not run")
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["captions"] = "modern {warm|bright"
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 1)
        self.assertIn("Wildcard syntax error in Style", outputs[0][18])
        self.assertIn("Missing closing }", outputs[0][18])
        self.assertEqual(state["warning_messages"], [outputs[0][18]])
        self.assertEqual(len(state["store_calls"]), 0)

    def test_allow_lm_batch_stores_multiple_codes(self):
        """Batch mode should store a list of generated codes up to batch size."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield a result carrying a list of generated codes."""
            result = list(build_progress_result(length=56))
            result[55] = [f"code-{idx}" for idx in range(8)]
            yield tuple(result)

        kwargs = _build_call_kwargs(module)
        kwargs["allow_lm_batch"] = True
        kwargs["batch_size_input"] = 3
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(state["store_calls"][0]["codes"], ["code-0", "code-1", "code-2"])

    def test_multi_song_run_stores_multiple_codes_without_lm_batch(self):
        """Sequential Songs should store multiple codes even when LM batching is off."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield a result carrying a list of generated codes."""
            result = list(build_progress_result(length=56))
            result[55] = [f"code-{idx}" for idx in range(8)]
            yield tuple(result)

        kwargs = _build_call_kwargs(module)
        kwargs["allow_lm_batch"] = False
        kwargs["batch_size_input"] = 3
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(state["store_calls"][0]["codes"], ["code-0", "code-1", "code-2"])

    def test_resolved_seed_updates_saved_and_next_params(self):
        """Stored params should reflect the seed displayed in the seed textbox."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield a result carrying the resolved generation seed."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["random_seed_checkbox"] = True
        kwargs["seed"] = "-1"
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        saved_params = state["store_calls"][0]["generation_params"]
        next_params = outputs[-1][57]
        self.assertEqual(saved_params["seed"], "42")
        self.assertEqual(next_params["seed"], "42")
        self.assertTrue(next_params["random_seed_checkbox"])

    def test_no_fsq_forwards_to_generation_and_saved_params(self):
        """Wrapper should pass and persist the Remix no_fsq checkbox."""
        module, state = load_batch_management_module(is_windows=False)
        seen = {}

        def _gen(*args, **_kwargs):
            """Capture positional generation args and yield a standard result."""
            seen["args"] = args
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "cover"
        kwargs["no_fsq"] = True
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertTrue(seen["args"][23])
        saved_params = state["store_calls"][0]["generation_params"]
        self.assertTrue(saved_params["no_fsq"])
        self.assertEqual(saved_params["task_type"], "cover")

    def test_trimmed_source_preview_forwards_to_generation_and_saved_params(self):
        """Edited Source Audio Preview should replace the original source path."""

        module, state = load_batch_management_module(is_windows=False)
        seen = {}

        def _gen(*args, **kwargs):
            """Capture positional generation args and yield a standard result."""
            seen["args"] = args
            seen["kwargs"] = kwargs
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "cover"
        kwargs["src_audio"] = "source_video.mp4"
        kwargs["src_audio_preview"] = "trimmed_source.wav"
        kwargs["src_audio_preview_original"] = "source_video_audio_preview.wav"

        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(seen["args"][15], "trimmed_source.wav")
        saved_params = state["store_calls"][0]["generation_params"]
        self.assertEqual(saved_params["src_audio"], "trimmed_source.wav")

    def test_remix_source_range_keeps_full_source_for_generation_and_saved_params(self):
        """Remix start/end should not trim the source sent into generation."""

        module, state = load_batch_management_module(is_windows=False)
        seen = {}

        def _gen(*args, **kwargs):
            """Capture positional generation args and yield a standard result."""
            seen["args"] = args
            seen["kwargs"] = kwargs
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "cover"
        kwargs["src_audio"] = "source.wav"
        kwargs["repainting_start"] = 4.0
        kwargs["repainting_end"] = 9.0

        with patch.dict(
            module.generate_with_batch_management.__globals__,
            {"generate_with_progress": _gen},
        ):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(seen["args"][15], "source.wav")
        saved_params = state["store_calls"][0]["generation_params"]
        self.assertEqual(saved_params["src_audio"], "source.wav")

    def test_extract_requires_track_name_before_generation(self):
        """Extract should stop before backend generation when no track is selected."""

        module, state = load_batch_management_module(is_windows=False)
        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "extract"
        kwargs["track_name"] = None

        outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 1)
        self.assertIn("Select Track Name", outputs[0][18])
        self.assertEqual(state["store_calls"], [])
        self.assertTrue(state["warning_messages"])

    def test_extract_all_stems_ignores_stale_track_name(self):
        """Extract-all-stems should not leak a selected Track Name."""

        module, state = load_batch_management_module(is_windows=False)
        seen = {}

        def _gen(*args, **kwargs):
            """Capture generation args and yield a standard result."""
            seen["args"] = args
            seen["kwargs"] = kwargs
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "extract"
        kwargs["track_name"] = "guitar"
        kwargs["extract_all_stems"] = True
        kwargs["src_audio"] = "source.wav"
        kwargs["extract_output_format"] = "wav"

        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(seen["kwargs"]["track_name"], None)
        self.assertTrue(seen["kwargs"]["extract_all_stems"])
        self.assertEqual("", seen["args"][2])
        self.assertEqual("Extract the track from the audio:", seen["args"][19])
        saved_params = state["store_calls"][0]["generation_params"]
        self.assertIsNone(saved_params["track_name"])
        self.assertTrue(saved_params["extract_all_stems"])

    def test_complete_requires_source_audio_before_generation(self):
        """Complete should stop before backend generation when source audio is missing."""

        module, state = load_batch_management_module(is_windows=False)
        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "complete"
        kwargs["src_audio"] = None
        kwargs["complete_track_classes"] = ["drums"]

        outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 1)
        self.assertIn("Upload Source Audio", outputs[0][18])
        self.assertEqual(state["store_calls"], [])
        self.assertTrue(state["warning_messages"])

    def test_complete_requires_track_classes_before_generation(self):
        """Complete should stop before backend generation when no tracks are selected."""

        module, state = load_batch_management_module(is_windows=False)
        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "complete"
        kwargs["src_audio"] = "source.wav"
        kwargs["complete_track_classes"] = []

        outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 1)
        self.assertIn("Select at least one Track Name", outputs[0][18])
        self.assertEqual(state["store_calls"], [])
        self.assertTrue(state["warning_messages"])

    def test_complete_track_classes_update_instruction_and_range(self):
        """Complete should pass selected tracks and section range to generation."""

        module, state = load_batch_management_module(is_windows=False)
        seen = {}

        def _gen(*args, **_kwargs):
            """Capture positional generation args and yield a standard result."""
            seen["args"] = args
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "complete"
        kwargs["src_audio"] = "source.wav"
        kwargs["complete_track_classes"] = ["drums", "bass"]
        kwargs["repainting_start"] = 5.0
        kwargs["repainting_end"] = 15.0
        kwargs["instruction_display_gen"] = ""

        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(seen["args"][15], "source.wav")
        self.assertEqual(seen["args"][17], 5.0)
        self.assertEqual(seen["args"][18], 15.0)
        self.assertEqual(
            seen["args"][19],
            "Complete the input track with DRUMS | BASS:",
        )
        saved_params = state["store_calls"][0]["generation_params"]
        self.assertEqual(saved_params["complete_track_classes"], ["drums", "bass"])

    def test_extract_track_name_updates_instruction_and_caption(self):
        """Extract should pass selected track context into the generation request."""

        module, state = load_batch_management_module(is_windows=False)
        seen = {}

        def _gen(*args, **kwargs):
            """Capture positional generation args and yield a standard result."""
            seen["args"] = args
            seen["kwargs"] = kwargs
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "extract"
        kwargs["track_name"] = " Vocals "
        kwargs["extract_output_format"] = "wav"
        kwargs["captions"] = ""
        kwargs["instruction_display_gen"] = ""
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        saved_params = state["store_calls"][0]["generation_params"]
        self.assertEqual(seen["args"][2], "vocals")
        self.assertEqual(seen["args"][19], "Extract the VOCALS track from the audio:")
        self.assertEqual(seen["kwargs"]["track_name"], "vocals")
        self.assertEqual(saved_params["track_name"], "vocals")
        self.assertEqual(saved_params["audio_format"], "wav")
        self.assertEqual(saved_params["extract_output_format"], "wav")

    def test_extract_rejects_invalid_track_name(self):
        """Extract should stop before generation for unsupported stems."""

        module, state = load_batch_management_module(is_windows=False)
        kwargs = _build_call_kwargs(module)
        kwargs["task_type"] = "extract"
        kwargs["track_name"] = "lead kazoo"

        outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 1)
        self.assertIn("Unsupported Extract Track Name", outputs[0][18])
        self.assertEqual(state["store_calls"], [])

    def test_auto_lrc_copies_lrc_fields_to_batch_queue(self):
        """Auto-LRC mode should copy LRC/subtitle payloads into stored queue entry."""
        module, _state = load_batch_management_module(is_windows=False)

        lrcs = [f"lrc-{idx}" for idx in range(8)]
        subtitles = [f"sub-{idx}" for idx in range(8)]

        def _gen(*_args, **_kwargs):
            """Yield a result with explicit LRC/subtitle payload."""
            result = list(build_progress_result(length=56))
            result[54] = {"lrcs": lrcs, "subtitles": subtitles}
            yield tuple(result)

        kwargs = _build_call_kwargs(module)
        kwargs["auto_lrc"] = True
        outputs = []
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        final_batch_queue = outputs[-1][56]
        self.assertEqual(final_batch_queue[0]["lrcs"], lrcs)
        self.assertEqual(final_batch_queue[0]["subtitles"], subtitles)

    def test_auto_lrc_sets_lrc_display_in_final_yield(self):
        """Final yield should carry gr.update(value=lrc) at positions 44-51."""
        module, _state = load_batch_management_module(is_windows=False)

        lrcs = [f"[00:01.00]Line {idx}" for idx in range(8)]
        subtitles = [f"sub-{idx}" for idx in range(8)]

        def _gen(*_args, **_kwargs):
            """Yield a result with LRC data in extra_outputs."""
            result = list(build_progress_result(length=56))
            result[54] = {"lrcs": lrcs, "subtitles": subtitles}
            yield tuple(result)

        kwargs = _build_call_kwargs(module)
        kwargs["auto_lrc"] = True
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        final_yield = outputs[-1]
        for i in range(8):
            lrc_val = final_yield[44 + i]
            self.assertIsInstance(lrc_val, dict, f"LRC position {44 + i} should be a gr.update dict")
            self.assertEqual(
                lrc_val.get("value"), lrcs[i],
                f"LRC position {44 + i} should contain the LRC text",
            )

    def test_auto_lrc_disabled_preserves_passthrough_values(self):
        """When auto_lrc is off, LRC positions pass through from inner generator."""
        module, _state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield a standard result without auto_lrc."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs["auto_lrc"] = False
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        final_yield = outputs[-1]
        for i in range(8):
            lrc_val = final_yield[44 + i]
            self.assertIsNone(lrc_val, f"LRC position {44 + i} should be None when auto_lrc is off")

    def test_empty_inner_generator_returns_skip_tuple_and_warning(self):
        """Empty inner generator should fail gracefully without indexing None."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield nothing to simulate a defensive empty-generator edge case."""
            if False:
                yield None

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            outputs = list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(outputs), 1)
        self.assertEqual(len(outputs[0]), 63)
        self.assertTrue(all(item.get("kind") == "skip" for item in outputs[0]))
        self.assertEqual(len(state["store_calls"]), 0)
        self.assertTrue(state["warning_messages"])
        self.assertIn("messages.batch_failed", state["warning_messages"][0])

    # ------------------------------------------------------------------
    # Score persistence regression tests (foreground batch fix)
    # ------------------------------------------------------------------

    def test_foreground_scores_passed_to_store_batch_in_queue(self):
        """Foreground generation must extract and pass scores to batch storage."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield a result with score values at indices 20-27."""
            result = list(build_progress_result(length=56))
            for i in range(8):
                result[20 + i] = f"8.{i}"
            yield tuple(result)

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(state["store_calls"]), 1)
        scores = state["store_calls"][0]["scores"]
        self.assertEqual(len(scores), 8)
        self.assertEqual(scores[0], "8.0")
        self.assertEqual(scores[7], "8.7")

    def test_foreground_scores_default_empty_when_absent(self):
        """When result tuple lacks score indices, scores should be empty strings."""
        module, state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield a short result with no score data."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(len(state["store_calls"]), 1)
        scores = state["store_calls"][0]["scores"]
        self.assertEqual(len(scores), 8)
        self.assertTrue(all(s == "" for s in scores), "Absent scores should default to empty strings")

    # ------------------------------------------------------------------
    # MPS cache-clearing regression tests (macOS audio-mute fix)
    # ------------------------------------------------------------------

    def test_mps_cache_cleared_before_and_after_generation_on_mac(self):
        """On MPS, empty_cache must be called both before and after generation."""
        module, state = load_batch_management_module(is_windows=False, mps_available=True)

        def _gen(*_args, **_kwargs):
            """Yield one result for MPS cache-clearing path."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertGreaterEqual(
            state["mps_empty_cache_calls"],
            2,
            "torch.mps.empty_cache() must be called before and after generation "
            "on macOS to prevent system audio mute",
        )

    def test_mps_cache_not_called_when_mps_unavailable(self):
        """MPS cache clear must not be called when MPS is absent (non-Mac hosts)."""
        module, state = load_batch_management_module(is_windows=False, mps_available=False)

        def _gen(*_args, **_kwargs):
            """Yield one result for non-MPS path."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        self.assertEqual(
            state["mps_empty_cache_calls"],
            0,
            "torch.mps.empty_cache() must not be called when MPS is unavailable",
        )

    def test_foreground_generation_logs_model_and_inference_steps(self):
        """Foreground generation should print the selected model and step count."""
        module, state = load_batch_management_module(is_windows=True)

        def _gen(*_args, **_kwargs):
            """Yield one standard progress result."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs.update(
            {
                "config_path": "acestep-v15-xl-turbo",
                "inference_steps": 8,
                "batch_size_input": 1,
                "audio_duration": 60,
            }
        )
        with patch.dict(module.generate_with_batch_management.__globals__, {"generate_with_progress": _gen}):
            list(module.generate_with_batch_management(None, None, **kwargs))

        log_text = "\n".join(state["log_info"])
        self.assertIn("Starting generation", log_text)
        self.assertIn("model=acestep-v15-xl-turbo", log_text)
        self.assertIn("inference_steps=8", log_text)

    def test_foreground_generation_forwards_runtime_settings_to_manifest_payload(self):
        """Foreground manifests should receive the runtime UI settings."""

        module, _state = load_batch_management_module(is_windows=True)
        seen = {}

        def _gen(*_args, **kwargs):
            """Capture forwarded runtime settings for manifest generation."""

            seen["ui_runtime_settings"] = kwargs.get("ui_runtime_settings")
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs.update(
            {
                "config_path": "acestep-v15-xl-turbo",
                "device": "cuda",
                "vae_checkpoint": "scragvae",
                "lm_model_path": "acestep-5Hz-lm-4B",
                "backend_dropdown": "vllm",
                "init_llm_checkbox": True,
                "lm_use_legacy_cfg_prompt": True,
                "compile_threads_slider": 13,
                "mlx_vae_chunk_size": 512,
            }
        )

        with patch.dict(
            module.generate_with_batch_management.__globals__,
            {"generate_with_progress": _gen},
        ):
            list(module.generate_with_batch_management(None, None, **kwargs))

        runtime = seen["ui_runtime_settings"]
        self.assertEqual(runtime["config_path"], "acestep-v15-xl-turbo")
        self.assertEqual(runtime["vae_checkpoint"], "scragvae")
        self.assertEqual(runtime["lm_model_path"], "acestep-5Hz-lm-4B")
        self.assertEqual(runtime["backend_dropdown"], "vllm")
        self.assertTrue(runtime["lm_use_legacy_cfg_prompt"])
        self.assertEqual(runtime["mlx_vae_chunk_size"], 512)
        self.assertEqual(runtime["compile_threads_slider"], 13)

    def test_foreground_generate_auto_initializes_dit_when_missing(self):
        """Generate should auto-initialize the foreground DiT service when needed."""
        module, _state = load_batch_management_module(is_windows=False)

        def _gen(*_args, **_kwargs):
            """Yield one result after the auto-init step completes."""
            yield build_progress_result(length=56)

        kwargs = _build_call_kwargs(module)
        kwargs.update(
            {
                "batch_size_input": 1,
                "config_path": "acestep-v15-xl-sft",
                "device": "cuda",
                "use_flash_attention_checkbox": False,
                "offload_to_cpu_checkbox": False,
                "offload_dit_to_cpu_checkbox": False,
                "compile_model_checkbox": False,
                "quantization_checkbox": False,
                "mlx_dit_checkbox": True,
                "init_llm_checkbox": False,
                "think_checkbox": False,
                "auto_score": False,
            }
        )

        dit_handler = MagicMock()
        dit_handler.model = None
        dit_handler.initialize_service.return_value = ("Initialized foreground DiT", True)
        llm_handler = MagicMock()
        llm_handler.llm_initialized = False

        with patch.dict(
            module.generate_with_batch_management.__globals__,
            {"generate_with_progress": _gen},
        ):
            outputs = list(
                module.generate_with_batch_management(
                    dit_handler,
                    llm_handler,
                    **kwargs,
                )
            )

        dit_handler.initialize_service.assert_called_once()
        self.assertGreaterEqual(len(outputs), 2)
        self.assertIn("Initializing DiT service", outputs[0][18])

    def test_foreground_dit_reinit_unloads_lm_before_loading_replacement(self):
        """Changing DiT checkpoints should free the foreground LM before DiT init."""

        module, _state = load_batch_management_module(is_windows=False)
        events = []
        dit_handler = _ForegroundDitHandler(events, config_path="acestep-v15-xl-turbo")
        llm_handler = _ForegroundLlmHandler(events, lm_model_path="acestep-5Hz-lm-4B")

        ok, status = _ensure_in_process_service_ready(module)(
            dit_handler,
            llm_handler,
            config_path="acestep-v15-xl-sft",
            device="cuda",
            vae_checkpoint="official",
            lm_model_path="acestep-5Hz-lm-4B",
            backend_dropdown="pt",
            init_llm_checkbox=True,
            use_flash_attention_checkbox=False,
            offload_to_cpu_checkbox=False,
            offload_dit_to_cpu_checkbox=False,
            compile_model_checkbox=False,
            quantization_checkbox=False,
            mlx_dit_checkbox=True,
            think_checkbox=True,
            auto_score=False,
        )

        self.assertTrue(ok)
        self.assertEqual(events, ["lm_unload", "dit_init", "lm_init"])
        self.assertIn("Unloaded 5Hz LM before DiT reinitialization.", status)

    def test_foreground_lm_reinit_unloads_old_lm_before_loading_replacement(self):
        """Changing LM settings should unload the old LM before loading the new one."""

        module, _state = load_batch_management_module(is_windows=False)
        events = []
        dit_handler = _ForegroundDitHandler(events, config_path="acestep-v15-xl-sft")
        llm_handler = _ForegroundLlmHandler(events, lm_model_path="lm-old")

        ok, status = _ensure_in_process_service_ready(module)(
            dit_handler,
            llm_handler,
            config_path="acestep-v15-xl-sft",
            device="cuda",
            vae_checkpoint="official",
            lm_model_path="lm-new",
            backend_dropdown="pt",
            init_llm_checkbox=True,
            use_flash_attention_checkbox=False,
            offload_to_cpu_checkbox=False,
            offload_dit_to_cpu_checkbox=False,
            compile_model_checkbox=False,
            quantization_checkbox=False,
            mlx_dit_checkbox=True,
            think_checkbox=True,
            auto_score=False,
        )

        self.assertTrue(ok)
        self.assertEqual(events, ["lm_unload", "lm_init"])
        self.assertIn("Unloaded 5Hz LM before 5Hz LM reinitialization.", status)

    def test_foreground_dit_reinits_when_vae_checkpoint_changes(self):
        """Changing the selected VAE should reinitialize the foreground DiT."""

        module, _state = load_batch_management_module(is_windows=False)
        events = []
        dit_handler = _ForegroundDitHandler(events, config_path="acestep-v15-xl-sft")
        llm_handler = _ForegroundLlmHandler(events, lm_model_path="acestep-5Hz-lm-4B")
        llm_handler.llm_initialized = False

        ok, status = _ensure_in_process_service_ready(module)(
            dit_handler,
            llm_handler,
            config_path="acestep-v15-xl-sft",
            device="cuda",
            vae_checkpoint="scragvae",
            lm_model_path="acestep-5Hz-lm-4B",
            backend_dropdown="pt",
            init_llm_checkbox=False,
            use_flash_attention_checkbox=False,
            offload_to_cpu_checkbox=False,
            offload_dit_to_cpu_checkbox=False,
            compile_model_checkbox=False,
            quantization_checkbox=False,
            mlx_dit_checkbox=True,
            think_checkbox=False,
            auto_score=False,
        )

        self.assertTrue(ok)
        self.assertEqual(events, ["lm_unload", "dit_init"])
        self.assertEqual(dit_handler.last_init_params["vae_checkpoint"], "scragvae")
        self.assertIn("Initializing DiT service", status)

    def test_foreground_auto_init_keeps_matching_lm_loaded(self):
        """Matching DiT and LM settings should not unload or reinitialize services."""

        module, _state = load_batch_management_module(is_windows=False)
        events = []
        dit_handler = _ForegroundDitHandler(events, config_path="acestep-v15-xl-sft")
        llm_handler = _ForegroundLlmHandler(events, lm_model_path="acestep-5Hz-lm-4B")

        ok, status = _ensure_in_process_service_ready(module)(
            dit_handler,
            llm_handler,
            config_path="acestep-v15-xl-sft",
            device="cuda",
            vae_checkpoint="official",
            lm_model_path="acestep-5Hz-lm-4B",
            backend_dropdown="pt",
            init_llm_checkbox=True,
            use_flash_attention_checkbox=False,
            offload_to_cpu_checkbox=False,
            offload_dit_to_cpu_checkbox=False,
            compile_model_checkbox=False,
            quantization_checkbox=False,
            mlx_dit_checkbox=True,
            think_checkbox=True,
            auto_score=False,
        )

        self.assertTrue(ok)
        self.assertEqual("", status)
        self.assertEqual(events, [])

    def test_foreground_generation_applies_lora_before_inner_generation(self):
        """Selected LoRA should be loaded and enabled before generation starts."""
        module, _state = load_batch_management_module(is_windows=True)

        with tempfile.TemporaryDirectory() as tmp:
            adapter = _write_peft_adapter(Path(tmp) / "voice")
            seen = {}

            def _gen(*args, **_kwargs):
                """Assert LoRA has been synchronized before generation."""
                seen["handler"] = args[0]
                seen["load_called_before_generation"] = args[0].load_lora.called
                yield build_progress_result(length=56)

            kwargs = _build_call_kwargs(module)
            kwargs.update(
                {
                    "lora_dropdown": str(adapter),
                    "lora_path": "",
                    "lora_scale_slider": 0.5,
                    "config_path": "acestep-v15-xl-sft",
                    "mlx_dit_checkbox": True,
                    "think_checkbox": False,
                    "auto_score": False,
                }
            )
            dit_handler = MagicMock()
            dit_handler.model = object()
            dit_handler.last_init_params = {
                "config_path": "acestep-v15-xl-sft",
                "use_flash_attention": False,
                "offload_to_cpu": False,
                "offload_dit_to_cpu": False,
                "compile_model": False,
                "use_mlx_dit": True,
                "quantization": None,
            }
            dit_handler.lora_loaded = False
            dit_handler.load_lora.return_value = "LoRA loaded"

            with patch.dict(
                module.generate_with_batch_management.__globals__,
                {"generate_with_progress": _gen},
            ):
                list(module.generate_with_batch_management(dit_handler, None, **kwargs))

        self.assertTrue(seen["load_called_before_generation"])
        dit_handler.load_lora.assert_called_once()
        self.assertEqual(Path(dit_handler.load_lora.call_args.args[0]).resolve(), adapter.resolve())
        dit_handler.set_lora_scale.assert_called_once_with(0.5)
        dit_handler.set_use_lora.assert_called_once_with(True)

    def test_subprocess_generation_payload_uses_effective_lora_selection(self):
        """Subprocess generation should receive resolved LoRA path and enabled flag."""
        module, state = load_batch_management_module(is_windows=True)

        with tempfile.TemporaryDirectory() as tmp:
            adapter = _write_peft_adapter(Path(tmp) / "voice")
            seen = {}

            def _stream(payload):
                seen["payload"] = payload
                raise RuntimeError("stop after payload capture")

            kwargs = _build_call_kwargs(module)
            kwargs.update(
                {
                    "subprocess_mode_checkbox": True,
                    "config_path": "acestep-v15-xl-turbo",
                    "device": "cuda",
                    "vae_checkpoint": "scragvae",
                    "inference_steps": 8,
                    "lm_use_legacy_cfg_prompt": True,
                    "compile_threads_slider": 21,
                    "mlx_vae_chunk_size": 512,
                    "lora_dropdown": str(adapter),
                    "lora_path": "",
                    "use_lora_checkbox": False,
                    "lora_scale_slider": 0.8,
                }
            )

            with patch.dict(
                module.generate_with_batch_management.__globals__,
                {"stream_subprocess_generation": _stream},
            ):
                target = Path(tmp) / "outputs"
                with use_results_dir(target), use_generation_run_name("batch-song"):
                    list(module.generate_with_batch_management(None, None, **kwargs))

        service_payload = seen["payload"]["service"]
        generation_payload = seen["payload"]["generation"]
        log_text = "\n".join(state["log_info"])
        self.assertEqual(service_payload["config_path"], "acestep-v15-xl-turbo")
        self.assertEqual(service_payload["vae_checkpoint"], "scragvae")
        self.assertTrue(service_payload["lm_use_legacy_cfg_prompt"])
        self.assertEqual(service_payload["compile_threads"], 21)
        self.assertEqual(service_payload["mlx_vae_chunk_size"], 512)
        self.assertEqual(generation_payload["inference_steps"], 8)
        self.assertIn("model=acestep-v15-xl-turbo", log_text)
        self.assertIn("inference_steps=8", log_text)
        self.assertEqual(Path(seen["payload"]["output_dir"]).resolve(), target.resolve())
        self.assertEqual(seen["payload"]["run_name"], "batch-song")
        self.assertEqual(Path(service_payload["lora_path"]).resolve(), adapter.resolve())
        self.assertTrue(service_payload["use_lora"])
        self.assertEqual(service_payload["lora_scale"], 0.8)


if __name__ == "__main__":
    unittest.main()
