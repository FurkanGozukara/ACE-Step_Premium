"""Audio-code parsing and conversion helpers for handler decomposition."""

import re
import traceback
from typing import List, Optional

import torch
from loguru import logger


class AudioCodesMixin:
    """Mixin containing audio-code parsing and latent conversion helpers.

    Depends on host members:
    - Attributes: ``model``, ``vae``, ``device``, ``dtype``, ``silence_latent``.
    - Methods: ``_load_model_context``, ``process_src_audio``, ``is_silence``,
      ``_encode_audio_to_latents``.
    """

    def _parse_audio_code_string(self, code_str: str) -> List[int]:
        """Extract integer audio codes from tokens like ``<|audio_code_123|>``."""
        if not code_str:
            return []
        try:
            max_audio_code = 63999
            codes = []
            clamped_count = 0
            for x in re.findall(r"<\|audio_code_(\d+)\|>", code_str):
                code_value = int(x)
                clamped_value = max(0, min(code_value, max_audio_code))
                if clamped_value != code_value:
                    clamped_count += 1
                    logger.warning(
                        f"[_parse_audio_code_string] Clamped audio code value from {code_value} to {clamped_value}"
                    )
                codes.append(clamped_value)
            if clamped_count > 0:
                logger.warning(
                    f"[_parse_audio_code_string] Clamped {clamped_count} audio code value(s) "
                    f"to valid range [0, {max_audio_code}]"
                )
            return codes
        except Exception as e:
            logger.debug(f"[_parse_audio_code_string] Failed to parse audio code string: {e}")
            return []

    def _align_quantizer_decode_buffers(self, quantizer: torch.nn.Module) -> None:
        """Align ResidualFSQ decode buffers with projection parameters.

        Direct BF16 model loading can leave non-persistent quantizer buffers such
        as ``scales`` in float32 while ``project_out`` is BF16.  ResidualFSQ
        multiplies codebooks by ``scales`` before calling ``project_out``, so the
        buffer has to match before ``get_output_from_indices`` runs.
        """
        project_out = getattr(quantizer, "project_out", None)
        if not isinstance(project_out, torch.nn.Module):
            return

        target_tensor = next(
            (param for param in project_out.parameters() if param.is_floating_point()),
            None,
        )
        if target_tensor is None:
            return

        for buffer_name in ("scales",):
            buffer = getattr(quantizer, buffer_name, None)
            if (
                isinstance(buffer, torch.Tensor)
                and buffer.is_floating_point()
                and (buffer.dtype != target_tensor.dtype or buffer.device != target_tensor.device)
            ):
                setattr(
                    quantizer,
                    buffer_name,
                    buffer.to(device=target_tensor.device, dtype=target_tensor.dtype),
                )

    def _decode_audio_codes_to_latents(self, code_str: str) -> Optional[torch.Tensor]:
        """Convert serialized audio-code string into 25Hz latents."""
        if self.model is None or not hasattr(self.model, "tokenizer") or not hasattr(self.model, "detokenizer"):
            return None

        code_ids = self._parse_audio_code_string(code_str)
        if len(code_ids) == 0:
            return None

        with self._load_model_context("model"):
            quantizer = self.model.tokenizer.quantizer
            detokenizer = self.model.detokenizer
            self._align_quantizer_decode_buffers(quantizer)
            indices = torch.tensor(code_ids, device=self.device, dtype=torch.long)
            indices = indices.unsqueeze(0).unsqueeze(-1)

            quantized = quantizer.get_output_from_indices(indices)
            if quantized.dtype != self.dtype:
                quantized = quantized.to(self.dtype)
            lm_hints_25hz = detokenizer(quantized)
            return lm_hints_25hz

    def convert_src_audio_to_codes(self, audio_file) -> str:
        """Convert uploaded source audio into serialized audio code tokens."""
        if audio_file is None:
            return "❌ Please upload source audio first"
        if self.model is None or self.vae is None:
            return "❌ Model not initialized. Please initialize the service first."

        try:
            processed_audio = self.process_src_audio(audio_file)
            if processed_audio is None:
                return "❌ Failed to process audio file"

            with torch.inference_mode():
                with self._load_model_context("vae"):
                    if self.is_silence(processed_audio.unsqueeze(0)):
                        return "❌ Audio file appears to be silent"
                    latents = self._encode_audio_to_latents(processed_audio)

                attention_mask = torch.ones(latents.shape[0], dtype=torch.bool, device=self.device)
                with self._load_model_context("model"):
                    hidden_states = latents.unsqueeze(0)
                    _, indices, _ = self.model.tokenize(
                        hidden_states, self.silence_latent, attention_mask.unsqueeze(0)
                    )
                    indices_flat = indices.flatten().cpu().tolist()
                    codes_string = "".join([f"<|audio_code_{idx}|>" for idx in indices_flat])
                    logger.info(f"[convert_src_audio_to_codes] Generated {len(indices_flat)} audio codes")
                    return codes_string
        except Exception as e:
            error_msg = f"❌ Error converting audio to codes: {str(e)}\n{traceback.format_exc()}"
            logger.exception("[convert_src_audio_to_codes] Error converting audio to codes")
            return error_msg
