"""SAM-Audio Large model configuration helpers."""

from __future__ import annotations

from copy import deepcopy

from .judge_assets import prepare_local_judge_model_dir, resolve_local_judge_checkpoint
from .ranker_availability import normalize_ranker_mode
from .settings import SamAudioSettings


SAM_AUDIO_LARGE_CONFIG: dict = {
    "in_channels": 768,
    "audio_codec": {
        "encoder_dim": 64,
        "encoder_rates": [2, 8, 10, 12],
        "latent_dim": 1024,
        "decoder_dim": 1536,
        "decoder_rates": [12, 10, 8, 2],
        "n_codebooks": 16,
        "codebook_size": 1024,
        "codebook_dim": 128,
        "quantizer_dropout": False,
        "sample_rate": 48000,
        "mean": 0.0,
        "std": 1.0,
    },
    "text_encoder": {
        "dim": 768,
        "name": "t5-base",
        "max_length": 512,
        "pad_mode": "longest",
    },
    "vision_encoder": {
        "dim": 1024,
        "batch_size": 300,
        "name": "PE-Core-L14-336",
        "normalize_feature": True,
        "interpolation_mode": "BICUBIC",
        "image_size": 336,
    },
    "transformer": {
        "dim": 2816,
        "n_heads": 22,
        "n_layers": 22,
        "dropout": 0.1,
        "norm_eps": 1e-05,
        "qk_norm": True,
        "fc_bias": False,
        "ffn_exp": 4,
        "ffn_dim_multiplier": 1,
        "multiple_of": 64,
        "non_linearity": "swiglu",
        "use_rope": True,
        "max_positions": 10000,
        "frequency_embedding_dim": 256,
        "timestep_non_linearity": "swiglu",
        "t_block_non_linearity": "silu",
        "t_block_bias": True,
        "context_dim": 2816,
        "context_non_linearity": "swiglu",
        "context_embedder_dropout": 0.0,
        "context_norm": False,
        "out_channels": 256,
        "in_channels": None,
    },
    "num_anchors": 3,
    "anchor_embedding_dim": 128,
    "visual_ranker": None,
    "text_ranker": None,
    "span_predictor": "pe-a-frame-large",
}


def config_for_settings(settings: SamAudioSettings) -> dict:
    """Return a SAM-Audio Large config adjusted for selected optional rankers."""

    config = deepcopy(SAM_AUDIO_LARGE_CONFIG)
    if not settings.predict_spans:
        config["span_predictor"] = None
    mode = normalize_ranker_mode(settings.ranker_mode)
    if mode == "imagebind":
        config["visual_ranker"] = {"checkpoint": None, "kind": "imagebind"}
    elif mode == "clap":
        config["text_ranker"] = {"checkpoint": None, "kind": "clap"}
    elif mode == "judge":
        config["text_ranker"] = _local_judge_ranker_config()
    elif mode == "text_ensemble":
        config["text_ranker"] = {
            "rankers": {
                "clap": [{"checkpoint": None, "kind": "clap"}, 5.0],
                "judge": [_local_judge_ranker_config(), 1.0],
            },
            "kind": "ensemble",
        }
    return config


def _local_judge_ranker_config() -> dict[str, str]:
    """Return a Judge ranker config using only local model assets."""

    return {
        "checkpoint_or_model_id": str(prepare_local_judge_model_dir()),
        "checkpoint_path": str(resolve_local_judge_checkpoint()),
        "kind": "judge",
    }
