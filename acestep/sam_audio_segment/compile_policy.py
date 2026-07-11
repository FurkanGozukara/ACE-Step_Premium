"""Measured component policy for SAM-Audio ``torch.compile`` use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


SamCompileComponent = Literal[
    "diffusion_forward",
    "codec_encoder",
    "codec_decoder",
    "text_encoder",
    "span_predictor",
    "rankers",
]

_COMPONENT_FIELDS: dict[SamCompileComponent, str] = {
    "diffusion_forward": "diffusion_forward",
    "codec_encoder": "codec_encoder",
    "codec_decoder": "codec_decoder",
    "text_encoder": "text_encoder",
    "span_predictor": "span_predictor",
    "rankers": "rankers",
}

SAM_EAGER_DETAILS: dict[SamCompileComponent, str] = {
    "codec_encoder": "measured slower warm; called once per separation pass",
    "codec_decoder": "measured slower warm; called once per separation pass",
    "text_encoder": "no measured warm benefit; called once per prompt",
    "span_predictor": "optional dynamic conditioning path; not a universal compile target",
    "rankers": "optional external reranking paths; not a universal compile target",
    "diffusion_forward": "disabled by user option",
}


@dataclass(frozen=True)
class SamCompilePolicy:
    """Resolved compile targets for one SAM-Audio service configuration."""

    requested: bool
    diffusion_forward: bool
    codec_encoder: bool
    codec_decoder: bool
    text_encoder: bool
    span_predictor: bool
    rankers: bool

    def enabled_for(self, component: SamCompileComponent) -> bool:
        """Return whether a component should receive a compiled callable."""

        try:
            field = _COMPONENT_FIELDS[component]
        except KeyError as exc:
            raise ValueError(f"unknown SAM compile component: {component}") from exc
        return bool(getattr(self, field))

    def targets(self) -> dict[str, bool]:
        """Return JSON-safe component decisions for logs and metadata."""

        return {
            component: self.enabled_for(component)
            for component in _COMPONENT_FIELDS
        }

    def disabled_detail(self, component: SamCompileComponent) -> str:
        """Explain why a disabled component remains eager."""

        if not self.requested:
            return "disabled by user option"
        if self.enabled_for(component):
            return ""
        return SAM_EAGER_DETAILS[component]


def resolve_sam_compile_policy(requested: bool) -> SamCompilePolicy:
    """Compile only SAM-Audio's repeatedly invoked diffusion/ODE forward.

    Controlled warm benchmarks showed a material speedup for the diffusion
    forward. Codec encode/decode and T5 text encoding were neutral or slower,
    while optional span/ranker paths are dynamic and request dependent.
    """

    enabled = bool(requested)
    return SamCompilePolicy(
        requested=enabled,
        diffusion_forward=enabled,
        codec_encoder=False,
        codec_decoder=False,
        text_encoder=False,
        span_predictor=False,
        rankers=False,
    )
