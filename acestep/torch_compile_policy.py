"""Measured component policy for ACE-Step inference ``torch.compile`` use."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


InferenceCompileComponent = Literal["5hz_lm", "dit", "vae"]

DIT_EAGER_DETAIL = (
    "kept eager by selective inference policy: variable-length ACE-Step DiT "
    "generation measured slower with torch.compile"
)
VAE_EAGER_DETAIL = (
    "kept eager by selective inference policy: tiled ACE-Step VAE decode "
    "measured slower with torch.compile"
)


@dataclass(frozen=True)
class InferenceCompilePolicy:
    """Resolved compile targets for one user inference request."""

    requested: bool
    lm: bool
    dit: bool
    vae: bool

    def enabled_for(self, component: InferenceCompileComponent) -> bool:
        """Return whether a component should receive a compiled callable."""

        if component == "5hz_lm":
            return self.lm
        if component == "dit":
            return self.dit
        if component == "vae":
            return self.vae
        raise ValueError(f"unknown inference compile component: {component}")

    def disabled_detail(self, component: InferenceCompileComponent) -> str:
        """Explain why a disabled component remains eager."""

        if not self.requested:
            return "disabled by user option"
        if self.enabled_for(component):
            return ""
        if component == "dit":
            return DIT_EAGER_DETAIL
        if component == "vae":
            return VAE_EAGER_DETAIL
        return "disabled by selective inference policy"


def resolve_inference_compile_policy(requested: bool) -> InferenceCompilePolicy:
    """Compile only the measured-beneficial PyTorch 5Hz LM inference path.

    ACE-Step DiT and tiled VAE shapes vary with generated audio-code length. The
    measured warm run improved LM throughput while DiT and VAE were slower and hit
    Dynamo recompilation limits, so those components intentionally stay eager.
    """

    enabled = bool(requested)
    return InferenceCompilePolicy(
        requested=enabled,
        lm=enabled,
        dit=False,
        vae=False,
    )
