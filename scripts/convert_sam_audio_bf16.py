"""Convert a SAM-Audio checkpoint to BF16 safetensors."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCES = (
    PROJECT_ROOT / "models" / "SAM-Audio-Large.pt",
    PROJECT_ROOT / "models" / "SAM-Audio-Large" / "checkpoint.pt",
)
DEFAULT_OUTPUT = PROJECT_ROOT / "models" / "SAM-Audio-Large-BF16.safetensors"


def main() -> int:
    """Run the checkpoint conversion CLI."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    source = _resolve_source(args.source)
    output = args.output.expanduser().resolve()
    if output.exists() and not args.overwrite:
        print(f"Output already exists: {output}")
        _verify_bf16(output)
        return 0

    print(f"Loading: {source}")
    state_dict = _load_state_dict(source)
    print(f"Converting {len(state_dict)} tensors to BF16 where applicable.")
    converted = {
        key: value.to(torch.bfloat16) if value.is_floating_point() else value
        for key, value in state_dict.items()
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        converted,
        str(output),
        metadata={
            "format": "pt-to-bf16-safetensors",
            "source": str(source),
            "floating_dtype": "bfloat16",
        },
    )
    print(f"Wrote: {output}")
    _verify_bf16(output)
    return 0


def _resolve_source(source: Path | None) -> Path:
    """Return the source checkpoint path to convert."""

    if source is not None:
        resolved = source.expanduser().resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"SAM-Audio checkpoint not found: {resolved}")
        return resolved
    for candidate in DEFAULT_SOURCES:
        if candidate.is_file():
            return candidate.resolve()
    searched = ", ".join(str(path) for path in DEFAULT_SOURCES)
    raise FileNotFoundError(f"SAM-Audio checkpoint not found. Searched: {searched}")


def _load_state_dict(source: Path) -> dict[str, torch.Tensor]:
    """Load a torch checkpoint state dict from disk."""

    try:
        payload = torch.load(source, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        payload = torch.load(source, map_location="cpu", weights_only=True)
    if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
        payload = payload["state_dict"]
    if not isinstance(payload, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(payload)!r}")
    tensors = {key: value for key, value in payload.items() if isinstance(value, torch.Tensor)}
    if not tensors:
        raise ValueError("Checkpoint does not contain tensor weights.")
    return tensors


def _verify_bf16(path: Path) -> None:
    """Raise if any floating-point tensor is not BF16."""

    with safe_open(str(path), framework="pt", device="cpu") as handle:
        floating_total = 0
        for key in handle.keys():
            tensor = handle.get_tensor(key)
            if not tensor.is_floating_point():
                continue
            floating_total += 1
            if tensor.dtype is not torch.bfloat16:
                raise RuntimeError(f"{key} is {tensor.dtype}, expected torch.bfloat16")
    print(f"Verified BF16 floating tensors: {floating_total}")


if __name__ == "__main__":
    raise SystemExit(main())
