"""Command-line diagnostic for the portable torch.compile toolchain package."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from .environment import compile_environment_report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Discover and validate the local torch.compile build toolchain."
    )
    parser.add_argument("--project-root", type=Path, default=None)
    parser.add_argument("--cache-dir", type=Path, default=None)
    parser.add_argument("--require-cuda-toolkit", action="store_true")
    parser.add_argument("--require-ninja", action="store_true")
    parser.add_argument("--json", action="store_true", dest="as_json")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    status = compile_environment_report(
        project_root=args.project_root,
        cache_dir=args.cache_dir,
        require_cuda_toolkit=args.require_cuda_toolkit,
        require_ninja=args.require_ninja,
    )
    if args.as_json:
        print(json.dumps(status.as_dict(), indent=2))
    else:
        print("torch.compile toolchain:", "ready" if status.ok else "unavailable")
        print(status.detail)
        for label, value in (
            ("compiler", status.compiler_path),
            ("CUDA root", status.cuda_root),
            ("Ninja", status.ninja_path),
            ("cache", status.cache_root),
        ):
            if value:
                print(f"{label}: {value}")
    return 0 if status.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
