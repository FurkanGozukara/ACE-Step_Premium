"""Compatibility imports for :mod:`torch_compile_toolchain.msvc`."""

from torch_compile_toolchain.msvc import (
    MsvcLoadStatus,
    describe_msvc_environment,
    has_cl_exe,
    load_msvc_environment,
    visual_studio_install_roots,
)

__all__ = [
    "MsvcLoadStatus",
    "describe_msvc_environment",
    "has_cl_exe",
    "load_msvc_environment",
    "visual_studio_install_roots",
]
