"""Portable automatic toolchain setup for :func:`torch.compile`.

This package deliberately has no ACE-Step dependencies. Copy this entire directory
into another Python 3.10+ application and import the public helpers from here.
See ``README.md`` beside this file for integration patterns.
"""

from .compiler_probe import CompilerProbeStatus, probe_cuda_compiler
from .discovery import (
    CudaDiscoveryStatus,
    ExecutableStatus,
    discover_cuda_environment,
    discover_ninja,
    discover_posix_compiler,
)
from .environment import (
    CompileToolchainStatus,
    compile_environment_report,
    ensure_compile_environment,
    prepare_compile_subprocess_env,
)
from .msvc import (
    MsvcLoadStatus,
    describe_msvc_environment,
    has_cl_exe,
    load_msvc_environment,
    visual_studio_install_roots,
)
from .runtime import SafeCompileResult, compile_callable, compile_module_callable

__version__ = "1.0.0"

__all__ = [
    "CompileToolchainStatus",
    "CompilerProbeStatus",
    "CudaDiscoveryStatus",
    "ExecutableStatus",
    "MsvcLoadStatus",
    "SafeCompileResult",
    "compile_callable",
    "compile_environment_report",
    "compile_module_callable",
    "describe_msvc_environment",
    "discover_cuda_environment",
    "discover_ninja",
    "discover_posix_compiler",
    "ensure_compile_environment",
    "has_cl_exe",
    "load_msvc_environment",
    "prepare_compile_subprocess_env",
    "probe_cuda_compiler",
    "visual_studio_install_roots",
]
