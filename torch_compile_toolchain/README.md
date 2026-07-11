# Portable torch.compile Toolchain

This directory is a self-contained Python package for automatically discovering and
activating the native build tools used by `torch.compile`. It has no ACE-Step imports
and uses only the Python standard library until PyTorch itself is needed.

To reuse it in another application, copy the entire `torch_compile_toolchain/`
directory into that application's repository. Do not copy the similarly named
compatibility files under `acestep/`.

## Quick start

Prepare the current process before calling `torch.compile`:

```python
from pathlib import Path

import torch
from torch_compile_toolchain import ensure_compile_environment

status = ensure_compile_environment(project_root=Path(__file__).resolve().parent)
print(status.detail)

if status.ok:
    model = torch.compile(model, backend="inductor")
```

For in-process compilation, omit the `env` argument so the helper updates
`os.environ`. Use an explicit mapping only for inspection or child-process setup.

For an application that builds CUDA/C++ extensions and therefore requires both a
CUDA Toolkit and Ninja, enable strict requirements:

```python
status = ensure_compile_environment(
    project_root=PROJECT_ROOT,
    require_cuda_toolkit=True,
    require_ninja=True,
)
```

## Safe callable integration

The high-level helper installs a compiled method and automatically keeps an eager
callable available if the cold compiled call fails:

```python
from torch_compile_toolchain import compile_module_callable

result = compile_module_callable(
    model,
    "forward",
    enabled=True,
    backend="inductor",
    mode="default",
    project_root=PROJECT_ROOT,
)

output = model(inputs)
print(result.compiled, result.verified, result.detail)
```

`result.verified` becomes `True` after the first compiled call succeeds. If that call
fails and fallback is enabled, the helper retries through the original eager callable
and uses eager execution afterward. Avoid this retry behavior for callables with
non-idempotent side effects, or set `fallback_on_first_error=False`.

## Subprocess integration

Prepare a child environment without changing the parent process:

```python
import subprocess
import sys

from torch_compile_toolchain import prepare_compile_subprocess_env

child_env = prepare_compile_subprocess_env(
    project_root=PROJECT_ROOT,
    compile_requested=True,
)
subprocess.Popen([sys.executable, "worker.py"], env=child_env)
```

## Diagnostic command

Run discovery from the same virtual environment as the application:

```text
python -m torch_compile_toolchain
python -m torch_compile_toolchain --json
python -m torch_compile_toolchain --require-cuda-toolkit --require-ninja
python -m torch_compile_toolchain.example
```

The command reports the selected compiler, CUDA root, Ninja executable, cache root,
and the exact reason when the environment is unavailable.

## Public API

| API | Purpose |
| --- | --- |
| `ensure_compile_environment()` | Mutate the current/supplied environment and return a structured status. |
| `prepare_compile_subprocess_env()` | Return a prepared copy suitable for `subprocess.Popen(env=...)`. |
| `compile_environment_report()` | Prepare and report the current process environment. |
| `compile_callable()` | Return a compiled callable guarded by optional cold-call fallback. |
| `compile_module_callable()` | Replace `module.forward`, `module.decode`, or another callable in place. |
| `CompileToolchainStatus.as_dict()` | Produce a JSON-serializable diagnostic payload. |

## What discovery covers

Windows:

- Existing `cl.exe` developer environments.
- Full Visual Studio and standalone Visual Studio Build Tools.
- Community, Professional, Enterprise, Build Tools, preview, custom-drive, registry,
  `vswhere`, environment-variable, standard, future-version, and legacy layouts.
- Every installed x64 MSVC toolset, ordered by its real toolset version.
- Exact and major/minor `-vcvars_ver` selection through `VsDevCmd.bat`, `vcvars64.bat`,
  and `vcvarsall.bat`.
- A real `nvcc` compile probe for each candidate; incompatible newer MSVC versions
  are skipped automatically.
- Ninja from environment overrides, PATH, virtual environments, Conda, the Python
  `ninja` package, and Visual Studio's CMake installation.

Linux and other POSIX systems:

- `NVCC_CCBIN`, `CUDAHOSTCXX`, `CXX`, and `CC` overrides.
- GCC, Clang, NVIDIA HPC, Intel oneAPI, ARM Clang, versioned executables, Conda
  target-prefixed compilers, and common GCC/LLVM installation directories.
- A real `nvcc -ccbin` probe, with fallback to another installed compiler when a
  candidate is incompatible.
- Ninja from PATH, virtual environments, Conda, Python, and common system paths.

CUDA selection:

- `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT`, `CUDA_ROOT`, versioned CUDA
  variables, `CUDACXX`, `NVCC`, Conda targets, standard Windows/Linux locations,
  and the `nvcc` already on PATH.
- Exact `torch.version.cuda` first, then the nearest toolkit with the same major
  version. The selected toolkit's `bin` directory is prioritized over stale PATH
  entries.

## Copy checklist

1. Copy this entire directory without renaming its internal files.
2. Add the parent directory to the application's Python import path.
3. Call `ensure_compile_environment()` before the first `torch.compile` call, or use
   `compile_module_callable()`.
4. For worker processes, call `prepare_compile_subprocess_env()` in the parent.
5. Surface `status.detail` in logs so support reports identify the exact selected or
   rejected compiler.
6. Run `python -m torch_compile_toolchain --json` on each deployment image.

The package discovers and validates installed tools; it never downloads, installs,
or modifies Visual Studio, CUDA, GCC, Clang, or Ninja.

For architecture, override precedence, testing, and migration details, see the
guide that travels with this package: [`INTEGRATION.md`](INTEGRATION.md).
