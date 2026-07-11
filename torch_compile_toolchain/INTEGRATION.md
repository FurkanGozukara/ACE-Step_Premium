# Reusing the Automatic torch.compile Toolchain

This guide travels inside the canonical `torch_compile_toolchain/` package. The
package is intentionally separate from ACE-Step so this directory can be copied
unchanged into another app.

## Integration contract

An application owns three decisions:

1. Whether compilation is enabled for a particular model or callable.
2. Whether CUDA Toolkit and Ninja are hard requirements for that workload.
3. How compilation success/fallback is displayed to its users.

The package owns discovery, environment activation, compatibility probing, cache
paths, and structured diagnostics. It does not install system software.

The minimum integration is:

```python
from torch_compile_toolchain import ensure_compile_environment

toolchain = ensure_compile_environment(project_root=PROJECT_ROOT)
if toolchain.ok:
    compiled_forward = torch.compile(model.forward, backend="inductor")
    model.forward = compiled_forward
else:
    logger.warning("torch.compile unavailable: %s", toolchain.detail)
```

In-process compilation should use the default `env=None`, which updates
`os.environ`. A separate mapping is useful for diagnostics, but PyTorch will not see
that mapping unless it is passed to a child process.

For a reusable guarded integration:

```python
from torch_compile_toolchain import compile_module_callable

compile_result = compile_module_callable(
    model,
    attribute_name="forward",
    enabled=settings.compile_model,
    backend="inductor",
    project_root=PROJECT_ROOT,
    on_status=lambda state, detail: logger.info("%s: %s", state, detail),
)
```

The same function supports methods such as `decode`:

```python
compile_module_callable(vae, "decode", project_root=PROJECT_ROOT)
```

## Package layout

```text
torch_compile_toolchain/
|-- __init__.py       Public API and version
|-- __main__.py       `python -m` diagnostic CLI
|-- environment.py    Orchestration, status, caches, subprocess environment
|-- discovery.py      CUDA, Ninja, and POSIX compiler discovery
|-- cuda_toolkit.py   PyTorch-aware CUDA Toolkit selection
|-- msvc.py           Visual Studio/MSVC discovery and activation
|-- compiler_probe.py Real nvcc host-compiler compatibility probe
|-- runtime.py        Optional safe torch.compile callable integration
|-- example.py        Executable end-to-end CUDA/CPU example
|-- runtime_test.py   Copy-isolation and safe-fallback regression tests
|-- py.typed          Type-checker marker
|-- README.md         Copy-first quick-start documentation
`-- INTEGRATION.md    Detailed architecture and migration guide
```

No file in this package imports `acestep`. ACE-Step's files named
`acestep/torch_compile_*.py` are compatibility facades and should not be copied into
another application.

## Public status

`ensure_compile_environment()` returns `CompileToolchainStatus`:

```python
{
    "ok": True,
    "detail": "CUDA Toolkit found ...; MSVC ...; ninja found ...",
    "changed": True,
    "platform": "win32",
    "cuda_root": "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v13.1",
    "compiler_path": "C:/.../cl.exe",
    "ninja_path": "C:/.../ninja.exe",
    "cache_root": "D:/my-app/.cache/torch_compile_toolchain",
}
```

Use `status.as_dict()` when returning this information through an API or support
bundle. `ok` means a validated host compiler is ready. Set
`require_cuda_toolkit=True` and/or `require_ninja=True` when those components must
also be present for the app's compile path.

## Discovery order

### Windows

1. Select the CUDA Toolkit that best matches `torch.version.cuda`.
2. If `cl.exe` is already active, compile a tiny CUDA source with it.
3. Otherwise find Visual Studio roots through configured environment variables,
   `vswhere`, registry keys, and dynamically scanned Program Files locations.
4. Enumerate all x64 MSVC toolset directories across all installations.
5. Try exact `-vcvars_ver`, family `-vcvars_ver`, then default developer-script
   activation for each candidate.
6. Accept a toolset only after `nvcc` compiles a real `.cu` file. Continue to older
   candidates after an unsupported-host-compiler result.
7. Discover and prioritize Ninja after the Visual Studio environment has loaded.

This avoids hard-coding a Visual Studio year or assuming that the newest MSVC is
compatible with the installed CUDA release.

### Linux/POSIX

1. Honor `NVCC_CCBIN`, `CUDAHOSTCXX`, and `CXX` in that order.
2. Resolve wrapper-style values such as `ccache g++` and compiler-directory values.
3. Scan default, dynamically discovered versioned, Conda-prefixed, GCC, Clang,
   NVIDIA HPC, Intel, and ARM compiler candidates.
4. Probe candidates through `nvcc -ccbin`; continue after incompatibility.
5. Export the accepted path through `CXX`, `CUDAHOSTCXX`, and `NVCC_CCBIN`.

### CUDA Toolkit

Explicit and standard candidates are collected first. When PyTorch reports a CUDA
version, an exact toolkit wins; otherwise, the nearest minor release with the same
major version wins. This permits, for example, a PyTorch CUDA 13.0 build to use an
installed CUDA 13.1 toolkit after the real compile probe validates the combination.

## Environment overrides

The package recognizes these common overrides:

| Purpose | Variables |
| --- | --- |
| CUDA root | `CUDA_HOME`, `CUDA_PATH`, `CUDAToolkit_ROOT`, `CUDA_ROOT`, `CUDA_PATH_V*` |
| CUDA compiler | `CUDACXX`, `NVCC` |
| Linux host compiler | `NVCC_CCBIN`, `CUDAHOSTCXX`, `CXX`, `CC` |
| Visual Studio | `VSWHERE`, `VSINSTALLDIR`, `VCINSTALLDIR`, `VCToolsInstallDir` |
| Ninja | `NINJA`, `CMAKE_MAKE_PROGRAM` |
| Cache | `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR` |
| Project root | `TORCH_COMPILE_PROJECT_ROOT` |

Explicit valid values are preferred, but an invalid or CUDA-incompatible compiler
does not prevent fallback to another installed candidate. Existing user cache
variables are preserved.

## Worker and service processes

Visual Studio developer scripts only modify the process that runs them. For an app
that launches generation or training workers, prepare the environment passed to the
child:

```python
child_env = prepare_compile_subprocess_env(
    os.environ,
    project_root=PROJECT_ROOT,
    compile_requested=request.compile_model,
)
subprocess.Popen(command, env=child_env)
```

Do not build a second hand-maintained PATH in the worker. The returned mapping
already contains CUDA, compiler, Windows SDK, include/library, Ninja, and cache
variables.

## Cache behavior

By default, portable integrations use:

```text
<project>/.cache/torch_compile_toolchain/inductor
<project>/.cache/torch_compile_toolchain/triton
```

Pass `cache_dir=` for another location. If the project directory is read-only, the
package falls back to the operating-system temporary directory. Explicit
`TORCHINDUCTOR_CACHE_DIR` and `TRITON_CACHE_DIR` values are not overwritten.

ACE-Step's compatibility facade continues using its historical
`.cache/acestep/torch_compile` location.

## Error and retry behavior

- Discovery failures are not permanently cached. A later service initialization can
  detect tools installed after application startup.
- Every distinct MSVC/Windows SDK environment can be probed; a rejected result does
  not hide another installation.
- `compile_callable()` catches setup errors and returns the original callable.
- Its optional cold-call fallback changes the result to eager mode after a failed
  first compiled call.
- After one successful compiled call, later exceptions are propagated normally so
  application errors are not silently converted into eager retries.

## Logging

The package uses Python's standard `logging` module and accepts an `on_status`
callback in the high-level runtime API. It does not require Loguru or another
application logging framework.

```python
def report_compile_status(state: str, detail: str) -> None:
    app_logger.info("[torch_compile] status=%s detail=%s", state, detail)

result = compile_module_callable(model, on_status=report_compile_status)
```

## Validation after copying

From the destination application's virtual environment, run:

```text
python -m torch_compile_toolchain --json
python -m torch_compile_toolchain.example
python -m unittest torch_compile_toolchain.runtime_test
```

Then run one representative model forward. `setup_ready` proves that a compiled
wrapper was installed; `first_call_ok` proves that the first real graph compiled and
executed. Tool discovery alone cannot guarantee that every model graph is supported
by TorchDynamo/Inductor, so retaining application-level eager fallback is recommended.

## Updating another repository

Treat `torch_compile_toolchain/` as one vendored unit. Replace the whole directory
when taking a newer version rather than copying individual files. The public imports
from `torch_compile_toolchain/__init__.py` are the stable integration surface; the
underscored helpers inside implementation modules are internal.

The implementation follows Microsoft's Visual Studio developer-command and
`vswhere` mechanisms, NVIDIA's documented CUDA host-compiler overrides, and
PyTorch's documented compiler/CUDA environment conventions:

- https://learn.microsoft.com/cpp/build/building-on-the-command-line
- https://github.com/microsoft/vswhere/wiki
- https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/
- https://docs.nvidia.com/cuda/cuda-installation-guide-linux/
- https://docs.pytorch.org/docs/stable/cpp_extension.html
