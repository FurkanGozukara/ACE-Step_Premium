# ACE-Step Selective torch.compile Policy

ACE-Step inference intentionally compiles only the PyTorch 5Hz language model. The
ACE-Step DiT decoder and tiled VAE decoder remain in eager PyTorch mode.

This is an application performance policy, separate from the reusable
`torch_compile_toolchain/` package. The portable package can prepare toolchains for
any component; ACE-Step chooses only the component that measured faster in its real
variable-length music workflow.

## Measured result

The July 11, 2026 RTX 5090 Laptop GPU test used XL Turbo, eight inference steps, one
song, DCW double, PyTorch 2.11.0+cu130, and approximately 205-second outputs.

| Component | Eager warm | Compiled warm | Result |
| --- | ---: | ---: | --- |
| 5Hz LM | 32.01s / 950 codes | 18.13s / 840 codes | 56% higher normalized code throughput |
| DiT diffusion | 7.26s / 4750 frames | 10.85s / 4200 frames | 69% slower normalized per frame-step |
| Tiled VAE | 2.44s / 4750 frames | 8.18s / 4200 frames | 3.79x slower normalized per frame |

The cold compile run also added approximately 79 seconds, and TorchDynamo reached
its recompilation limit for dynamic FlashAttention/cache guards. The warm whole-run
gain was only about 8%, driven by the LM while DiT and VAE regressed.

## Selective-policy validation

After applying this policy, a clean-process test used a fixed 60-second request,
fixed seed, 300 LM audio codes, and eight DiT steps. The eager baseline and both
warm selective runs had the same output geometry.

| Measurement | Eager | Selective cold | Selective warm 1 | Selective warm 2 |
| --- | ---: | ---: | ---: | ---: |
| 5Hz LM (300 codes) | 8.35s | 97.06s | 5.44s | 5.31s |
| DiT diffusion | 1.71s | 1.90s | 1.73s | 1.73s |
| Tiled VAE kernel | 0.62s | 0.77s | 0.62s | 0.67s |
| Client request to final result | 15.52s | 132.63s | 14.96s | 14.85s |

The two warm LM runs averaged 5.38 seconds: 35.6% less LM time and 1.55x the
eager code throughput. DiT and VAE used their eager callables and remained within
normal run-to-run variance. The server reported `setup_ready` and
`first_forward_ok` for the LM, with no compile fallback, setup failure,
recompilation-limit warning, traceback, or application error. The cold cost remains
large, so compilation is an explicit option intended for repeated generation.

## Current targets

| Inference component | Policy |
| --- | --- |
| PyTorch 5Hz LM | Compile when the user enables the checkbox |
| ACE-Step DiT decoder | Eager |
| ACE-Step tiled VAE decode | Eager |
| MLX on Apple Silicon | Existing `mx.compile` behavior is unchanged |
| SAM-Audio | Separate setting; unchanged because this benchmark did not measure it |
| Training decoder | Separate setting; unchanged because this benchmark did not measure it |

The source of truth is `acestep/torch_compile_policy.py`. Do not re-enable DiT or VAE
globally without a controlled benchmark using fixed prompt, seed, duration, audio
codes/latent shape, multiple warm runs, and both component and end-to-end timings.
