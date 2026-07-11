# SAM-Audio Selective torch.compile Policy

SAM-Audio intentionally compiles only `SAMAudio.forward`, the diffusion/ODE
function evaluated repeatedly during separation. Codec, text, span, and ranker
paths remain in eager PyTorch mode.

This application policy is separate from the reusable
`torch_compile_toolchain/` package. The package discovers and prepares a working
compiler; this policy decides which SAM-Audio component is worth compiling.

## Controlled warm benchmark

The July 12, 2026 test used an RTX 5090 Laptop GPU, PyTorch 2.11.0+cu130, a
10-second 48 kHz stereo input, BF16 SAM-Audio Large, text prompting, one candidate,
16 ODE steps, no ranker, fixed seed 99, and identical request settings. Each
component experiment used a clean model process, an eager warm-up, and two compiled
warm repetitions.

| Compile target | Eager warm | Compiled warm runs | Decision |
| --- | ---: | ---: | --- |
| Diffusion/ODE `SAMAudio.forward` | 1.714s | 1.316s, 1.329s | Compile: 22.8% less time, 1.30x speed |
| Audio codec encoder | 1.674s | 1.745s, 1.801s | Eager: 5.9% slower compiled |
| Audio codec decoder | 1.762s | 1.873s, 1.727s | Eager: 2.2% slower compiled |
| T5 tensor encoder | 1.762s | 1.768s, 1.772s | Eager: no warm benefit |
| Codec encoder + decoder | 1.678s | 1.661s, 1.704s | Eager: no repeatable benefit |

The clean diffusion compile run took 27.107 seconds on its first separation versus
1.714 seconds eager. Compilation is therefore optional and intended for repeated
runs, especially longer inputs, chunk batches, or higher ODE-step counts.

## Long-input validation

The implemented policy was then retested through the production service with a
60-second input split into four overlapping chunks (20-second chunks, 5-second
overlap). All other settings remained fixed at 16 ODE steps and one candidate.

| Run | Separation | Result |
| --- | ---: | --- |
| Eager warm | 12.037s | Baseline |
| Selective compile cold | 33.717s | Includes graph compilation for chunk shapes |
| Selective compile warm 1 | 9.169s | 23.8% less time |
| Selective compile warm 2 | 9.407s | 21.9% less time |

The two warm runs averaged 9.288 seconds, again 22.8% below eager and 1.30x as
fast. The metadata reported only `diffusion_forward=true`; all codec, T5, span,
and ranker targets remained false. No fallback, setup failure, graph-break warning,
recompilation-limit warning, traceback, or out-of-memory error was logged.

The 24GB Balanced UI workload was also checked with 32 ODE steps, two candidates,
and Judge reranking. Its eager warm separation took 5.157 seconds; selective warm
runs took 3.910 and 3.989 seconds, averaging 23.4% less time and 1.31x speed. Peak
CUDA allocation/reservation remained unchanged at 11.774/12.793 GiB.

## Current targets

| SAM-Audio component | Policy | Reason |
| --- | --- | --- |
| Diffusion/ODE forward | Compile when enabled | Repeated many times per ODE solve; measured warm gain |
| Audio codec encoder | Eager | One call per pass; measured slower compiled |
| Audio codec decoder | Eager | One call per pass; measured slower compiled |
| T5 text encoder | Eager | One call per prompt; no measured warm gain |
| Span predictor | Eager | Optional request-dependent conditioning path |
| Text/visual rankers | Eager | Optional external reranking paths with dynamic inputs |
| Tokenization, processor, saving | Eager | Python/I/O orchestration, not tensor compile targets |

The source of truth is `acestep/sam_audio_segment/compile_policy.py`. Rebenchmark
before adding targets, using fixed input, prompt, seed, candidate count, ODE steps,
attention backend, and multiple warm repetitions. Compare separation time, not only
the first compiled call.
