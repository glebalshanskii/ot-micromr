# ADR-0007: hybrid CPU/CUDA backend for P4

- Status: accepted
- Date: 2026-08-11
- Scope: P4 strategy and threshold evaluation

## Context

The powered P3V runs established the simulator gates on the existing NumPy event-path
generator. Figure 4 adds a policy sweep over many paths and thresholds. The generator is
an adaptive, branch-heavy event loop, while the band state machine across fixed path
observations and thresholds admits a batch representation. One backend need not be optimal
for both workloads.

Neither the paper nor the scientific claim requires `float64` or the exact
`numpy.random.PCG64DXSM` sequence. Precision and RNG choices must preserve the estimand,
valid distributions, independent streams and reproducibility; exact cross-library replay is
an engineering convenience rather than scientific evidence.

## Decision

1. Keep the benchmarked 10-process NumPy CPU path for adaptive event generation until a
   semantically equivalent alternative demonstrates a better end-to-end runtime.
2. Add PyTorch `2.13.0` as optional extra `gpu`, not as a default dependency.
3. Use CUDA with `torch.compile` for vectorisable P4 policy/threshold post-processing after
   the full continuous-crossing kernel passes regression against the CPU reference.
4. Use `float32` as the primary P4 CUDA candidate and keep `float64` as the regression
   oracle/fallback. Freeze `float32` for a claim-eligible run only after the full
   continuous-crossing kernel, rather than merely the endpoint proxy, passes downstream
   materiality/equivalence checks.
5. Benchmark cold compile, steady-state device-resident time and transfer-inclusive
   end-to-end time. A speed claim must name the workload and include correctness evidence.
6. RNG backends may differ. Every backend must record its algorithm, seed mapping and known
   nondeterminism; statistical equivalence replaces impossible NumPy/Torch bitwise identity.

## Evidence

On an RTX 3080 Ti Laptop GPU, an exact endpoint band state machine with 20 paths, 50000
observations and 21 thresholds took `0.69324 s` in vectorised NumPy `float64`. Compiled CUDA
took `0.00563 s` including transfers (`123.2x`) in `float64`; fill and flip counts matched
exactly and maximum rate error was `2.91e-13`. Compiled CUDA `float32` took `0.00445 s`
(`155.6x`) with maximum rate error `6.81e-6`. A larger 84-million-element check measured
`0.01516 s` for compiled `float32` versus `0.01912 s` for compiled `float64`: a direct
`1.26x` speedup and 20.7% runtime reduction. Counts again matched exactly; maximum rate
error was `7.19e-6`, or `1.82e-7` of the maximum reference-rate scale. Raw evidence is in
`outputs/p3v/band-backends.json` and
`outputs/p3v/band-backends-large.json`.

## Continuous-crossing amendment, 2026-08-11

The first full CPU pilot took `186.996 s`, so keeping policy crossing inside every scalar
adaptive step is rejected for P4 target execution. The semantically equivalent decomposition
is now:

1. CPU `float64` generates each adaptive market path once and records left, pre-book and
   post-book endpoints. Independent paths remain process-parallel.
2. A compiled CUDA `float32` kernel broadcasts those endpoints over all thresholds and seeds,
   evaluates exact one-sided bridge probabilities, orders diffusion before book jumps, carries
   the alternating position through `cummax`, and reduces fills/rewards on device.
3. NumPy `PCG64DXSM` remains the market-path RNG. CUDA bridge uniforms use a recorded
   deterministic Philox seed mapping; cross-backend bitwise RNG identity is not a scientific
   requirement.

On a real pilot row (six paths, 13 thresholds, 1,060,041 adaptive endpoints), scalar endpoint
generation took `3.13 s`; compiled continuous-crossing CUDA evaluation took `8.84 s`, including
`5.30 s` cold compilation. A one-path decomposition measured an `8.80x` endpoint-generator
speedup over the old policy-inside-step implementation. The full end-to-end pilot regression
and steady-state target timing remain required before claim execution.

## Consequences

The earlier endpoint proxy alone remains non-evidence. The new CUDA kernel does evaluate
intra-step crossings and is the target candidate, but `SIM-FIG4-002` remains blocked until
its pilot-level functional regression, statistical protocol and executable config are frozen.
Cold compilation is reported separately and included in every end-to-end decision.
