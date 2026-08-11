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
4. Use `float64` for the first P4 CUDA implementation. `float32` or mixed precision may
   replace it only after a downstream materiality/equivalence check, not from dtype custom.
5. Benchmark cold compile, steady-state device-resident time and transfer-inclusive
   end-to-end time. A speed claim must name the workload and include correctness evidence.
6. RNG backends may differ. Every backend must record its algorithm, seed mapping and known
   nondeterminism; statistical equivalence replaces impossible NumPy/Torch bitwise identity.

## Evidence

On an RTX 3080 Ti Laptop GPU, an exact endpoint band state machine with 20 paths, 50000
observations and 21 thresholds took `0.69324 s` in vectorised NumPy `float64`. Compiled CUDA
took `0.00563 s` including transfers (`123.2x`) in `float64`; fill and flip counts matched
exactly and maximum rate error was `2.91e-13`. Compiled CUDA `float32` took `0.00445 s`
(`155.6x`) but had maximum rate error `6.81e-6`. Raw evidence is
`outputs/p3v/band-backends.json`.

## Consequences

This benchmark selects the architecture but is not Figure 4 evidence: the endpoint proxy
does not detect intra-step Brownian crossings. `SIM-FIG4-002` remains blocked until the
continuous-crossing CPU reference, CUDA regression, statistical protocol and executable
config are frozen. GPU compilation cost is amortised only for repeated or sufficiently
large evaluations; small workloads may continue to use CPU automatically if a recorded
break-even benchmark supports that choice.
