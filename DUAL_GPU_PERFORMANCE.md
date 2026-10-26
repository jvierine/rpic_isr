# Dual-GPU performance budget

## Measured progress, 2026-10-26

The packed solver is now compiled and GPU-tested on Seid. The old long job
was stopped without deleting its prior data. A new bounded high-resolution
run was launched in `/scratch/data/juha/rpic_isr/results/high512-2hour-ijBorK`.
It is running, not a completed spectrum or a deployed viewer replacement.

Matched 512²/8ppc/physical-O+ tests use seed 9100, 2,973 steps (about 10 us),
no thermostat, and diagnostic stride 1000. These are short performance tests,
not stationarity or spectral-convergence tests. The unrelated tomography job
remained running throughout; timings are not exclusive-device performance.

| Variant | Observed steady ms/step |
|---|---:|
| Packed direct, one GPU | 3.71 |
| Split-species push | 3.82 |
| Split push with precomputed coordinate scale | 3.84 |
| Shared-memory tiles, rebuild each step | 6.24 |
| Shared-memory tiles, rebuild every 4 steps | 5.33 |
| Shared-memory tiles, rebuild every 16 steps | 5.07 |
| Packed direct, two GPUs | 5.02 |

The legacy matched run took 63.45 s versus 11.42 s for the initial packed
single-GPU version: about 5.6x total-run speedup. No claim of optimality or
repeated-trial confidence intervals is made. A profiler identified particle
push and deposition as the largest costs; the FFT/Poisson/pack stage was about
0.15 ms. Synchronized profiler samples at multiples of 16 overrepresent
rebinning steps for tiled variants; use whole-run slopes to compare throughput.

All seven 128² short variants passed the finite-data/energy checks and
the 1e-4 relative-L2 voltage/fast-trajectory regression against the legacy
297-step reference (voltage errors were around 2e-6 to 4e-6). At 2,973 steps,
float32 trajectories differ: relative spectral-input differences are about
1-2%; raw coordinate L2 differences also include periodic-wrap discontinuities.
This is not proof of long-time statistical agreement. HDF5 comparisons are
retained locally under `results/bench_20261026/` and on the remote data disk.

### Two-hour preset

Build `rpic_dual` with CUDA, then run `bash scripts/run_high_2hour.sh`.
The preset uses the winning direct packed single-GPU variant, no tiled
deposition and no profiler instrumentation. It keeps the full periodic field
solve: there is no Debye-distance force cutoff.

The minimum unthermostatted burn is 1 ms. A 0.25 ms two-half temperature
stationarity check, anisotropy and radial-kurtosis checks gate thermostat
activation. Separate mean-preserving weak velocity controls target 2000 K
electrons and 1000 K ions, with 2 ms relaxation time. After at least 0.5 ms
settling, stable temperatures within 1% of their respective targets gate the
4 ms signal recording. This is an operational two-temperature stationarity
criterion, not proof of kinetic equilibrium. The radial-kurtosis diagnostic
assumes negligible mean drift. Thermostat strengths need independent spectral
bias tests before quantitative ISR interpretation.

Energy before/after each control action and cumulative removed work are
recorded separately. The 1% guard applies to `E + cumulative_removed_work`,
so cooling cannot conceal numerical energy generation. Live HDF5 checkpoints
are atomically replaced. A 6900 s graceful deadline preserves diagnostics;
an outer timeout and kill fallback finish within 7200 s. Failure to satisfy
the gate or energy guard is reported as failure, not a completed spectrum.

The GPU control smoke test exercised Burn -> Settle -> Record, kept distinct
species targets, recorded thermostat work and completed output saving. It was
a deliberately short control-path test, not a physically equilibrated run.
The production binary has build label `packed-budget-2hour-v5` and SHA256
`d15ed72e6e89f0ea5f0ea58bf8665325a6aae852159b8bcf6eb3653e5fc50616`.

## Earlier analytical budget

The following is an analytical budget, not a measured speedup or completion forecast.
The experimental `rpic_dual` backend shards one physical particle population
across GPUs, reduces the density on GPU 0, solves Poisson there, and broadcasts
the packed real electric field. See the measured comparisons above.

## Reference problem

- 512 by 512 cells, 8 particles/cell/species: 2,097,152 electrons and as many ions.
- Physical oxygen/electron mass ratio; Te = 2000 K, Ti = 1000 K.
- ne = 1e11 m^-3, dt*omega_pe = 0.06: dt about 3.3633 ns.
- 10 ms burn plus 16 ms record: about 7.73 million timesteps.
- Thermostat settling adds time beyond this reference budget.

## Optimistic bounds for the current communication design

One packed density copy and one packed field copy each carry
512*512*2*4 = 2,097,152 bytes per step. They are causally sequential.
At the nominal PCIe 3.0 x8 encoding-limited bandwidth of about 7.88 GB/s,
the 4,194,304-byte exchange alone takes about 0.532 ms per step.
Protocol overhead, host staging if required, synchronization, and shared-root
contention can make this appreciably slower. This assumes a full-speed link
under load, not the idle downclocked state reported by nvidia-smi.

An optimistic particle-array traffic budget is 40 bytes/particle/step:
8 bytes position read for deposition plus 32 bytes position/velocity read/write
for pushing. At 4,194,304 particles and the two cards' combined advertised
968 GB/s memory bandwidth this is another 0.173 ms. This ignores deposition
atomics, field gathers, mesh clearing, FFTs, launch latency and cache behavior.
It is an optimistic bound for this explicit kernel/dataflow design, not a
universal lower bound for all possible PIC algorithms.

Thus roughly 0.7 ms/step, or 1.5 hours for the reference run, is an optimistic
floor, not a credible promised runtime. A provisional engineering target is
2-5 ms/step (4.3-10.7 hours), pending actual hardware measurements. The last
legacy production progress interval was about 26 ms/step, roughly 56 hours
for the same 26 ms physical span; contention and diagnostics affect this.

## Optimization order

1. Packed real density and electric fields; no full-particle CPU diagnostic copies.
2. GPU moment reductions and a separately accounted thermostat-work ledger.
3. Measure one versus two GPUs with identical particles, settings and diagnostics.
4. Measure peer-copy bandwidth, staging behavior, kernels and FFTs separately.
5. If atomics dominate, evaluate cell binning/shared-memory deposition while
   retaining stable particle IDs and checking conservation and spectra.
6. If PCIe dominates, evaluate replicated field solves or domain decomposition;
   these are different communication designs, not free speedups.

Never trade physical mass ratio, timestep stability, mesh resolution, or the
requested T_e/T_i for a misleading speedup. Temperature control is not an
energy-conserving integrator and may affect fluctuation spectra.

Hardware bandwidth source: PNY GTX 1080 Ti manufacturer datasheet,
https://www.pny.com/File%20Library/Company/Support/Product%20Brochures/GeForce%20Graphics/English/PNY-NVIDIA-GeForce-GTX1080Ti-Blower.pdf
