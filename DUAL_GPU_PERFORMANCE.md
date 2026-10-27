# Dual-GPU performance budget

## Replacement below one million total particles, 2026-10-27

The user requested a longer run, up to 1,000,000 TOTAL macroparticles and
enough actual replay tracks for 1,000 dots inside the camera. A new run is
active in `/scratch/data/juha/rpic_isr/results/submillion-long-FLXOr5`
(PID 90972, outer timeout 90970): 256², 7 ppc/species = 458,752 electrons
and 458,752 O+ ions, 917,504 total. It keeps ne=1e11, Te=2000 K, Ti=1000 K,
physical mass ratio and dt*omega_pe=.06. The periodic box is ONE Bragg period,
not two: the scattering k is retained, but the finite-box modes change.
This is not evidence of box-size or particle-number convergence.

Target record is 16 ms after the same minimum 1 ms unthermostatted burn,
stationarity gate and >=0.5 ms settling. The 1% work-corrected energy guard,
6900 s internal deadline and 7190+5 s outer cap remain in force. Never restart
or extend past the budget without user permission. Pilot wall time predicts
roughly 75–90 minutes, not a completion guarantee. The 0.2 ms pilot completed
in 49.85 s with +0.009585% energy change; it is a short numerical test, not an
equilibrium or spectral validation. The direct two-Bragg, half-million pilots
heated more quickly; reducing dt alone did not solve their spatial noise.

`RPIC_WEB_POOL=16384` saves 32,768 actual globally distributed tracks. A separate
GPU capture smoke test with pool32768/species passed finite data and had
1342–1432 actual tracks in the smaller camera even in the TWO-period box.
The new single-period capture should contain more, but its actual count must
be verified after completion. Pool size is now explicit metadata, not fixed8192.
The browser still loads only three compact animation chunks, not these full
capture arrays. `RPIC_WEB_SIGNAL_STRIDE=16` retains the high plasma band with
an explicit anti-alias passband check, reducing the 16 ms browser signal to
about 2.38 MB. Display waveform reduction now operates on typed arrays and
does not allocate four nested JavaScript point arrays for every source sample.

Production binary SHA256:
`f546b24a109c4cf4105000ede3ba17dd0e01bd9d38778f2bcd7a9afedbb06541`,
build label `half-million-wide-pool-v1`. Presets: `run_submillion_long.sh` and
`benchmark_halfmillion.sh`. Old code, data and the public run remain intact
until the replacement completes and passes checks. Follow-up automation
`finish-longer-sub-million-pic-replay` watches actual progress and verification.

## Measured progress, 2026-10-26

The packed solver is now compiled and GPU-tested on Seid. The old long job
was stopped without deleting its prior data. A new bounded high-resolution
run is in `/scratch/data/juha/rpic_isr/results/high512-2hour-8y3Ktw`
(PID 2691000 at launch). Its short predecessor `high512-2hour-ijBorK` was
stopped early, with diagnostics preserved, to shorten the target from 4 to 3 ms
after initial production throughput measured about 5 ms/step rather than the
3.7 ms/step short benchmark. The resolution and plasma parameters are unchanged.
It completed with exit 0 in 6708.07 s (111.8 minutes), inside the two-hour
budget. The finite capture, raw-signal match and control ledger passed checks.
The recorded signal spans 3 ms; the movie contains 4096 frames covering its
first 110.18 us. Final Te/Ti are 2002.14/1000.38 K. During recording the
temperature changes were -0.02177%/-0.00160%. Raw energy changed +0.01743%
(maximum excursion 0.04054%); maximum work-corrected numerical energy drift
was 0.18149%. These are different quantities: the weak thermostat removed
0.00246368 normalized energy units. It does not make this an energy-conserving
solver. Early/late plasma-band power ratios are 0.7922/0.8297; a 3 ms
thermostatted realization is not a converged equilibrium ISR spectrum.
Checks are reproducible with `scripts/check_bounded_run.py` and
`scripts/check_web_spectrum.py`; outputs remain HDF5.

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
3 ms signal recording. This is an operational two-temperature stationarity
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
