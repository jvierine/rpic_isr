# Executed validation record

This repository is a reduced, unmagnetized 1D1V kinetic experiment, not a
validated three-dimensional ionospheric plasma simulation.
The 64-realization run, additional 20 ms burn-in comparison, and refined CPU
backup are complete (25 October 2026). Reduced dimensionality was explicitly
retained at the author's request. Precision convergence is **not** established.

## Checks already executed

* Eleven CPU unit tests pass (CIC, analytic periodic Poisson, zero mode, plasma
  dispersion function, FIR rejection, Welch Parseval/ACF convention, and
  parameter validation, and Hann-response normalization).
* Optional CUDA and plotting targets compile. The actual Seid CUDA batch test
  passes for four independent seeds against the CPU's first 20 timesteps,
  at both (64 cells, dt=0.12) and (128 cells, dt=0.06), with a 5e-5 tolerance
  in the dimensionless density mode. Clippy passes with warnings as errors.
* The CPU reference was actually run: four seeds, each 1 ms burn-in plus an
  8 ms record, 32768 particles per species, 64 cells, dt*omega_pe=0.12.
  Maximum energy drift was 0.529%; its ensemble ion-band power was 1.0668 times
  the unsmoothed analytical prediction. Finite ensemble scatter is substantial.
* Two GTX 1080 Ti GPUs completed four seeds each at those parameters. Their
  combined ion-band power was 1.0169 times theory. This is an encouraging pilot,
  **not** proof that initialization and discretization errors are negligible.
* A matched short timestep/mesh check (seed 200, 0.2 ms burn-in + 2 ms record)
  gave maximum energy drift 0.126% (64 cells, dt=0.12), 0.0814% (64, 0.06),
  and 0.0184% (128, 0.06). The fine grid keeps the same total particle count
  (256 particles/cell versus 512). This checks heating, not spectral convergence:
  two Welch windows per short run are insufficient for that claim.

## Initialization and burn-in qualification

Independent uniform random positions and Maxwellian velocities are a standard
random loading, but not an exactly screened equilibrium distribution. Removing
the initial mean drift and matching the velocity variance do not establish
equilibrium correlations. Thermal fluctuations are the desired ISR signal.

The original 1 ms burn-in is about five ion plasma periods. A matched four-seed
study at the coarser 64-cell, dt=0.12 configuration gave power/theory ratios
1.0250 (1 ms), 1.0712 +/- 0.0889 (5 ms), and 0.9824 +/- 0.0726 (10 ms).
Uncertainties here and below are one standard error across independent seeds.

The refined production ensemble uses 10 ms burn-in (about 53 ion plasma
periods), 128 cells, dt=0.06, and 64 fresh independent seeds. A further matched
eight-seed test at this same mesh/timestep compares 10 and 20 ms burn-ins,
each followed by an equal 8 ms record:

| Burn-in | Mean ion-band power/theory | Standard error |
|---|---:|---:|
| 10 ms | 1.037363 | 0.059421 |
| 20 ms | 1.147515 | 0.032247 |

The paired change is **+0.110152 +/- 0.070080**. An eight-pair Student-t 95%
interval includes zero but is broad. This comparison is **inconclusive**, not
a demonstration of negligible initialization effects. The 20 ms subset also
has a larger excess relative to theory. The data do not justify a precision
convergence claim. More matched seeds, particle-count, box-size, and position-
precision studies are needed to distinguish sampling variability, aging, and
numerical effects reliably. Energy conservation alone is insufficient.

The individual GPU checkpoints are separately usable HDF5 files. CUDA batches
only share launches and FFT plans; they have disjoint grids and do not exchange
particles, charges, or forces. Float32 atomic ordering prevents bitwise replay.

Use `cargo run --release --bin inspect_results -- INPUT.h5 ...` to reproduce
energy, temperature, charge, and ion-band power checks.

## Completed primary ensemble

Both Seid GTX 1080 Ti GPUs ran 32 fresh independent seeds: 1000–1031 and
2000–2031. Conditions: 440.2 MHz, n_e=1e10 m^-3, initial Te/Ti=2000/1000 K,
O+ ions (16 u), physical mass ratio 29166.2158, periodic length 0.681037 m,
128 cells, 256 particles/cell/species (32768 per species), dx/lambda_De=0.172401,
omega_pe*dt=0.06 (10.6355 ns). Each record has 10 ms burn-in plus 8 ms sampled
data. Anti-aliased sampling interval: 3.99896 us. Welch: 512-point periodic
Hann, 50% overlap, 488.4 Hz bin spacing.

There are **64 independent realizations**, not 384 independent observations:
the latter is the count of overlapping Welch windows. Neither the old eight-
seed pilot nor the repeated-seed burn-in check is mixed into the main ensemble.
The source revision `9a1d814` is recorded in its HDF5 files. Subsequent commits
add analysis, documentation, progress reporting and output handling, without
changing its particle equations or CUDA deposition kernel.

| Quantity | Result |
|---|---:|
| Integrated power/theory, absolute frequency below 12 kHz | 1.042676 +/- 0.017837 |
| First half of measurement / theory | 1.051502 +/- 0.025041 |
| Second half / theory | 1.022698 +/- 0.025163 |
| Paired late-minus-early / theory | -0.028803 +/- 0.031788 |
| Worst energy excursion, including burn-in | 0.40914% |
| Worst electron / ion temperature excursion | 0.89823% / 0.30128% |
| Worst mean normalized charge imbalance | 2.605e-9 |

The early/late check does not resolve a trend at its present precision.
The approximately **4.3% power excess is about 2.4 estimated standard errors**;
it is a possible remaining finite-particle/numerical bias, not silently declared
agreement within uncertainty. Relative integrated absolute spectral difference
from the window-matched analytical curve is 0.05345. No amplitude fitting,
frequency adjustment, enforced spectral symmetry, or frequency folding was used.

The final selected velocities include 1024 particles/species across the runs.
Electron/ion standard deviations: 176023 / 736.1 m/s; kurtoses: 3.150 / 3.182
(Gaussian reference: 3). These are selected samples, not all-particle velocity
distributions. The plotted normal-approximation 95% bands are **pointwise
sampling intervals**, not simultaneous bands or bounds on model error.

The toy model and ion-line spectrum are delivered as illustrative results;
do not use them yet for percent-level calibration or plasma inversion.

## Refined CPU backup

A full CPU realization at the same refined 10 + 8 ms configuration was executed
with seed 9000, source revision `10d4153`: energy drift 0.32752%, power/theory
1.026861. A single record is noisy and is not a precision spectrum benchmark.
Its source version left the final CPU mean-charge diagnostic at the default
zero; other rows are measured. This diagnostic-only issue is corrected in the
current source and does not change particle trajectories or the spectrum.

## Reproduce and inspect

Download the [v0.1.0 HDF5 data release](https://github.com/jvierine/rpic_isr/releases/tag/v0.1.0).

```sh
cargo build --release --features plot --bins
target/release/check_ensemble gpu0.h5 gpu1.h5
target/release/compare_burn gpu0.h5 burn20.h5
target/release/inspect_results cpu_refined.h5
target/release/plot_results figures gpu0.h5 gpu1.h5
```

`examples/440mhz/ensemble.h5` contains compact per-seed spectra, early/late
spectra, uncertainties, physical configurations and build metadata.
`example_seed1000.h5` is a complete raw/decimated/trajectory record. The release
retains all 64 production records, the CPU record, additional long-burn-in data,
and earlier validation runs. Data products are HDF5, not CSV.
