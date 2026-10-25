# Executed 2D2V validation: 2026-10-25

Experimental release v0.2.0 retains the original 1D code and v0.1.0 results.
The 2D model evolves x, y, vx and vy self-consistently using a periodic 2D
FFT Poisson solver, quadratic TSC particle shapes, and physical masses.

## Completed ensemble

Both GTX 1080 Ti cards on Seid completed their jobs, without stopping the
pre-existing GPU workload. GPU 0 processed seeds 4000--4007 in 4207.8 seconds;
GPU 1 processed seeds 5000--5007 in 5052.6 seconds. These are shared-host
elapsed times, not controlled hardware performance benchmarks.

The 16 independent realizations each used a 128x128 square mesh,
131072 particles per species (8 per cell per species), 440.2 MHz monostatic
Bragg wave number 18.45181957 rad/m, electron density 1e10 m^-3, Te=2000 K,
Ti=1000 K and O+ mass 16 u. The ion/electron mass ratio is 29166.2157805.
The square side is 0.681036933 m; dx/lambda_De=0.17240112;
dt*omega_pe=0.06, or 10.635544 ns. Each run has 10 ms burn-in and 8 ms
recording. Independent uniform positions and componentwise Maxwellian
velocities are used; this is not Debye-correlated equilibrium loading.

Each realization contains 752195 full-rate complex density samples, 1969
filtered samples, 848 diagnostic rows and particle trajectories [848,32,2].
High-cadence initial trajectories are [2048,32,2]. Both components move.
All 138 floating-point datasets across the two aggregate files are finite.
Six 50%-overlapping Hann Welch windows are used per seed (96 total); these
windows are **not** counted as 96 independent realizations. The figure's
pointwise 95% intervals use the normal approximation across the 16 seed PSDs.
The theoretical curve includes the Hann window response. No fitted amplitude
scale, forced symmetry, or spectral folding is applied.

## Measurements

Power is integrated over |f|<12 kHz and divided by the theoretical power over
the same bins. Uncertainties below are one standard error across independent
seeds, not errors obtained by treating overlapping windows as independent.

| Quantity | Result |
| --- | --- |
| Full-record power / theory | 0.951148 +/- 0.032437 |
| First-half power / theory | 1.017940 +/- 0.058899 |
| Second-half power / theory | 0.967930 +/- 0.058813 |
| Paired late-minus-early power / theory | -0.050010 +/- 0.067867 |
| Maximum energy excursion over burn-in plus recording | 0.42824% |
| Maximum electron-temperature change | 0.66747% |
| Maximum ion-temperature change | 0.24312% |
| Maximum absolute mean charge / ne | 5.697e-9 |

The full power is 4.89% below theory, approximately 1.51 standard errors.
The early/late difference is not statistically significant at this sample
count. Full-record and half-record Welch estimates use different windows,
so the full estimate need not equal the average of the two half estimates.
There remains measurable secular energy and temperature drift.

Selected final x velocities (only 16 tracked particles per species per seed,
not the entire distribution) give electron std 178563 m/s, kurtosis 3.032,
and ion std 685.6 m/s, kurtosis 3.462. This small subset does not establish
that the full distribution is Maxwellian at late times.

## Backend and artifact checks

All 16 library tests and strict Clippy checks pass. Tests include charge
partition, constant-field interpolation, transverse acceleration, oblique
Poisson modes and preservation of the other electric-field component on
a Nyquist plane. A deterministic short CPU/CUDA comparison on Seid gives
relative L2 errors of 1.104e-6 and 2.156e-6 for the density record components,
1.411e-7 for selected positions and 4.081e-7 for selected velocities.
Both backends have 0.007956% maximum energy excursion in that short test.
This verifies implementation agreement, not long-time CPU/GPU equivalence.

The exact model build revision is
`65b8281c8f65362e4c56819db231254ef2e6d7ba`, recorded in every production
HDF5 file. The recorded crate version remains 0.1.0; v0.2.0 is the release
tag packaging the 2D extension and these results. SHA256 values matched
between the execution host and downloaded aggregate files:

* gpu0.h5: `87d303d81339b1dfda7256840427355daab95110dc23ccbe20b1a1f0cb5be5e9`
* gpu1.h5: `2cb5fb8fa072e748f8237e87d5cebd7dde331957634b7db822130a71e91c9101`

The compact ensemble HDF5 contains the individual PSDs, early/late PSDs,
mean, standard errors, theoretical curves and input metadata. A full seed
4000 example is in the repository; the release contains both complete GPU
aggregate files and the CPU/CUDA regression records. All four final PNG
figures were visually inspected at their delivered sizes. Figures use native
Rustplotlib and TeX typography.

## Limits

This is an unmagnetized electrostatic 2D2V model, not a 3D electromagnetic
simulation or calibrated received radar voltage. The near-theoretical ion
line and nonsignificant early/late difference do not prove equilibration.
No matched longer-burn 2D ensemble or production-spectrum grid, particle-count
and timestep convergence study has been completed. Float32 position/velocity
precision effects have not been separately quantified. The observed energy
drift and finite-ensemble uncertainty preclude a percent-level calibration
claim. Do not use this release as a validated precision plasma inversion.

Recheck the downloaded complete files with:

```sh
conda run -n base python scripts/validate_2d.py gpu0.h5 gpu1.h5
cargo run --release --bin check_ensemble -- gpu0.h5 gpu1.h5
cargo run --release --bin check_2d_backend -- cpu_final.h5 gpu_final.h5
```
