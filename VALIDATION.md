# Validation record

This repository is a reduced, unmagnetized 1D1V kinetic experiment, not a
validated three-dimensional ionospheric plasma simulation.

## Checks already executed

* Ten CPU unit tests pass (CIC, analytic periodic Poisson, zero mode, plasma
  dispersion function, FIR rejection, Welch Parseval/ACF convention, and
  parameter validation).
* Optional CUDA and plotting targets compile. The actual Seid CUDA batch test
  passes for four independent seeds against the CPU's first 20 timesteps.
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

## Initialization and current production qualification

Independent uniform random positions and Maxwellian velocities are a standard
random loading, but not an exactly screened equilibrium distribution. Removing
the initial mean drift and matching the velocity variance do not establish
equilibrium correlations. Thermal fluctuations are the desired ISR signal.

The original 1 ms burn-in is about five ion plasma periods. A matched-seed
1/5/10 ms comparison is being evaluated; no burn-in sufficiency claim is made
from the elapsed time alone. The expanded production configuration uses a
10 ms burn-in (about 53 ion plasma periods), 128 cells, dt=0.06, and 64 fresh
independent seeds. Production results and stationarity checks will be recorded
below when complete. The eight-seed figures currently in `examples/440mhz`
are pilot results and will be replaced with the qualified larger ensemble.

The individual GPU checkpoints are separately usable HDF5 files. CUDA batches
only share launches and FFT plans; they have disjoint grids and do not exchange
particles, charges, or forces. Float32 atomic ordering prevents bitwise replay.

Use `cargo run --release --bin inspect_results -- INPUT.h5 ...` to reproduce
energy, temperature, charge, and ion-band power checks.
