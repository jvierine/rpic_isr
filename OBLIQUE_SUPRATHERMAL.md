# Oblique suprathermal comparison

The new `rpic_isr_2d3v` executable retains two spatial dimensions and evolves
three velocity components with the Boris mover. The uniform magnetic field
lies in the simulation plane, 3 degrees away from perpendicular to the radar
wave vector. Thus the resonance condition is
`omega - k_parallel v_parallel = n Omega_e`, with nonzero `k_parallel`.
The electron cyclotron and plasma frequencies are 1 and 3 MHz.

The matched pair uses 917,504 macroparticles total, a 256-square mesh, one
Bragg wavelength per side, physical O+/electron mass ratio, core electrons at
2000 K and ions at 1000 K. One run replaces 1% of the electrons with a
zero-drift isotropic Maxwellian population of 20 eV mean kinetic energy;
the control has only the thermal population. Density and macro weights stay
fixed. This is a test distribution, not a fitted photoelectron distribution.

Both runs use the same seed, dt*omega_pe=0.01, 0.5 ms preparation and 2 ms
recording. Preparation is not proof of equilibrium. All eight saved density
modes include preparation; the principal voltage and replay start afterwards.
GPU 0 runs the hot-tail case and GPU 1 the thermal control. Each has a hard
7190-second timeout with a 5-second termination fallback. No thermostat is
enabled in this primary comparison. Optional core-only control is restricted
to preparation, with its work recorded explicitly.

Validation before launch: 31 library tests passed, including arbitrary-field
Boris invariants, the previous perpendicular limit, population temperatures,
thermostat isolation and a prescribed-wave parallel-velocity resonance test.
A 1-microsecond CPU/CUDA comparison gave relative errors below 8e-7 in
voltage, particle position and velocity. Full-size 30-microsecond hot-tail
pilots at dt*omega_pe=0.02 and 0.01 had total energy excursions of 0.000674%
and 0.000694%, respectively. They took 27.8 and 51.0 seconds. These are short
numerical checks, not convergence or equilibration evidence.

The production result remains pending. Compare spectra on the same absolute
normalization, including integrated power below and above 3 MHz, time evolution,
and separate core/tail/ion energy and parallel/perpendicular temperatures.
Do not interpret numerical relaxation as demonstrated cyclotron energy transfer.
Do not reuse the exactly-perpendicular Maxwellian dispersion markers for this
oblique, two-population run. A hot tail does not guarantee enhancement.

## Initial-distribution resonance check

`scripts/check_oblique_resonance.py` evaluates the oblique Maxwellian kinetic
susceptibility separately for core and tail (MIT PSFC/RR-14-01, equations
2.45-2.48), with cold ions as the MHz approximation. Complex roots converge
between 96 and 192 positive/negative gyroharmonics. An independent quadrature
checks the third-harmonic Bessel coupling average.

At the actual radar wave number, k_parallel = 0.965687 rad/m. The thermal
control roots are 2.904799 and 3.365396 MHz. The initial 1% hot-tail mixture
roots are 2.902307 and 3.353551 MHz, selecting parallel velocities -635.6 and
2300.4 km/s, or parallel kinetic energies 1.149 and 15.043 eV. Both are within
the hot distribution. The density-weighted third-harmonic resonant overlap
of the hot electrons exceeds that of the core at both roots.

These hot-tail roots are damped: Im(omega)/(2 pi) = -1116 and -2108 Hz with
the exp(-i omega t) convention. Therefore the initial isotropic hot Maxwellian
does not drive exponential growth of these two modes. Resonance overlap is
verified, but enhanced spontaneous scattering power must still be measured;
it cannot be inferred from overlap alone. This calculation describes the
initial distribution, which may relax during the PIC preparation interval.

## Growing-mode candidate (linear theory, not a PIC result)

The user subsequently requested a growing mode. `scripts/find_unstable_ring.py`
checks a gyrotropic phase-space ring proportional to
exp[-(v_perp-v_ring)^2/(2 sigma_ring^2)] times a parallel Maxwellian.
The radial sampling probability includes the v_perp Jacobian. This is a
different, anisotropic test distribution, not the isotropic hot-tail run.
Its physical motivation is Eliasson et al. (2016),
doi:10.1088/0741-3335/58/9/095002.

With the same density, magnetic field, radar wave number and 3-degree angle,
a 10% ring with m_e v_ring^2/2 = 10 eV, sigma_ring/v_ring = 0.1 and parallel
temperature 2000 K yields omega/(2 pi) = (3.036495 + 0.139378 i) MHz.
Thus the predicted amplitude e-folding time is 1.142 microseconds. Core and
ions retain 2000 and 1000 K. The ring energy parameter is NOT the mean total
particle energy. No growing roots were found for the 1%, 3% or 5% cases in
the limited tested grid; this is not a general stability proof.

The solver recovers the isotropic Maxwellian susceptibility independently,
and checks the growing root with 48/96/192 harmonics and 256/512/768 radial
quadrature nodes. The ion response is the cold-ion MHz approximation.
This candidate has not been implemented or verified in PIC. A subsequent
experiment must introduce the ring after any core preparation, record from
injection, resolve the microsecond exponential stage and saturation, and
measure the loss of perpendicular particle energy alongside field growth.
The isotropic pair was subsequently stopped at the user's change to a delayed
ring experiment. Its live diagnostics and logs remain preserved; it did not
produce a completed comparison and must not be described as completed.

## Delayed ring experiment

`scripts/run_ring_pair.sh` starts identical thermal electron/ion realizations
on two GPUs. At 20 microseconds GPU 0 converts 10% of electron velocities to
the ring above; GPU 1 makes no change. Electron positions and charge remain
unchanged. This is an instantaneous external preparation, not a self-generated
ring or sustained injection. The exact kinetic energy added at that event is
recorded separately in `ring_injection_time_work_0`; the cumulative external
work column includes it so the numerical-energy guard does not mistake the
deliberate injection for heating. No thermostat is enabled.

Both record all 120 microseconds, including 20 microseconds before injection.
That short baseline is not a claim of equilibrium. Diagnostics every 200 steps
resolve the predicted growth interval. The 4096-frame movie spans the full
record; exporters must read `frame_stride_steps` instead of assuming 8 steps.
The control and ring use the same seed, 917504 total particles and dt*omega_pe
=0.01. The ring radial probability includes the cylindrical velocity Jacobian.
Antithetic velocity samples suppress injected bulk momentum without changing
the ring's marginal distribution.

Validation: all 32 library tests passed; a CPU/CUDA delayed-injection regression
had relative voltage/trajectory errors below 7e-6 and work-corrected energy
excursions about 0.0072%. Production growth, saturation and energy transfer
remain to be measured. A spectral peak alone is not sufficient evidence.

Run launch and validation helpers are `scripts/run_oblique_pair.sh` and
`scripts/check_oblique.py`. The latter accepts two files only for a CPU/GPU
regression, not for comparing physically different tail/control outputs.

## Completed delayed-ring pair: preliminary quantitative analysis

Both runs exited successfully: ring 306.284 s and control 312.582 s, in
`/scratch/data/juha/rpic_isr/results/ring-pair-v1`. Production binary SHA256:
`f120fc249c52cfdb15667977bd310f773bd188092caf54b636979a68f3b98a00`.
Finite complete captures contain 4096 frames, 32768 tracks and 226195 raw
voltage samples; raw capture voltage exactly matches the scientific output.
Frame stride is 55 timesteps. Injection occurred at 19.999941 microseconds.

After subtracting the separately measured injection work, maximum numerical
energy excursions are 0.003676% (ring) and 0.005444% (control). Ring injection
adds 5.884730 in units N_e k_B T_core; the resulting deliberate total-energy
increase must not be called numerical heating.

`scripts/analyze_ring_pair.py` saves the comparison to HDF5. A fixed 20.5–23.5
microsecond fit to log power in 0.5-microsecond bins gives amplitude growth
gamma/(2 pi)=125.6 kHz, R-squared 0.9971, versus the initial linear prediction
139.4 kHz. Control has no comparable exponential trend (R-squared 0.014).
This fits total radar density-mode power, not an isolated eigenmode, so it
is preliminary agreement rather than a converged eigenfrequency validation.
Power saturates around 24–26 microseconds. Ring perpendicular temperature
falls from about 119677 K just after injection to 108853 K at 25 microseconds,
while core perpendicular temperature rises from about 2002 K to 3133 K.

For the same post-30-microsecond interval and identical absolute PSD estimator,
integrated power in |f|=2.7–3.4 MHz is 37.1 dB above the thermal control.
The nonlinear spectrum's strongest bin in that band is about 3.22 MHz;
do not label that late peak as the initial 3.0365 MHz eigenfrequency.
The transient is not a stationary ISR spectrum or a model fit to observed
photoelectrons.

Web preparation: compact validated exports now exist locally at
`web/public/data/ring-v1` and `ring-control-v1`, including native scalar JPEG
potential, bounded particle chunks and uncompressed float32 voltage. Each
supports 1000 visible particles; frame interval is 29.1784 ns and the movie
spans 119.5 microseconds. Source UI/catalog changes add the two choices and
turn-on status, disable inappropriate equilibrium overlays, and distinguish
the in-plane 2D3V geometry. Annotation/navigation/loading/potential/waveform/help
tests passed. Both exports were transferred to `/mnt/shovel/picisr/data`,
checked with checksum dry runs, and exposed through public symlinks. Public
signal and first/last particle/potential chunks matched local SHA256 hashes.
The app, index and catalog are deployed at `https://juha.no/picisr/?run=ring`
and `?run=ring-control`. Browser checks verified both new runs, actual particles
and potential, 20-microsecond turn-on status, clock/rulers, full 120-microsecond
voltage, computed spectra, disabled inappropriate theory, drag zoom/reset,
panel close and natural stop at the final frame. Screenshots are saved in
`results/ring-pair-v1/ring-live.png` and `control-live.png`.

Browser QA caught and fixed a missing FFT-menu option for the short records.
The new defaults use 4096 samples (28.76 kHz bins, five overlapping windows);
this short run cannot resolve a detailed ion-line shape. The ring/control
spectrum is a time-average over the entire transient, including the baseline;
the separate scientific comparison above uses a post-30-microsecond interval.
