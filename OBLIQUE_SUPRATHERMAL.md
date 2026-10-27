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
The currently running isotropic pair remains a separate control experiment.

Run launch and validation helpers are `scripts/run_oblique_pair.sh` and
`scripts/check_oblique.py`. The latter accepts two files only for a CPU/GPU
regression, not for comparing physically different tail/control outputs.
