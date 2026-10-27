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

Run launch and validation helpers are `scripts/run_oblique_pair.sh` and
`scripts/check_oblique.py`. The latter accepts two files only for a CPU/GPU
regression, not for comparing physically different tail/control outputs.
