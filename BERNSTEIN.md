# Perpendicular magnetic-field pilot

The explicit `rpic_isr_2d` CPU and CUDA movers accept `--bz-t` (tesla).
The uniform field is along z; both positions and velocities remain in x,y.
All resolved wavevectors are perpendicular to B. The FFT electrostatic
Poisson solve and TSC particle shape are unchanged. B=0 retains the previous
electric kick. The packed/dual-GPU executable rejects nonzero B for now.

The Boris rotation uses the physical species q/m, not macroparticle charge
alone. Weight multiplies charge and mass equally. In normalized units the
electron q/m is -1, the ion q/m is 1/mass_ratio, and the field parameter is
e B/(m_e omega_pe). Diagnostics time-center velocities with the same Boris
update. Magnetic-only rotation preserves speed to roundoff, but the coupled
electrostatic scheme is not exactly energy conserving.

Initial pilot: fpe=3 MHz, fce=1 MHz, B=35.72386758 microtesla,
ne=1.116398347e11 m^-3, Te=2000 K, Ti=1000 K, physical O+.
256 squared cells, 7 particles/cell/species, 917504 total, one 440.2-MHz
radar Bragg wavelength per box side. dt=3.1831 ns. Independent random
positions and Maxwellian velocities; 100 microsecond burn and 200 microsecond
record, no thermostat. This is a pilot, not a convergence or equilibration claim.

The capture includes native cropped potential, 16384 fixed particle tracks
per species, full-rate scattered voltage, and complex electron density modes
1 through 8 along x (TSC-corrected, normalized by sqrt(N)). The latter span
k rho_e=0.511 through 4.09. They permit a dispersion comparison beyond the
single radar mode. The existing unmagnetized theoretical PSD is omitted from
magnetic-run output; it is not a Bernstein model. Existing web datasets remain
unchanged until validation and magnetic-field labeling are ready.

Validation: CPU unit tests cover gyro-phase and sign, magnetic-only energy,
crossed-field drift, reversibility and the existing zero-field force test.
The Boris electron gyrofrequency error at this timestep is approximately
-0.00333 percent. CUDA/CPU regression and pilot diagnostics must be checked
before interpreting any spectral peaks. A gyroharmonic peak alone is not
proof of a collective Bernstein mode; compare with dielectric roots.

Initial verification (2026-10-27): all 26 library tests passed. The 64-square,
16384-particle/species, 1-microsecond CPU/CUDA regression passed: voltage
relative L2 errors 8.39e-7 and 1.55e-6; position 2.84e-7; velocity 2.53e-6.
Both backends had 0.005557% maximum energy excursion. These are short-run
implementation checks, not validation of the Bernstein spectrum.

The larger pilot was launched on Seid GPU0 in
`/scratch/data/juha/rpic_isr/results/bernstein-boris-v1`, with an external
two-hour timeout. Build label `bernstein-boris-v1`; source commit `5408378`.
The pilot completed in 85.67 s. Completed-output checks passed: finite data,
matching raw voltage and Bragg-mode record, 4096 frames, and 200.00047 us
recording. Maximum total-energy excursion was 0.017488%; recording-phase
electron and ion temperature changes were +0.008738% and +0.016944%.
This does not yet establish the identification of the Bernstein branches.

`scripts/run_bernstein_long.sh` extends burn-in to 0.5 ms and recording to
2 ms on GPU0 only, seed 9302, without a thermostat. The 4096-frame movie
still covers the first approximately 104 us after burn-in; the full-rate
voltage and all eight density modes cover the full 2 ms. Full raw HDF5
remains a scientific output, not a browser download. The previous pilot
is preserved. The two-hour outer timeout remains in force.

The longer run completed in 713.07 s on GPU0. It saved 2.0000015 ms of
full-rate voltage and eight density modes plus 4096 replay frames. Finite
data, record duration, raw-voltage consistency and replay checks passed.
Maximum total-energy excursion was 0.136060%; maximum recording-phase
temperature excursions were 0.091288% (electrons) and 0.149676% (ions).
There was no thermostat.

However, late-half voltage variance was only 0.329813 of early-half variance.
This remains a transient demonstration, not a stationary thermal spectrum.
For the radar mode, electron-only perpendicular Maxwellian dielectric roots
are 1.78006, 2.91138 and 3.36536 MHz; nearby finite-record Welch maxima are
1.77367, 2.90498 and 3.36517 MHz. This supports a frequency comparison but
does not establish equilibrium amplitudes or numerical convergence.
`scripts/check_bernstein_spectrum.py` stores the comparison for eight modes
in HDF5 and explicitly uses immobile ions for the MHz theory comparison.

Web transports are installed under juha-no:/mnt/shovel/picisr/data/
bernstein-long, with a symlink from the public data directory. Potential
uses scalar JPEG chunks (12.6 MB total); real particle endpoints use bounded
chunks (32.8 MB total), and the anti-aliased voltage uses 0.63 MB. The browser
loads animation chunks on demand. Full raw HDF5 remains on Seid under
/scratch/data/juha/rpic_isr/results/bernstein-long-v1.

The transient replay is deployed at https://juha.no/picisr/?run=bernstein.
Live browser checks confirmed the potential and real particle tracks, physical
clock and rulers, full 2 ms signal, magnetic spectra, disabled unmagnetized
theory, drag zoom/reset, close control, natural final-frame stop, and switching
back to both existing densities. Help explicitly describes the variance decay
and does not claim stationary thermal equilibrium.
