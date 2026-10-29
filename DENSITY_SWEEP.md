# Density scan across the third gyroharmonic

The user requested the measurement-like progression with plasma density varying
and gyrofrequency and ring energy fixed. This campaign uses independent
homogeneous boxes, not a density ramp within one periodic box.

- Plasma frequency: 2.40 to 3.40 MHz, 50 kHz spacing (21 points).
- Density: `111639834702.32187 * (fpe / 3e6)^2` in cubic metres inverse.
- Bx = 1.869630096 microtesla, By = 35.67491076 microtesla, Bz = 0;
  electron gyrofrequency approximately 1 MHz, three degrees from k perpendicular B.
- Ring parameter 10 eV, radial width 0.1 vring, fraction 1%, one-time turn-on
  at 20 microseconds. Initial Te = 2000 K, Ti = 1000 K; no thermostat.
- Physical O+ mass, 2D3V, 256 square mesh, 7 particles/cell/species, one Bragg
  wavelength, 440.2 MHz radar. Same physical box, charge neutrality and seed 9501.
- 120 microsecond record, dt omega_pe = 0.01. Physical timestep changes with
  density; physical spectral axes must be used when comparing rows.
- GPU 0 ring and GPU 1 thermal baseline at each new density. The existing 3 MHz
  1% ring and original thermal control are reused without rerunning them.

The density range includes the upper-hybrid branch crossing; plasma-line
frequency is not identical to fpe. The final plot should use frequency horizontally
and fpe or density vertically, with gyroharmonic markers. No altitude assignment
is justified without an observed density profile. Compare absolute post-injection
PSD and ring/control enhancement on identical physical intervals. Do not normalize
each row to its own peak or infer growth merely from a brighter spectral bin.

Driver: `scripts/run_density_sweep.py`. Remote campaign:
`/scratch/data/juha/rpic_isr/results/density-sweep-1pct-v1` on Seid.
The manifest records complete rows, source paths and work-corrected energy errors
in HDF5. No large movie captures are made for this spectral scan. Existing replay
data remain untouched. First pair launched at 2.8 MHz; further rows expand outward.
Campaign has a 6900 second internal budget and 7190 second external timeout plus
5 second kill fallback. No automatic restart/extension. Outputs remain pending
verification; a single realization per density is not a convergence result.

Binary SHA256: f120fc249c52cfdb15667977bd310f773bd188092caf54b636979a68f3b98a00.

## Completed 1 ms, 1% ring sweep (29 October 2026)

All 21 independent density points from 2.50 to 3.50 MHz in 50 kHz steps
completed within the overnight wall budget. The user reduced the requested
8 ms duration to 1 ms; the superseded 8 ms and 0.1% runs were deliberately
stopped and their diagnostics retained. No new thermal controls were run.

Remote raw data: `/scratch/data/juha/rpic_isr/results/ring-1pct-1ms-sweep-v1`.
`scripts/analyze_long_sweep.py` independently checked full finite raw records,
1 ms duration, 1% fraction, fixed field, density, particle count, temperatures,
seed, and one-time injection at 20 microseconds. The maximum injection-corrected
numerical energy excursion is 0.118004%. Final raw total energy increases by
25.999–26.064%; the injected ring supplies 25.944–25.964% of initial energy.
These are different quantities: the injected energy is not numerical heating.
Final population-averaged electron temperatures are 2780.9–2783.4 K, including
the energetic ring, and ion temperatures are 1000.87–1002.38 K.

The derived HDF5 and native Rustplotlib figure separate 30–1000, 30–250,
250–600 and 600–1000 microsecond PSD estimates on identical absolute scales.
All use 20 microsecond Hann segments with 50% overlap (50 kHz bin spacing).
The late/early integrated power ratio above |f|=2 MHz within the plotted
frequency range varies from 0.523 to 10.943 across densities. This is an
evolving, single-seed experiment, not evidence of stationary or converged ISR.

The separate enhancement figure uses ONLY matched 30–120 microsecond windows
against the existing thermal controls. Controls exist through 3.40 MHz, so
this comparison contains 19 density points, not all 21. There is no matched
late-time or full-1-ms thermal baseline. No extrapolated control is inserted.

Local derived products: `results/ring-1pct-1ms-sweep-v1/analysis.h5`,
`analysis-matched.h5`, `spectra.png/pdf`, and `early-enhancement.png/pdf`.
