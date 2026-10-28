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
