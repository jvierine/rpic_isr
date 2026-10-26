# Native field display and high-density refinement

2026-10-26: the native-node display correction is deployed. The high-density
recording itself has not yet been replaced. Potential is not electric field;
do not force potential features to one Debye length with a display filter.

The former every-other-node potential export discarded spatial information.
The new export crops a centered region with halo, preserving every native
mesh node and its spacing. Both the field and particle camera remain centered
on the same physical location. Browser transport checks verify crop alignment;
direct HDF5 comparisons verify retained-node values within i16 quantization.

Matched high-density refinement uses the same 0.681037-m periodic square,
440.2-MHz Bragg mode 2, density 1e11 m^-3, initial Te=2000 K/Ti=1000 K,
physical O+/electron mass ratio and dt*omega_pe=.06. Only mesh and particle
number change, keeping 8 particles/cell/species:

| Mesh | dx/lambda_De | Particles/species |
|---|---:|---:|
|128²|0.545180|131072|
|256²|0.272590|524288|
|512²|0.136295|2097152|

Short pilots cover 0.1 ms, not equilibrium. At matched 0.1 ms, the original
128² run changed energy by +0.190996%, versus +0.027173% for the 256² pilot.
These are short-time observations, not extrapolated long-run guarantees.
The 512² pilot is pending. Full captures use 10 ms burn-in plus 4 ms record;
the full 256² capture is running. Scripts refuse to overwrite prior HDF5.

Before replacing the public high-density recording: check finite complete
records, energy and temperature histories, early/late spectral stationarity,
both native-grid transports, and deployed browser playback/navigation. Target
less than 1% total energy excursion and small recording temperature drift;
passing those checks still does not prove equilibrium or spectral convergence.
Keep limitations in Help/documentation, not in the removed on-screen banner.

Remote builds use the checked source copy, not a Git checkout. The launched
256²/pilot binary SHA256 is
b0398080e573a4c48fd3dd5491fc084702708228c26e28496cb9bd889ad34674.
Its gpu2d.rs SHA256 is
5d689e1384bdd34c87ce5c2ccb37109721fcce1a5dde9df1c769fc4aa30f53f8,
matching the committed source. Capture metadata may say `unrecorded` because
the build did not set RPIC_BUILD_REV; do not replace that with an invented ID.
