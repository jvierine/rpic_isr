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
The 512² pilot finished with +0.004476% energy change at matched 0.1 ms;
its 828.45-s runtime includes frequent CPU diagnostics and is not a reliable
production ETA. The full 256² validation (10 ms burn + 4 ms record) finished
in 4917.45 s. All records are finite and transport checks passed, but energy
excursion was 3.762401%, final Te/Ti 2109.60/1010.57 K, and recording temperature
changes +1.389751%/+0.275036%. Early/late plasma-band power ratios were
0.452747/1.080000. It is not accepted as the public replacement: lower drift
than the original 128² case does not establish stationarity/convergence.
Validation HDF5 is retained in results/web_resolution/validation_256.h5.
The user subsequently requested 4× longer recordings and a low-pass voltage
overlay. New production captures use 4096 dense frames and 16 ms record,
with unchanged 10 ms burn-in. The low case completed: maximum energy excursion
0.571131%, recording Te/Ti changes +0.457484%/+0.145322%, all data finite and
4096 frames verified. The extended 512² high case is now running on GPU1;
the public high selection remains the old short diagnostic until validation.
Scripts refuse to overwrite prior HDF5. Raw extended-low capture SHA256:
75f8abd781e07ef0e0ada5be9e1c7f6f51a6affbb487c9951e97282263dc1328;
spectrum SHA256:
b4fd738cd580e44c3711d51a494fd11268d70fc6f2642d0c947aa16b762c814d.

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

The extended native-crop capture binary uses build tag
`extended-web-native-gaussian-v1` and SHA256
5e7cafea4bfd3d267cb56abcd4b484b0fc72b2d034e6f267383220dca0a16bcf.
Its gpu2d.rs SHA256 is
a52363904a1dea81d932fc0fac006046b1e146ac205bcf10b56bf816295bc278.
Running old processes retain their original binary; rebuilding did not
interrupt the 256² validation job.
