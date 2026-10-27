# PIC ISR web replay

## Video preview export

With local replay assets and built Wasm present, serve `web/public` locally and
open `video.html` in a WebCodecs-capable Chromium browser. It renders the existing
low/high recordings and Rustplotlib plots into 1800 deterministic 1920x1080
frames at 30 fps, with 30 seconds per density. It does not record wall-clock UI
playback and does not upload to YouTube. Package the downloaded Annex-B stream
with `bash scripts/package_video_preview.sh INPUT.h264 OUTPUT.mp4`.
The description in `web/youtube-description.txt` distinguishes the current
numerically heated high-density diagnostic from the newer completed simulation.
Inspect the exact MP4 and get user preview approval before publication.

Public route: https://juha.no/picisr/

## Lightweight replay transport

`scripts/compress_web.py` follows `export_web.py`. It leaves the original
scientific HDF5 and original export unchanged and writes `metadata-lossy.json`.
Scalar potential is quantized to 8 bits with one symmetric scale across time,
then JPEG quality 95 in independently fetched 8x8-frame atlases. Native mesh
nodes are retained; each frame starts at a DCT-block-aligned boundary. The
viewer applies the colormap/contrast after decoding, not before compression.
No compression is used for the voltage or spectral input.

The high-resolution 512² run's potential shrinks from 134.22 MB to 13.45 MB;
RMSE is 0.001002 kBTe/e (about 0.95% of its 97th-percentile display scale),
maximum error 0.004666. The low run uses 4.45 MB of potential JPEGs. Only three
decoded chunks are cached (12.58 MB high, 3.15 MB low). Arbitrary seeking
requests the relevant chunk, not the preceding movie. In-flight requests are
also capped at three.

At each frame the export selects up to 500 visible actual particles per species,
with their true next-frame endpoints, rather than joining different IDs as
the subset changes. Particle endpoints are streamed in the same 64-frame chunks
as the field, rather than fetched up front. The low recording supports a 1000
particle cap; the high recording contains 141–211 saved tracks inside the view
(a padded 240 cap exposes them all). Default remains 100. The slider label
always shows the selected limit, never a frame-dependent actual/limit fraction;
tabular fixed-width count elements prevent layout shifts during playback.
The full high-run voltage is 0.892 MB and remains available for zoom/FFT.
Original HDF5 downloads are separate from browser loading, on the data share.
`node web/test_potential.mjs` checks the same-ID interpolation and bounded cache.

The accepted longer low-density record has 4096 dense frames (348.42 us)
and 16 ms voltage, with unchanged 10 ms burn-in. Native potential nodes are
cropped before writing HDF5; crop geometry remains in metadata. It has
0.57113% maximum energy excursion, final Te=2015.85 K/Ti=1002.82 K, and
recording temperature changes +0.45748%/+0.14532%. Early/late electron-band
power ratios are 0.98701/1.02138; these checks do not establish equilibrium
or spectral convergence. Wasm checks passed at 4096 frames, with 44 overlapping
8192-point Hann windows and PSD integral 0.462093 vs raw power 0.463624.
The original short data remains intact. The prior extended 512² job was stopped
and superseded by the bounded packed run: 512², 8 ppc/species, 2,097,152 particles
per species, two Bragg wavelengths, Te=2000 K and Ti=1000 K. It finished in
111.8 minutes. Unthermostatted burn and stationarity gating preceded weak
separate-temperature control and settling; actual recording begins at 1.51346
ms. Its 3 ms signal and 4096 movie frames replace the old high diagnostic in
the run catalog. Maximum work-corrected drift is 0.18149%, versus 0.04054%
raw energy excursion; thermostat action and short-record spectral fluctuations
still preclude an equilibrium/convergence claim. The completed 256² validation
is preserved but not accepted (3.7624% energy excursion).

The frontend adds a Rust/Wasm, zero-phase Gaussian FIR overlay on
the full complex voltage: thick blue/orange are filtered Re/Im, thin pale
lines are unfiltered Re/Im. The cutoff is 30 kHz at -3 dB power. Linear FFT
convolution avoids circular wrap, and unsupported edge intervals are omitted.
The original signal and its Welch spectrum are not altered. A checkbox hides
the overlay. Tests check gain/phase at 10 kHz, rejection at 1 MHz, omitted
edges, and preservation of the input array. Browser QA passed locally with
the original recording and numerical checks passed on the longer low record.

Spatial-resolution correction: the potential transport now crops native mesh
nodes around the camera, without skipping nodes or resampling the data. The
crop includes an interpolation halo and its own physical extent; particle
coordinates still refer to the full simulation box. This avoids the former
every-other-node spatial aliasing while retaining small downloads. The solver
grid spacing, crop origin and extent are recorded in metadata and checked by
the Wasm transport tests. Old raw HDF5 recordings remain unchanged.

The high-density heating problem was not fixed by the display correction.
The completed bounded 512² run above supersedes the old diagnostic recording;
its thermostat and work-corrected energy ledger are documented separately.

## Historical high-density diagnostic (preserved, no longer selected)

The earlier run selector used a 10^11 m^-3 diagnostic recording (seed 8300):
26.5735% maximum energy excursion, final Te 2802.1 K and Ti 1039.7 K; Te rises
7.92% during its 4 ms record. Its narrow peaks near -3.0352/+3.0318 MHz must
not be interpreted as an equilibrium/converged thermal plasma-line result.
It is no longer selected by the public catalog. Its files remain preserved.

The interface uses a bone/light theme, retaining the seismic potential map.
Every signal/spectrum plot connects Rustplotlib's Rust/Wasm viewport
and DOM navigation bridge: zoom buttons, pointer-anchored wheel zoom,
plain-drag rectangle, Shift-drag pan, and reset/double-click/right-click. Views persist
across animation frames and reset when switching recordings.
The minimum normalized zoom span is 1e-6, not 1e-3, so the MHz plot can inspect
narrow features. The redundant frequency-band menu is removed; the ion panel
starts at ±30 kHz, and the plasma panel already provides the full spectrum.
The panel has an explicit top-right close button. `node web/test_navigation.mjs`
checks gesture dispatch; actual drag zoom and closing were also checked live.

`scripts/compare_ion_records.py` saves matched 3 ms voltage PSDs in HDF5 at
8192/16384/32768 FFT sizes, with 26/12/5 overlapping windows respectively.
The old full 4 ms record had 8 windows at 32768, versus 5 for the new 3 ms
record. Neither overlapping-window count is an independent ensemble size.
The different stochastic realization, grid and thermostat prevent attributing
all spectral differences to duration. Display-particle selection and JPEG
potential compression do not alter the voltage or these spectra.
The MathJax scale bar spans exactly 1/10 of the 10-Debye-length horizontal
view. The lower-left clock is elapsed time after burn-in, in ns and electron
plasma periods: t*fp = t*omega_pe/(2*pi). It follows the selected recording.
Run `node web/test_annotations.mjs` to check the clock conversion.

High-density export and numerical checks:
```sh
conda run -n base python scripts/export_web.py \
  results/web_high_capture.h5 results/web_high_spectrum.h5 web/public/data/high
node web/verify.mjs ./public/data/high/
conda run -n base python scripts/check_web_spectrum.py \
  results/web_high_capture.h5 results/web_high_spectrum.h5 \
  web/public/data/high/metadata.json results/web_high_validation.h5
```
The high recording has 1024 frames at 26.906 ns cadence (27.525 us dense movie),
a 69.783-Debye-length full box, and a 9.759 cm wide camera crop. The fixed-ID
pool can contain fewer dots inside this crop than the requested slider cap;
the counter reports actual/cap rather than inventing additional particles.
Default high-density FFT length 32768 gives 1.134 kHz bins. Early/late
plasma-band power ratios are 0.669/0.488: stationarity is not established.
Full HDF5 source hashes are in the exported metadata; browser binaries are
transport products, not replacements for the authoritative HDF5.

## Original short-record configuration (retained)

Full-viewport WebGL2 view, 100 dots by default (50 electrons, 50 O+ ions),
10×10 Debye-length crop, a display-only 10–1,000 particle cap, time controls,
potential contrast/toggle (0.3× by default), parameter definitions and equations/help, and collapsible voltage/PSD panels.
Potential is blue–light-yellow–red seismic, with zero at the light-yellow centre.
Screen aspect can differ from the square physical crop; resizing never changes
the underlying coordinates or dynamics.

## Responsibilities

- Rust GPU PIC: physical 128×128, 131,072 particles per species, TSC/leapfrog,
  10 ms burn followed by 4 ms recording, seed 8200. This is the existing explicit
  solver, not the separate implicit experiment. Energy excursion: 0.33965%.
- Rust/Wasm: shortest-periodic-path interpolation, crop/particle selection,
  potential interpolation, and the two-sided Hann Welch FFT calculation.
- Rustplotlib Rust/Wasm/WebGL2: potential texture, particle markers, plot axes
  and GPU curves. JavaScript connects controls and transports the recorded data;
  it does not implement particle dynamics or the plot renderer.

The dense movie spans 87.042 µs after burn-in (1,024 frames, 85.084 ns cadence).
It records fixed IDs `p=j*32`, j=0…4095, per species. The camera sees a subset
of those IDs, not a freshly resampled particle cloud. Both species share one
clock. Positions are u16-quantized over the full box and the native-node cropped
display field is i16-quantized using one fixed scale. Authoritative output stays float32 HDF5.
Field and particle interpolation is display-only, not extra PIC steps.

The mesh Fourier electron-density mode, corrected for the TSC window, supplies
the signal, independent of the camera or particle cap. A 257-tap Blackman sinc
FIR precedes 8-fold signal decimation; both filter edges are discarded. The
browser processes the resulting 4 ms finite record, not new live GPU data.
8,192-point windows give 1.435 kHz bins and 10 overlapping periodograms. The
longer voltage record and shorter dense movie are distinguished in the help.
Neither burn-in nor this single realization establishes spectral convergence.

## Build and verify

Requires Rust with wasm32-unknown-unknown, wasm-bindgen-cli 0.2.129, Node, and
base-conda Python (NumPy, SciPy and h5py for export).

```sh
bash web/build.sh
conda run -n base python scripts/export_web.py \
  results/web_capture.h5 results/web_spectrum.h5 web/public/data
node web/verify.mjs
conda run -n base python -m http.server 8768 --directory web/public
```

The build runs three Rust numerical tests; verify.mjs exercises the exact built
Wasm against the exported recording at multiple frames and counts, verifies
interpolation, and checks integrated spectral power. This record gave PSD
integral 0.50891 versus mean squared signal 0.51237; Hann window weighting
need not produce the identical finite-record sample mean.

For capture, build the normal CUDA executable from revision
721100d49cab22b31b5a049343883f19a5818889, then run on Seid with the existing
CUDA library path:

```sh
RPIC_WEB_CAPTURE=results/web_capture.h5 target/release/rpic_isr_2d \
  --backend cuda --device 0 --cells 128 --ppc 8 --bragg-mode 2 --dt .06 \
  --burn-ms 10 --duration-ms 4 --sample-us 1 --window 1024 --seed 8200 \
  --diagnostic-stride 10000 --output results/web_spectrum.h5
```

Capture HDF5 SHA256: a564591753ac9f2e5577d7befc89c58e234c670248c0b1a9c9c9930d72b4ea23.
Spectrum HDF5 SHA256: c09ca2f7b62d1c9fabb77d60bd5af9fd781c957ded10cefb3137ea6168a981a5.
Source hashes, quantization and timing are retained in data/metadata.json.

## Renderer vendoring

`vendor/rustplotlib-web` is an MIT-licensed source snapshot from the author's
local Rustplotlib browser library (parent revision
9aeb0a67dc67ff88d7e0ec6e78692b65b959cc5a, including its then-uncommitted
LinePlot and SeriesView work). The original checkout is untouched. Local
adaptations: round point sprites in LinePlot and bilinear regular-grid field
display. Numerical simulation values are not spatially filtered by those
display adaptations. The vendored license is included.

## Loading and plasma-line panel

The loading bar aggregates the actual streamed byte counts for the Wasm module,
preview, dense movie and voltage files. The small post-burn preview is shown
while the full recording loads. Run `node web/test_loading.mjs` to check streamed
buffers, aggregate progress, and rejection of failed or truncated downloads.

A separate Rustplotlib/WebGL2 plasma-line panel shares the same browser Welch
estimate as the ion-line panel. It uses a MHz frequency axis and retains
all frequencies, including the ion line at zero. The recorded electron-frequency peaks are
at −1.2740 and +1.2037 MHz for the default 8,192-point estimate. They are measured
bins in this finite record, not fitted theoretical frequencies. `verify.mjs`
checks both peaks in addition to integrated PSD power and the replay tests.
