"""Build compact browser transports from authoritative GPU-capture HDF5.

Invocation: conda run -n base python scripts/export_web.py CAPTURE.h5 SPECTRUM.h5 OUTPUT
Transport quantization affects only replay, never simulation or PSD input.
"""
import hashlib
import json
import pathlib
import sys
import os

import h5py
import numpy as np
from scipy.signal import fftconvolve

capture, spectrum, destination = sys.argv[1:]
out = pathlib.Path(destination)
out.mkdir(parents=True, exist_ok=True)
with h5py.File(capture) as f, h5py.File(spectrum) as sf:
    config = json.loads(bytes(f["config_json"][:]))
    scales = json.loads(bytes(f["scales_json"][:]))
    source = json.loads(bytes(f["metadata_json"][:]))
    x = f["positions_lambda_de"][:]
    phi = f["potential_kbte_e"][:]
    raw = f["raw_signal"][:]
    frames = x.shape[0]
    source_grid = source.get("potential_grid", config["cells"])
    pool=x.shape[1]//2
    assert x.shape == (frames, 2*pool, 2) and frames >= 1024
    assert phi.shape == (frames, source_grid, source_grid)
    assert all(np.isfinite(a).all() for a in (x, phi, raw))
    assert x.min() >= 0 and x.max() <= scales["length"] * (1 + 1e-6)
    # The bounded packed run uses an explicit stationarity/settling protocol,
    # not the older fixed 10 ms protocol. Never accept an arbitrary short burn.
    control = source.get("control")
    if config["burn_ms"] < 10:
        assert control and control["thermostatted"]
        assert control["activation_time_s"] >= .001
        assert config["burn_ms"]*1e-3-control["activation_time_s"] >= .0005
        history = sf["control_0/history"][:]
        assert np.max(np.abs(history[:, 8])) < 1
        assert np.any(history[:, 9] == 2)
    assert config["seed"] == json.loads(bytes(sf["config_json"][:]))["seed"]
    assert np.array_equal(raw[:, 0], sf["raw/0/real"][:])
    assert np.array_equal(raw[:, 1], sf["raw/0/imag"][:])
    diagnostics = sf["diagnostics/0"][:]
    energy = float(100 * np.max(np.abs(diagnostics[:, 1] / diagnostics[0, 1] - 1)))
    # Crop, never decimate: retain native nodes around the fixed 10-lambda camera.
    # A power-of-two crop includes interpolation halo while limiting download size.
    dx = scales["length"] / config["cells"]
    grid = min(source_grid, 2**int(np.ceil(np.log2(10/dx+2))))
    start = (source_grid-grid)//2
    field = phi[:, start:start+grid, start:start+grid]
    quant_scale = float(np.max(np.abs(field)))
    display_scale = float(np.quantile(np.abs(field), .97))
    encoded_x = np.rint(x / scales["length"] * 65535).clip(0, 65535).astype("<u2")
    encoded_phi = np.rint(field / quant_scale * 32767).clip(-32767, 32767).astype("<i2")
    assert np.max(np.abs(encoded_x.astype(float)/65535*scales["length"]-x)) < scales["length"]/65535
    assert np.max(np.abs(encoded_phi.astype(float)/32767*quant_scale-field)) < quant_scale/32767
    # Same anti-alias filter and edge trimming as the Rust spectrum::decimate.
    stride = int(os.environ.get('RPIC_WEB_SIGNAL_STRIDE','8'))
    assert stride>=1 and stride & (stride-1)==0
    # The displayed plasma band must lie safely within the FIR passband.
    band=max(2,int(np.ceil(1.6*scales['omega_pe']/(2*np.pi*1e6))))*1e6
    assert .4/(stride*scales['dt_s'])>1.1*band
    half = 16 * stride
    t = np.arange(-half, half+1, dtype=float)
    fc = .4 / stride
    u = (t + half) / (2*half)
    h = (2*fc*np.sinc(2*fc*t)*(.42-.5*np.cos(2*np.pi*u)+.08*np.cos(4*np.pi*u))).astype(np.float32)
    h /= h.sum()
    voltage = raw[:, 0] + 1j*raw[:, 1]
    filtered = fftconvolve(voltage, h, mode="same")[half:-half:stride]
    signal = np.stack([filtered.real, filtered.imag], axis=-1).astype("<f4")
    files = {}
    for name, array in [("positions", encoded_x), ("potential", encoded_phi), ("signal", signal)]:
        digest = hashlib.sha256(array.tobytes()).hexdigest()
        filename = f"{name}-{digest[:12]}.bin"
        array.tofile(out / filename)
        files[name] = filename
    for name, array in [("preview_positions",encoded_x[:2]),("preview_potential",encoded_phi[:2])]:
        digest=hashlib.sha256(array.tobytes()).hexdigest()
        filename=f"{name}-{digest[:12]}.bin"
        array.tofile(out / filename)
        files[name]=filename
    metadata = dict(
        positions_file=files["positions"], potential_file=files["potential"], signal_file=files["signal"],
        preview_positions_file=files["preview_positions"],preview_potential_file=files["preview_potential"],
        frames=frames, pool_per_species=pool, field_grid=grid,
        potential_length_lambda_de=grid*dx,
        potential_grid_spacing_lambda_de=dx,
        potential_crop_start_node=source.get("potential_start_node", 0)+start,
        length_lambda_de=scales["length"], lambda_de_m=scales["lambda_de"],
        particles_per_species=config["cells"]**2*config["ppc"],
        frame_dt_s=scales["dt_s"]*8, burn_ms=config["burn_ms"],
        signal_dt_s=scales["dt_s"]*stride, signal_first_s=half*scales["dt_s"],
        signal_samples=len(signal), phi_quant_scale=quant_scale,
        density_m3=config["density"], te_K=config["te"], ti_K=config["ti"],
        ion_mass_amu=config["ion_mass_amu"], mass_ratio=scales["mass_ratio"],
        radar_mhz=config["radar_mhz"], k_bragg_rad_m=scales["k_bragg"],
        k_lambda_de=scales["k_bragg"]*scales["lambda_de"],
        omega_pe_rad_s=scales["omega_pe"], seed=config["seed"],
        field_solver_grid=config["cells"],
        plasma_band_mhz=max(2, int(np.ceil(1.6*scales["omega_pe"]/(2*np.pi*1e6)))),
        default_fft=int(2**np.clip(np.round(np.log2(1/(scales["dt_s"]*stride*1500))), 6, 16)),
        phi_display_scale=display_scale, potential_to_mv=config["te"]*8.617333262145e-2,
        max_energy_excursion_percent=energy, source=source,
        transport="little-endian u16 positions; i16 potential; interleaved f32 signal",
        capture_sha256=hashlib.sha256(pathlib.Path(capture).read_bytes()).hexdigest(),
        spectrum_sha256=hashlib.sha256(pathlib.Path(spectrum).read_bytes()).hexdigest(),
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))
