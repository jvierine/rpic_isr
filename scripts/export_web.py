"""Build compact browser transports from authoritative GPU-capture HDF5.

Invocation: conda run -n base python scripts/export_web.py CAPTURE.h5 SPECTRUM.h5 OUTPUT
Transport quantization affects only replay, never simulation or PSD input.
"""
import hashlib
import json
import pathlib
import sys

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
    assert x.shape == (1024, 8192, 2)
    assert phi.shape == (1024, 128, 128)
    assert all(np.isfinite(a).all() for a in (x, phi, raw))
    assert x.min() >= 0 and x.max() <= scales["length"] * (1 + 1e-6)
    assert config["burn_ms"] >= 10
    assert config["seed"] == json.loads(bytes(sf["config_json"][:]))["seed"]
    assert np.array_equal(raw[:, 0], sf["raw/0/real"][:])
    assert np.array_equal(raw[:, 1], sf["raw/0/imag"][:])
    diagnostics = sf["diagnostics/0"][:]
    energy = float(100 * np.max(np.abs(diagnostics[:, 1] / diagnostics[0, 1] - 1)))
    # Shared even grid nodes retain the potential's periodic coordinate origin.
    field = phi[:, ::2, ::2]
    quant_scale = float(np.max(np.abs(field)))
    display_scale = float(np.quantile(np.abs(field), .97))
    encoded_x = np.rint(x / scales["length"] * 65535).clip(0, 65535).astype("<u2")
    encoded_phi = np.rint(field / quant_scale * 32767).clip(-32767, 32767).astype("<i2")
    assert np.max(np.abs(encoded_x.astype(float)/65535*scales["length"]-x)) < scales["length"]/65535
    assert np.max(np.abs(encoded_phi.astype(float)/32767*quant_scale-field)) < quant_scale/32767
    # Same anti-alias filter and edge trimming as the Rust spectrum::decimate.
    stride = 8
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
        frames=1024, pool_per_species=4096, field_grid=64,
        length_lambda_de=scales["length"], lambda_de_m=scales["lambda_de"],
        particles_per_species=config["cells"]**2*config["ppc"],
        frame_dt_s=scales["dt_s"]*8, burn_ms=config["burn_ms"],
        signal_dt_s=scales["dt_s"]*stride, signal_first_s=half*scales["dt_s"],
        signal_samples=len(signal), phi_quant_scale=quant_scale,
        phi_display_scale=display_scale, potential_to_mv=config["te"]*8.617333262145e-2,
        max_energy_excursion_percent=energy, source=source,
        transport="little-endian u16 positions; i16 potential; interleaved f32 signal",
        capture_sha256=hashlib.sha256(pathlib.Path(capture).read_bytes()).hexdigest(),
        spectrum_sha256=hashlib.sha256(pathlib.Path(spectrum).read_bytes()).hexdigest(),
    )
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))
