"""Validate one GPU web capture and save spectral diagnostics as HDF5.

conda run -n base python scripts/check_web_spectrum.py CAPTURE SPECTRUM METADATA OUTPUT
This checks a finite demonstration recording, not equilibrium/convergence.
"""
import json
import pathlib
import sys

import h5py
import numpy as np
from scipy.signal import welch

capture_path, spectrum_path, metadata_path, output_path = sys.argv[1:]
metadata_file = pathlib.Path(metadata_path)
m = json.loads(metadata_file.read_text())
with h5py.File(capture_path) as c, h5py.File(spectrum_path) as s:
    config = json.loads(bytes(c["config_json"][:]))
    scales = json.loads(bytes(c["scales_json"][:]))
    other_config = json.loads(bytes(s["config_json"][:]))
    assert config == other_config
    positions = c["positions_lambda_de"][:]
    phi = c["potential_kbte_e"][:]
    raw = c["raw_signal"][:]
    d = s["diagnostics/0"][:]
    assert all(np.isfinite(a).all() for a in (positions, phi, raw, d))
    assert positions.shape == (1024, 8192, 2)
    assert phi.shape == (1024, 128, 128)
    assert raw.shape == (scales["steps"], 2)
    assert np.array_equal(raw[:, 0], s["raw/0/real"][:])
    assert np.array_equal(raw[:, 1], s["raw/0/imag"][:])
    assert positions.min() >= 0 and positions.max() <= scales["length"]*(1+1e-6)
    assert m["density_m3"] == config["density"]
    assert abs(d[-1, 0] - (scales["burn_steps"]+scales["steps"])*scales["dt_s"]) < 1e-12
    a = np.fromfile(metadata_file.parent/m["signal_file"], dtype="<f4").reshape(-1, 2)
    z = a[:, 0]+1j*a[:, 1]
    n = m["default_fft"]
    def psd(v):
        f, p = welch(v, fs=1/m["signal_dt_s"], nperseg=n, noverlap=n//2,
                     detrend=False, return_onesided=False, window="hann")
        order = np.argsort(f)
        return f[order], p[order]
    f, full = psd(z)
    _, early = psd(z[:len(z)//2])
    _, late = psd(z[len(z)//2:])
    fpe = scales["omega_pe"]/(2*np.pi)
    # Widths of noisy local maxima are misleading. Report the envelope after a
    # stated nine-bin boxcar, not an intrinsic or deconvolved damping width.
    envelope = np.convolve(full, np.ones(9)/9, mode="same")
    widths, peaks, power_ratios = [], [], []
    for sign in [-1, 1]:
        band = (f*sign > .7*fpe) & (f*sign < m["plasma_band_mhz"]*1e6)
        ids = np.flatnonzero(band)
        j = ids[np.argmax(full[band])]
        centre = ids[np.argmax(envelope[band])]
        half = envelope[centre]/2
        left, right = centre, centre
        while left > 0 and envelope[left] > half:
            left -= 1
        while right+1 < len(full) and envelope[right] > half:
            right += 1
        peaks.append(float(f[j]))
        widths.append(float(f[right]-f[left]))
        power_ratios.append(float(late[band].sum()/early[band].sum()))
    post = d[d[:, 0] >= config["burn_ms"]*1e-3]
    report = dict(
        density_m3=config["density"], seed=config["seed"],
        k_lambda_de=scales["k_bragg"]*scales["lambda_de"],
        fft=n, bin_width_hz=float(f[1]-f[0]),
        plasma_peaks_hz=peaks, nine_bin_envelope_half_max_widths_hz=widths,
        envelope_smoothing_width_hz=float(9*(f[1]-f[0])), intrinsic_linewidth_claim=False,
        plasma_late_early_power_ratios=power_ratios,
        max_energy_excursion_percent=float(100*np.max(np.abs(d[:, 1]/d[0, 1]-1))),
        final_te_K=float(d[-1, 5]), final_ti_K=float(d[-1, 6]),
        record_te_change_percent=float(100*(post[-1, 5]/post[0, 5]-1)),
        record_ti_change_percent=float(100*(post[-1, 6]/post[0, 6]-1)),
        equilibrium_claim=False, precision_convergence_claim=False,
    )
    target = pathlib.Path(output_path)
    assert not target.exists(), f"Refusing to overwrite {target}"
    with h5py.File(target, "w") as out:
        for name, array in [("frequency_hz", f), ("psd", full), ("early_psd", early),
                            ("late_psd", late), ("diagnostics", d)]:
            out.create_dataset(name, data=array)
        out.create_dataset("report_json", data=np.frombuffer(json.dumps(report).encode(), dtype=np.uint8))
    print(json.dumps(report, indent=2))
