"""Validate implicit PIC records; invoke with conda run -n base python."""
import json
import sys

import h5py
import numpy as np

for path in sys.argv[1:]:
    with h5py.File(path) as f:
        options = json.loads(bytes(f["implicit_options_json"][:]))
        scales = json.loads(bytes(f["scales_json"][:]))
        metadata = json.loads(bytes(f["metadata_json"][:]))
        for name in f:
            assert np.isfinite(f[name][:]).all(), (path, name)
        d = f["diagnostics"][:]
        steps = options["steps"]
        assert f["iterations"].shape == (steps,)
        assert f["residual_rms"].shape == (steps,)
        assert f["positions_m"].shape == (len(d), 32, 2)
        assert f["density_real"].shape == (len(d),)
        assert abs(d[-1, 0] - steps * scales["dt_s"]) < 1e-12
        assert (f["iterations"][:] <= options["max_iterations"]).all()
        rho, phi = f["final_rho"][:], f["final_phi"][:]
        length = scales["length"]
        if options["backend"] == "Cuda":
            length = float(np.float32(length))
        k = 2 * np.pi * np.fft.fftfreq(len(rho), length / len(rho))
        k2 = k[:, None] ** 2 + k[None, :] ** 2
        rh = np.fft.fft2(rho)
        rh[0, 0] = 0
        poisson = np.linalg.norm(k2 * np.fft.fft2(phi) - rh) / np.linalg.norm(rh)
        assert poisson < 1e-3, poisson
        assert abs(0.5 * np.mean(rho * phi) - d[-1, 4]) < 1e-10
        assert np.allclose(d[:, 1], d[:, 2:5].sum(axis=1), rtol=1e-12)
        excursion = 100 * np.max(np.abs(d[:, 1] / d[0, 1] - 1))
        print(f"{path}: PASS; energy excursion={excursion:.9g}%; "
              f"Poisson relative residual={poisson:.5g}; "
              f"mean iterations={f['iterations'][:].mean():.4f}; "
              f"elapsed={f.attrs['elapsed_s']:.3f}s; "
              f"max mean charge={np.max(np.abs(d[:,11])):.5g}; "
              f"momentum change={d[-1,7:9]-d[0,7:9]}; "
              f"revision={metadata['source_revision']}")
