"""Validate matched PIC microbenchmarks and retain the comparison as HDF5.

Use the base conda environment. Timings include the live diagnostics and the
specified instrumentation, not just a CUDA kernel. Short-run agreement is not
a test of thermal equilibrium or of long-time ISR spectral convergence.
"""
import json
from pathlib import Path
import re
import sys
import h5py
import numpy as np

directory = Path(sys.argv[1])
reference = Path(sys.argv[2]) if len(sys.argv) > 2 else directory / "single-direct.h5"
destination = directory / (sys.argv[3] if len(sys.argv)>3 else "comparison.h5")
assert not destination.exists(), f"Refusing overwrite: {destination}"
keys = ["raw/0/real", "raw/0/imag", "trajectories/0/fast_x_m"]
with h5py.File(reference) as source:
    ref_config = json.loads(bytes(source["config_json"][:]))
    refs = {key: source[key][:] for key in keys}

with h5py.File(destination, "x") as report:
    report.attrs["reference"] = str(reference)
    report.attrs["scope"] = "short deterministic regression, not long-time convergence"
    report.attrs["source_quantity"] = "full-population electron density mode"
    for path in sorted(directory.glob("*.h5")):
        if path.name == destination.name:
            continue
        if not path.with_suffix(".log").exists() or path.name.endswith(".live.h5"):
            continue
        with h5py.File(path) as data:
            config = json.loads(bytes(data["config_json"][:]))
            for key in ["cells", "ppc", "density", "te", "ti", "seed", "dt", "duration_ms"]:
                assert config[key] == ref_config[key], (path, key)
            d = data["diagnostics/0"][:]
            assert np.isfinite(d).all()
            drift = 100 * np.max(np.abs(d[:, 1] / d[0, 1] - 1))
            assert drift < 0.1, (path, "short-run energy regression", drift)
            errors = []
            for key in keys:
                values = data[key][:]
                assert values.shape == refs[key].shape and np.isfinite(values).all()
                error = np.linalg.norm(values.astype(float) - refs[key]) / np.linalg.norm(refs[key])
                errors.append(float(error))
                # Float32 trajectories diverge over time; the strict regression
                # threshold applies only to the 297-step short tests.
                if len(data["raw/0/real"]) <= 300:
                    assert error < 1e-4, (path, key, error)
            control = json.loads(bytes(data["control_0/metadata_json"][:]))
            assert not control["thermostatted"]
            rows = re.findall(r"t=([0-9.]+)ms .*?wall=([0-9.]+)s", path.with_suffix(".log").read_text())
            times = np.array(rows, dtype=float)
            scales = json.loads(bytes(data["scales_json"][:]))
            steps = (times[:, 0] * 1e-3 / scales["dt_s"]).round()
            # Fit only after the first diagnostic to reduce initialization and
            # early profiling overhead; two-point short cases use both rows.
            selected = slice(1, None) if len(times) >= 3 else slice(None)
            ms_per_step = float(np.polyfit(steps[selected], times[selected, 1], 1)[0] * 1000)
            group = report.create_group(path.stem)
            group.attrs["ms_per_step"] = ms_per_step
            group.attrs["energy_excursion_percent"] = drift
            group.attrs["regression_keys"] = json.dumps(keys)
            group.attrs["control_json"] = json.dumps(control)
            group.create_dataset("relative_l2_errors", data=errors)
            group.create_dataset("diagnostics", data=d)
            group.create_dataset("progress_time_ms_wall_s", data=times)
            print(json.dumps(dict(case=path.stem, ms_per_step=ms_per_step,
                                  energy_excursion_percent=float(drift), relative_l2_errors=errors)))
print(f"Saved {destination}")
