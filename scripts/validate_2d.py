"""Validate complete production HDF5 records; run with conda run -n base python.

This validates the data contract, not numerical convergence or equilibration.
"""
import json
import argparse

import h5py
import numpy as np

seeds = set()
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--expected-realizations", type=int, default=16)
parser.add_argument("paths", nargs="+")
args = parser.parse_args()
for path in args.paths:
    with h5py.File(path, "r") as f:
        config = json.loads(bytes(f["config_json"][:]))
        scales = json.loads(bytes(f["scales_json"][:]))
        metadata = json.loads(bytes(f["metadata_json"][:]))
        assert metadata["dimensions"] == 2
        assert metadata["grid_shape"] == [128, 128]
        assert metadata["particles_per_species"] == 131072
        assert metadata["particle_shape"] == "quadratic B-spline (TSC)"
        assert metadata["source_revision"] == "65b8281c8f65362e4c56819db231254ef2e6d7ba"
        assert config["duration_ms"] == 8 and config["burn_ms"] == 10
        assert abs(scales["mass_ratio"] - 29166.215780524162) < 1e-8
        checked = []

        def finite(name, item):
            if isinstance(item, h5py.Dataset) and item.dtype.kind in "fc":
                assert np.isfinite(item[:]).all(), name
                checked.append(name)

        f.visititems(finite)
        for r in range(config["realizations"]):
            seed = config["seed"] + r
            assert seed not in seeds, seed
            seeds.add(seed)
            assert f[f"raw/{r}/real"].shape == (scales["steps"],)
            assert f[f"raw/{r}/imag"].shape == (scales["steps"],)
            assert f[f"sampled/{r}/real"].shape == (1969,)
            diag = f[f"diagnostics/{r}"][:]
            assert diag.shape == (848, 9)
            assert np.allclose(diag[0, 5:7], [2000, 1000], rtol=1e-6)
            assert np.isclose(diag[-1, 0], (scales["burn_steps"] + scales["steps"]) * scales["dt_s"])
            for key in ("x_m", "v_m_s"):
                assert f[f"trajectories/{r}/{key}"].shape == (848, 32, 2)
            assert f[f"trajectories/{r}/fast_x_m"].shape == (2048, 32, 2)
            positions = f[f"trajectories/{r}/x_m"][:]
            assert np.all(positions >= 0) and np.all(positions <= scales["length_m"])
            assert np.all(np.std(positions, axis=0) > 0), "Frozen position component"
        print(f"{path}: {len(checked)} finite floating datasets, complete 2D records")
assert len(seeds) == args.expected_realizations, f"Expected {args.expected_realizations} independent seeds, found {len(seeds)}"
print(f"PASS: {len(seeds)} complete independent realizations", sorted(seeds))
