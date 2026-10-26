"""Compare matched-time PIC energy/temperature drift; these short pilots are not equilibrated spectra."""
import json
import sys
import h5py
import numpy as np

for path in sys.argv[1:]:
    with h5py.File(path) as f:
        c = json.loads(bytes(f["config_json"][:]))
        s = json.loads(bytes(f["scales_json"][:]))
        d = f["diagnostics/0"][:]
        assert np.isfinite(d).all()
        stop = min(.0001, d[-1, 0])
        energy = np.interp(stop, d[:, 0], d[:, 1])
        te = np.interp(stop, d[:, 0], d[:, 5])
        print(json.dumps(dict(cells=c["cells"], ppc=c["ppc"], dx_lambda=s["dx"],
            compared_time_ms=stop*1000,
            energy_change_percent=100*(energy/d[0, 1]-1),
            te_change_percent=100*(te/d[0, 5]-1),
            full_duration_ms=float(d[-1, 0]*1000),
            full_max_energy_excursion_percent=float(100*np.max(np.abs(d[:, 1]/d[0, 1]-1))),
            equilibrium_claim=False)))
