"""Extract completed sweep rows with matched physical Welch windows into HDF5."""
import sys, json
import h5py
import numpy as np
from scipy.signal import welch

with h5py.File(sys.argv[1], 'r') as manifest:
    complete = manifest['complete'][:]
    fps = manifest['plasma_frequency_hz'][:]
    paths = manifest['paths'].asstr()[:]
with h5py.File(sys.argv[2], 'w') as out:
    out.attrs['interval_s'] = [30e-6, 120e-6]
    out.attrs['segment_s'] = 20e-6
    out.attrs['normalization'] = 'Absolute density-mode PSD; no individual peak normalization'
    for i in np.flatnonzero(np.all(complete == 1, axis=1)):
        row = out.create_group(f'fp_{int(round(fps[i]))}')
        row.attrs['plasma_frequency_hz'] = fps[i]
        for j, name in enumerate(['ring', 'control']):
            with h5py.File(paths[i,j], 'r') as src:
                dt = json.loads(bytes(src['scales_json'][:]))['dt_s']
                z = src['raw/0/real'][:] + 1j*src['raw/0/imag'][:]
                assert np.isfinite(z).all()
                t = np.arange(z.size)*dt
                z = z[(t >= 30e-6) & (t < 120e-6)]
                n = round(20e-6/dt)
                f,p = welch(z, fs=1/dt, window='hann', nperseg=n,
                            noverlap=n//2, detrend=False, return_onesided=False)
                order = np.argsort(f)
                keep = order[np.abs(f[order]) <= 6e6]
                g = row.create_group(name)
                g['frequency_hz'] = f[keep]
                g['psd'] = p[keep]
                g.attrs['source'] = paths[i,j]
                g.attrs['frequency_bin_hz'] = 1/(n*dt)
        print(fps[i]/1e6, flush=True)
