"""Fixed-B, fixed-ring density scan. Independent homogeneous PIC volumes.

Run with base Python on Seid. Reuses the existing 3 MHz ring/control. Saves
scientific metadata in HDF5; never changes or restarts an existing campaign.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import h5py
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('binary', type=Path)
parser.add_argument('output', type=Path)
parser.add_argument('--results', type=Path, required=True)
args = parser.parse_args()
args.output.mkdir()  # Deliberately refuses to overwrite a campaign.
deadline = time.monotonic() + 6900
expected = 'f120fc249c52cfdb15667977bd310f773bd188092caf54b636979a68f3b98a00'
assert hashlib.sha256(args.binary.read_bytes()).hexdigest() == expected
fps = np.arange(2400, 3401, 50) * 1e3
densities = 111639834702.32187 * (fps / 3e6)**2
manifest = args.output / 'manifest.h5'
with h5py.File(manifest, 'w') as f:
    f['plasma_frequency_hz'] = fps
    f['density_m3'] = densities
    f['complete'] = np.zeros((len(fps), 2), dtype='u1')
    f['work_corrected_energy_error_percent'] = np.full((len(fps), 2), np.nan)
    f['paths'] = np.full((len(fps), 2), '', dtype=object).astype(h5py.string_dtype())
    f.attrs['cases'] = 'ring 1 percent, thermal control'
    f.attrs['binary_sha256'] = expected
    f.attrs['ring_energy_parameter_eV'] = 10.
    f.attrs['ring_fraction'] = .01
    f.attrs['gyrofrequency_hz'] = 1e6
    f.attrs['geometry'] = 'independent homogeneous 2D3V boxes; 3 degrees from k perpendicular B'
    f.attrs['interpretation'] = 'density scan, not temporal density ramp or altitude profile'
    f.attrs['seed'] = 9501

def validate(path, density, fraction):
    with h5py.File(path) as f:
        c = json.loads(bytes(f['config_json'][:]))
        if 'base' in c:
            c = {**c.pop('base'), **c}
        s = json.loads(bytes(f['scales_json'][:]))
        assert np.isclose(c['density'], density, rtol=1e-12)
        assert c['hot_fraction'] == fraction and c['ring_energy_ev'] == 10
        assert c['te'] == 2000 and c['ti'] == 1000 and c['seed'] == 9501
        assert c['ring_turn_on_us'] == 20 and c['core_thermostat_us'] == 0
        assert c['bx_t'] == .000001869630096 and c['by_t'] == .00003567491076
        for key in ['raw/0/real', 'raw/0/imag']:
            x = f[key][:]
            assert len(x) == s['steps'] and np.isfinite(x).all()
        d = f['diagnostics/0'][:]
        p = f['population_energy_0'][:]
        assert np.isfinite(d).all() and np.isfinite(p).all()
        error = 100 * np.max(abs((d[:, 1] - p[:, -1] - d[0, 1]) / d[0, 1]))
        assert error < 1, f'Numerical energy guard: {error}'
        injection = f['ring_injection_time_work_0'][:]
        assert injection.shape == (1, 2)
        assert abs(injection[0, 0] - 20e-6) <= s['dt_s']
        return error

def run(case, device, fraction, density, directory):
    path = directory / f'{case}.h5'
    remaining = int(deadline - time.monotonic())
    if remaining < 30:
        raise RuntimeError('Campaign wall-time budget exhausted')
    command = [str(args.binary), '--backend', 'cuda', '--device', str(device),
        '--cells', '256', '--ppc', '7', '--bragg-mode', '1',
        '--density', str(density), '--te', '2000', '--ti', '1000',
        '--bx-t', '.000001869630096', '--by-t', '.00003567491076',
        '--hot-fraction', str(fraction), '--ring-energy-ev', '10',
        '--ring-width', '.1', '--ring-turn-on-us', '20',
        '--core-thermostat-us', '0', '--dt', '.01', '--burn-ms', '0',
        '--duration-ms', '.12', '--sample-us', '.005', '--window', '1024',
        '--diagnostic-stride', '200', '--seed', '9501', '--output', str(path)]
    env = {k:v for k,v in os.environ.items() if not k.startswith('RPIC_WEB_')}
    env['RPIC_LIVE_PATH'] = str(directory / f'{case}-live.h5')
    with (directory / f'{case}.log').open('w') as log:
        completed = subprocess.run(['timeout', '--signal=TERM', '--kill-after=5s',
            str(remaining), *command], stdout=log, stderr=subprocess.STDOUT, env=env)
    (directory / f'{case}.exit').write_text(str(completed.returncode)+'\n')
    completed.check_returncode()
    return path, validate(path, density, fraction)

# Visit the expected crossing first, then expand outwards. Final rows stay sorted.
order = sorted(range(len(fps)), key=lambda i: abs(fps[i]-2.8e6))
for i in order:
    fp = fps[i]
    print(f'Start fpe={fp/1e6:.2f} MHz, density={densities[i]:.9g}', flush=True)
    if fp == 3e6:
        paths = [args.results/'ring-pair-1pct-v1/ring/spectrum.h5',
                 args.results/'ring-pair-v1/control/spectrum.h5']
        results = [(p, validate(p, densities[i], fraction))
                   for p, fraction in zip(paths, [.01, 0.])]
    else:
        directory = args.output / f'fp-{int(fp/1000)}khz'
        directory.mkdir()
        with ThreadPoolExecutor(max_workers=2) as pool:
            jobs = [pool.submit(run, case, dev, fraction, densities[i], directory)
                    for case, dev, fraction in [('ring', 0, .01), ('control', 1, 0.)]]
            results = [j.result() for j in jobs]
    with h5py.File(manifest, 'r+') as f:
        for j, (path, error) in enumerate(results):
            f['paths'][i, j] = str(path)
            f['work_corrected_energy_error_percent'][i, j] = error
            f['complete'][i, j] = 1
    print(f'PASS {fp/1e6:.2f} MHz; numerical energy errors {[r[1] for r in results]} percent', flush=True)
print('COMPLETE: 21 density points, both cases validated', flush=True)
