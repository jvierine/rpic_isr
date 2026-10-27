"""Export perpendicular Maxwellian electron-mode positions for the web plot.

Immobile ions are the MHz approximation. These are dielectric roots, not
cyclotron harmonics, spectral amplitudes, or fitted PIC peaks.
Reference: https://doi.org/10.1063/5.0316617
"""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.optimize import brentq
from scipy.special import ive

root = Path(__file__).resolve().parents[1]
m = json.loads((root/'web/public/data/bernstein-long/metadata-lossy.json').read_text())
keys = ['magnetic_field_z_t', 'omega_pe_rad_s', 'te_K', 'k_bragg_rad_m']
omega_c = abs(m['magnetic_field_z_t'])*1.602176634e-19/9.1093837139e-31
b = (m['k_bragg_rad_m']*np.sqrt(1.380649e-23*m['te_K']/9.1093837139e-31)/omega_c)**2
def solve(terms):
    n = np.arange(1, terms+1)
    a = 2*(m['omega_pe_rad_s']/omega_c)**2*n*n*ive(n,b)/b
    def epsilon(w):
        return 1-np.sum(a/(w*w-n*n))
    roots = np.array([brentq(epsilon,j+1e-8,j+1-1e-8,xtol=1e-14) for j in range(1,6)])
    assert max(abs(epsilon(w)) for w in roots) < 1e-6
    return roots*omega_c/(2*np.pi)
frequencies = solve(96)
assert np.allclose(frequencies, solve(192), rtol=1e-12)
with h5py.File(root/'results/bernstein-long-v1/dispersion-check.h5') as f:
    assert np.allclose(frequencies, f['electron_bernstein_roots_hz'][0], rtol=1e-9)
with h5py.File(root/'results/bernstein-long-v1/analytic-lines.h5','w') as f:
    f['frequency_hz'] = frequencies
    f.attrs['b'] = b
    for k in keys: f.attrs[k] = m[k]
out = {'parameters': {k:m[k] for k in keys}, 'frequency_hz':frequencies.tolist()}
(root/'web/public/data/theory/bernstein-lines.json').write_text(json.dumps(out,indent=2)+'\n')
print('Verified Bernstein roots [MHz]:',frequencies/1e6)
