"""Electron Bernstein roots and observed perpendicular-mode spectra in HDF5.

Electrostatic Maxwellian perpendicular dielectric; ions treated as immobile
for this MHz electron-branch comparison only. Not an ion-line theory.
Roots are compared with finite-record Welch maxima, not declared detections.
"""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from scipy.special import ive
from scipy.optimize import brentq
from scipy.signal import welch

source, target = map(Path, sys.argv[1:])
assert not target.exists()
with h5py.File(source) as f:
    c=json.loads(bytes(f['config_json'][:]))
    s=json.loads(bytes(f['scales_json'][:]))
    a=f['electron_density_modes_1_to_8'][:]
omega_c=abs(c['bz_t'])*1.602176634e-19/9.1093837139e-31
fc=omega_c/(2*np.pi)
rho=s['vte']/omega_c
harmonic=np.arange(1,97,dtype=float)
frequency=None
power=[]; roots=[]; peaks=[]
for mode in range(1,9):
    k=2*np.pi*mode/s['length_m']; b=(k*rho)**2
    coeff=2*(s['omega_pe']/omega_c)**2/b*harmonic**2*ive(harmonic,b)
    def dielectric(w):
        return 1-np.sum(coeff/(w*w-harmonic**2))
    r=np.array([brentq(dielectric,j+1e-7,j+1-1e-7)*fc for j in range(1,6)])
    z=a[:,mode-1,0]+1j*a[:,mode-1,1]
    f,p=welch(z,fs=1/s['dt_s'],window='hann',nperseg=min(32768,len(z)),
              noverlap=min(32768,len(z))//2,detrend=False,return_onesided=False,scaling='density')
    order=np.argsort(f);frequency=f[order];p=p[order]
    maxima=[]
    for value in r:
        selected=(frequency>value-.05*fc)&(frequency<value+.05*fc)
        maxima.append(frequency[selected][np.argmax(p[selected])])
    power.append(p);roots.append(r);peaks.append(maxima)
    print(f'mode {mode}, k*rho={k*rho:.3f}: roots MHz {np.round(r/1e6,5)}; nearby maxima MHz {np.round(np.array(maxima)/1e6,5)}')
with h5py.File(target,'x') as f:
    f['frequency_hz']=frequency;f['psd']=power
    f['electron_bernstein_roots_hz']=roots;f['nearby_welch_maxima_hz']=peaks
    f['k_perp_rad_m']=2*np.pi*np.arange(1,9)/s['length_m']
    f.attrs['theory']='Perpendicular collisionless Maxwellian electron dielectric, immobile-ion MHz approximation'
    f.attrs['interpretation']='Nearby maximum is not a detection criterion or an amplitude/convergence validation'
    f.attrs['source_capture']=str(source)
