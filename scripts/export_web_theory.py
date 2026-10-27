"""Collisionless unequal-temperature Maxwellian reference; no fitted amplitude.

Rosenbluth & Rostoker (1962), doi:10.1063/1.1724446, Eqs. 16/26.
Same dielectric expression as src/spectrum.rs, evaluated with scipy's Faddeeva
function. Integrate narrow resonances before Hann convolution, rather than
sampling a possibly unresolved plasma line at FFT bin centres.
Scientific results/validation: HDF5. Float32 files are browser transport only.
"""
import hashlib
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import wofz

ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / 'web/public'
OUT = PUBLIC / 'data/theory'
OUT.mkdir(parents=True, exist_ok=True)
KB, QE, EPS, ME, AMU = 1.380649e-23, 1.602176634e-19, 8.8541878128e-12, 9.1093837139e-31, 1.66053906660e-27

def response(f, m):
    k = m['k_bragg_rad_m']
    def species(temp, mass):
        v = np.sqrt(KB*temp/mass)
        z = 2*np.pi*f/(np.sqrt(2)*k*v)
        chi = m['density_m3']*QE**2/(EPS*KB*temp*k*k)*(1+z*1j*np.sqrt(np.pi)*wofz(z))
        sf = k*v/(2*np.pi)
        g = np.exp(-.5*(f/sf)**2)/(np.sqrt(2*np.pi)*sf)
        return chi, g
    ce, ge = species(m['te_K'], ME)
    ci, gi = species(m['ti_K'], m['ion_mass_amu']*AMU)
    eps = 1+ce+ci
    return np.abs((1+ci)/eps)**2*ge+np.abs(ce/eps)**2*gi, eps

def expected(m, n, over):
    dt = m['signal_dt_s']; length = n*over
    f = np.fft.fftfreq(length, dt); df = 1/(length*dt)
    density = response(f, m)[0]
    fp = m['omega_pe_rad_s']/(2*np.pi)
    probe = np.linspace(.8*fp, 2*fp, 1000)
    er = response(probe, m)[1].real
    roots = [brentq(lambda x: response(x,m)[1].real,a,b) for a,b,x,y in zip(probe[:-1],probe[1:],er[:-1],er[1:]) if x*y<0]
    # Integrate each bin within 32 bins of every dielectric resonance.
    for root in roots:
        for sign in [-1,1]:
            centre = sign*root
            for j in np.flatnonzero(np.abs(f-centre)<32*df):
                a,b=f[j]-df/2,f[j]+df/2
                density[j]=quad(lambda x: float(response(x,m)[0]),a,b,
                    points=[centre] if a<centre<b else None,
                    epsabs=1e-12,epsrel=2e-7,limit=200)[0]/df
    # Apply exactly the export anti-alias filter's power response.
    raw_dt=.06/m['omega_pe_rad_s'];stride=round(dt/raw_dt)
    half=16*stride;t=np.arange(-half,half+1);u=(t+half)/(2*half)
    h=(2*(.4/stride)*np.sinc(2*(.4/stride)*t)*(.42-.5*np.cos(2*np.pi*u)+.08*np.cos(4*np.pi*u))).astype(np.float32)
    h/=h.sum()
    # FFT evaluation avoids O(grid size * filter length) freqz work.
    nh=length*stride
    hf=np.abs(np.fft.fft(h,nh))**2
    density*=hf[np.round(f*raw_dt*nh).astype(int)%nh]
    w=.5-.5*np.cos(2*np.pi*np.arange(n)/n)
    kernel=np.abs(np.fft.fft(w,length))**2/(length*np.sum(w*w))
    smoothed=np.fft.ifft(np.fft.fft(density)*np.fft.fft(kernel)).real
    out=np.fft.fftshift(smoothed[::over])
    assert np.isfinite(out).all() and out.min()>-1e-15
    out=np.maximum(out,0)
    assert np.max(np.abs(out-np.roll(out[::-1],1)))<out.max()*1e-7
    return out,roots

if __name__ == '__main__':
    catalog={}
    with h5py.File(OUT/'validation.h5','w') as h5:
        for run in json.loads((PUBLIC/'runs.json').read_text())['runs']:
            m=json.loads((PUBLIC/run['metadata']).read_text());entry={'parameters':{k:m[k] for k in ['density_m3','te_K','ti_K','ion_mass_amu','k_bragg_rad_m','signal_dt_s']},'fft':{}}
            group=h5.create_group(run['id']);group.attrs['metadata']=json.dumps(m)
            for n in [8192,16384,32768,65536]:
                a,roots=expected(m,n,8);b,_=expected(m,n,16)
                err=np.sum(np.abs(a-b))/np.sum(b)
                peakerr=np.max(np.abs(a-b))/np.max(b)
                assert err<.01 and peakerr<.01,(run['id'],n,err,peakerr)
                data=b.astype('<f4').tobytes();digest=hashlib.sha256(data).hexdigest()
                filename=f'{run["id"]}-{n}-{digest[:12]}.bin';(OUT/filename).write_bytes(data)
                g=group.create_group(str(n));g.create_dataset('psd',data=b);g.attrs.update(l1_refinement_error=err,peak_refinement_error=peakerr,dielectric_roots_hz=roots)
                entry['fft'][str(n)]=filename
                print(run['id'],n,'refinement',err,peakerr,'roots',roots,flush=True)
            catalog[run['id']]=entry
    (OUT/'catalog.json').write_text(json.dumps(catalog,indent=2)+'\n')
