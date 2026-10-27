"""Initial-distribution electrostatic dispersion and n=3 resonance check.

Isotropic magnetized Maxwellian susceptibility: MIT PSFC/RR-14-01,
equations 2.45-2.48. Sum core and tail susceptibilities at fixed density.
Cold ions are the MHz approximation; no prediction of PIC equilibration.
Convention exp(i k.x - i omega t): negative Im(omega) is damping.
"""
from pathlib import Path
import numpy as np
import h5py
from scipy.special import ive, wofz, jv
from scipy.integrate import quad
from scipy.optimize import root

qe=1.602176634e-19; me=9.1093837139e-31; kb=1.380649e-23
k=4*np.pi*440.2e6/299792458
bx=1.869630096e-6; by=3.567491076e-5
oc=qe*np.hypot(bx,by)/me; op=2*np.pi*3e6
kp=k*bx/np.hypot(bx,by); kt=np.sqrt(k*k-kp*kp)
core=2000.; hot=2*20*qe/(3*kb)
mass_ratio=16*1.66053906660e-27/me

def components(w, fraction, terms=96):
    n=np.arange(-terms,terms+1)
    out=[]
    for weight,temp in [(1-fraction,core),(fraction,hot)]:
        vt=np.sqrt(2*kb*temp/me)
        b=(kt*vt/oc)**2/2
        z=(w*oc-n*oc)/(kp*vt)
        susceptibility=weight*2*op**2/(k*k*vt*vt)*(1+w*oc/(kp*vt)*np.sum(ive(n,b)*1j*np.sqrt(np.pi)*wofz(z)))
        out.append(susceptibility)
    return out

def dielectric(w,fraction,terms=96):
    return 1+sum(components(w,fraction,terms))-op**2/(mass_ratio*(w*oc)**2)

def solve(fraction,terms):
    roots=[]
    for guess in [2.9,3.37]:
        def fn(a):
            e=dielectric(a[0]+1j*a[1],fraction,terms)
            return [e.real,e.imag]
        sol=root(fn,[guess,-.003],tol=1e-10)
        assert sol.success and np.linalg.norm(fn(sol.x))<1e-7,sol
        roots.append(sol.x[0]+1j*sol.x[1])
    return np.array(roots)*oc/(2*np.pi)

out=Path(__file__).resolve().parents[1]/'results/oblique-resonance'
out.mkdir(parents=True,exist_ok=True)
with h5py.File(out/'initial-dispersion.h5','w') as f:
    f.attrs['approximation']='Initial homogeneous isotropic Maxwellian mixture; cold ions; not a measured PIC spectrum'
    f.attrs['k_parallel_rad_m']=kp
    f.attrs['k_perpendicular_rad_m']=kt
    f.attrs['f_ce_hz']=oc/(2*np.pi)
    for label,fraction in [('control',0.),('tail',.01)]:
        freq=solve(fraction,96)
        assert np.allclose(freq,solve(fraction,192),rtol=1e-8,atol=.01)
        v=(2*np.pi*freq.real-3*oc)/kp
        energy=.5*me*v*v/qe
        g=f.create_group(label)
        g['frequency_hz']=freq.real;g['damping_hz']=freq.imag
        g['resonant_v_parallel_m_s']=v;g['parallel_energy_ev']=energy
        # Perpendicular-averaged J_3^2 times parallel velocity PDF and density fraction.
        weights=[]
        for pop,weight,temp in [('core',1-fraction,core),('hot',fraction,hot)]:
            sigma=np.sqrt(kb*temp/me);b=(kt*sigma/oc)**2
            # Independently verify the perpendicular gyro-orbit coupling average.
            numerical=quad(lambda u: u*np.exp(-u*u/2)*jv(3,np.sqrt(b)*u)**2,0,14,
                           epsabs=1e-11)[0]
            assert np.isclose(numerical,ive(3,b),rtol=1e-7,atol=1e-11)
            overlap=weight*ive(3,b)*np.exp(-v*v/(2*sigma*sigma))/(np.sqrt(2*np.pi)*sigma)
            g[pop+'_third_harmonic_overlap_s_m']=overlap
            g.attrs[pop+'_b']=b;g.attrs[pop+'_Gamma3']=ive(3,b)
            weights.append(overlap)
        print(label,'frequency MHz',freq.real/1e6,'damping Hz',freq.imag,
              'v_parallel km/s',v/1e3,'E_parallel eV',energy,
              'tail/core resonant overlap',weights[1]/weights[0])
    print('k_parallel',kp,'fce',oc/(2*np.pi),'saved',f.filename)
