"""Find an unstable electrostatic n~3 mode for a gyrotropic electron ring.

General gyrotropic Vlasov susceptibility integrated over perpendicular speed;
parallel Gaussian integrated with the Landau plasma dispersion function.
Reference motivation: Eliasson et al. 2016, doi:10.1088/0741-3335/58/9/095002.
Ring is f_perp proportional to exp(-(v_perp-v_ring)^2/(2 sigma_ring^2)),
NOT a Gaussian radial probability density. Cold-ion MHz approximation.
"""
from pathlib import Path
import numpy as np
import h5py
from scipy.special import jv,ive,wofz
from scipy.optimize import root
from numpy.polynomial.legendre import leggauss

qe=1.602176634e-19;me=9.1093837139e-31;kb=1.380649e-23
oc=2*np.pi*1e6;op=3*oc;k=4*np.pi*440.2e6/299792458
kp=k*np.sin(np.deg2rad(3));kt=k*np.cos(np.deg2rad(3))
vc=np.sqrt(2*kb*2000/me);ratio=16*1.6605390666e-27/me

def coefficients(energy,width,terms=48,points=256,gaussian=False):
    n=np.arange(-terms,terms+1)
    vr=np.sqrt(2*energy*qe/me);sigma=width*vr
    if gaussian:vr=0;sigma=vc/np.sqrt(2)
    x,w=leggauss(points);v=(x+1)*(vr+9*sigma)/2;w=w*(vr+9*sigma)/2
    f=np.exp(-.5*((v-vr)/sigma)**2);f/=np.sum(w*v*f)
    derivative=-(v-vr)/sigma**2*f
    jj=jv(n[:,None],kt*v[None,:]/oc)**2
    return n,jj@(w*v*f),jj@(w*derivative)

def chi(w,n,b,a,vt=vc):
    z=(w-n)*oc/(kp*vt);Z=1j*np.sqrt(np.pi)*wofz(z)
    return op**2/k**2*np.sum(-n*oc*a*Z/(kp*vt)+2*b/vt**2*(1+z*Z))

def core(w,n):
    b=(kt*vc/oc)**2/2;z=(w-n)*oc/(kp*vc)
    return 2*op**2/(k*k*vc*vc)*(1+w*oc/(kp*vc)*np.sum(ive(n,b)*1j*np.sqrt(np.pi)*wofz(z)))

# Independent isotropic Maxwellian limit checks sign and normalization.
n,b,a=coefficients(20,.1,gaussian=True)
for w in [2.91+.01j,3.35+.02j,3.01+.03j]:
    assert np.allclose(chi(w,n,b,a),core(w,n),rtol=1e-8,atol=1e-8)

found=[]
for fraction in [.01,.03,.05,.1]:
 for energy in [1.,2.,5.,10.,20.,40.]:
  n,b,a=coefficients(energy,.1)
  def dielectric(w):return 1+(1-fraction)*core(w,n)+fraction*chi(w,n,b,a)-op**2/(ratio*(w*oc)**2)
  for guess in [2.9,2.97,3.02,3.1,3.25,3.35]:
   def fun(x):
    e=dielectric(x[0]+1j*x[1]);return [e.real,e.imag]
   sol=root(fun,[guess,.02],tol=1e-10)
   wr,wi=sol.x
   if sol.success and np.linalg.norm(fun(sol.x))<1e-7 and 2.7<wr<3.6 and wi>1e-5:
    row=[fraction,energy,.1,wr*1e6,wi*1e6]
    if not any(np.allclose(row,r) for r in found):found.append(row);print(row,flush=True)

out=Path(__file__).resolve().parents[1]/'results/oblique-resonance/ring-search.h5'
checks=[]
for fraction,energy,width,fr,gi in found:
 for terms,points in [(96,512),(192,768)]:
  n,b,a=coefficients(energy,width,terms,points)
  def verify(x):
   z=x[0]+1j*x[1]
   e=1+(1-fraction)*core(z,n)+fraction*chi(z,n,b,a)-op**2/(ratio*(z*oc)**2)
   return [e.real,e.imag]
  sol=root(verify,[fr/1e6,gi/1e6],tol=1e-11)
  assert sol.success and np.linalg.norm(verify(sol.x))<1e-8
  assert np.allclose(sol.x,np.array([fr,gi])/1e6,rtol=1e-7,atol=1e-8)
  checks.append([terms,points,*sol.x])
  print('converged',terms,points,sol.x,flush=True)
with h5py.File(out,'w') as f:
 f['unstable_roots']=np.array(found).reshape(-1,5)
 f.attrs['columns']='fraction,ring_parameter_energy_eV,width_fraction,frequency_Hz,gamma_over_2pi_Hz'
 f.attrs['parallel_temperature_K']=2000.
 f.attrs['angle_from_perpendicular_degrees']=3.
 f.attrs['distribution']='gyrotropic radial phase-space Gaussian ring; parameter energy is not mean total energy'
 f['convergence_checks']=np.array(checks)
print('unstable candidates',len(found))
