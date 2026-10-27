"""Verify 2D3V CPU/CUDA regression or a production record. HDF5 only."""
import json,sys
import h5py
import numpy as np
def check(path):
 with h5py.File(path) as f:
  c=json.loads(bytes(f['config_json'][:]));s=json.loads(bytes(f['scales_json'][:]))
  if 'base' in c: c={**c.pop('base'),**c}
  d=f['diagnostics/0'][:];p=f['population_energy_0'][:];t=f['parallel_perpendicular_temperatures_K_0'][:]
  for name in ['raw/0/real','raw/0/imag','trajectories/0/v_m_s','trajectories/0/x_m']:
   assert np.isfinite(f[name][:]).all(),name
  assert f['trajectories/0/v_m_s'].shape[-1]==3
  assert f['trajectories/0/x_m'].shape[-1]==2
  assert len(f['raw/0/real'])==s['steps']
  assert np.isfinite(d).all() and np.isfinite(p).all() and np.isfinite(t).all()
  error=(d[:,1]-p[:,-1]-d[0,1])/d[0,1]
  assert max(abs(error))<.01
  assert np.allclose(d[:,1],p[:,1:4].sum(axis=1)+d[:,4])
  assert np.allclose(t[:,::2]/3+2*t[:,1::2]/3,p[:,4:7],rtol=1e-5,atol=1e-5)
  assert abs(p[0,4]-c['te'])<.1 and abs(p[0,6]-c['ti'])<.1
  if c['hot_fraction']>0: assert abs(p[0,5]-2*c['hot_energy_ev']*1.602176634e-19/(3*1.380649e-23))<1
  recording=p[:,0]>=s['burn_steps']*s['dt_s']
  if recording.any(): assert np.ptp(p[recording,-1])<1e-12,'thermostat active during recording'
  print(path,'energy excursion %',100*max(abs(error)),'final core/tail/ion K',p[-1,4:7])
for name in sys.argv[1:]:check(name)
if len(sys.argv)==3:
 with h5py.File(sys.argv[1]) as a,h5py.File(sys.argv[2]) as b:
  for name in ['raw/0/real','raw/0/imag','trajectories/0/x_m','trajectories/0/v_m_s','population_energy_0']:
   x=a[name][:];y=b[name][:];err=np.linalg.norm(x-y)/max(np.linalg.norm(x),1e-20)
   print(name,'relative L2',err);assert err<1e-3
