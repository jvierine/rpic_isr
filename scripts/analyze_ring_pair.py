"""Matched ring/control diagnostics; no peak normalization or forced growth fit."""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import welch

base=Path(__file__).resolve().parents[1]/'results/ring-pair-v1'
with h5py.File(base/'comparison.h5','w') as out:
 out.attrs['growth_fit_warning']='Total radar density-mode power, not isolated eigenmode; linear theory comparison is provisional.'
 for name in ['ring','control']:
  with h5py.File(base/name/'spectrum.h5') as f:
   s=json.loads(bytes(f['scales_json'][:]));dt=s['dt_s']
   z=f['raw/0/real'][:]+1j*f['raw/0/imag'][:];t=np.arange(len(z))*dt
   d=f['diagnostics/0'][:];p=f['population_energy_0'][:];temps=f['parallel_perpendicular_temperatures_K_0'][:]
   edges=np.arange(0,120.0001,.5)*1e-6
   power=np.array([np.mean(abs(z[(t>=a)&(t<b)])**2) for a,b in zip(edges[:-1],edges[1:])])
   centers=(edges[1:]+edges[:-1])/2
   group=out.create_group(name);group['time_s']=centers;group['half_us_mean_density_mode_power']=power
   group['diagnostics']=d;group['population_energy']=p;group['directional_temperatures_K']=temps
   fit=(centers>=20.5e-6)&(centers<23.5e-6)
   slope,intercept=np.polyfit(centers[fit],np.log(power[fit]),1)
   pred=slope*centers[fit]+intercept
   r2=1-np.sum((np.log(power[fit])-pred)**2)/np.sum((np.log(power[fit])-np.log(power[fit]).mean())**2)
   group.attrs['amplitude_growth_rate_s_inverse']=slope/2
   group.attrs['fit_log_power_r_squared']=r2
   group.attrs['fit_start_s']=20.5e-6;group.attrs['fit_end_s']=23.5e-6
   freq,ps=welch(z[t>=30e-6],fs=1/dt,nperseg=32768,return_onesided=False,detrend=False)
   group['frequency_hz']=freq;group['post30us_psd']=ps
   mask=(abs(freq)>2.7e6)&(abs(freq)<3.4e6)
   group.attrs['post30us_band_power']=np.sum(ps[mask])*(freq[1]-freq[0])
   group.attrs['max_work_corrected_energy_error_percent']=100*np.max(abs((d[:,1]-p[:,-1]-d[0,1])/d[0,1]))
   print(name,'gamma/2pi Hz',slope/(4*np.pi),'R2',r2,'bandpower',group.attrs['post30us_band_power'])
 ratio=out['ring'].attrs['post30us_band_power']/out['control'].attrs['post30us_band_power']
 out.attrs['ring_control_band_power_ratio']=ratio
 print('band enhancement dB',10*np.log10(ratio))
