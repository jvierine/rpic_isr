"""Validate complete 1 ms ring-only sweep; retain physical windows and provenance.

Old thermal records support ONLY the matched 30--120 us comparison. They are
never used as a stationary baseline for the late or full 1 ms ring spectra.
"""
import sys,json
from pathlib import Path
import h5py,numpy as np
from scipy.signal import welch
b=Path(sys.argv[1]); old=Path(sys.argv[2]); out=Path(sys.argv[3])
with h5py.File(b/'manifest.h5') as m:
    assert np.all(m['complete'][:]==1)
    fps=m['plasma_frequency_hz'][:]; paths=m['paths'].asstr()[:]
with h5py.File(old/'manifest.h5') as m:
    controls={int(fp):path for fp,path,done in zip(m['plasma_frequency_hz'][:],m['paths'].asstr()[:,1],m['complete'][:,1]) if done}
windows={'full':(30,1000),'early':(30,250),'middle':(250,600),'late':(600,1000)}
freq=np.arange(-119,120)*50000.
def read(path):
    with h5py.File(path) as f:
        c=json.loads(bytes(f['config_json'][:]))
        if 'base' in c:c={**c.pop('base'),**c}
        s=json.loads(bytes(f['scales_json'][:]))
        z=f['raw/0/real'][:].astype(float)+1j*f['raw/0/imag'][:]
        assert len(z)==s['steps'] and np.isfinite(z).all()
        return c,s,z
def psd(z,dt,start,end):
    t=np.arange(len(z))*dt;n=round(20e-6/dt)
    f,p=welch(z[(t>=start*1e-6)&(t<end*1e-6)],fs=1/dt,nperseg=n,noverlap=n//2,detrend=False,return_onesided=False)
    assert np.isfinite(p).all() and np.all(p>0)
    ix=np.array([np.argmin(abs(f-x)) for x in freq])
    return p[ix]
with h5py.File(out,'w') as dst,h5py.File(str(out).replace('.h5','-matched.h5'),'w') as matched:
    dst['plasma_frequency_hz']=fps;dst['doppler_frequency_hz']=freq
    dst.attrs['ring_fraction']=.01;dst.attrs['segment_us']=20.;dst.attrs['seed']=9501
    dst.attrs['interpretation']='Transient single realization; late/full PSD has no matched long thermal baseline'
    arrays={name:[] for name in windows};metrics=[]
    for fp,path in zip(fps,paths):
        c,s,z=read(path);dt=s['dt_s']
        assert c['duration_ms']==1 and c['hot_fraction']==.01 and c['cells']==256 and c['ppc']==7
        assert c['te']==2000 and c['ti']==1000 and c['core_thermostat_us']==0
        assert c['ring_energy_ev']==10 and c['ring_turn_on_us']==20 and c['seed']==9501
        assert c['bx_t']==.000001869630096 and c['by_t']==.00003567491076
        assert np.isclose(c['density'],111639834702.32187*(fp/3e6)**2,rtol=1e-12)
        assert abs(len(z)*dt-.001)<=dt
        with h5py.File(path) as src:
            d=src['diagnostics/0'][:];e=src['population_energy_0'][:];inj=src['ring_injection_time_work_0'][:]
            assert np.isfinite(d).all() and np.isfinite(e).all()
            assert inj.shape==(1,2) and abs(inj[0,0]-20e-6)<=dt
            err=100*np.max(abs((d[:,1]-e[:,-1]-d[0,1])/d[0,1]));assert err<1
            raw=100*(d[-1,1]/d[0,1]-1);work=100*e[-1,-1]/d[0,1]
            metrics.append([err,raw,work,d[-1,5],d[-1,6]])
            g=dst.create_group('diagnostics/'+str(int(fp)))
            g['energy']=d;g['population_energy']=e;g.attrs['source']=path
        for name,(start,end) in windows.items():arrays[name].append(psd(z,dt,start,end))
        if int(fp) in controls:
            cc,ss,zz=read(controls[int(fp)])
            assert cc['hot_fraction']==0 and ss['dt_s']==dt
            g=matched.create_group('fp_'+str(int(fp)));g.attrs['plasma_frequency_hz']=fp
            for case,data in [('ring',z),('control',zz)]:
                q=g.create_group(case);q['frequency_hz']=freq;q['psd']=psd(data,dt,30,120)
        print(fp/1e6,'error/raw/injection percent, final Te/Ti',metrics[-1],flush=True)
    dst['metrics']=metrics
    dst.attrs['metric_columns']='max_corrected_energy_error_percent,raw_final_energy_change_percent,injection_work_percent,final_Te_K,final_Ti_K'
    for name,v in arrays.items():
        g=dst.create_group(name);g['psd']=np.array(v).T;g.attrs['interval_us']=windows[name]
    ratio=np.sum(np.array(arrays['late'])[:,abs(freq)>2e6],axis=1)/np.sum(np.array(arrays['early'])[:,abs(freq)>2e6],axis=1)
    dst['late_early_above_2mhz_power_ratio']=ratio
    print('late/early band power range',ratio.min(),ratio.max())
