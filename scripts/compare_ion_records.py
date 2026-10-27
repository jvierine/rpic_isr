"""Matched-window old/new web-voltage PSD checks. No scientific input changes."""
import json
from pathlib import Path
import h5py
import numpy as np
from scipy.signal import welch

records={}
for name,folder in [('old','high'),('new','bounded-high')]:
    root=Path('web/public/data')/folder
    m=json.loads((root/'metadata.json').read_text())
    v=np.fromfile(root/m['signal_file'],'<f4').reshape(-1,2)
    records[name]=(v[:,0]+1j*v[:,1],m['signal_dt_s'])
length=min(len(v[0]) for v in records.values())
with h5py.File('results/high512-2hour-8y3Ktw-complete/ion_comparison.h5','w') as out:
    for name,(v,dt) in records.items():
        for nfft in [8192,16384,32768]:
            f,p=welch(v[:length],fs=1/dt,window='hann',nperseg=nfft,noverlap=nfft//2,detrend=False,return_onesided=False)
            order=np.argsort(f);f=f[order];p=p[order]
            group=out.create_group(f'{name}/{nfft}')
            group.create_dataset('frequency_hz',data=f)
            group.create_dataset('psd',data=p)
            group.attrs['matched_samples']=length
            group.attrs['matched_duration_ms']=length*dt*1000
            group.attrs['segments']=1+(length-nfft)//(nfft//2)
            peak=[]
            for sign in [-1,1]:
                ids=np.flatnonzero((sign*f>1000)&(sign*f<20000))
                peak.append(float(f[ids[np.argmax(p[ids])]]))
            print(json.dumps(dict(record=name,nfft=nfft,segments=int(group.attrs['segments']),bin_hz=1/(nfft*dt),ion_peaks_hz=peak)))
