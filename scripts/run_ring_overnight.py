"""Two GPU workers, fixed-B 1% ring density sweep; bounded overnight budget.

Each case is independent. No thermal or 0.1% runs. HDF5 manifest records
completed points; pending points are not silently treated as complete.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import h5py
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('binary',type=Path)
p.add_argument('output',type=Path)
p.add_argument('--hours',type=float,default=12)
p.add_argument('--duration-ms',type=float,default=8)
a=p.parse_args()
assert a.duration_ms>0.02 and a.hours>0
a.output.mkdir()
sha=hashlib.sha256(a.binary.read_bytes()).hexdigest()
assert sha=='f120fc249c52cfdb15667977bd310f773bd188092caf54b636979a68f3b98a00'
fps=np.arange(2500,3501,50)*1000.
order=sorted(range(len(fps)),key=lambda i:(abs(fps[i]-3e6),fps[i]))
lock=threading.Lock()
deadline=time.monotonic()+a.hours*3600
manifest=a.output/'manifest.h5'
with h5py.File(manifest,'w') as f:
    f['plasma_frequency_hz']=fps
    f['density_m3']=111639834702.32187*(fps/3e6)**2
    f['complete']=np.zeros(len(fps),dtype='u1')
    f['paths']=np.full(len(fps),'',dtype=object).astype(h5py.string_dtype())
    f['energy_error_percent']=np.full(len(fps),np.nan)
    f.attrs.update(duration_ms=a.duration_ms,ring_fraction=.01,ring_turn_on_us=20.,
                   ring_energy_ev=10.,wall_budget_hours=a.hours,binary_sha256=sha)

def worker(device):
    while True:
        with lock:
            if not order: return
            i=order[0]
            # Measured 8 ms cases need 5--6 h; reserve seven, scaled with duration.
            if deadline-time.monotonic()<7*3600*a.duration_ms/8:
                print(f'GPU {device}: insufficient overnight budget for another full case; pending points retained',flush=True)
                return
            order.pop(0)
        directory=a.output/f'fp-{int(fps[i]/1000)}khz'
        directory.mkdir()
        density=111639834702.32187*(fps[i]/3e6)**2
        path=directory/'ring.h5'
        cmd=[str(a.binary),'--backend','cuda','--device',str(device),
             '--cells','256','--ppc','7','--bragg-mode','1','--density',str(density),
             '--te','2000','--ti','1000','--bx-t','.000001869630096',
             '--by-t','.00003567491076','--hot-fraction','.01','--ring-energy-ev','10',
             '--ring-width','.1','--ring-turn-on-us','20','--core-thermostat-us','0',
             '--dt','.01','--burn-ms','0','--duration-ms',str(a.duration_ms),'--sample-us','.005',
             '--window','1024','--diagnostic-stride','200','--seed','9501','--output',str(path)]
        env={k:v for k,v in os.environ.items() if not k.startswith('RPIC_WEB_')}
        env['RPIC_LIVE_PATH']=str(directory/'live.h5')
        print(f'START GPU{device} fpe={fps[i]/1e6:.2f}MHz 1% ring {a.duration_ms:g}ms',flush=True)
        with (directory/'run.log').open('w') as log:
            result=subprocess.run(['timeout','--signal=TERM','--kill-after=5s',
                                   str(int(deadline-time.monotonic())),*cmd],
                                  env=env,stdout=log,stderr=subprocess.STDOUT)
        (directory/'exit').write_text(str(result.returncode)+'\n')
        result.check_returncode()
        with h5py.File(path) as f:
            config=json.loads(bytes(f['config_json'][:]))
            if 'base' in config: config={**config.pop('base'),**config}
            assert config['hot_fraction']==.01 and config['duration_ms']==a.duration_ms
            assert np.isclose(config['density'],density)
            scales=json.loads(bytes(f['scales_json'][:]))
            for key in ['raw/0/real','raw/0/imag']:
                ds=f[key]; assert len(ds)==scales['steps']
                for start in range(0,len(ds),1000000):
                    assert np.isfinite(ds[start:start+1000000]).all()
            d=f['diagnostics/0'][:];e=f['population_energy_0'][:]
            assert np.isfinite(d).all() and np.isfinite(e).all()
            error=100*np.max(abs((d[:,1]-e[:,-1]-d[0,1])/d[0,1]))
            assert error<1,f'Numerical energy error {error}%'
        with lock,h5py.File(manifest,'r+') as f:
            f['paths'][i]=str(path); f['energy_error_percent'][i]=error; f['complete'][i]=1
        print(f'PASS fpe={fps[i]/1e6:.2f}MHz corrected energy error {error:.6g}%',flush=True)

with ThreadPoolExecutor(max_workers=2) as pool:
    jobs=[pool.submit(worker,gpu) for gpu in [0,1]]
    for job in jobs: job.result()
with h5py.File(manifest) as f:
    print(f'FINISHED {int(f["complete"][:].sum())}/{len(fps)} points within wall budget',flush=True)
