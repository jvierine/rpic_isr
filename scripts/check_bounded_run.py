"""Verify the bounded run's phase and energy ledger; save HDF5 diagnostics."""
import json
import pathlib
import sys
import h5py
import numpy as np

p=pathlib.Path(sys.argv[1])
with h5py.File(p/'live.h5') as f, h5py.File(p/'spectrum.h5') as sf:
    assert bytes(f['status'][:])==b'complete'
    h=f['control_history'][:]; d=f['diagnostics'][:]
    c=json.loads(bytes(sf['config_json'][:]))
    control=json.loads(bytes(sf['control_0/metadata_json'][:]))
    assert np.isfinite(h).all() and np.isfinite(d).all()
    assert np.array_equal(h,sf['control_0/history'][:])
    assert c['te']==2000 and c['ti']==1000 and c['cells']==512
    assert control['elapsed_s']<7200
    assert control['activation_time_s']>=.001
    assert c['burn_ms']*.001-control['activation_time_s']>=.0005
    assert np.all(h[h[:,0]<control['activation_time_s'],10:12]==1)
    assert np.allclose(h[:,1]-h[:,2],h[:,5],rtol=0,atol=1e-12)
    assert np.allclose(np.cumsum(h[:,5]),h[:,6],rtol=0,atol=1e-12)
    assert np.allclose(h[:,2]+h[:,6],h[:,7],rtol=0,atol=1e-12)
    assert np.max(np.abs(h[:,8]))<1
    r=h[h[:,9]==2]
    assert np.all(np.abs(r[0,3:5]/[2000,1000]-1)<.01)
    result=dict(elapsed_s=control['elapsed_s'],record_ms=c['duration_ms'],
        raw_energy_change_percent=100*(d[-1,1]/d[0,1]-1),
        max_raw_energy_excursion_percent=100*np.max(np.abs(d[:,1]/d[0,1]-1)),
        max_work_corrected_energy_drift_percent=np.max(np.abs(h[:,8])),
        final_te_K=d[-1,5],final_ti_K=d[-1,6],
        record_te_change_percent=100*(r[-1,3]/r[0,3]-1),
        record_ti_change_percent=100*(r[-1,4]/r[0,4]-1),
        thermostat_removed_energy=h[-1,6],equilibrium_claim=False)
with h5py.File(p/'control_validation.h5','w') as out:
    for k,v in result.items():out.attrs[k]=v
    out.create_dataset('history',data=h)
print(json.dumps(result,indent=2))
