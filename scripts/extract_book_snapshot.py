"""Extract an uncompressed native PIC snapshot and stable-ID tracks for a book figure."""
import sys,json,hashlib
import h5py
import numpy as np
source,dest=sys.argv[1:]
with h5py.File(source) as f:
    m=json.loads(bytes(f['metadata_json'][:])); s=json.loads(bytes(f['scales_json'][:]))
    c=json.loads(bytes(f['config_json'][:]))
    frame=2048; history=80; pool=m['recorded_particles_per_species']
    # Common history interval for both species, no fabricated or rescaled paths.
    p=f['positions_lambda_de'][frame-history:frame+2].astype(float)
    L=s['length']; now=p[history]-L/2
    chosen=[]
    for species in range(2):
        ids=np.flatnonzero(np.max(abs(now[species*pool:(species+1)*pool]),axis=1)<4.25)+species*pool
        # Spatially separated deterministic selection, not curvature-selected.
        selected=[]
        for i in ids:
            if all(np.linalg.norm(now[i]-now[j])>.95 for j in selected):selected.append(i)
            if len(selected)==20:break
        chosen.extend(selected)
    p=p[:,chosen]; delta=np.diff(p,axis=0); delta-=L*np.round(delta/L)
    track=np.empty_like(p);track[-2]=now[chosen]
    for j in range(history-1,-1,-1):track[j]=track[j+1]-delta[j]
    track[-1]=track[-2]+delta[-1]
    dt=8*s['dt_s']; velocity=(track[history+1]-track[history-1])*s['lambda_de']/(2*dt)
    phi=f['potential_kbte_e'][frame].astype(float)*c['te']*1.380649e-23/1.602176634e-19*1000
    with h5py.File(dest,'w') as out:
        out.create_dataset('tracks',data=track[:history+1]);out.create_dataset('velocity_m_s',data=velocity)
        out.create_dataset('species',data=np.array(chosen)//pool);out.create_dataset('ids',data=chosen)
        out.create_dataset('potential_mV',data=phi)
        out.attrs.update(frame=frame,history_steps=history,frame_dt_s=dt,lambda_de_m=s['lambda_de'],length_lambda_de=L,
            grid_spacing=L/c['cells'],crop_start=m['potential_start_node'],Te_K=c['te'],Ti_K=c['ti'],mass_ratio=29166.215780524162,
            source_capture_sha256=hashlib.sha256(open(source,'rb').read()).hexdigest(),velocity_method='centered position difference over 16 PIC steps',density_m3=c['density'])
print('Extracted',len(chosen),'real tracks; history ns',history*dt*1e9)
