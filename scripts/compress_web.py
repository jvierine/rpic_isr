"""Lossy scalar JPEG atlases for bounded-memory web replay; originals untouched.
Run after export_web.py. JPEG stores normalized scalar values, not baked colors.
"""
import hashlib
import json
import pathlib
import sys
import numpy as np
from PIL import Image

root = pathlib.Path(sys.argv[1])
m = json.loads((root / 'metadata.json').read_text())
assert not m.get('potential_atlases'), 'Use an original export as input'
n, g, pool = m['frames'], m['field_grid'], m['pool_per_species']
x = np.fromfile(root / m['positions_file'], '<u2').reshape(n, 2, pool, 2)
# Select at most 50 visible actual IDs/species/frame, as the original viewer
# does. Store their TRUE next-frame endpoints too: changing the display subset
# must never interpolate between two different particles.
small=np.full((n,2,50,4),65535,dtype='<u2')
for frame in range(n):
    distance=np.max(np.abs(x[frame].astype(float)/65535*m['length_lambda_de']-m['length_lambda_de']/2),axis=-1)
    for s in range(2):
        ids=np.flatnonzero(distance[s]<5)[:50]
        small[frame,s,:len(ids),:2]=x[frame,s,ids]
        small[frame,s,:len(ids),2:]=x[min(frame+1,n-1),s,ids]
name = 'positions-' + hashlib.sha256(small.tobytes()).hexdigest()[:12] + '.bin'
small.tofile(root / name)
phi = np.fromfile(root / m['potential_file'], '<i2').reshape(n, g, g)
# A fixed symmetric scale preserves zero and contrast across every chunk.
scale = m['phi_quant_scale']
encoded = np.rint((phi.astype(float)/32767 + 1)*127.5).clip(0,255).astype('uint8')
atlases=[]; errors=[]; maximum=0.; compressed=0
for start in range(0,n,64):
    frames=encoded[start:start+64]
    if len(frames)<64: frames=np.concatenate([frames,np.repeat(frames[-1:],64-len(frames),axis=0)])
    atlas=frames.reshape(8,8,g,g).transpose(0,2,1,3).reshape(8*g,8*g)
    filename=f'potential-jpeg-{start//64:03d}.jpg'
    Image.fromarray(atlas).save(root/filename,quality=95,subsampling=0)
    decoded=np.asarray(Image.open(root/filename)).astype(float)
    delta=(decoded-atlas.astype(float))*scale/127.5
    errors.append(np.mean(delta**2));maximum=max(maximum,float(np.max(np.abs(delta))))
    compressed+=(root/filename).stat().st_size;atlases.append(filename)
m.update(positions_file=name,pool_per_species=50,potential_atlases=atlases,
         positions_layout='frame-species-particle-xy-nextxy-u16',
         potential_chunk_frames=64,potential_codec='jpeg-scalar-95',
         potential_decode_scale=scale,animation_particles=100,
         compression=dict(bytes=compressed,rmse=float(np.sqrt(np.mean(errors))),
                          max_absolute_error=maximum,units='kB Te / e'))
if m['source'].get('control'):
    m['scientific_download']='downloads/high512-spectrum.h5'
    m['capture_download']='downloads/high512-capture.h5'
(root/'metadata-lossy.json').write_text(json.dumps(m,indent=2)+'\n')
print(json.dumps(dict(potential_bytes=compressed,positions_bytes=small.nbytes,**m['compression']),indent=2))
