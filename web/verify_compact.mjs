import fs from 'node:fs';
import path from 'node:path';
import assert from 'node:assert/strict';
import {initSync,Welch} from './public/pkg/picisr_web.js';
import {particlePoints} from './public/potential.js';
const root=process.argv[2];
const m=JSON.parse(fs.readFileSync(path.join(root,'metadata-lossy.json')));
const typed=(name,T)=>{const b=fs.readFileSync(path.join(root,name));return new T(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
assert.equal(m.positions_chunks.length,Math.ceil(m.frames/64));
assert.equal(m.potential_atlases.length,m.positions_chunks.length);
for(let c=0;c<m.positions_chunks.length;c++){
 const p=typed(m.positions_chunks[c],Uint16Array);
 assert.equal(p.length,64*2*m.pool_per_species*4);
 assert(fs.statSync(path.join(root,m.potential_atlases[c])).size>0);
 for(const f of [0,31.5,63]){
  const xy=particlePoints(p,{...m,frames:65},f,1000,true,true);
  assert(xy.length<=4000&&xy.length>0);
  for(let i=0;i<xy.length;i+=4)assert(xy[i]>=0&&xy[i]<1&&xy[i+1]>=0&&xy[i+1]<1);
 }
}
const signal=typed(m.signal_file,Float32Array);
assert.equal(signal.length,m.signal_samples*2);assert(signal.every(Number.isFinite));
initSync({module:fs.readFileSync(new URL('./public/pkg/picisr_web_bg.wasm',import.meta.url))});
const w=new Welch(m.default_fft,m.signal_dt_s);
for(let i=0;i+m.default_fft<=m.signal_samples;i+=m.default_fft/2)assert(w.add_window(signal.subarray(2*i,2*(i+m.default_fft))));
assert(w.segments()>0);assert(w.spectrum().every(Number.isFinite));
console.log(`PASS compact replay: ${m.frames} frames, ${m.animation_particles} display limit, ${(m.signal_samples*m.signal_dt_s*1e3).toFixed(3)} ms signal, ${w.segments()} Welch windows`);
w.free();
