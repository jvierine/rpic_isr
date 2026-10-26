// Numerical tests of the exact built Wasm and exported scientific transports.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {initSync, Replay, Welch} from './public/pkg/picisr_web.js';
initSync({module:fs.readFileSync(new URL('./public/pkg/picisr_web_bg.wasm',import.meta.url))});
const root=new URL('./public/data/',import.meta.url);
const m=JSON.parse(fs.readFileSync(new URL('metadata.json',root),'utf8'));
const typed=(file,Type)=>{const b=fs.readFileSync(new URL(file,root));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const x=typed(m.positions_file,Uint16Array),phi=typed(m.potential_file,Int16Array),s=typed(m.signal_file,Float32Array);
const r=new Replay(x,phi,m.pool_per_species,m.field_grid,m.length_lambda_de,10,m.phi_quant_scale);
for(const frame of [16,100.5,700.25,1022.9]){
 for(const count of [10,100,1000]){
  const points=r.points(frame,count,true,true);assert.equal(points.length,count*4);
  for(let j=0;j<points.length;j+=4){assert(points[j]>=0&&points[j]<1&&points[j+1]>=0&&points[j+1]<1);}
 }
 assert.equal(r.points(frame,100,false,true).length,400);
 const a=r.field(Math.floor(frame)),b=r.field(Math.floor(frame)+1),mixed=r.field(frame),t=Math.fround(frame)%1;
 for(let j=0;j<a.length;j++)assert(Math.abs(mixed[j]-((1-t)*a[j]+t*b[j]))<1e-6);
}
const w=new Welch(8192,m.signal_dt_s);
for(let start=0;start+8192<=s.length/2;start+=4096)assert(w.add_window(s.subarray(2*start,2*(start+8192))));
const p=w.spectrum();assert(p.every(v=>Number.isFinite(v)&&v>=0));
const spectralPower=p.reduce((a,b)=>a+b,0)/(8192*m.signal_dt_s);
const rawPower=s.reduce((a,b)=>a+b*b,0)/(s.length/2);
assert(Math.abs(spectralPower/rawPower-1)<.25);
const plasmaPeaksMHz=[];
for(const sign of [-1,1]){
 let peak=-1,power=-Infinity;
 for(let j=0;j<p.length;j++){const f=(j-p.length/2)/(p.length*m.signal_dt_s)/1e6;if(sign*f>.6&&sign*f<2&&p[j]>power){power=p[j];peak=f;}}
 assert(Number.isFinite(power)&&power>0);assert(Math.abs(peak)>.7&&Math.abs(peak)<1.7);
 plasmaPeaksMHz.push(peak);
}
console.log(JSON.stringify({passed:true,frames:m.frames,displayCounts:[10,100,1000],segments:w.segments(),spectralPower,rawPower,plasmaPeaksMHz}));
