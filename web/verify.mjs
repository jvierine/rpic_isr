// Numerical tests of the exact built Wasm and exported scientific transports.
import fs from 'node:fs';
import assert from 'node:assert/strict';
import {initSync, Replay, Welch, PlotViewport} from './public/pkg/picisr_web.js';
initSync({module:fs.readFileSync(new URL('./public/pkg/picisr_web_bg.wasm',import.meta.url))});
const nav=new PlotViewport();nav.zoom(.5,.5,.7);
assert(Math.abs(nav.bounds()[2]-.85)<1e-12);nav.pan(.1,0);
assert(nav.bounds()[0]>.15);assert(nav.zoom_box(.1,.2,.8,.9));
nav.reset();assert.deepEqual(Array.from(nav.bounds()),[0,0,1,1]);nav.free();
const root=new URL(process.argv[2]||'./public/data/',import.meta.url);
const m=JSON.parse(fs.readFileSync(new URL('metadata.json',root),'utf8'));
const typed=(file,Type)=>{const b=fs.readFileSync(new URL(file,root));return new Type(b.buffer.slice(b.byteOffset,b.byteOffset+b.byteLength));};
const x=typed(m.positions_file,Uint16Array),phi=typed(m.potential_file,Int16Array),s=typed(m.signal_file,Float32Array);
const r=new Replay(x,phi,m.pool_per_species,m.field_grid,m.length_lambda_de,10,m.phi_quant_scale);
for(const frame of [16,100.5,700.25,1022.9]){
 const available=r.points(frame,2*m.pool_per_species,true,true);let ne=0,ni=0;
 for(let j=2;j<available.length;j+=4){if(available[j]<.5)ne++;else ni++;}
 for(const count of [10,100,1000]){
  const points=r.points(frame,count,true,true);assert.equal(points.length,4*(Math.min(ne,count/2)+Math.min(ni,count/2)));
  for(let j=0;j<points.length;j+=4){assert(points[j]>=0&&points[j]<1&&points[j+1]>=0&&points[j+1]<1);}
 }
 assert.equal(r.points(frame,100,false,true).length,4*Math.min(ni,100));
 const a=r.field(Math.floor(frame)),b=r.field(Math.floor(frame)+1),mixed=r.field(frame),t=Math.fround(frame)%1;
 for(let j=0;j<a.length;j++)assert(Math.abs(mixed[j]-((1-t)*a[j]+t*b[j]))<1e-6);
}
const n=m.default_fft||8192,w=new Welch(n,m.signal_dt_s);
for(let start=0;start+n<=s.length/2;start+=n/2)assert(w.add_window(s.subarray(2*start,2*(start+n))));
const p=w.spectrum();assert(p.every(v=>Number.isFinite(v)&&v>=0));
const spectralPower=p.reduce((a,b)=>a+b,0)/(n*m.signal_dt_s);
const rawPower=s.reduce((a,b)=>a+b*b,0)/(s.length/2);
assert(Math.abs(spectralPower/rawPower-1)<.25);
const plasmaPeaksMHz=[];
const fpe=(m.omega_pe_rad_s||5641460.225)/(2*Math.PI*1e6),band=m.plasma_band_mhz||2;
const expected=fpe*Math.sqrt(1+3*(m.k_lambda_de||.5694555)**2);
for(const sign of [-1,1]){
 let peak=-1,power=-Infinity;
 for(let j=0;j<p.length;j++){const f=(j-p.length/2)/(p.length*m.signal_dt_s)/1e6;if(sign*f>.7*fpe&&sign*f<band&&p[j]>power){power=p[j];peak=f;}}
 assert(Number.isFinite(power)&&power>0);assert(Math.abs(Math.abs(peak)/expected-1)<.4);
 plasmaPeaksMHz.push(peak);
}
console.log(JSON.stringify({passed:true,density_m3:m.density_m3,frames:m.frames,displayLimits:[10,100,1000],fft:n,segments:w.segments(),spectralPower,rawPower,plasmaPeaksMHz}));
