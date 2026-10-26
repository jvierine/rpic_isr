// Deterministic offline video frames from authoritative browser replay assets.
// Scientific panels use the same Rust/Wasm/Rustplotlib renderer as the viewer.
// WebCodecs timestamps, not UI/screenshot timing, determine the frame cadence.
import init,{Replay,Welch,ScientificPlot,PlasmaView,ion_lowpass} from './pkg/picisr_web.js';
await init();
const output=document.querySelector('#output'),ctx=output.getContext('2d'),status=document.querySelector('#status'),button=document.querySelector('#render');
const W=1920,H=1080,FPS=30,SECONDS=30,bone='#e9e4d6',ink='#263437';
const canvas=()=>{const c=document.createElement('canvas');document.querySelector('#scratch').append(c);return c;};
const field=canvas(),particles=canvas(),plasma=new PlasmaView(field,particles);
const plots=Array.from({length:3},()=>{const axes=canvas(),lines=canvas();return {axes,lines,renderer:new ScientificPlot(axes,lines)};});
const colours={real:[.07,.42,.64],imag:[.71,.29,.09],paleReal:[.55,.73,.84],paleImag:[.85,.68,.54],spectrum:[.09,.33,.36],plasma:[.48,.16,.56]};
function series(p,xy,rgb,width=1){p.series(new Float64Array(xy),...rgb,width,false);}
function extrema(signal,component,dt,first,width=1800){
 const n=signal.length/2,step=Math.max(1,Math.ceil(n/width)),xy=[];
 for(let j=0;j<n;j+=step){let a=-1,b=-1;for(let k=j;k<Math.min(n,j+step);k++){const y=signal[2*k+component];if(!Number.isFinite(y))continue;if(a<0||y<signal[2*a+component])a=k;if(b<0||y>signal[2*b+component])b=k;}
  if(a<0)continue;for(const k of a<=b?[a,b]:[b,a])xy.push((first+k*dt)*1e3,signal[2*k+component]);}
 return xy;
}
function cachePlot(index,bounds,xlabel,ylabel,curves,height){
 const {axes,lines,renderer:p}=plots[index],width=490;
 p.begin(width,height,2,...bounds,xlabel,ylabel);
 for(const [xy,rgb,lineWidth] of curves)series(p,xy,rgb,lineWidth);
 const saved=document.createElement('canvas');saved.width=width*2;saved.height=height*2;
 const c=saved.getContext('2d');c.drawImage(axes,0,0);c.drawImage(lines,0,0);return saved;
}
function prepare(run){
 const {meta:m,signal}=run,n=m.default_fft,welch=new Welch(n,m.signal_dt_s);
 for(let j=0;j+n<=signal.length/2;j+=n/2)if(!welch.add_window(signal.subarray(2*j,2*(j+n))))throw Error('Invalid Welch window');
 const psd=welch.spectrum();run.segments=welch.segments();welch.free();
 let amplitude=0;for(const v of signal)amplitude=Math.max(amplitude,Math.abs(v));amplitude*=1.08;
 const slow=ion_lowpass(signal,m.signal_dt_s,30000),duration=(m.signal_first_s+(signal.length/2-1)*m.signal_dt_s)*1e3;
 run.duration=duration;run.amplitude=amplitude;
 run.voltage=cachePlot(0,[0,duration,-amplitude,amplitude],'Time after burn-in [ms]','Normalized amplitude',[
  [extrema(signal,0,m.signal_dt_s,m.signal_first_s),colours.paleReal,.7],[extrema(signal,1,m.signal_dt_s,m.signal_first_s),colours.paleImag,.7],
  [extrema(slow,0,m.signal_dt_s,m.signal_first_s),colours.real,1.5],[extrema(slow,1,m.signal_dt_s,m.signal_first_s),colours.imag,1.5]],130);
 for(const [key,index,unit,band,span,rgb] of [['ion',1,1000,30,75,colours.spectrum],['wide',2,1e6,m.plasma_band_mhz,80,colours.plasma]]){
  const xy=[];let peak=-200;for(let j=0;j<n;j++){const f=(j-n/2)/(n*m.signal_dt_s)/unit;if(Math.abs(f)<=band){const y=10*Math.log10(Math.max(psd[j],1e-20));xy.push(f,y);peak=Math.max(peak,y);}}
  const top=Math.ceil(peak/5)*5+5;run[key]=cachePlot(index,[-band,band,top-span,top],`Frequency offset [${unit===1000?'kHz':'MHz'}]`,'PSD [dB re 1 Hz⁻¹]',[[xy,rgb,.9]],120);
 }
}
async function load(id,path){
 const url=new URL(path,location.href),m=await(await fetch(url)).json();
 const binary=async name=>{const r=await fetch(new URL(m[name],url));if(!r.ok)throw Error(`Missing ${name}`);return r.arrayBuffer();};
 const [x,p,s]=await Promise.all(['positions_file','potential_file','signal_file'].map(binary));
 const run={id,meta:m,signal:new Float32Array(s),replay:new Replay(new Uint16Array(x),new Int16Array(p),m.pool_per_species,m.field_grid,m.length_lambda_de,10,m.phi_quant_scale)};
 prepare(run);return run;
}
function text(message,x,y,size=26,colour=ink){ctx.fillStyle=colour;ctx.font=`${size}px Arial`;ctx.fillText(message,x,y);}
function draw(run,j){
 const m=run.meta,frame=16+j*5/FPS,t=frame*m.frame_dt_s,periods=t*m.omega_pe_rad_s/(2*Math.PI),high=run.id==='high';
 ctx.fillStyle=bone;ctx.fillRect(0,0,W,H);
 text('Collective Thomson scattering',35,52,40);text(`nₑ = ${high?'10¹¹':'10¹⁰'} m⁻³    Initial Tₑ = 2000 K, Tᵢ = 1000 K    O⁺ + electrons`,35,98,27);
 plasma.draw(run.replay.field(frame),run.replay.points(frame,100,true,true),800,800,1,m.potential_length_lambda_de,m.phi_display_scale/.3,true);
 ctx.drawImage(field,35,145);ctx.drawImage(particles,35,145);
 text('Electric potential · 10 × 10 electron Debye lengths',35,133,23);
 text('● electrons',55,975,24,'#087c9e');text('● oxygen ions',235,975,24,'#9b5b00');
 text(`t = ${(t*1e9).toFixed(1)} ns = ${periods.toFixed(2)} plasma periods`,55,1010,24);
 // One-tenth of the square camera width is one electron Debye length.
 ctx.strokeStyle=ink;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(715,915);ctx.lineTo(715,925);ctx.lineTo(795,925);ctx.lineTo(795,915);ctx.stroke();text('1 λDe',727,905,23);
 text('Scattered complex amplitude',900,134,27);text('Re',1530,134,24,'#126ca4');text('Im',1590,134,24,'#b64a16');text('30 kHz low-pass: thick',1660,134,20);
 ctx.drawImage(run.voltage,900,150);
 const markerX=900+140+(980-176)*(t*1e3/run.duration);ctx.strokeStyle='#356f59';ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(markerX,210);ctx.lineTo(markerX,314);ctx.stroke();
 text(`Full ${run.duration.toFixed(2)} ms voltage record; particle movie is a short excerpt`,920,440,22);
 text('Ion-line spectrum',900,475,28);ctx.drawImage(run.ion,900,495);
 text('Full spectrum: ion line + plasma lines',900,770,28);ctx.drawImage(run.wide,900,785);
 const third=Math.floor(j/(FPS*10));
 const caption=third===0?(high?'Higher density raises the electron plasma frequency.':'Electrons scatter the radar wave; their phases add to form the complex signal.'):
  third===1?'Langmuir waves give the fast electron-frequency response and the plasma lines.':
  'Slower ion-acoustic fluctuations are easier to follow in the low-pass complex amplitude.';
 text(caption,35,1060,24);
}
let runs;
button.onclick=async()=>{
 button.disabled=true;const chunks=[];let failure;
 try{
  const config={codec:'avc1.640028',width:W,height:H,framerate:FPS,bitrate:8000000,latencyMode:'realtime',avc:{format:'annexb'}};
  const supported=await VideoEncoder.isConfigSupported(config);if(!supported.supported)throw Error('H.264 WebCodecs encoder unavailable');
  const encoder=new VideoEncoder({output:chunk=>{const bytes=new Uint8Array(chunk.byteLength);chunk.copyTo(bytes);chunks.push(bytes);},error:e=>{failure=e;}});encoder.configure(config);
  let number=0;
  for(const run of runs)for(let j=0;j<FPS*SECONDS;j++){
   if(failure)throw failure;draw(run,j);
   const frame=new VideoFrame(output,{timestamp:Math.round(number*1e6/FPS),duration:Math.round(1e6/FPS)});encoder.encode(frame,{keyFrame:number%60===0});frame.close();number++;
   document.querySelector('#progress').value=number;
   if(encoder.encodeQueueSize>8)await encoder.flush();
   if(number%15===0){status.textContent=`Rendering ${number}/1800 frames at fixed 30 fps`;await new Promise(requestAnimationFrame);}
  }
  await encoder.flush();encoder.close();if(failure)throw failure;
  const link=document.querySelector('#download');link.href=URL.createObjectURL(new Blob(chunks,{type:'video/h264'}));link.download='collective-thomson-scattering-preview.h264';link.hidden=false;
  status.textContent='Complete: 1800 frames, 60 seconds, fixed 30 fps. Not uploaded.';
 }catch(e){status.textContent=String(e.message||e);console.error(e);}finally{button.disabled=false;}
};
try{
 await init();runs=[];
 for(const [id,path] of [['low','data/extended-low/metadata.json'],['high','data/high/metadata.json']]){status.textContent=`Loading and calculating ${id}-density spectra…`;runs.push(await load(id,path));}
 draw(runs[0],0);button.disabled=false;status.textContent='Ready. Offline rendering preserves every output frame; no screen-recording drops.';
}catch(e){status.textContent=String(e.message||e);console.error(e);}
