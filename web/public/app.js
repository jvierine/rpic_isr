import init, {Replay, Welch, ScientificPlot, PlasmaView} from './pkg/picisr_web.js?v=9ddd504a6ec4';
const $=id=>document.getElementById(id);
let meta, replay, signal, spectrum=[], running=true, frame=16, last=0, pending=0, welch;
const state={ready:false,frame:16,shown:0,segments:0};
// Small browser-QA surface: no file paths or generator metadata are rendered.
window.picisrState=state;
const status=(text,error=false)=>{$('statusText').textContent=text;$('status').classList.toggle('error',error);};
const dialog=$('help');$('helpOpen').onclick=()=>dialog.showModal();
$('plotsToggle').onclick=()=>{const panel=$('plots');panel.hidden=!panel.hidden;$('plotsToggle').setAttribute('aria-expanded',String(!panel.hidden));if(!panel.hidden){drawSignal();drawSpectrum();}};
$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen();};
$('helpClose').onclick=()=>dialog.close();
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});

const canvas=$('plasma');let plasmaView;
function renderPlasma(){
 const points=replay.points(frame,Number($('count').value),$('electrons').checked,$('ions').checked);
 const r=canvas.getBoundingClientRect();
 plasmaView.draw(replay.field(frame),points,r.width,r.height,Math.min(devicePixelRatio||1,2),meta.length_lambda_de,meta.phi_display_scale/Number($('contrast').value),$('potential').checked);
 let ne=0,ni=0;for(let j=2;j<points.length;j+=4){if(points[j]<.5)ne++;else ni++;}
 $('shown').textContent=`${ne} electrons · ${ni} ions`;
 state.frame=frame;state.shown=ne+ni;state.electrons=ne;state.ions=ni;
 $('clock').textContent=`t = ${(meta.burn_ms+frame*meta.frame_dt_s*1e3).toFixed(6)} ms`;
 $('scrub').value=frame;
 const v=meta.phi_display_scale/Number($('contrast').value)*meta.potential_to_mv;
 $('phiMinus').textContent=`−${v.toFixed(1)} mV`;$('phiPlus').textContent=`+${v.toFixed(1)} mV`;
}
const plotRenderers=new Map();
function chart(c,xmin,xmax,ymin,ymax,xlabel,ylabel){
 if(!plotRenderers.has(c.id))plotRenderers.set(c.id,new ScientificPlot(c,$(`${c.id}Lines`)));
 const p=plotRenderers.get(c.id),r=c.getBoundingClientRect();
 p.begin(r.width,r.height,Math.min(devicePixelRatio||1,2),xmin,xmax,ymin,ymax,xlabel,ylabel);return p;
}
function curve(plot,values,colour){const rgb=colour.match(/\w\w/g).map(v=>parseInt(v,16)/255);plot.series(new Float64Array(values.flat()),...rgb,1.25,false);}
let waveform;
function prepareWaveform(){const duration=(meta.frames-1)*meta.frame_dt_s;const n=Math.min(signal.length/2,Math.floor((duration-meta.signal_first_s)/meta.signal_dt_s));let amplitude=0;const real=[],imag=[];for(let j=0;j<n;j++){const t=(meta.signal_first_s+j*meta.signal_dt_s)*1e6;real.push([t,signal[2*j]]);imag.push([t,signal[2*j+1]]);amplitude=Math.max(amplitude,Math.abs(signal[2*j]),Math.abs(signal[2*j+1]));}waveform={real,imag,amplitude:amplitude*1.08,duration};}
function drawSignal(){if(!waveform||$('plots').hidden)return;const p=chart($('signal'),0,waveform.duration*1e6,-waveform.amplitude,waveform.amplitude,'Time after burn-in [µs]','Normalized amplitude');curve(p,waveform.real,'#126ca4');curve(p,waveform.imag,'#b64a16');const t=frame*meta.frame_dt_s*1e6;p.series(new Float64Array([t,-waveform.amplitude,t,waveform.amplitude]),.1,.4,.3,1,false);}
function drawSpectrum(){if(!meta||$('plots').hidden)return;const band=Number($('band').value),n=spectrum.length;let points=[];for(let j=0;j<n;j++){const f=(j-n/2)/(n*meta.signal_dt_s)/1000;if(Math.abs(f)<=band)points.push([f,10*Math.log10(Math.max(spectrum[j],1e-20))]);}const max=points.length?Math.max(...points.map(v=>v[1])):-40;const ymax=Math.ceil(max/10)*10+5,ymin=ymax-55;const p=chart($('spectrum'),-band,band,ymin,ymax,'Frequency offset [kHz]','PSD [dB re 1 Hz⁻¹]');if(points.length)curve(p,points,'#18555c');}
async function calculate(){
 if(!signal)return;const token=++pending;if(welch)welch.free();
 const n=Number($('fft').value);welch=new Welch(n,meta.signal_dt_s);spectrum=[];drawSpectrum();
 for(let j=0;j+n<=signal.length/2;j+=n/2){if(token!==pending)return;welch.add_window(signal.subarray(2*j,2*(j+n)));spectrum=welch.spectrum();state.segments=welch.segments();$('segments').textContent=`${state.segments} Hann windows`;
 $('specCaption').textContent=`Computed here from the full ${(meta.signal_samples*meta.signal_dt_s*1e3).toFixed(2)} ms record · Δf = ${(1/(n*meta.signal_dt_s)/1000).toFixed(2)} kHz · 50% overlap`;
 drawSpectrum();await new Promise(r=>setTimeout(r,100));}
}
function animate(now){if(state.ready){if(running&&last){frame+=Math.min((now-last)/1000,.1)*20*Number($('speed').value);if(frame>=meta.frames-1.001){frame=meta.frames-1.001;running=false;$('play').textContent='Play';$('playtime').textContent='End of recording';}}renderPlasma();drawSignal();}last=now;requestAnimationFrame(animate);}
$('play').onclick=()=>{if(frame>=meta.frames-1.01)frame=16;running=!running;$('play').textContent=running?'Pause':'Play';$('playtime').textContent=running?'Recorded time, slowed down':'Paused';};
$('restart').onclick=()=>{frame=16;running=true;$('play').textContent='Pause';$('playtime').textContent='Recorded time, slowed down';};
$('scrub').oninput=()=>{frame=Number($('scrub').value);running=false;$('play').textContent='Play';$('playtime').textContent='Paused';};
$('count').oninput=()=>{$('countLabel').textContent=$('count').value;};
$('contrast').oninput=()=>{$('contrastLabel').textContent=`${Number($('contrast').value).toFixed(1)}×`;};
$('band').onchange=drawSpectrum;$('fft').onchange=calculate;$('recalculate').onclick=calculate;
window.addEventListener('resize',()=>{drawSpectrum();drawSignal();});
canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();state.ready=false;status('The graphics context was lost. Reload to restore the recording.',true);});
async function loadBinary(url){const r=await fetch(url);if(!r.ok)throw Error(`Recording download failed (${r.status})`);return r.arrayBuffer();}
try{
 await init({module_or_path:new URL('./pkg/picisr_web_bg.wasm?v=15b074a7f9bb',import.meta.url)});plasmaView=new PlasmaView(canvas,$('particles'));const response=await fetch('data/metadata.json');if(!response.ok)throw Error('Recording metadata is not available');meta=await response.json();
 const [previewPositions,previewPotential]=await Promise.all([loadBinary(`data/${meta.preview_positions_file}`),loadBinary(`data/${meta.preview_potential_file}`)]);
 replay=new Replay(new Uint16Array(previewPositions),new Int16Array(previewPotential),meta.pool_per_species,meta.field_grid,meta.length_lambda_de,10,meta.phi_quant_scale);
 frame=0;running=false;state.ready=true;requestAnimationFrame(animate);
 status('Post-burn-in particles shown. Loading the motion recording…');
 const [positions,potential]=await Promise.all([loadBinary(`data/${meta.positions_file}`),loadBinary(`data/${meta.potential_file}`)]);
 replay.free();
 replay=new Replay(new Uint16Array(positions),new Int16Array(potential),meta.pool_per_species,meta.field_grid,meta.length_lambda_de,10,meta.phi_quant_scale);
 status('Loading the full-population scattered signal…');signal=new Float32Array(await loadBinary(`data/${meta.signal_file}`));
 if(signal.length!==meta.signal_samples*2)throw Error('Incomplete signal recording');prepareWaveform();
 $('energyDrift').textContent=`${meta.max_energy_excursion_percent.toFixed(3)}% maximum energy excursion`;
 $('frameCadence').textContent=`${(meta.frame_dt_s*1e9).toFixed(2)} ns (8 PIC steps)`;
 $('playbackScale').textContent=`${(20*meta.frame_dt_s*1e6).toFixed(2)} µs of plasma time`;
 $('recordLength').textContent=`${(meta.signal_samples*meta.signal_dt_s*1e3).toFixed(2)} ms`;
 for(const id of ['play','restart','scrub','recalculate'])$(id).disabled=false;
 frame=16;running=true;state.ready=true;state.totalParticles=meta.particles_per_species*2;state.clipDurationUs=(meta.frames-1)*meta.frame_dt_s*1e6;
 status('Recorded GPU data · 10 ms burn-in · display controls do not change the physics');$('status').classList.add('ready');
 drawSpectrum();calculate();
}catch(e){console.error(e);status(e.message,true);}
