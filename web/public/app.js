import init, {Replay, Welch, ScientificPlot, PlasmaView} from './pkg/picisr_web.js?v=652fb638f3c7';
import {ProgressDownloads} from './loading.js?v=ebb138dbfd70';
import {simulationClock} from './annotations.js?v=e0a325e6b590';
import {bindPlotNavigation} from './navigation.js?v=cf02acb86355';
const $=id=>document.getElementById(id);
let meta, replay, signal, spectrum=[], running=true, frame=16, last=0, pending=0, welch;
const state={ready:false,frame:16,shown:0,segments:0};
// Small browser-QA surface: no file paths or generator metadata are rendered.
window.picisrState=state;
const status=(text,error=false)=>{$('statusText').textContent=text;$('status').classList.toggle('error',error);};
let downloads,runList=[],wasmLoaded=false,loadSerial=0;
const makeDownloads=()=>new ProgressDownloads(({loaded,total,fraction})=>{
 $('loadingProgress').value=Math.min(fraction*100,99.9);
 $('loadingDetails').textContent=`${Math.floor(fraction*100)}% · ${(loaded/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`;
});
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
 const limit=Number($('count').value);$('countLabel').textContent=ne+ni<limit?`${ne+ni}/${limit}`:String(limit);
 $('clock').textContent=`t = ${(meta.burn_ms+frame*meta.frame_dt_s*1e3).toFixed(6)} ms`;
 const time=simulationClock(frame,meta.frame_dt_s,meta.omega_pe_rad_s);
 $('timeNs').textContent=time.nanoseconds.toFixed(1);$('timePeriods').textContent=time.plasmaPeriods.toFixed(2);
 $('scrub').value=frame;
 const v=meta.phi_display_scale/Number($('contrast').value)*meta.potential_to_mv;
 $('phiMinus').textContent=`−${v.toFixed(1)} mV`;$('phiPlus').textContent=`+${v.toFixed(1)} mV`;
}
const plotRenderers=new Map();
function chart(c,xmin,xmax,ymin,ymax,xlabel,ylabel){
 if(!plotRenderers.has(c.id)){
  const overlay=$(`${c.id}Lines`),plot=new ScientificPlot(c,overlay);
  plotRenderers.set(c.id,plot);
  const redraw=c.id==='signal'?drawSignal:c.id==='plasmaLine'?drawPlasmaLine:drawSpectrum;
  bindPlotNavigation(overlay,plot,redraw,()=>({x:70,y:30,width:overlay.clientWidth-88,height:overlay.clientHeight-78}));
  const controls=document.createElement('div');controls.className='plot-navigation';
  for(const [label,title,action] of [ ['+','Zoom in',()=>plot.zoom(.5,.5,.7)],['−','Zoom out',()=>plot.zoom(.5,.5,1/.7)],['Reset','Reset view',()=>plot.reset()] ]){
   const b=document.createElement('button');b.textContent=label;b.title=title;b.setAttribute('aria-label',`${c.id} ${title}`);b.onclick=()=>{action();redraw();};controls.append(b);
  }
  c.parentElement.append(controls);
 }
 const p=plotRenderers.get(c.id),r=c.getBoundingClientRect();
 p.begin(r.width,r.height,Math.min(devicePixelRatio||1,2),xmin,xmax,ymin,ymax,xlabel,ylabel);
 const view=Array.from(p.viewport_bounds());c.dataset.viewport=JSON.stringify(view);
 c.setAttribute('aria-label',`${xlabel}: ${(xmin+view[0]*(xmax-xmin)).toPrecision(4)} to ${(xmin+view[2]*(xmax-xmin)).toPrecision(4)}; ${ylabel}: ${(ymin+view[1]*(ymax-ymin)).toPrecision(4)} to ${(ymin+view[3]*(ymax-ymin)).toPrecision(4)}`);return p;
}
function curve(plot,values,colour){const rgb=colour.match(/\w\w/g).map(v=>parseInt(v,16)/255);plot.series(new Float64Array(values.flat()),...rgb,1.25,false);}
let waveform;
function prepareWaveform(){const duration=(meta.frames-1)*meta.frame_dt_s;const n=Math.min(signal.length/2,Math.floor((duration-meta.signal_first_s)/meta.signal_dt_s));let amplitude=0;const real=[],imag=[];for(let j=0;j<n;j++){const t=(meta.signal_first_s+j*meta.signal_dt_s)*1e6;real.push([t,signal[2*j]]);imag.push([t,signal[2*j+1]]);amplitude=Math.max(amplitude,Math.abs(signal[2*j]),Math.abs(signal[2*j+1]));}waveform={real,imag,amplitude:amplitude*1.08,duration};}
function drawSignal(){if(!waveform||$('plots').hidden)return;const p=chart($('signal'),0,waveform.duration*1e6,-waveform.amplitude,waveform.amplitude,'Time after burn-in [µs]','Normalized amplitude');curve(p,waveform.real,'#126ca4');curve(p,waveform.imag,'#b64a16');const t=frame*meta.frame_dt_s*1e6;p.series(new Float64Array([t,-waveform.amplitude,t,waveform.amplitude]),.1,.4,.3,1,false);}
function drawSpectrum(){if(!meta||$('plots').hidden)return;const band=Number($('band').value),n=spectrum.length;let points=[];for(let j=0;j<n;j++){const f=(j-n/2)/(n*meta.signal_dt_s)/1000;if(Math.abs(f)<=band)points.push([f,10*Math.log10(Math.max(spectrum[j],1e-20))]);}const max=points.length?Math.max(...points.map(v=>v[1])):-40;const ymax=Math.ceil(max/10)*10+5,ymin=ymax-55;const p=chart($('spectrum'),-band,band,ymin,ymax,'Frequency offset [kHz]','PSD [dB re 1 Hz⁻¹]');if(points.length)curve(p,points,'#18555c');drawPlasmaLine();}
function drawPlasmaLine(){
 if(!meta||$('plots').hidden)return;
 const n=spectrum.length,points=[],band=meta.plasma_band_mhz||2;let peak=-100;
 for(let j=0;j<n;j++){
  const f=(j-n/2)/(n*meta.signal_dt_s)/1e6;
  if(Math.abs(f)>band)continue;
  const power=10*Math.log10(Math.max(spectrum[j],1e-20));
  if(Math.abs(f)<.2){points.push([f,NaN]);continue;}
  points.push([f,power]);peak=Math.max(peak,power);
 }
 const ymax=Math.ceil(peak/5)*5+5,ymin=ymax-40;
 const p=chart($('plasmaLine'),-band,band,ymin,ymax,'Frequency offset [MHz]','PSD [dB re 1 Hz⁻¹]');
 if(points.length)curve(p,points,'#762a83');
}
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
 $('band').onchange=()=>{plotRenderers.get('spectrum')?.reset();drawSpectrum();};$('fft').onchange=calculate;$('recalculate').onclick=calculate;
window.addEventListener('resize',()=>{drawSpectrum();drawSignal();});
canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();state.ready=false;status('The graphics context was lost. Reload to restore the recording.',true);});
const annotationLayout=()=>{document.documentElement.style.setProperty('--toolbar-height',`${$('toolbar').getBoundingClientRect().height}px`);document.documentElement.style.setProperty('--header-height',`${document.querySelector('header').getBoundingClientRect().height}px`);};
const annotationObserver=new ResizeObserver(annotationLayout);annotationObserver.observe($('toolbar'));annotationObserver.observe(document.querySelector('header'));annotationLayout();
const loadBinary=url=>downloads.load(url);
function densityText(n){const exponent=Math.round(Math.log10(n)),digits='⁰¹²³⁴⁵⁶⁷⁸⁹';return `10${String(exponent).split('').map(v=>digits[Number(v)]).join('')} m⁻³`;}
function updateRunHelp(){
 for(const element of document.querySelectorAll('.run-density'))element.textContent=densityText(meta.density_m3);
 $('selectedParameters').textContent=`Selected run: nₑ = ${densityText(meta.density_m3)}; Tₑ = ${meta.te_K.toLocaleString()} K; Tᵢ = ${meta.ti_K.toLocaleString()} K; initial Bragg kλDe = ${meta.k_lambda_de.toFixed(3)}. Physical O⁺/electron mass ratio ${meta.mass_ratio.toFixed(0)}.`;
 $('debyeLengthCm').textContent=(meta.lambda_de_m*100).toFixed(3);
 $('viewWidthCm').textContent=(meta.lambda_de_m*1000).toFixed(2);
 $('fullBoxDebye').textContent=meta.length_lambda_de.toFixed(2);
 $('energyDrift').textContent=`${meta.max_energy_excursion_percent.toFixed(3)}% maximum energy excursion`;
 $('frameCadence').textContent=`${(meta.frame_dt_s*1e9).toFixed(2)} ns (8 PIC steps)`;
 $('playbackScale').textContent=`${(20*meta.frame_dt_s*1e6).toFixed(2)} µs of plasma time`;
 $('recordLength').textContent=`${(meta.signal_samples*meta.signal_dt_s*1e3).toFixed(2)} ms`;
}
async function loadRun(id){
 const run=runList.find(v=>v.id===id);if(!run)throw Error('Unknown simulation run');
 const token=++loadSerial,plotsWereOpen=!$('plots').hidden;++pending;
 state.ready=false;running=false;signal=null;waveform=null;spectrum=[];state.segments=0;document.body.classList.remove('has-recording');
 for(const plot of plotRenderers.values())plot.reset();
 $('plots').hidden=true;$('plotsToggle').setAttribute('aria-expanded','false');
 $('clock').textContent='Loading…';$('segments').textContent='Waiting for data';
 $('status').classList.remove('ready','error');$('loadingProgress').removeAttribute('value');$('loadingDetails').textContent='Connecting…';status(`Loading ${run.label}…`);
 for(const control of ['run','play','restart','scrub','recalculate'])$(control).disabled=true;
 if(plasmaView){const r=canvas.getBoundingClientRect();plasmaView.draw(new Float32Array(4096),new Float32Array(),r.width,r.height,Math.min(devicePixelRatio||1,2),22.,1.,false);}
 if(replay){replay.free();replay=null;}
 downloads=makeDownloads();
 try{
 const wasmUrl=new URL('./pkg/picisr_web_bg.wasm?v=6c98680c8f0b',import.meta.url);
 const metadataUrl=new URL(run.metadata,location.href);
 const response=await fetch(metadataUrl,{cache:'no-cache'});if(!response.ok)throw Error('Recording metadata is not available');meta=await response.json();
 const asset=name=>new URL(meta[name],metadataUrl);
 if(!wasmLoaded)downloads.plan(wasmUrl);
 downloads.plan(asset('preview_positions_file'),2*meta.pool_per_species*8);
 downloads.plan(asset('preview_potential_file'),2*meta.field_grid**2*2);
 downloads.plan(asset('positions_file'),meta.frames*meta.pool_per_species*8);
 downloads.plan(asset('potential_file'),meta.frames*meta.field_grid**2*2);
 downloads.plan(asset('signal_file'),meta.signal_samples*8);
 updateRunHelp();
 if(!wasmLoaded){status('Loading WebAssembly…');await init({module_or_path:await loadBinary(wasmUrl)});plasmaView=new PlasmaView(canvas,$('particles'));wasmLoaded=true;}
 const [previewPositions,previewPotential]=await Promise.all([loadBinary(asset('preview_positions_file')),loadBinary(asset('preview_potential_file'))]);
 replay=new Replay(new Uint16Array(previewPositions),new Int16Array(previewPotential),meta.pool_per_species,meta.field_grid,meta.length_lambda_de,10,meta.phi_quant_scale);
 frame=0;running=false;state.ready=true;document.body.classList.add('has-recording');
 status('Post-burn-in particles shown. Loading the motion recording…');
 const [positions,potential]=await Promise.all([loadBinary(asset('positions_file')),loadBinary(asset('potential_file'))]);
 replay.free();
 replay=new Replay(new Uint16Array(positions),new Int16Array(potential),meta.pool_per_species,meta.field_grid,meta.length_lambda_de,10,meta.phi_quant_scale);
 status('Loading the full-population scattered signal…');signal=new Float32Array(await loadBinary(asset('signal_file')));
 if(signal.length!==meta.signal_samples*2)throw Error('Incomplete signal recording');prepareWaveform();
 for(const option of $('fft').options)option.disabled=Number(option.value)>meta.signal_samples;
 $('fft').value=String(meta.default_fft);$('play').textContent='Pause';$('playtime').textContent='Recorded time, slowed down';
 for(const control of ['run','play','restart','scrub','recalculate'])$(control).disabled=false;
 frame=16;running=true;state.ready=true;state.totalParticles=meta.particles_per_species*2;state.clipDurationUs=(meta.frames-1)*meta.frame_dt_s*1e6;
 state.run=run.id;$('run').value=run.id;
 const url=new URL(location.href);url.searchParams.set('run',run.id);history.replaceState(null,'',url);
 $('plots').hidden=!plotsWereOpen;$('plotsToggle').setAttribute('aria-expanded',String(plotsWereOpen));
 $('loadingProgress').value=100;$('loadingDetails').textContent='100%';status('Loaded');$('status').classList.add('ready');
 drawSpectrum();calculate();
 }catch(e){if(token===loadSerial){console.error(e);status(e.message,true);$('run').disabled=false;}}
}
$('run').onchange=()=>loadRun($('run').value);
requestAnimationFrame(animate);
try{
 const response=await fetch('runs.json',{cache:'no-cache'});if(!response.ok)throw Error('Simulation list is unavailable');
 const catalog=await response.json();runList=catalog.runs;
 for(const run of runList){const option=document.createElement('option');option.value=run.id;option.textContent=run.label;$('run').append(option);}
 const wanted=new URL(location.href).searchParams.get('run'),id=runList.some(v=>v.id===wanted)?wanted:catalog.default;
 $('run').value=id;await loadRun(id);
}catch(e){console.error(e);status(e.message,true);}
