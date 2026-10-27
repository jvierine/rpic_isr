import init, {Replay, Welch, ScientificPlot, PlasmaView, ion_lowpass, inset_legend} from './pkg/picisr_web.js?v=31d16ba49250';
import {ProgressDownloads} from './loading.js?v=ebb138dbfd70';
import {simulationClock} from './annotations.js?v=e0a325e6b590';
import {bindPlotNavigation} from './navigation.js?v=a0857bc752dc';
import {PotentialFrames,particlePoints} from './potential.js?v=01bfbefb4745';
import {signalTrace} from './waveform.js?v=3cb07a38a949';
const $=id=>document.getElementById(id);
let meta, replay, signal, spectrum=[], running=true, frame=16, last=0, pending=0, welch;
let theory=[],activeRunId='',theoryCatalog;
const theoryCache=new Map();
async function loadTheory(n,token){
 if(meta.magnetic_field_z_t){theory=[];state.theoryReady=false;$('theory').disabled=true;$('theoryStatus').textContent='Magnetized run: unmagnetized theory does not apply';drawSpectrum();return;}
 $('theory').disabled=false;
 try{
  theoryCatalog??=await fetch('data/theory/catalog.json',{cache:'no-cache'}).then(r=>{if(!r.ok)throw Error('Theory unavailable');return r.json();});
  const entry=theoryCatalog[activeRunId];
  if(!entry)throw Error('Theory unavailable');
  for(const [key,value] of Object.entries(entry.parameters))if(Math.abs(meta[key]-value)>Math.abs(value)*1e-10)throw Error('Theory parameters do not match this run');
  const file=entry.fft[n];if(!file)throw Error('Theory unavailable for this FFT');
  if(!theoryCache.has(file)){const r=await fetch(`data/theory/${file}`);if(!r.ok)throw Error('Theory unavailable');theoryCache.set(file,new Float32Array(await r.arrayBuffer()));}
  if(token!==pending)return;
  theory=theoryCache.get(file);if(theory.length!==n)throw Error('Invalid theory length');
  state.theoryReady=true;state.theoryRun=activeRunId;state.theoryBins=n;$('theoryStatus').textContent='Theory: Maxwellian, collisionless · same Hann window';drawSpectrum();
 }catch(e){if(token===pending){theory=[];state.theoryReady=false;$('theoryStatus').textContent=e.message;}}
}
function spectrumValue(value){return $('linearSpectrum').checked?value*1e6:10*Math.log10(Math.max(value,1e-20));}
function drawTheory(plot,scale,band){if(!$('theory').checked||theory.length!==spectrum.length)return;const n=theory.length,points=[];for(let j=0;j<n;j++){const f=(j-n/2)/(n*meta.signal_dt_s)/scale;if(Math.abs(f)<=band)points.push([f,spectrumValue(theory[j])]);}curve(plot,points,'#b66b14',1.7);}
const state={ready:false,frame:16,shown:0,segments:0};
// Small browser-QA surface: no file paths or generator metadata are rendered.
window.picisrState=state;
const status=(text,error=false)=>{$('statusText').textContent=text;$('status').classList.toggle('error',error);};
let downloads,runList=[],wasmLoaded=false,loadSerial=0,potentialFrames,particleData;
const makeDownloads=()=>new ProgressDownloads(({loaded,total,fraction})=>{
 $('loadingProgress').value=Math.min(fraction*100,99.9);
 $('loadingDetails').textContent=`${Math.floor(fraction*100)}% · ${(loaded/1048576).toFixed(1)} / ${(total/1048576).toFixed(1)} MB`;
});
const dialog=$('help');$('helpOpen').onclick=()=>dialog.showModal();
$('plotsToggle').onclick=()=>{const panel=$('plots');panel.hidden=!panel.hidden;$('plotsToggle').setAttribute('aria-expanded',String(!panel.hidden));if(!panel.hidden){drawSignal();drawSpectrum();}};
$('plotsClose').onclick=()=>{$('plots').hidden=true;$('plotsToggle').setAttribute('aria-expanded','false');};
$('fullscreen').onclick=()=>{if(document.fullscreenElement)document.exitFullscreen();else document.documentElement.requestFullscreen();};
$('helpClose').onclick=()=>dialog.close();
dialog.addEventListener('click',e=>{if(e.target===dialog){const r=dialog.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)dialog.close();}});

const canvas=$('plasma');let plasmaView;
function renderPlasma(){
 const field=potentialFrames?potentialFrames.field(frame):replay.field(frame);
 if(!field){state.buffering=true;$('playtime').textContent='Buffering…';if(potentialFrames?.error){status(potentialFrames.error.message,true);$('status').classList.remove('ready');}return;}
 if(state.buffering)$('playtime').textContent=running?'Recorded time, slowed down':'Paused';state.buffering=false;
 if(potentialFrames){state.potentialCacheBytes=[...potentialFrames.cache.values()].reduce((s,v)=>s+v.field.byteLength+(v.particles?.byteLength||0),0);state.potentialCacheChunks=potentialFrames.cache.size;state.potentialDownloadedBytes=potentialFrames.bytes;canvas.dataset.potentialCacheBytes=state.potentialCacheBytes;canvas.dataset.potentialCacheChunks=state.potentialCacheChunks;}
 const count=Number($('count').value),electrons=$('electrons').checked,ions=$('ions').checked;
 const points=meta.positions_chunks?potentialFrames.points(frame,count,electrons,ions):potentialFrames?particlePoints(particleData,meta,frame,count,electrons,ions):replay.points(frame,count,electrons,ions);
 const r=canvas.getBoundingClientRect();
 plasmaView.draw(field,points,r.width,r.height,Math.min(devicePixelRatio||1,2),meta.potential_length_lambda_de||meta.length_lambda_de,meta.phi_display_scale/Number($('contrast').value),$('potential').checked);
 let ne=0,ni=0;for(let j=2;j<points.length;j+=4){if(points[j]<.5)ne++;else ni++;}
 $('shown').textContent=`${ne} electrons · ${ni} ions`;
 $('displaySubsetSummary').textContent=`The display limit is ${$('count').value} particles. This frame shows ${ne} electrons and ${ni} O⁺ ions. These are actual saved trajectories inside the 10 × 10 Debye-length camera view, not extra particles or a separate small simulation. The full simulated electron population contributes to the scattered signal, regardless of this display limit.`;
 state.frame=frame;state.shown=ne+ni;state.electrons=ne;state.ions=ni;
 $('countLabel').textContent=$('count').value;
 $('clock').textContent=`t = ${(meta.burn_ms+frame*meta.frame_dt_s*1e3).toFixed(6)} ms`;
 const time=simulationClock(frame,meta.frame_dt_s,meta.omega_pe_rad_s);
 $('timeNs').textContent=time.nanoseconds.toFixed(1);$('timePeriods').textContent=time.plasmaPeriods.toFixed(2);
 $('scrub').value=frame;
 const v=meta.phi_display_scale/Number($('contrast').value)*meta.potential_to_mv;
 $('phiMinus').textContent=`−${v.toFixed(1)} mV`;$('phiPlus').textContent=`+${v.toFixed(1)} mV`;
}
const plotRenderers=new Map();
function drawLegend(id){
 const c=$(id);let legend=$(`${id}Legend`);
 if(!legend){legend=document.createElement('canvas');legend.id=`${id}Legend`;legend.style.pointerEvents='none';legend.style.zIndex='1';legend.setAttribute('aria-label',id==='signal'?'Real: blue; Imaginary: brown':`PIC: ${id==='spectrum'?'teal':'purple'}; Theory: ochre`);c.parentElement.append(legend);}
 const r=c.getBoundingClientRect();inset_legend(legend,r.width,r.height,Math.min(devicePixelRatio||1,2),id==='signal',$('theory').checked&&theory.length===spectrum.length,id==='plasmaLine');
}
function chart(c,xmin,xmax,ymin,ymax,xlabel,ylabel){
 if(!plotRenderers.has(c.id)){
  const overlay=$(`${c.id}Lines`),plot=new ScientificPlot(c,overlay);
  plotRenderers.set(c.id,plot);
  const redraw=c.id==='signal'?drawSignal:c.id==='plasmaLine'?drawPlasmaLine:drawSpectrum;
  bindPlotNavigation(overlay,plot,redraw,()=>({x:70,y:30,width:overlay.clientWidth-88,height:overlay.clientHeight-78}));
 }
 const p=plotRenderers.get(c.id),r=c.getBoundingClientRect();
 p.begin(r.width,r.height,Math.min(devicePixelRatio||1,2),xmin,xmax,ymin,ymax,xlabel,ylabel);
 const view=Array.from(p.viewport_bounds());c.dataset.viewport=JSON.stringify(view);
 c.setAttribute('aria-label',`${xlabel}: ${(xmin+view[0]*(xmax-xmin)).toPrecision(4)} to ${(xmin+view[2]*(xmax-xmin)).toPrecision(4)}; ${ylabel}: ${(ymin+view[1]*(ymax-ymin)).toPrecision(4)} to ${(ymin+view[3]*(ymax-ymin)).toPrecision(4)}`);return p;
}
function curve(plot,values,colour,width=1.25){const rgb=colour.match(/\w\w/g).map(v=>parseInt(v,16)/255);plot.series(new Float64Array(values.flat()),...rgb,width,false);}
let waveform;
function prepareWaveform(){
 const n=signal.length/2,duration=meta.signal_first_s+(n-1)*meta.signal_dt_s;
 const filtered=ion_lowpass(signal,meta.signal_dt_s,30000);
 let amplitude=0;
 for(let j=0;j<signal.length;j++)amplitude=Math.max(amplitude,Math.abs(signal[j]));
 waveform={real:[signal,0],imag:[signal,1],slowReal:[filtered,0],slowImag:[filtered,1],amplitude:amplitude*1.08,duration};
}
// Display-only extrema reduction: retain peaks at overview scale and recover
// individual source samples on zoom. Never changes the signal used for the PSD.
function drawSignal(){if(!waveform||$('plots').hidden)return;const duration=waveform.duration*1e3,p=chart($('signal'),0,duration,-waveform.amplitude,waveform.amplitude,'Time after burn-in [ms]','Normalized amplitude');
 const view=Array.from(p.viewport_bounds()),pixels=Math.ceil($('signal').clientWidth),key=JSON.stringify([view[0],view[2],pixels]);
 if(waveform.cacheKey!==key){waveform.traces=[waveform.real,waveform.imag,waveform.slowReal,waveform.slowImag].map(v=>signalTrace(v,view[0]*duration,view[2]*duration,pixels,meta.signal_dt_s*1e3,meta.signal_first_s*1e3));waveform.cacheKey=key;}
 const [real,imag,slowReal,slowImag]=waveform.traces,show=$('lowpass').checked;curve(p,real,show?'#8dbbd6':'#126ca4',1);curve(p,imag,show?'#d8ad89':'#b64a16',1);
 if(show){curve(p,slowReal,'#126ca4',2.5);curve(p,slowImag,'#b64a16',2.5);}
 const t=frame*meta.frame_dt_s*1e3;p.series(new Float64Array([t,-waveform.amplitude,t,waveform.amplitude]),.1,.4,.3,1,false);drawLegend('signal');}
function spectrumPanel(id,scale,band,colour,range){
 if(!meta||$('plots').hidden)return;
 const n=spectrum.length,points=[];let peak=$('linearSpectrum').checked?0:-100;
 for(let j=0;j<n;j++){const f=(j-n/2)/(n*meta.signal_dt_s)/scale;if(Math.abs(f)>band)continue;const value=spectrumValue(spectrum[j]);points.push([f,value]);peak=Math.max(peak,value);if($('theory').checked&&theory.length===n)peak=Math.max(peak,spectrumValue(theory[j]));}
 const linear=$('linearSpectrum').checked,ymax=linear?Math.max(peak*1.12,1e-6):Math.ceil(peak/5)*5+5,ymin=linear?0:ymax-range;
 const p=chart($(id),-band,band,ymin,ymax,`Frequency offset [${scale===1000?'kHz':'MHz'}]`,linear?'PSD [10⁻⁶ Hz⁻¹]':'PSD [dB re 1 Hz⁻¹]');
 if(points.length)curve(p,points,colour);drawTheory(p,scale,band);drawLegend(id);
}
function drawSpectrum(){spectrumPanel('spectrum',1000,30,'#18555c',75);drawPlasmaLine();}
function drawPlasmaLine(){spectrumPanel('plasmaLine',1e6,meta?.plasma_band_mhz||2,'#762a83',80);}
async function calculate(){
 if(!signal)return;const token=++pending;if(welch)welch.free();
 const n=Number($('fft').value);welch=new Welch(n,meta.signal_dt_s);spectrum=[];theory=[];state.theoryReady=false;$('theoryStatus').textContent='Loading theory…';loadTheory(n,token);drawSpectrum();
 for(let j=0;j+n<=signal.length/2;j+=n/2){if(token!==pending)return;welch.add_window(signal.subarray(2*j,2*(j+n)));spectrum=welch.spectrum();state.segments=welch.segments();$('segments').textContent=`${state.segments} Hann windows`;
 $('specCaption').textContent=`Computed here from the full ${(meta.signal_samples*meta.signal_dt_s*1e3).toFixed(2)} ms record · Δf = ${(1/(n*meta.signal_dt_s)/1000).toFixed(2)} kHz · 50% overlap`;
 drawSpectrum();await new Promise(r=>setTimeout(r,100));}
}
function animate(now){if(state.ready){if(running&&last&&!state.buffering){frame+=Math.min((now-last)/1000,.1)*20*Number($('speed').value);if(frame>=meta.frames-1.001){frame=meta.frames-1.001;running=false;$('play').textContent='Play';$('playtime').textContent='End of recording';}}renderPlasma();drawSignal();}last=now;requestAnimationFrame(animate);}
$('play').onclick=()=>{if(frame>=meta.frames-1.01)frame=16;running=!running;$('play').textContent=running?'Pause':'Play';$('playtime').textContent=running?'Recorded time, slowed down':'Paused';};
$('restart').onclick=()=>{frame=16;running=true;$('play').textContent='Pause';$('playtime').textContent='Recorded time, slowed down';};
$('scrub').oninput=()=>{frame=Number($('scrub').value);running=false;$('play').textContent='Play';$('playtime').textContent='Paused';};
$('count').oninput=()=>{$('countLabel').textContent=$('count').value;};
 $('lowpass').onchange=drawSignal;
 $('theory').onchange=drawSpectrum;
 $('linearSpectrum').onchange=()=>{for(const id of ['spectrum','plasmaLine'])plotRenderers.get(id)?.reset();drawSpectrum();};
$('contrast').oninput=()=>{$('contrastLabel').textContent=`${Number($('contrast').value).toFixed(1)}×`;};
 $('fft').onchange=calculate;$('recalculate').onclick=calculate;
window.addEventListener('resize',()=>{drawSpectrum();drawSignal();});
canvas.addEventListener('webglcontextlost',e=>{e.preventDefault();state.ready=false;status('The graphics context was lost. Reload to restore the recording.',true);});
const annotationLayout=()=>{document.documentElement.style.setProperty('--toolbar-height',`${$('toolbar').getBoundingClientRect().height}px`);document.documentElement.style.setProperty('--header-height',`${document.querySelector('header').getBoundingClientRect().height}px`);};
const annotationObserver=new ResizeObserver(annotationLayout);annotationObserver.observe($('toolbar'));annotationObserver.observe(document.querySelector('header'));annotationLayout();
const loadBinary=url=>downloads.load(url);
function densityText(n){const exponent=Math.floor(Math.log10(n)),digits='⁰¹²³⁴⁵⁶⁷⁸⁹',mantissa=n/10**exponent;return `${Math.abs(mantissa-1)<1e-8?'':mantissa.toFixed(3)+' × '}10${String(exponent).split('').map(v=>digits[Number(v)]).join('')} m⁻³`;}
function updateRunHelp(){
 document.querySelector('.spectrum-card h2').textContent=meta.magnetic_field_z_t?'Low-frequency spectrum':'Ion-line spectrum';
 document.querySelector('.plasma-line-card h2').textContent=meta.magnetic_field_z_t?'Electron-frequency spectrum':'Plasma-line spectrum';
 const per=meta.particles_per_species,saved=meta.recorded_particles_per_species||4096;
 for(const element of document.querySelectorAll('.saved-pool-count'))element.textContent=saved.toLocaleString();
 $('populationSummary').textContent=`This run contains ${(2*per).toLocaleString()} computational macroparticles: ${per.toLocaleString()} electrons and ${per.toLocaleString()} O⁺ ions. The original replay capture saved trajectories for ${(2*saved).toLocaleString()} of them (${saved.toLocaleString()} per species). Across the animation, ${meta.visible_particles_min??'some'}–${meta.visible_particles_max??'some'} of those saved tracks lie inside the camera view. The web display offers a limit up to ${(meta.animation_particles||100).toLocaleString()} particles; it cannot show tracks that were not saved. The default display limit is 1,000.`;
 for(const element of document.querySelectorAll('.population-count'))element.textContent=meta.particles_per_species.toLocaleString();
 $('fieldGrid').textContent=`${meta.field_solver_grid}×${meta.field_solver_grid}`;
 for(const element of document.querySelectorAll('.run-density'))element.textContent=densityText(meta.density_m3);
 $('selectedParameters').textContent=`Selected run: nₑ = ${densityText(meta.density_m3)}; Tₑ = ${meta.te_K.toLocaleString()} K; Tᵢ = ${meta.ti_K.toLocaleString()} K; initial Bragg kλDe = ${meta.k_lambda_de.toFixed(3)}. Physical O⁺/electron mass ratio ${meta.mass_ratio.toFixed(0)}.`;
 $('magneticParameters').textContent=meta.magnetic_field_z_t?`Uniform out-of-plane B = ${(meta.magnetic_field_z_t*1e6).toFixed(2)} μT; electron gyrofrequency ${(meta.electron_gyrofrequency_hz/1e6).toFixed(3)} MHz; plasma frequency ${(meta.omega_pe_rad_s/(2*Math.PI*1e6)).toFixed(3)} MHz. All resolved wavevectors are perpendicular to B. The Boris mover rotates both species with their physical charge-to-mass ratios.`:'B = 0 for this run; particle acceleration is electrostatic.';
 $('debyeLengthCm').textContent=(meta.lambda_de_m*100).toFixed(3);
 const braggLength=2*Math.PI/meta.k_bragg_rad_m;
 $('braggLengthCm').textContent=(100*braggLength).toFixed(2);
 $('braggScale').style.width=`${braggLength/meta.lambda_de_m}vw`;
 $('viewWidthCm').textContent=(meta.lambda_de_m*1000).toFixed(2);
 $('fullBoxDebye').textContent=meta.length_lambda_de.toFixed(2);
 $('energyDrift').textContent=`${meta.max_energy_excursion_percent.toFixed(3)}% maximum energy excursion`;
 for(const el of document.querySelectorAll('.burn-duration'))el.textContent=`${meta.burn_ms.toFixed(3)} ms`;
 $('controlProtocol').textContent=meta.source?.control?.thermostatted?'After an unthermostatted 1 ms minimum burn and stationarity checks, separate weak velocity controls target 2000 K electrons and 1000 K ions. Recording follows at least 0.5 ms settling. This thermostat can affect spectra; the stationarity gate is not proof of kinetic equilibrium.':'This run uses an unthermostatted fixed-duration burn-in; that duration alone is not proof of equilibrium.';
 $('frameCadence').textContent=`${(meta.frame_dt_s*1e9).toFixed(2)} ns (8 PIC steps)`;
 if(meta.magnetic_field_z_t&&meta.late_early_voltage_variance_ratio) $('controlProtocol').textContent+=` This is a transient magnetic demonstration: the late-half scattered-voltage variance is ${(100*meta.late_early_voltage_variance_ratio).toFixed(0)}% of the early-half value. Its spectrum is not a stationary thermal-equilibrium estimate; peak frequencies and spectral amplitudes must be assessed separately.`;
 $('playbackScale').textContent=`${(20*meta.frame_dt_s*1e6).toFixed(2)} µs of plasma time`;
 $('recordLength').textContent=`${(meta.signal_samples*meta.signal_dt_s*1e3).toFixed(2)} ms`;
}
async function loadRun(id){
 const run=runList.find(v=>v.id===id);if(!run)throw Error('Unknown simulation run');
 const token=++loadSerial,plotsWereOpen=!$('plots').hidden;++pending;
 state.ready=false;running=false;signal=null;waveform=null;spectrum=[];theory=[];activeRunId=run.id;state.theoryReady=false;state.segments=0;document.body.classList.remove('has-recording');
 for(const plot of plotRenderers.values())plot.reset();
 $('plots').hidden=true;$('plotsToggle').setAttribute('aria-expanded','false');
 $('clock').textContent='Loading…';$('segments').textContent='Waiting for data';
 $('status').classList.remove('ready','error');$('loadingProgress').removeAttribute('value');$('loadingDetails').textContent='Connecting…';status(`Loading ${run.label}…`);
 for(const control of ['run','play','restart','scrub','recalculate'])$(control).disabled=true;
 if(plasmaView){const r=canvas.getBoundingClientRect();plasmaView.draw(new Float32Array(4096),new Float32Array(),r.width,r.height,Math.min(devicePixelRatio||1,2),22.,1.,false);}
 if(replay){replay.free();replay=null;}
 if(potentialFrames){potentialFrames.close();potentialFrames=null;}
 downloads=makeDownloads();
 try{
 const wasmUrl=new URL('./pkg/picisr_web_bg.wasm?v=e53c7f6aa404',import.meta.url);
 const metadataUrl=new URL(run.metadata,location.href);
 const response=await fetch(metadataUrl,{cache:'no-cache'});if(!response.ok)throw Error('Recording metadata is not available');meta=await response.json();
 const asset=name=>new URL(meta[name],metadataUrl);
 if(!wasmLoaded)downloads.plan(wasmUrl);
 if(!meta.potential_atlases){
 downloads.plan(asset('preview_positions_file'),2*meta.pool_per_species*8);
 downloads.plan(asset('preview_potential_file'),2*meta.field_grid**2*2);
 downloads.plan(asset('potential_file'),meta.frames*meta.field_grid**2*2);
 }
 if(!meta.positions_chunks)downloads.plan(asset('positions_file'),meta.frames*meta.pool_per_species*(meta.positions_layout?16:8));
 downloads.plan(asset('signal_file'),meta.signal_samples*8);
 updateRunHelp();
 if(!wasmLoaded){status('Loading WebAssembly…');await init({module_or_path:await loadBinary(wasmUrl)});plasmaView=new PlasmaView(canvas,$('particles'));wasmLoaded=true;}
 if(meta.potential_atlases){
  potentialFrames=new PotentialFrames(meta,metadataUrl);
  if(meta.positions_chunks){
   const [voltage]=await Promise.all([loadBinary(asset('signal_file')),potentialFrames.load(0)]);signal=new Float32Array(voltage);
  }else{
  const [positions,voltage]=await Promise.all([loadBinary(asset('positions_file')),loadBinary(asset('signal_file')),potentialFrames.load(0)]);
  particleData=new Uint16Array(positions);
  signal=new Float32Array(voltage);
  }
  $('count').max=String(meta.animation_particles||100);$('count').value=String(Math.min(1000,Number($('count').max)));
 }else{
 const [previewPositions,previewPotential]=await Promise.all([loadBinary(asset('preview_positions_file')),loadBinary(asset('preview_potential_file'))]);
 replay=new Replay(new Uint16Array(previewPositions),new Int16Array(previewPotential),meta.pool_per_species,meta.field_grid,meta.length_lambda_de,10,meta.phi_quant_scale);
 frame=0;running=false;state.ready=true;document.body.classList.add('has-recording');
 status('Post-burn-in particles shown. Loading the motion recording…');
 const [positions,potential]=await Promise.all([loadBinary(asset('positions_file')),loadBinary(asset('potential_file'))]);
 replay.free();
 replay=new Replay(new Uint16Array(positions),new Int16Array(potential),meta.pool_per_species,meta.field_grid,meta.length_lambda_de,10,meta.phi_quant_scale);
 status('Loading the full-population scattered signal…');signal=new Float32Array(await loadBinary(asset('signal_file')));
 }
 if(signal.length!==meta.signal_samples*2)throw Error('Incomplete signal recording');prepareWaveform();
 for(const option of $('fft').options)option.disabled=Number(option.value)>meta.signal_samples;
 $('fft').value=String(meta.default_fft);$('play').textContent='Pause';$('playtime').textContent='Recorded time, slowed down';
 for(const control of ['run','play','restart','scrub','recalculate'])$(control).disabled=false;
 frame=16;running=true;state.ready=true;document.body.classList.add('has-recording');state.totalParticles=meta.particles_per_species*2;state.clipDurationUs=(meta.frames-1)*meta.frame_dt_s*1e6;
 $('scrub').max=String(meta.frames-1.001);
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
