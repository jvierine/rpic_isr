// Actual replay fields and tracks; deterministic 60 fps; no synthetic physics.
import init,{PlasmaView} from './pkg/picisr_web.js';
import {PotentialFrames} from './potential.js';
await init();
const W=1920,H=1080,FPS=60,MOTION=30,HOLD=15;
const output=document.querySelector('#output'),ctx=output.getContext('2d'),status=document.querySelector('#status'),button=document.querySelector('#render');
const field=document.createElement('canvas'),dots=document.createElement('canvas');
document.querySelector('#scratch').append(field,dots);
const plasma=new PlasmaView(field,dots);
const catalog=await(await fetch('runs.json')).json();
async function makeVideo(id){
 const entry=catalog.runs.find(r=>r.id===id),url=new URL(entry.metadata,location.href),meta=await(await fetch(url)).json();
 const frames=new PotentialFrames(meta,url);
 const end=await createImageBitmap(await(await fetch(`video-local-${id}.png`)).blob());
 const chunks=[];let failure;
 const config={codec:'avc1.64002a',width:W,height:H,framerate:FPS,bitrate:12000000,avc:{format:'annexb'}};
 if(!(await VideoEncoder.isConfigSupported(config)).supported)throw Error('H264 encoder unavailable');
 const encoder=new VideoEncoder({output:chunk=>{const bytes=new Uint8Array(chunk.byteLength);chunk.copyTo(bytes);chunks.push(bytes);},error:e=>{failure=e;}});
 encoder.configure(config);
 for(let j=0;j<FPS*(MOTION+HOLD);j++){
  if(failure)throw failure;
  if(j<FPS*MOTION){
   const frame=16+j*5/FPS,chunk=Math.floor(frame/64);
   await frames.load(chunk);if(Math.floor((Math.floor(frame)+1)/64)!==chunk)await frames.load(chunk+1);
   const potential=frames.field(frame);if(!potential)throw Error('Missing replay frame');
   plasma.draw(potential,frames.points(frame,1000,true,true),W,H,1,meta.potential_length_lambda_de,meta.phi_display_scale/.3,true);
   ctx.drawImage(field,0,0);ctx.drawImage(dots,0,0);
   ctx.fillStyle='#f5f0e3dd';ctx.fillRect(18,18,410,46);
   ctx.fillStyle='#27372f';ctx.font='26px system-ui';ctx.fillText(`nₑ = ${id==='low'?'10¹⁰':'10¹¹'} m⁻³`,32,50);
   ctx.fillStyle='#f5f0e3dd';ctx.fillRect(18,H-60,700,42);
   ctx.fillStyle='#27372f';ctx.font='24px system-ui';
   const t=frame*meta.frame_dt_s;
   ctx.fillText(`t = ${(t*1e9).toFixed(1)} ns = ${(t*meta.omega_pe_rad_s/(2*Math.PI)).toFixed(2)} plasma periods`,32,H-31);
  }else ctx.drawImage(end,0,0,W,H);
  const f=new VideoFrame(output,{timestamp:Math.round(j*1e6/FPS),duration:Math.round(1e6/FPS)});
  encoder.encode(f,{keyFrame:j%(2*FPS)===0});f.close();
  if(encoder.encodeQueueSize>6)await encoder.flush();
  if(j%30===0){status.textContent=`${id}: ${j}/${FPS*(MOTION+HOLD)} frames`;await new Promise(requestAnimationFrame);}
 }
 await encoder.flush();encoder.close();frames.close();end.close();if(failure)throw failure;
 const response=await fetch(`/save/${id}.h264`,{method:'POST',body:new Blob(chunks,{type:'video/h264'})});
 if(!response.ok)throw Error(await response.text());
}
button.disabled=false;status.textContent='Ready: 30 seconds fullscreen plasma, then 15 seconds GUI.';
button.onclick=async()=>{button.disabled=true;try{for(const id of ['low','high'])await makeVideo(id);status.textContent='Both previews saved at 60 fps. Not published.';}catch(e){status.textContent=String(e);console.error(e);}};
