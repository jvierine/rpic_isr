// Scalar JPEG transport, bounded cache; color and contrast remain in WebGL.
export class PotentialFrames {
 constructor(meta,url){this.meta=meta;this.url=url;this.cache=new Map();this.pending=new Map();this.closed=false;this.bytes=0;}
 close(){this.closed=true;this.cache.clear();}
 async load(index){
  if(index<0||index>=this.meta.potential_atlases.length||this.closed)return;
  if(this.cache.has(index))return this.cache.get(index);
  if(this.pending.has(index))return this.pending.get(index);
  if(this.pending.size>=3)return;
  const task=(async()=>{
   const response=await fetch(new URL(this.meta.potential_atlases[index],this.url));
   if(!response.ok)throw Error('Potential animation chunk unavailable');
   const blob=await response.blob();this.bytes+=blob.size;
   const bitmap=await createImageBitmap(blob,{colorSpaceConversion:'none'});
   const c=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=c.getContext('2d',{willReadFrequently:true});
   ctx.drawImage(bitmap,0,0);bitmap.close();
   const rgba=ctx.getImageData(0,0,c.width,c.height).data,g=this.meta.field_grid;
   const values=new Float32Array(64*g*g),scale=this.meta.potential_decode_scale;
   for(let f=0;f<64;f++)for(let y=0;y<g;y++)for(let x=0;x<g;x++){
    const p=((Math.floor(f/8)*g+y)*c.width+(f%8)*g+x)*4;
    values[f*g*g+y*g+x]=(rgba[p]/127.5-1)*scale;
   }
   if(!this.closed){this.cache.set(index,values);while(this.cache.size>3)this.cache.delete(this.cache.keys().next().value);}
   return values;
  })().finally(()=>this.pending.delete(index));
  this.pending.set(index,task);return task;
 }
 field(frame){
  const a=Math.min(Math.floor(frame),this.meta.frames-2),b=a+1,t=frame-a,g=this.meta.field_grid;
  const ia=Math.floor(a/64),ib=Math.floor(b/64),left=this.cache.get(ia),right=this.cache.get(ib);
  for(const i of new Set([ia,ib,ib+1]))this.load(i).catch(e=>{this.error=e;});
  if(!left||!right)return null;
  const out=new Float32Array(g*g),offsetA=(a%64)*g*g,offsetB=(b%64)*g*g;
  for(let j=0;j<out.length;j++)out[j]=(1-t)*left[offsetA+j]+t*right[offsetB+j];
  return out;
 }
}

// Endpoints belong to the SAME actual particle. Display selection can change
// as particles leave the camera; it never joins unrelated particle tracks.
export function particlePoints(data,meta,frame,count,electrons,ions){
 const a=Math.min(Math.floor(frame),meta.frames-2),t=frame-a,L=meta.length_lambda_de,origin=(L-10)/2;
 const species=Number(electrons)+Number(ions),out=[];if(!species)return new Float32Array();
 for(let s=0;s<2;s++){
  if((s===0&&!electrons)||(s===1&&!ions))continue;
  let shown=0;const cap=Math.min(50,Math.floor(count/species));
  for(let p=0;p<50&&shown<cap;p++){
   const j=((a*2+s)*50+p)*4,xy=[];
   for(let d=0;d<2;d++){const x=data[j+d]/65535*L,y=data[j+2+d]/65535*L;let delta=y-x;if(delta>L/2)delta-=L;if(delta< -L/2)delta+=L;xy.push(((x+t*delta+L)%L-origin)/10);}
   if(xy.every(v=>v>=0&&v<1)){out.push(...xy,s,p);shown++;}
  }
 }
 return new Float32Array(out);
}
