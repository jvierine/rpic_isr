// Byte-counted downloads. Plan the recording files before starting transfers so
// parallel fetches share one denominator. Decoded transport bytes are counted.
export class ProgressDownloads {
 constructor(onProgress){this.files=new Map();this.onProgress=onProgress;}
 plan(url,total=0){this.files.set(String(url),{loaded:0,total});this.report();}
 report(){let loaded=0,total=0;for(const file of this.files.values()){loaded+=file.loaded;total+=Math.max(file.total,file.loaded);}this.onProgress({loaded,total,fraction:total?Math.min(loaded/total,1):0});}
 async load(url,fetcher=fetch){
  const key=String(url);if(!this.files.has(key))this.plan(key);
  const file=this.files.get(key),response=await fetcher(url);
  if(!response.ok)throw Error(`Recording download failed (${response.status})`);
  const declared=Number(response.headers.get('content-length'));
  if(declared>0&&!response.headers.get('content-encoding'))file.total=declared;
  const chunks=[];
  if(response.body){const reader=response.body.getReader();for(;;){const {done,value}=await reader.read();if(done)break;chunks.push(value);file.loaded+=value.byteLength;this.report();}}
  else{const value=new Uint8Array(await response.arrayBuffer());chunks.push(value);file.loaded=value.byteLength;}
  if(file.total&&file.loaded!==file.total)throw Error(`Incomplete recording download (${file.loaded} of ${file.total} bytes)`);
  file.total=file.loaded;this.report();
  const result=new Uint8Array(file.loaded);let offset=0;for(const chunk of chunks){result.set(chunk,offset);offset+=chunk.byteLength;}
  return result.buffer;
 }
}
