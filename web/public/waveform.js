// Peak-preserving display reduction directly from interleaved typed arrays.
// No per-sample JavaScript objects, even for long voltage records.
export function signalTrace([samples,component],lo,hi,pixels,dt,first){
 const n=samples.length/2,value=j=>samples[2*j+component];
 const start=Math.max(0,Math.floor((lo-first)/dt)-1),end=Math.min(n,Math.ceil((hi-first)/dt)+2);
 const step=Math.max(1,Math.ceil((end-start)/Math.max(1,pixels))),out=[];
 const point=j=>[first+j*dt,value(j)];
 for(let j=start;j<end;j+=step){
  let min=j,max=j;
  for(let k=j;k<Math.min(j+step,end);k++)if(Number.isFinite(value(k))){
   if(!Number.isFinite(value(min))||value(k)<value(min))min=k;
   if(!Number.isFinite(value(max))||value(k)>value(max))max=k;
  }
  if(min<=max){out.push(point(min));if(max!==min)out.push(point(max));}
  else out.push(point(max),point(min));
 }
 return out;
}
