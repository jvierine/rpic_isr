import assert from 'node:assert/strict';
import {signalTrace} from './public/waveform.js';
const values=new Float32Array(200000);
for(let j=0;j<values.length/2;j++){values[2*j]=Math.sin(j);values[2*j+1]=Math.cos(j);}
values[2468]=100;values[3000]=-100;
const copy=values.slice();
const overview=signalTrace([values,0],0,100000,100,1,0);
assert.ok(overview.length<=200);
assert.ok(overview.some(p=>p[1]===100));assert.ok(overview.some(p=>p[1]===-100));
const zoom=signalTrace([values,1],40,44,100,1,0);
for(const [t,v] of zoom)assert.equal(v,values[2*t+1]);
assert.deepEqual(values,copy);
values[0]=NaN;assert.ok(Number.isNaN(signalTrace([values,0],0,0,100,1,0)[0][1]));
console.log('PASS: bounded typed-array overview, retained extrema, exact zoom samples, unchanged input');
