import assert from 'node:assert/strict';
import {simulationClock} from './public/annotations.js';
for(const omega of [5641460.225429499,17839863.641604174]){
 const dt=.48/omega,onePeriod=2*Math.PI/omega;
 const c=simulationClock(onePeriod/dt,dt,omega);
 assert(Math.abs(c.plasmaPeriods-1)<1e-12);
 assert(Math.abs(c.nanoseconds-onePeriod*1e9)<1e-9);
 assert.equal(simulationClock(0,dt,omega).nanoseconds,0);
}
assert.throws(()=>simulationClock(1,0,1),/Invalid/);
console.log('PASS: ns clock, electron plasma periods, zero origin, and 2 pi conversion');
