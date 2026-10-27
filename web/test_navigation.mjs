import assert from 'node:assert/strict';
import {bindPlotNavigation} from './public/navigation.js';
class Target{
 constructor(){this.events=new Map();this.style={};this.clientWidth=600;this.clientHeight=200;}
 addEventListener(name,fn){this.events.set(name,fn);}
 removeEventListener(name){this.events.delete(name);}
 setAttribute(){} setPointerCapture(){} remove(){}
 getBoundingClientRect(){return{left:0,top:0};}
 emit(name,extra={}){let prevented=false;this.events.get(name)?.({clientX:200,clientY:100,button:0,pointerId:1,deltaY:-100,preventDefault(){prevented=true;},stopImmediatePropagation(){},...extra});return prevented;}
}
globalThis.document={createElement:()=>new Target(),body:{append(){}}};
globalThis.window=new Target();
const canvas=new Target(),calls=[];
const viewport=Object.fromEntries(['zoom','zoom_box','pan','reset'].map(k=>[k,(...v)=>calls.push([k,...v])]));
const dispose=bindPlotNavigation(canvas,viewport,()=>{},()=>({x:70,y:30,width:512,height:122}));
assert.ok(canvas.emit('wheel'));assert.equal(calls.pop()[0],'zoom');
canvas.emit('pointerdown');canvas.emit('pointermove',{clientX:350,clientY:70});canvas.emit('pointerup',{clientX:350,clientY:70});
assert.equal(calls.pop()[0],'zoom_box');assert.equal(calls.length,0);
canvas.emit('pointerdown',{shiftKey:true});canvas.emit('pointermove',{clientX:250});canvas.emit('pointerup',{clientX:250});
assert.equal(calls.pop()[0],'pan');
canvas.emit('dblclick');assert.equal(calls.pop()[0],'reset');
dispose();assert.equal(canvas.events.size,0);
console.log('PASS: plain wheel zoom, plain drag box, Shift-drag pan, double-click reset');
