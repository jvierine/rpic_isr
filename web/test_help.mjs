import fs from 'node:fs';
import assert from 'node:assert/strict';
const html=fs.readFileSync(new URL('./public/index.html',import.meta.url),'utf8');
const help=html.slice(html.indexOf('<dialog id="help"'),html.indexOf('</dialog>'));
const details=help.indexOf('<h3>Run and display details</h3>');
assert.equal((help.slice(0,details).match(/class="equation"/g)||[]).length,3);
for(const id of ['selectedParameters','populationSummary','displaySubsetSummary','controlProtocol']){
 assert.ok(help.indexOf(`id="${id}"`)>details);
 assert.equal((help.match(new RegExp(`id="${id}"`,'g'))||[]).length,1);
}
assert.ok(help.indexOf('class="equation"')<350);
console.log('PASS: equations first; run, subset and numerical details below key physics');
