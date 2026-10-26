// Mechanical cache versioning; no build annotations are rendered in the UI.
import fs from 'node:fs';import crypto from 'node:crypto';
const root=new URL('./public/',import.meta.url);
const hash=file=>crypto.createHash('sha256').update(fs.readFileSync(new URL(file,root))).digest('hex').slice(0,12);
let app=fs.readFileSync(new URL('app.js',root),'utf8');
app=app.replace(/\.\/pkg\/picisr_web\.js(?:\?v=[a-z0-9]+)?/,'./pkg/picisr_web.js?v='+hash('pkg/picisr_web.js'));
app=app.replace(/\.\/loading\.js(?:\?v=[a-z0-9]+)?/,'./loading.js?v='+hash('loading.js'));
app=app.replace(/\.\/annotations\.js(?:\?v=[a-z0-9]+)?/,'./annotations.js?v='+hash('annotations.js'));
app=app.replace(/\.\/navigation\.js(?:\?v=[a-z0-9]+)?/,'./navigation.js?v='+hash('navigation.js'));
app=app.replace(/picisr_web_bg\.wasm\?v=[A-Za-z0-9_]+/,'picisr_web_bg.wasm?v='+hash('pkg/picisr_web_bg.wasm'));
fs.writeFileSync(new URL('app.js',root),app);
let html=fs.readFileSync(new URL('index.html',root),'utf8');
html=html.replace(/style\.css\?v=[a-z0-9]+/,'style.css?v='+hash('style.css')).replace(/app\.js\?v=[a-z0-9]+/,'app.js?v='+hash('app.js'));
fs.writeFileSync(new URL('index.html',root),html);
