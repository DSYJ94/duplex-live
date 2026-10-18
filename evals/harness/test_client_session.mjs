// Actual client module with fake browser devices; never physical-browser acceptance.
import { readFileSync } from 'node:fs';
import vm from 'node:vm';
import test from 'node:test';
import assert from 'node:assert/strict';
const code = readFileSync(new URL('../../serving/hybrid/client.js', import.meta.url), 'utf8').replace("import { AudioPlayout } from '/playout.js';", '');
function client(getUserMedia) {
  const elements = Object.fromEntries(['token','start','end','status','settings','activity','input-level'].map(id=>[id,{value:'',disabled:id==='end',textContent:''}]));
  const sockets=[];const nodes=[];
  const node=()=>({gain:{value:1},connect(target){return target;},disconnect(){}});
  class AudioContext {
    sampleRate=24000;destination={};audioWorklet={addModule:async()=>{}};
    async resume(){} async close(){} createMediaStreamSource(){return node();} createGain(){return node();}
  }
  class AudioWorkletNode { constructor(){this.port={};nodes.push(this);} connect(target){return target;} disconnect(){} }
  class WebSocket {
    static OPEN=1;readyState=0;bufferedAmount=0;sent=[];
    constructor(){sockets.push(this);} send(value){this.sent.push(value);} close(){this.readyState=3;}
  }
  vm.runInNewContext(code,{document:{getElementById:id=>elements[id]},location:{hash:'',protocol:'http:',host:'localhost:8767'},URLSearchParams,
    fetch:async()=>({ok:true,json:async()=>({token:'fixture-only',websocket_url:'ws://localhost:8779/ws'})}),
    AudioContext,AudioWorkletNode,WebSocket,AudioPlayout:class {stop(){} append(){}},
    navigator:{mediaDevices:{getUserMedia}},setTimeout:fn=>fn()});
  return {elements,sockets,nodes};
}
function microphone(){let stops=0;return {getTracks:()=>[{stop(){stops++;}}],getAudioTracks:()=>[{getSettings:()=>({sampleRate:24000})}],get stops(){return stops;}};}
test('explicit Start replaces a stale session and complete releases UI/microphone',async()=>{
  const mic=microphone(),c=client(async()=>mic);await c.elements.start.onclick();
  assert.equal(c.elements.end.disabled,false);const socket=c.sockets[0];socket.readyState=1;socket.onopen();
  assert.equal(JSON.parse(socket.sent[0]).replace_existing,true);
  socket.onmessage({data:JSON.stringify({kind:'session_ready'})});
  c.nodes[0].port.onmessage({data:new Float32Array(1920).fill(.1).buffer});
  assert.match(c.elements['input-level'].textContent,/sound detected/);
  socket.onmessage({data:JSON.stringify({kind:'session_complete'})});
  assert.equal(c.elements.start.disabled,false);assert.equal(c.elements.end.disabled,true);assert.equal(mic.stops,1);
});
test('End during pending microphone permission cannot leave a late capture or socket',async()=>{
  let release;const mic=microphone(),c=client(()=>new Promise(resolve=>{release=resolve;}));
  const starting=c.elements.start.onclick();
  for(let i=0;i<20&&!release;i++)await Promise.resolve();
  assert.ok(release);assert.equal(c.elements.end.disabled,false);c.elements.end.onclick();release(mic);await starting;
  assert.equal(mic.stops,1);assert.equal(c.sockets.length,0);assert.equal(c.elements.start.disabled,false);
});
