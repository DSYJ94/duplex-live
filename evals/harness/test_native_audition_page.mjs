import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const script=fs.readFileSync(new URL('../../portable/context/native_audition.html',import.meta.url),'utf8').split('<script>')[1].split('</script>')[0];
const turn=()=>new Promise(resolve=>setImmediate(resolve));

function page(external){
  const nodes=Object.fromEntries(['start','stop','allow','consent','status','text','disclosure'].map(id=>[id,{disabled:false,checked:false,hidden:false,textContent:''}]));
  const sockets=[],contexts=[];let permit,requests=0,stopped=0;
  const stream={getTracks:()=>[{stop:()=>stopped++}]};
  class Socket{
    static OPEN=1;
    constructor(){this.readyState=1;this.closed=false;sockets.push(this)}
    close(){this.closed=true;this.readyState=3;if(this.onclose)this.onclose()}
    send(){}
  }
  class Audio{
    constructor(options){this.sampleRate=options.sampleRate;this.state='running';this.audioWorklet={addModule:async()=>{}};contexts.push(this)}
    async resume(){}
    async close(){this.state='closed'}
    createGain(){return {gain:{},connect(){return this}}}
    createMediaStreamSource(){return {connect(){}}}
  }
  class Worklet{constructor(){this.port={}}connect(node){return node}}
  vm.runInNewContext(script,{document:{getElementById:id=>nodes[id]},window:{addEventListener(){}},
    location:{protocol:'http:',host:'localhost:8765'},WebSocket:Socket,AudioContext:Audio,AudioWorkletNode:Worklet,
    fetch:async()=>({ok:true,json:async()=>({external_backend_enabled:external})}),
    navigator:{mediaDevices:{getUserMedia:()=>{requests++;return new Promise(resolve=>{permit=()=>resolve(stream)})}}}});
  return {nodes,sockets,contexts,allowMicrophone:()=>permit(),requests:()=>requests,stopped:()=>stopped};
}

test('external reasoning requires opt-in before opening a microphone connection',async()=>{
  const p=page(true);await turn();
  assert.equal(p.nodes.start.disabled,true);
  await p.nodes.start.onclick();
  assert.equal(p.sockets.length,0);
  assert.equal(p.requests(),0);
  p.nodes.allow.checked=true;p.nodes.allow.onchange();
  assert.equal(p.nodes.start.disabled,false);
});

test('Stop retires an in-flight microphone permission and obsolete socket events',async()=>{
  const p=page(true);await turn();p.nodes.allow.checked=true;p.nodes.allow.onchange();
  await p.nodes.start.onclick();const old=p.sockets[0];
  const ready=old.onmessage({data:JSON.stringify({type:'ready',sample_rate:24000,frame_size:1920})});
  await turn();assert.equal(p.requests(),1);
  await p.nodes.stop.onclick();p.allowMicrophone();await ready;
  assert.equal(p.stopped(),1);
  assert.equal(p.contexts[0].state,'closed');
  await p.nodes.start.onclick();const next=p.sockets[1];
  await old.onclose();await old.onmessage({data:JSON.stringify({type:'ready',sample_rate:24000,frame_size:1920})});
  assert.equal(next.closed,false);
  assert.equal(p.requests(),1);
  await p.nodes.stop.onclick();
});

test('voice-only mode is explicit and does not require external reasoning consent',async()=>{
  const p=page(false);await turn();
  assert.equal(p.nodes.consent.hidden,true);
  assert.equal(p.nodes.start.disabled,false);
  assert.match(p.nodes.disclosure.textContent,/External reasoning is disconnected/);
});

test('checkbox events cannot enable a second connection during an active session',async()=>{
  const p=page(true);await turn();p.nodes.allow.checked=true;await p.nodes.allow.onchange();
  await p.nodes.start.onclick();const original=p.sockets[0];
  await p.nodes.allow.onchange();
  assert.equal(p.nodes.start.disabled,true);
  await p.nodes.start.onclick();
  assert.equal(p.sockets.length,1);
  assert.equal(original.closed,false);
  await p.nodes.stop.onclick();
});

test('withdrawing reasoning consent retires the socket and an in-flight microphone',async()=>{
  const p=page(true);await turn();p.nodes.allow.checked=true;await p.nodes.allow.onchange();
  await p.nodes.start.onclick();const original=p.sockets[0];
  const ready=original.onmessage({data:JSON.stringify({type:'ready',sample_rate:24000,frame_size:1920})});
  await turn();assert.equal(p.requests(),1);
  p.nodes.allow.checked=false;await p.nodes.allow.onchange();
  assert.equal(original.closed,true);
  assert.equal(p.nodes.start.disabled,true);
  assert.equal(p.nodes.stop.disabled,true);
  p.allowMicrophone();await ready;
  assert.equal(p.stopped(),1);
  assert.equal(p.contexts[0].state,'closed');
  await p.nodes.start.onclick();assert.equal(p.sockets.length,1);
  p.nodes.allow.checked=true;await p.nodes.allow.onchange();
  await p.nodes.start.onclick();assert.equal(p.sockets.length,2);
  await original.onclose();assert.equal(p.sockets[1].closed,false);
  await p.nodes.stop.onclick();
});

test('voice-only Start is guarded against duplicate connection events',async()=>{
  const p=page(false);await turn();await p.nodes.start.onclick();
  await p.nodes.start.onclick();
  assert.equal(p.sockets.length,1);
  await p.nodes.stop.onclick();
});

test('withdrawing consent stops an already installed microphone and audio context',async()=>{
  const p=page(true);await turn();p.nodes.allow.checked=true;await p.nodes.allow.onchange();
  await p.nodes.start.onclick();const original=p.sockets[0];
  const ready=original.onmessage({data:JSON.stringify({type:'ready',sample_rate:24000,frame_size:1920})});
  await turn();p.allowMicrophone();await ready;
  assert.equal(p.stopped(),0);assert.equal(p.contexts[0].state,'running');
  p.nodes.allow.checked=false;await p.nodes.allow.onchange();
  assert.equal(p.stopped(),1);assert.equal(p.contexts[0].state,'closed');
  assert.equal(original.closed,true);assert.equal(p.nodes.start.disabled,true);
});
