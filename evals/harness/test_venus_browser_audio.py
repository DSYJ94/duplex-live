"""Execute the shipped page script: stale playback and microphone-start races."""
import json,subprocess

HARNESS=r'''
const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync('portable/venus/app.html','utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements={};const el=id=>elements[id]||(elements[id]={disabled:false,textContent:'',value:'released',hidden:true});
let ws,resolveMic;const nodes=[];const track={stopped:false,stop(){this.stopped=true}};
const media={getTracks:()=>[track]};
class Socket{constructor(){ws=this;this.readyState=1;}close(){}send(){}}
class Audio{constructor(){this.sampleRate=16000;this.currentTime=0;this.destination={};this.audioWorklet={addModule:async()=>{}};}
 async resume(){}async close(){}createMediaStreamSource(){return{connect(){},disconnect(){}}}
 createBuffer(ch,len,rate){return{duration:len/rate,copyToChannel(){}}}
 createBufferSource(){let n={connect(){},disconnect(){},start(t){this.time=t;},stop(){this.stopped=true}};nodes.push(n);return n}}
class Worklet{constructor(){this.port={}}connect(){}disconnect(){}}
const context={document:{querySelector:el},fetch:async()=>({json:async()=>({comparison_available:true})}),
 AudioContext:Audio,AudioWorkletNode:Worklet,WebSocket:Socket,Blob:class{},URL:{createObjectURL:()=>'',revokeObjectURL(){}},
 location:{protocol:'http:',host:'127.0.0.1:8037'},navigator:{mediaDevices:{getUserMedia:()=>MODE==='race'?new Promise(r=>resolveMic=r):Promise.resolve(media)}},
 Uint8Array,Float32Array,Int16Array,atob:b=>Buffer.from(b,'base64').toString('binary'),performance:{now:()=>0}};
vm.runInNewContext(script,context);
const audio=x=>Buffer.from(new Float32Array(24000).fill(x).buffer).toString('base64');
(async()=>{await new Promise(r=>setImmediate(r));await el('#start').onclick();
 const ready=ws.onmessage({data:JSON.stringify({ready:true})});
 if(MODE==='race'){
   await new Promise(r=>setImmediate(r));await el('#stop').onclick();resolveMic(media);
   let error=null;try{await ready}catch(e){error=e.message}
   console.log(JSON.stringify({trackStopped:track.stopped,startEnabled:!el('#start').disabled,error}));return;
 }
 await ready;
 await ws.onmessage({data:JSON.stringify({is_listen:false,end_of_turn:MODE==='final',audio:audio(.2),audio_sample_rate:24000})});
 const speech=nodes[0];const count=nodes.length;
 for(let i=0;i<2;i++)await ws.onmessage({data:JSON.stringify({is_listen:true,audio:audio(0),audio_sample_rate:24000})});
 console.log(JSON.stringify({speechStopped:!!speech.stopped,sourcesBeforeListen:count,sourcesAfterListen:nodes.length}));
})().catch(e=>{console.error(e);process.exitCode=1});
'''

def run(mode):
    p=subprocess.run(['node','-e',"const MODE="+json.dumps(mode)+';'+HARNESS],capture_output=True,text=True,check=True)
    return json.loads(p.stdout)


def test_native_listen_cancels_stale_speech_and_does_not_queue_silence():
    state=run('playback')
    assert state['speechStopped'],state
    assert state['sourcesAfterListen']==state['sourcesBeforeListen'],state


def test_end_during_microphone_permission_does_not_leave_a_late_microphone_stream():
    state=run('race')
    assert state['error'] is None and state['trackStopped'] and state['startEnabled'],state


def test_final_speech_is_not_cut_off_when_native_turn_has_ended():
    state=run('final')
    assert not state['speechStopped'],state
    assert state['sourcesAfterListen']==state['sourcesBeforeListen'],state


def completion_frame(payload):
    # Exercise the actual shipped ws.onmessage handler with the continuation
    # metadata exported by the native frame. Silent legacy listen stays silent.
    inject=r'''
 await ws.onmessage({data:JSON.stringify({is_listen:true,audio:audio(.3),wire_audio_bytes:96000,audio_sample_rate:24000,
  speech_completion:{version:'native-terminal-resume-v1',pending:true,acoustic_complete:false,events:[{kind:'terminal_acoustic_piece'}]}})});
 const continuation=nodes[1];
 PAYLOAD
 console.log(JSON.stringify({sources:nodes.length,oldStopped:!!speech.stopped,continuationStopped:!!continuation?.stopped}));
'''.replace('PAYLOAD',payload)
    start=HARNESS.index(' for(let i=0;i<2;i++)')
    finish=HARNESS.index('\n})().catch',start)
    code=HARNESS[:start]+inject+HARNESS[finish:]
    p=subprocess.run(['node','-e',"const MODE='final';"+code],capture_output=True,text=True,check=True)
    return json.loads(p.stdout)


def test_native_continuation_audio_plays_while_llm_is_listening():
    result=completion_frame('')
    assert result['sources']==2 and not result['oldStopped'] and not result['continuationStopped'],result


def test_new_native_speech_transition_clears_queued_old_continuation_audio():
    result=completion_frame(r'''await ws.onmessage({data:JSON.stringify({is_listen:false,end_of_turn:true,audio:audio(.4),audio_sample_rate:24000,
 speech_completion:{version:'native-terminal-resume-v1',pending:false,acoustic_complete:true,events:[{kind:'pending_acoustic_cancelled',reason:'new_native_speaking_turn'}]}})});''')
    assert result['sources']==3 and result['oldStopped'] and result['continuationStopped'],result
