import json
from pathlib import Path
import subprocess
import pytest


@pytest.mark.parametrize('error',[
    "ERROR: gcloud crashed (ConnectionError): ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))",
    "ERROR: gcloud crashed (SSLError): SSLEOFError UNEXPECTED_EOF_WHILE_READING",
])
def test_current_gcloud_remote_disconnect_gets_one_fresh_read(monkeypatch,error):
    from portable.venus import guard
    calls=[]
    def cloud(*args,**kwargs):
        calls.append(kwargs['timeout'])
        if len(calls)==1:
            raise RuntimeError(error)
        return {'fresh_provider_read':True}
    monkeypatch.setattr(guard,'cloud',cloud)
    assert guard.bounded_cloud_read('instances','list')=={'fresh_provider_read':True}
    assert calls==[12,8]


def test_browser_websocket_failure_before_microphone_ready_recovers_controls():
    script=r'''
const fs=require('fs'),vm=require('vm');
const html=fs.readFileSync('portable/venus/app.html','utf8');
const script=html.match(/<script>([\s\S]*?)<\/script>/)[1];
const elements={}; const el=id=>elements[id]||(elements[id]={disabled:false,textContent:'',value:'candidate',hidden:true});
let ws;
class Socket {constructor(url){ws=this;}close(){} }
class Audio {constructor(){this.sampleRate=16000;}async resume(){}async close(){}}
const context={document:{querySelector:el},fetch:async()=>({json:async()=>({comparison_available:true})}),
 AudioContext:Audio,WebSocket:Socket,location:{protocol:'http:',host:'127.0.0.1:8037'},
 navigator:{mediaDevices:{getUserMedia:async()=>{throw Error('Microphone must not start before socket ready');}}}};
vm.runInNewContext(script,context);
(async()=>{await new Promise(r=>setImmediate(r));await el('#start').onclick();
 ws.onerror({});if(ws.onclose)ws.onclose({code:1006});await new Promise(r=>setImmediate(r));
 console.log(JSON.stringify({startDisabled:el('#start').disabled,variantDisabled:el('#variant').disabled,
 stopDisabled:el('#stop').disabled,state:el('#state').textContent}));
})().catch(e=>{console.error(e);process.exitCode=1;});
'''
    result=subprocess.run(['node','-e',script],text=True,capture_output=True,check=True)
    state=json.loads(result.stdout)
    assert not state['startDisabled'] and not state['variantDisabled'],state
    assert state['stopDisabled'],state


def test_authentication_errors_still_fail_immediately(monkeypatch):
    from portable.venus import guard
    calls=[]
    def cloud(*args,**kwargs):
        calls.append(1);raise RuntimeError('Reauthentication failed. cannot prompt')
    monkeypatch.setattr(guard,'cloud',cloud)
    with pytest.raises(RuntimeError,match='Reauthentication'):guard.bounded_cloud_read('instances','list')
    assert len(calls)==1


def test_real_websocket_route_accepts_same_origin_and_exchanges_pcm(monkeypatch):
    from types import SimpleNamespace
    from fastapi.testclient import TestClient
    from portable.venus import native_app
    class LLM:
        def modules(self):return []
    class Engine:
        def __init__(self,*args):
            self.model=SimpleNamespace(llm=LLM())
            self.profile=dict(model='fixture-only',optimizer_updates=244,candidate={})
        def prepare(self):pass
        def step(self,pcm,feedback):
            assert len(pcm)==32000
            return dict(raw_text='<|listen|></unit>',text='',is_listen=True,audio='')
    monkeypatch.setattr(native_app,'NativeEngine',Engine)
    with TestClient(native_app.create_app('fixture',adapter_checkpoint='fixture',compare_foundation=True)) as c:
        for variant,steps in [('released',0),('candidate',244)]:
            with c.websocket_connect('/ws?variant='+variant,headers={'origin':'http://testserver'}) as ws:
                ready=ws.receive_json();assert ready['ready'] and ready['profile']['optimizer_updates']==steps
                ws.send_bytes(bytes(32000));frame=ws.receive_json()
                assert frame['sequence']==0 and frame['is_listen']
