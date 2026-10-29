"""Run the existing microphone prototype behind a loopback-only SSH tunnel.

No transcripts/audio are collected here. Provider STOP and both independent
guards remain mandatory; the trial also ends early when unused or disconnected.
"""
import json
import subprocess
import sys,shlex
import time
from urllib.request import build_opener, ProxyHandler

# Local tunnel health must never travel through an inherited HTTP proxy.
urlopen = build_opener(ProxyHandler({})).open


def websocket_probe(url):
    """Exercise the actual conversation route before publishing a live URL."""
    from websockets.sync.client import connect
    rows=[]
    for variant,updates in (('released',0),('candidate',244)):
        with connect(url.replace('http://','ws://')+'/ws?variant='+variant,
                     origin=url, proxy=None, open_timeout=10, close_timeout=3) as ws:
            ready=json.loads(ws.recv(timeout=40))
            assert ready['ready'] and ready['profile']['optimizer_updates']==updates
            # One second of synthetic silence, never the owner's microphone.
            ws.send(bytes(32000));frame=json.loads(ws.recv(timeout=40))
            assert frame['sequence']==0 and isinstance(frame['is_listen'],bool)
            rows.append(dict(variant=variant,optimizer_updates=updates,
                sequence=frame['sequence'],is_listen=frame['is_listen'],
                audio_samples=frame.get('audio_samples'),actual_pcm_roundtrip=True))
        for _ in range(50):
            with urlopen(url+'/health',timeout=3) as response:health=json.load(response)
            if not health['busy']:break
            time.sleep(.1)
        else:raise RuntimeError('Conversation slot did not reset after native probe')
    return dict(complete=True,conditions=rows,owner_microphone_used=False,
                behavioral_approval=False)


def serve(controller, python_command, end, unit_prefix, *, port=8037,unused_timeout=300,idle_timeout=90,voice_reference=None,voice_reference_sha256=None,benchmark_only=False,tts_attention='released',benchmark_dir='latency-benchmark'):
    if not 90<=idle_timeout<=300 or not 300<=unused_timeout<=1800:
        raise ValueError("Bounded live waiting windows required")
    if bool(voice_reference)!=bool(voice_reference_sha256):raise ValueError('Exact custom reference hash required')
    c = controller
    unit = unit_prefix + '-microphone'
    checkpoint = c.OUT + '/paced-candidate-v25/step-0054.pt'
    command = python_command + ' -u -m portable.venus.native_app --model-dir ' + c.MODEL
    command += ' --adapter-checkpoint ' + checkpoint + ' --compare-foundation --port ' + str(port)
    command+=' --tts-attention '+shlex.quote(tts_attention)
    if voice_reference:command+=' --voice-reference '+shlex.quote(voice_reference)+' --voice-reference-sha256 '+voice_reference_sha256
    c.start_unit(unit, command, max(1, int(end-time.time()-180)))
    # Options precede the host. Existing strict host keys/OSLogin/IAP are reused.
    separate = [('ControlMaster=no' if x.startswith('ControlMaster=') else
                 'ControlPath=none' if x.startswith('ControlPath=') else x)
                for x in c.ssh_argv[:-1]]
    argv = separate + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
        '127.0.0.1:%d:127.0.0.1:%d' % (port, port), c.ssh_argv[-1]]
    tunnel = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=(c.R/(unit_prefix+'-tunnel.log')).open('x'))
    (c.R/'live-tunnel.json').write_text(json.dumps(dict(pid=tunnel.pid, port=port,
        address='127.0.0.1', credentials_recorded=False))+'\n')
    url = 'http://127.0.0.1:%d' % port
    ready = None
    started = time.time()
    last_health = started
    first_conversation = False
    last_busy = None
    try:
        while time.time() < end-180:
            if tunnel.poll() is not None or c.guard.poll() is not None:
                raise RuntimeError('Live tunnel or independent guard ended')
            if (c.R/'end-live-trial').exists():
                c.record('live_trial_end_requested'); break
            try:
                with urlopen(url+'/health', timeout=5) as response:
                    health = json.load(response)
            except (OSError, ValueError):
                if ready is None and time.time()-started > 180:
                    raise RuntimeError('Native microphone server did not become healthy')
                if ready is not None and time.time()-last_health > 20:
                    raise RuntimeError('Live transport lost health after startup')
                time.sleep(2); continue
            last_health=time.time()
            if ready is None:
                assert health['ready'] and health['optimizer_updates']==244
                assert health['candidate']['checkpoint_sha256']=='c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270'
                proof=websocket_probe(url)
                (c.R/'native-websocket-proof.json').write_text(json.dumps(proof,indent=2)+'\n')
                if benchmark_only:
                    c.record('native_benchmark_service_ready',reference_sha256=health['voice_reference']['sha256'])
                    target=c.R/benchmark_dir
                    budget=min(300,int(end-time.time()-180))
                    if budget<60:raise TimeoutError('Preserve complete benchmark and backup reserve')
                    with (c.R/(benchmark_dir+'.log')).open('x') as log:
                        result=subprocess.run([sys.executable,'-u','-m','evals.venus_stream_benchmark','--url',url,'--output',str(target),'--save-synthetic-audio'],stdout=log,stderr=subprocess.STDOUT,timeout=budget)
                    if result.returncode:raise RuntimeError('Actual native latency benchmark failed; preserve logs')
                    measured=json.loads((target/'timings.json').read_text());assert measured['complete'] and len(measured['profiles'])==2 and len(measured['frames'])==22
                    assert all(p['profile']['voice_reference']['sha256']==voice_reference_sha256 for p in measured['profiles'])
                    c.record('complete_native_synthetic_benchmark',summary=measured['summaries'],owner_audio_used=False,optimizer_updates=0)
                    return
                ready = time.time()
                (c.R/'live-ready.json').write_text(json.dumps(dict(url=url, ready_unix=ready,
                    stop_unix=end, profile=health, experimental=True, microphone_recorded=False),indent=2)+'\n')
                c.record('native_microphone_ready', url=url, optimizer_updates=244,
                    microphone_recorded=False, general_reasoning_backend=False)
            if health['busy']:
                first_conversation=True; last_busy=time.time()
            elif first_conversation and time.time()-last_busy>idle_timeout:
                c.record('live_trial_idle_after_conversation'); break
            elif not first_conversation and time.time()-ready>unused_timeout:
                c.record('live_trial_unused_five_minutes' if unused_timeout==300 else 'live_trial_unused_window_elapsed',unused_seconds=unused_timeout); break
            time.sleep(3)
    finally:
        tunnel.terminate()
        try:tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:tunnel.kill()

