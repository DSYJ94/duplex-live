"""Run the existing microphone prototype behind a loopback-only SSH tunnel.

No transcripts/audio are collected here. Provider STOP and both independent
guards remain mandatory; the trial also ends early when unused or disconnected.
"""
import json
import subprocess
import time
from urllib.request import build_opener, ProxyHandler

# Local tunnel health must never travel through an inherited HTTP proxy.
urlopen = build_opener(ProxyHandler({})).open


def serve(controller, python_command, end, unit_prefix, *, port=8037):
    c = controller
    unit = unit_prefix + '-microphone'
    checkpoint = c.OUT + '/paced-candidate-v25/step-0054.pt'
    command = python_command + ' -u -m portable.venus.native_app --model-dir ' + c.MODEL
    command += ' --adapter-checkpoint ' + checkpoint + ' --compare-foundation --port ' + str(port)
    c.start_unit(unit, command, max(1, int(end-time.time()-180)))
    # Options precede the host. Existing strict host keys/OSLogin/IAP are reused.
    separate = [('ControlMaster=no' if x.startswith('ControlMaster=') else
                 'ControlPath=none' if x.startswith('ControlPath=') else x)
                for x in c.ssh_argv[:-1]]
    argv = separate + ['-o', 'ExitOnForwardFailure=yes', '-N', '-L',
        '127.0.0.1:%d:127.0.0.1:%d' % (port, port), c.ssh_argv[-1]]
    tunnel = subprocess.Popen(argv, stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=(c.R/'live-tunnel.log').open('x'))
    (c.R/'live-tunnel.json').write_text(json.dumps(dict(pid=tunnel.pid, port=port,
        address='127.0.0.1', credentials_recorded=False))+'\n')
    url = 'http://127.0.0.1:%d' % port
    ready = None
    started = time.time()
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
                if time.time()-started > 180:
                    raise RuntimeError('Native microphone server did not become healthy')
                time.sleep(2); continue
            if ready is None:
                assert health['ready'] and health['optimizer_updates']==244
                assert health['candidate']['checkpoint_sha256']=='c634e6cf38831210be0165ce7f6dd09c5454226f62ff622477fd4d051abde270'
                ready = time.time()
                (c.R/'live-ready.json').write_text(json.dumps(dict(url=url, ready_unix=ready,
                    stop_unix=end, profile=health, experimental=True, microphone_recorded=False),indent=2)+'\n')
                c.record('native_microphone_ready', url=url, optimizer_updates=244,
                    microphone_recorded=False, general_reasoning_backend=False)
            if health['busy']:
                first_conversation=True; last_busy=time.time()
            elif first_conversation and time.time()-last_busy>90:
                c.record('live_trial_idle_after_conversation'); break
            elif not first_conversation and time.time()-ready>300:
                c.record('live_trial_unused_five_minutes'); break
            time.sleep(3)
    finally:
        tunnel.terminate()
        try:tunnel.wait(timeout=10)
        except subprocess.TimeoutExpired:tunnel.kill()

