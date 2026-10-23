"""Bounded CPU-only recognition of all six preserved repair components."""
import argparse
import asyncio
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys


async def run(a):
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '': raise ValueError('CPU-only batch required')
    header=json.loads((a.recordings/'repair-components.json').read_text())
    if (header['status']!='raw_components_complete_unapproved' or header['failures']
            or len(header['recordings'])!=6 or a.out.exists()):
        raise ValueError('Exact completed six-component input and new output required')
    a.control.mkdir(parents=True,exist_ok=False)
    state=dict(pid=os.getpid(),birth=int(Path('/proc/self/stat').read_text().split()[21]),
               source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
               phase='recognizing_all6_replacement_components',results=[],children=[],
               failures=[],maximum_concurrency=2,provider_requests=0,
               whole32_approved=False,training_approved=False)
    def save():
        state['at_utc']=datetime.now(timezone.utc).isoformat()
        p=a.control/'status.partial';p.write_text(json.dumps(state,indent=2)+'\n');p.replace(a.control/'status.json')
    save();semaphore=asyncio.Semaphore(2)
    async def one(record):
        async with semaphore:
            identity=record['id']
            args=[sys.executable,'-m','evals.native_reasoning_component_speech',
                  '--recording',str(a.recordings/identity),'--weights',str(a.weights),
                  '--out',str(a.out/identity),'--wav-sha256',record['artifacts']['model_out.wav']['sha256']]
            with(a.control/(identity+'.log')).open('x')as log:
                child=await asyncio.create_subprocess_exec(*args,stdout=log,stderr=asyncio.subprocess.STDOUT)
                state['children'].append(dict(id=identity,pid=child.pid,
                    birth=int(Path('/proc',str(child.pid),'stat').read_text().split()[21])));save()
                try:
                    code=await asyncio.wait_for(child.wait(),600)
                finally:
                    if child.returncode is None: child.kill();await child.wait()
            item=dict(id=identity,actual_exit=code)
            target=a.out/identity/'component-speech.json'
            if target.exists():item['component_speech_passed']=json.loads(target.read_text())['component_speech_verdict']['passed']
            state['results'].append(item);save()
    errors=await asyncio.gather(*(one(r)for r in header['recordings']),return_exceptions=True)
    state['failures']=[dict(type=type(e).__name__,reason=str(e))for e in errors if isinstance(e,BaseException)]
    state['phase']=('complete6_components_still_no_whole32_approval'
                    if not state['failures']and len(state['results'])==6
                    and all(r['actual_exit']==0 for r in state['results'])else'failed_preserved_no_retry')
    save();print(json.dumps(state))
    return state


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('recordings','weights','out','control'):p.add_argument('--'+name,type=Path,required=True)
    result=asyncio.run(run(p.parse_args()))
    if result['phase']=='failed_preserved_no_retry':raise SystemExit(1)
