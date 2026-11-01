"""Exact34 assistant-source batch under the owner's training-data instruction."""
import argparse,hashlib,json
from pathlib import Path
from .render_context_source import _execute_admitted

PAYLOAD=Path('docs/NATIVE-VENUS-CONTEXT-ASSISTANT-SPEECH-PAYLOAD-2026-11-01.json')
ADMISSION=Path('docs/NATIVE-VENUS-CONTEXT-ASSISTANT-SPEECH-ADMISSION-2026-11-01.json')


def admission():
    raw=PAYLOAD.read_bytes();p=json.loads(raw);a=json.loads(ADMISSION.read_text())
    if a['status']!='APPROVED_WITHIN_EXISTING_TRAINING_DATA_INSTRUCTION' or a['payload_sha256']!=hashlib.sha256(raw).hexdigest():raise ValueError('Exact assistant batch admission required')
    if a['maximum_paid_calls']!=34 or a['maximum_api_usd']!=2.5:raise ValueError('Exact new batch limit required')
    if len(p['requests'])!=34 or sum(len(r['body']['text']) for r in p['requests'])!=4449:raise ValueError('Complete exact34 required')
    if len({r['id'] for r in p['requests']})!=34:raise ValueError('Duplicate request')
    for r in p['requests']:
        if r['payload_sha256']!=hashlib.sha256(json.dumps(r['body'],sort_keys=True).encode()).hexdigest():raise ValueError('Body changed')
        if r['voice_id']!='JBFqnCBsd6RMkjVDRZzb' or r['body']['model_id']!='eleven_v3':raise ValueError('No voice/model substitution')
        if r['fold'] not in ('TRAIN','DEV') or not r['body']['text'].strip():raise ValueError('Only spoken TRAIN/DEV targets')
    backend=json.loads(Path('docs/NATIVE-VENUS-CONTEXT-BACKEND21-V2-REVIEW-2026-10-31.json').read_text())
    sources=json.loads(Path('runs/venus-training-20261101/context-source-v1/ledger.json').read_text())
    if not sources['complete'] or len(sources['attempts'])!=72:raise ValueError('Previous source cost unknown')
    phase=30300*4/3600+24+backend['combined_conservative_cost_upper_usd']+sources['conservative_cost_upper_usd']+p['conservative_cost_upper_usd']
    if phase!=a['conservative_phase_upper_with_this_batch_usd'] or phase>70 or 91.98448055555555+phase>175:raise ValueError('Carried phase/project cap exceeded')
    return p


def execute(output):return _execute_admitted(output,admission(),PAYLOAD)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);execute(p.parse_args().output)
