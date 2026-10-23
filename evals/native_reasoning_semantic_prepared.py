"""Review a remotely verified whole32 corpus using only checked local metadata.

The CPU audio/transport audit runs remotely. This Mac sends exactly the same
source-bound reviewer messages without rerunning neural ASR or waveform audits.
No API key is transferred; no owner data or scoring oracle enters a payload.
"""
import argparse
import asyncio
from datetime import datetime,timezone
import json
from pathlib import Path
import subprocess

from evals import native_reasoning_semantic as semantic
from evals import native_reasoning_teacher_mixed as mixed
from evals import native_reasoning_teacher_mixed_speech as speech
from evals.native_reasoning_source import sha
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_teacher_quality import expected_output

SCHEMA='dl-native-reasoning32-prepared-semantic-inputs-v1'
METADATA=('report.json','environment.json','lifecycle.json','events.jsonl','transcripts.json')


def checked_messages(c,batch,approval):
    header=json.loads((batch/'teacher-collection.json').read_text());qualified=json.loads(approval.read_text())
    if (header['schema']!=mixed.SCHEMA or header['code_sha256']!=sha(Path(mixed.__file__))
            or header['complete_cases']!=32 or [r['id']for r in header['recordings']]!=mixed.IDS
            or header['input_manifest_hashes']!=mixed.source_pins(c)
            or [r['id']for r in qualified['results']]!=mixed.IDS
            or not all(r['passed']for r in qualified['results'])):
        raise ValueError('Exact complete32 approved corpus metadata required')
    speech.verify_value(c,batch,qualified,qualified['results'])
    manifest=json.loads((c.source/'source-audio.json').read_text());plan=json.loads((c.source/'source-plan.json').read_text())
    if (sha(c.source/'source-plan.json')!=manifest['plan_sha256']
            or [s['id']for s in manifest['scenes']]!=mixed.IDS or [s['id']for s in plan['cases']]!=mixed.IDS):
        raise ValueError('Exact complete authored synthetic source plan required')
    folders={}
    for record,audio in zip(header['recordings'],qualified['results']):
        identity=record['id'];kind=record['recorder'];raw=record['recording']
        roots={'v6':c.parent,'cutoff-v7':c.cutoff,'repair-v8':c.repairs}
        if hasattr(c,'ack'):roots.update({'ack-v9':c.ack,'ack-v10':c.ack})
        if kind not in roots or record['recording_sha256']!=canonical_hash(raw):
            raise ValueError('Unknown or changed actual recorder identity')
        folder=roots[kind]/identity;folders[identity]=folder
        for name in METADATA:
            path=folder/name;binding=raw['artifacts'][name]
            if path.is_symlink()or path.stat().st_size!=binding['bytes']or sha(path)!=binding['sha256']:
                raise ValueError('API-bound actual event/receipt metadata changed')
        if (audio['original_rendered_wav_sha256']!=raw['artifacts']['model_out.wav']['sha256']
                or audio['actual_provider_transcript']!=expected_output(folder)):
            raise ValueError('Approved speech differs from actual source-bound words')
    return semantic.build_messages(manifest,plan,qualified,batch,folders)


def envelope(c,batch,approval,sent):
    return dict(schema=SCHEMA,code_sha256=sha(Path(__file__)),semantic_code_sha256=sha(Path(semantic.__file__)),
        request_code_sha256=sha(Path('training/control/teacher_data.py')),model=semantic.MODEL,
        teacher_manifest_sha256=sha(batch/'teacher-collection.json'),speech_approval_sha256=sha(approval),
        source_quality_sha256=sha(c.source_quality),input_manifest_hashes=mixed.source_pins(c),
        selected_ids=list(sent),maximum_requests=32,maximum_concurrency=2,messages=sent,
        input_sha256={identity:canonical_hash(message)for identity,message in sent.items()},
        whole32_speech_passed=True,source_oracles_sent=False,training_approved=False)


def prepare(c,batch,approval,out):
    # This receipt is emitted ONLY after the actual full transport check,
    # independent twice-recomputed speech decisions and snapshot closure.
    proof_path=approval.parent/'independent-readback.json';proof=json.loads(proof_path.read_text())
    if (proof['complete_cases']!=32 or proof['whole_speech_passed']is not True
            or proof['all_original_transports_verified']is not True
            or proof['all_speech_decisions_recomputed_twice']is not True
            or proof['unchanged_input_snapshot']is not True or proof['report_sha256']!=sha(approval)
            or proof['source_commit']!=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()):
        raise ValueError('Fresh actual complete32 CPU readback required before preparing reviewer inputs')
    sent=checked_messages(c,batch,approval)
    value=dict(envelope(c,batch,approval,sent),created_utc=datetime.now(timezone.utc).isoformat(),
               verified_readback_sha256=sha(proof_path),source_commit=proof['source_commit'])
    with out.open('x')as stream:stream.write(json.dumps(value,indent=2)+'\n')
    return value


def validate(c,batch,approval,prepared):
    sent=checked_messages(c,batch,approval);value=json.loads(prepared.read_text());expected=envelope(c,batch,approval,sent)
    if (set(value)!=set(expected)|{'created_utc','verified_readback_sha256','source_commit'}
            or any(value[k]!=v for k,v in expected.items())
            or value['verified_readback_sha256']!=sha(approval.parent/'independent-readback.json')):
        raise ValueError('Exact source-bound whole32 reviewer payloads required; no edits or subset')
    proof=json.loads((approval.parent/'independent-readback.json').read_text())
    if (proof['report_sha256']!=sha(approval)or proof['whole_speech_passed']is not True
            or proof['complete_cases']!=32 or value['source_commit']!=proof['source_commit']
            or any(proof[k]is not True for k in ('all_original_transports_verified',
                'all_speech_decisions_recomputed_twice','unchanged_input_snapshot'))
            or proof['training_approved']is not False):
        raise ValueError('Independent complete32 speech receipt changed')
    return sent


async def review(c,batch,approval,prepared,out,env_file):
    sent=validate(c,batch,approval,prepared)
    result=await semantic.review_messages(sent,c.source_quality,batch,approval,out,env_file,prepared_inputs=prepared)
    sent=validate(c,batch,approval,prepared)
    semantic.verify_responses(sent,c.source_quality,batch,approval,out/'semantic.json',prepared_inputs=prepared)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('mode',choices=('prepare','review'))
    for name in ('config','batch','speech-approval','prepared'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('out','env-file'):p.add_argument('--'+name,type=Path)
    a=p.parse_args();c=mixed.load(a.config)
    if a.mode=='prepare':
        value=prepare(c,a.batch,a.speech_approval,a.prepared)
        print(json.dumps(dict(prepared=len(value['messages']),provider_requests=0,training_approved=False)))
    else:
        if a.out is None or a.env_file is None:p.error('Local review requires output and the existing local key file')
        value=asyncio.run(review(c,a.batch,a.speech_approval,a.prepared,a.out,a.env_file))
        print(json.dumps(dict(complete=len(value['results']),passed=value['passed'],training_approved=False)))
