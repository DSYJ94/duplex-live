"""Typed complete32 view over preserved original and diagnosed replacement traces.

Every source, original failed audit and actual recorder profile is checked. No
provider request, audio edit, semantic approval or training occurs here.
"""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from types import SimpleNamespace

from evals import native_reasoning_teacher_speech_v2 as region
from evals import native_reasoning_teacher_recovery_v6 as legacy
from evals import native_reasoning_teacher_v6 as v6
from evals import native_reasoning_teacher_v8 as v8
from evals import native_reasoning_teacher_ack_probe as v9
from evals import native_reasoning_teacher_cutoff_readback as cutoff
from evals import native_reasoning_teacher_repair_v8 as repair
from evals.native_reasoning_source import sha
from evals.native_reasoning_source_pacing import verify as source_manifest
from evals.native_reasoning_teacher import geometry
from evals.native_reasoning_teacher_lifecycle_review import case_review
from evals.native_reasoning_lifecycle import canonical_hash
from evals.native_reasoning_verification_session import source_readbacks

SCHEMA='dl-native-reasoning32-derived-teacher-collection-v1'
IDS=[f'native-reasoning4-train-{i:03d}'for i in range(32)]
PATHS=('source','source_quality','parent','old_failed','region_plan','region_worker',
       'cutoff','cutoff_proof','cutoff_speech','repairs','repair_speech','engine','tokenizer')


def load(path):
    value=json.loads(path.read_text())
    if set(value) not in (set(PATHS),set(PATHS)|{'ack','ack_speech'}):
        raise ValueError('Explicit complete input map required; no unknown/missing roots')
    return SimpleNamespace(**{k:Path(v)for k,v in value.items()})


def source_pins(c):
    files=dict(source_manifest=c.source/'source-audio.json',source_quality=c.source_quality,
               original_collection=c.parent/'teacher-collection.json',original_failed_speech=c.old_failed,
               original_region_plan=c.region_plan,original_region_speech=c.region_worker/'teacher-speech-quality.json',
               cutoff_component=c.cutoff/'cutoff-component.json',cutoff_readback=c.cutoff_proof,
               repair_components=c.repairs/'repair-components.json')
    if hasattr(c,'ack'):files['ack_component']=c.ack/'ack-component.json'
    return {k:sha(p)for k,p in files.items()}


def input_roots(c):
    # Both approvals reference independent ASR siblings, not only their header.
    return [*vars(c).values(),c.source_quality.parent,c.old_failed.parent]


def derive(c):
    previous=region.verify(c.source,c.source_quality,c.parent,c.engine,c.tokenizer,
                           c.old_failed,c.region_plan,c.region_worker,require_pass=False)
    if ([r['id']for r in previous['results']if not r['passed']]!=repair.FAILED
            or previous['passed']is not False or sha(c.parent/'teacher-collection.json')!=repair.PARENT
            or sha(c.region_worker/'teacher-speech-quality.json')!=repair.SPEECH):
        raise ValueError('Exact full32 failed parent and all25 unchanged passes required')
    parent=json.loads((c.parent/'teacher-collection.json').read_text())
    new=json.loads((c.repairs/'repair-components.json').read_text())
    if (new['status']!='raw_components_complete_unapproved' or new['failures']
            or new['source_commit']!='e762d82a2a83a53cffaae7597a05f000090a6c3e'
            or new['code_hashes']!=v8.code_hashes() or new['collector_code_sha256']!=sha(Path(repair.__file__))
            or new['failed_whole32_speech_sha256']!=repair.SPEECH
            or new['source_manifest_sha256']!=sha(c.source/'source-audio.json')
            or new['source_quality_sha256']!=sha(c.source_quality)
            or new['attempted_ids']!=repair.REPAIR or len(new['recordings'])!=6
            or sorted(r['id']for r in new['recordings'])!=repair.REPAIR
            or tuple(new[k]for k in ('maximum_new_voice_sessions','maximum_concurrent_sessions',
                'maximum_aggregate_voice_attempts','maximum_aggregate_backend_request_upper_bound'))!=(6,2,45,115)
            or any(new[k]is not False for k in ('automatic_retry','passing25_recaptured',
                'originals_or_source_words_changed','whole32_approved','semantic_approved','training_approved'))):
        raise ValueError('Complete bounded actual V8 six-repair collection required')
    # Reproduce the independent corrected readback rather than trusting its flag.
    with tempfile.TemporaryDirectory(prefix='duplex-cutoff-readback-')as temporary:
        actual=cutoff.review(c.source,c.source_quality,c.parent,c.old_failed,c.cutoff,
                             c.engine,c.tokenizer,Path(temporary)/'readback.json')
    recorded=json.loads(c.cutoff_proof.read_text())
    if {k:v for k,v in actual.items()if k not in ('at_utc','source_commit')}!={
            k:v for k,v in recorded.items()if k not in ('at_utc','source_commit')}:
        raise ValueError('Original independently corrected cutoff readback differs')
    component=json.loads((c.cutoff/'cutoff-component.json').read_text())
    records={r['id']:('v6',r,c.parent/r['id'])for r in parent['recordings']}
    records['native-reasoning4-train-001']=('cutoff-v7',component['recording'],c.cutoff/'native-reasoning4-train-001')
    for record in new['recordings']:
        if record['recorder_version']!='v8' or record['clarity_profile']!=(record['id']!='native-reasoning4-train-005'):
            raise ValueError('Exact diagnosed V8 recorder profile required')
        records[record['id']]=('repair-v8',record,c.repairs/record['id'])
    if hasattr(c,'ack'):
        v9.diagnosis(c.repairs,c.repair_speech/v9.CASE/'component-speech.json')
        ack=json.loads((c.ack/'ack-component.json').read_text())
        if (ack['status']!='raw_component_complete_independent_speech_and_semantics_required'
                or ack['case_id']!=v9.CASE or ack['code_hashes']!=v9.code_hashes()
                or ack['parent_repair_manifest_sha256']!=sha(c.repairs/'repair-components.json')
                or ack['failed_component_speech_sha256']!=v9.FAILED_SPEECH_SHA
                or ack['source_manifest_sha256']!=sha(c.source/'source-audio.json')
                or ack['source_quality_sha256']!=sha(c.source_quality)
                or tuple(ack[k]for k in ('maximum_new_voice_sessions','maximum_backend_requests',
                    'maximum_aggregate_voice_attempts','maximum_aggregate_backend_request_upper_bound'))!=(1,6,46,121)
                or any(ack[k]is not False for k in ('automatic_retry','original_audio_or_words_changed',
                    'teacher_approved','semantic_approved','training_approved'))):
            raise ValueError('Exact bounded diagnosed006 acknowledgment experiment required')
        records[v9.CASE]=('ack-v9',ack['recording'],c.ack/v9.CASE)
    manifest,plan=source_manifest(c.source)
    if sorted(records)!=IDS or [s['id']for s in manifest['scenes']]!=IDS:
        raise ValueError('Every original32 scene required')
    rows=[dict(id=identity,recorder=records[identity][0],recording=records[identity][1],
               recording_sha256=canonical_hash(records[identity][1]))for identity in IDS]
    expected=dict(schema=SCHEMA,code_sha256=sha(Path(__file__)),input_manifest_hashes=source_pins(c),
                  complete_cases=32,recordings=rows,original_audio_edited=False,
                  passing_subset_accepted=False,teacher_approved=False,training_approved=False)
    return expected,manifest,plan,records


def validate_view(c,batch,expected,manifest,plan,records):
    header=json.loads((batch/'teacher-collection.json').read_text())
    if (set(header)!=set(expected)|{'created_utc','prepared_source_commit'}
            or any(header[k]!=v for k,v in expected.items())
            or not re.fullmatch('[0-9a-f]{40}',header['prepared_source_commit'])):
        raise ValueError('Exact complete32 typed view and original source lineage required')
    if {p.name for p in batch.iterdir()}!=set(IDS)|{'teacher-collection.json'}:
        raise ValueError('Unknown or missing complete32 view member')
    tokenizer=geometry(c.engine,c.tokenizer);cases={r['id']:r for r in plan['cases']}
    for scene in manifest['scenes']:
        identity=scene['id'];kind,record,origin=records[identity];folder=batch/identity
        names={p.relative_to(folder).as_posix()for p in folder.rglob('*')if p.is_file()}
        if (folder.is_symlink()or any(p.is_symlink()for p in folder.rglob('*'))
                or names!=set(record['artifacts'])):raise ValueError('Complete exact raw artifact inventory required')
        for name,row in record['artifacts'].items():
            path=folder/name
            if (Path(name).is_absolute()or'..'in Path(name).parts or path.is_symlink()
                    or sha(path)!=row['sha256']or path.stat().st_size!=row['bytes']
                    or sha(origin/name)!=row['sha256']):raise ValueError('Changed raw teacher artifact or original')
        if kind=='v6':
            report=(legacy.old_readback(c.source,scene,folder,record,tokenizer)
                    if record['reused_in_v6']else v6.verify_record(c.source,scene,folder,record,tokenizer))
        elif kind=='cutoff-v7':report=cutoff.verify_record(c.source,scene,folder,record,tokenizer)
        elif kind=='repair-v8':report=v8.implementation(record['clarity_profile'])[1](c.source,scene,folder,record,tokenizer)
        elif kind=='ack-v9':report=v9.implementation()[1](c.source,scene,folder,record,tokenizer)
        else:raise ValueError('Unknown recorder profile')
        if (report['native_frame_budget']['frame_count_diagnostic_pass']is not True
                or not case_review(cases[identity],json.loads((folder/'lifecycle.json').read_text()))['structural_pass']):
            raise ValueError('All32 full frame/lifecycle checks required')
    return header,manifest,plan


def verify(c,batch):
    expected,manifest,plan,records=derive(c)
    return validate_view(c,batch,expected,manifest,plan,records)


def prepare(c,out):
    roots=input_roots(c)
    with source_readbacks(roots,include_old_speech=True)as memo:
        expected,manifest,plan,records=derive(c);out.mkdir(parents=True,exist_ok=False)
        for identity,(_,_,origin)in records.items():shutil.copytree(origin,out/identity)
        header=dict(expected,created_utc=datetime.now(timezone.utc).isoformat(),
                    prepared_source_commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip())
        (out/'teacher-collection.json').write_text(json.dumps(header,indent=2)+'\n')
        validate_view(c,out,expected,manifest,plan,records)
    return dict(complete_cases=32,source_checker_executions=memo.calls,
                duplicate_reads_reused=memo.hits,teacher_approved=False,training_approved=False)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);a=p.parse_args();print(json.dumps(prepare(load(a.config),a.out)))
