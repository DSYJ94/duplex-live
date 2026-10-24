"""Convert the complete32 clock-reapproved collection using unchanged geometry.

Rechecks all original transports, every original speech verdict, both independent
clock repairs and all32 semantic response bindings. The frozen converter's PCM,
word grid, lifecycle and full1100-frame checks are reused without modification.
"""
import argparse
from copy import deepcopy
import json
from pathlib import Path
from types import FunctionType

from evals import native_reasoning_convert as converter
from evals import native_reasoning_timing_semantic as reviewer
from evals import native_reasoning_timing_reapproval as timing
from evals import native_reasoning_burst_asr as burst
from evals import native_reasoning_teacher_mixed as mixed
from evals import native_reasoning_teacher_mixed_speech as speech
from evals import native_reasoning_source_quality as quality
from evals import native_reasoning_semantic as semantic
from evals import native_reasoning_teacher_speech_v2 as region
from evals.native_reasoning_source import sha
from evals.native_reasoning_verification_session import source_readbacks

SCHEMA='dl-native-reasoning32-clock-reapproved-measured-conversion-v1'


def code_pins():
    return dict(converter.code_pins(),**{str(Path(m.__file__).relative_to(Path.cwd())):sha(Path(m.__file__))
        for m in (timing,reviewer,burst)},**{str(Path(__file__).relative_to(Path.cwd())):sha(Path(__file__))})


def approved(a,version,semantic_approval):
    import soundfile as sf
    c,sent,_=reviewer.validated(a,version)
    semantic.verify_responses(sent,c.source_quality,a['batch'],version/'teacher-speech-quality.json',
        semantic_approval,prepared_inputs=version/'prepared-semantic.json')
    result=json.loads(semantic_approval.read_text())
    if (result['derived_reviewer_code_sha256']!=sha(Path(reviewer.__file__))
            or result['maximum_new_requests']!=2 or result['actual_new_requests_started']!=2
            or result['actual_reused_responses']!=30 or result['new_review_ids']!=list(timing.IDS)
            or result['original_failed_semantic_sha256']!=sha(a['failed_semantic'])):
        raise ValueError('Exact bounded two-request/30-reused complete semantic lineage required')
    header,manifest,plan=mixed.verify(c,a['batch'])
    old=json.loads(a['old_speech'].read_text())
    speech.verify_value(c,a['batch'],old,speech.rows(c,a['batch'],header))
    new=timing.checked_bursts(a,old);wanted=deepcopy(old['results'])
    for row in wanted:
        if row['id']not in new:
            continue
        raw,binding=new[row['id']];pcm,rate=sf.read(a['batch']/row['id']/'model_out.wav',dtype='float32')
        if region.regions(pcm,rate)!=binding['original_regions']:
            raise ValueError('Every original waveform region must remain complete')
        row.update(quality.assess(row['actual_provider_transcript'],raw,pcm,rate))
        row['recognition_binding']=dict(kind='independent-burst-clock-v11',
            wav_sha256=binding['wav_sha256'],asr_sha256=binding['asr_sha256'],
            recognition_sha256=sha(a['burst']/'recognition.json'))
    qualified=json.loads((version/'teacher-speech-quality.json').read_text())
    if wanted!=qualified['results'] or len(wanted)!=32 or not all(r['passed']for r in wanted):
        raise ValueError('All32 original speech decisions and clock repairs must reproduce exactly')
    return manifest,plan,qualified


def bind(function,**values):
    result=FunctionType(function.__code__,dict(function.__globals__,**values),
        function.__name__,function.__defaults__,function.__closure__)
    result.__kwdefaults__=function.__kwdefaults__
    return result


def run(a,version,semantic_approval,out,verify_only=False):
    c=mixed.load(a['config'])
    roots=[*mixed.input_roots(c),*a.values(),a['old_speech'].parent,
        a['failed_semantic'].parent,version,semantic_approval.parent]
    if any(out.resolve().is_relative_to(p.resolve())for p in roots if p.is_dir()):
        raise ValueError('New conversion must be outside all immutable inputs')
    if verify_only:roots.append(out)
    def approvals(cc,batch,speech_approval,prepared,semantic_path):
        if (vars(cc)!=vars(c)or batch!=a['batch']or speech_approval!=version/'teacher-speech-quality.json'
                or prepared!=version/'prepared-semantic.json'or semantic_path!=semantic_approval):
            raise ValueError('Exact complete32 conversion inputs required')
        return approved(a,version,semantic_approval)
    check=bind(converter.verify_derived,SCHEMA=SCHEMA,code_pins=code_pins)
    operation=bind(converter.verify if verify_only else converter.convert,
        approvals=approvals,SCHEMA=SCHEMA,code_pins=code_pins,verify_derived=check)
    with source_readbacks(roots,include_old_speech=True):
        return operation(c,a['batch'],version/'teacher-speech-quality.json',
            version/'prepared-semantic.json',semantic_approval,out)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for n in ('inputs','version','semantic','out'):
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--verify-only',action='store_true')
    a=p.parse_args();v=run(timing.load(a.inputs),a.version,a.semantic,a.out,a.verify_only)
    print(json.dumps(dict(complete32=v['complete32'],converted=len(v['accepted']),training_approved=False)))
