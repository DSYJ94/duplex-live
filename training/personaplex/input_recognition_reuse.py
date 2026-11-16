"""Reuse existing unbiased input recognition, verifying the exact evaluated PCM.

No recognizer call, student inference or gold answer is needed. This proves
source intelligibility to Scribe, not the native model's internal perception.
"""
import argparse,json
from pathlib import Path
from .local_reasoner_reference import digest


def run(data,parent,probe,out):
    import numpy as np
    import soundfile as sf
    data=Path(data);parent=Path(parent);probe=Path(probe);out=Path(out)
    manifest=json.loads((data/'manifest.json').read_text());quality=json.loads((data/'qualification/quality.json').read_text());prep=json.loads((parent/'input-preparation.json').read_text())
    if (digest(data/'manifest.json')!=prep['source_manifest_sha256']or digest(data/'qualification/quality.json')!=manifest['quality_sha256']
            or not quality['complete']or quality['requested']!=2144 or quality['protocol']!='complete2144_scribe3_no_reference_or_bias_v1'):raise ValueError('Exact complete unbiased original source audit required')
    source={r['id']:r for r in manifest['rows']if r['fold']=='DEV'};audit={r['id']:r for r in quality['rows']}
    questions=[r for r in json.loads((probe/'questions.json').read_text())if r['kind']=='quantitative-DEV']
    if len(source)!=64 or len(questions)!=64 or set(source)!={r['id']for r in questions}:raise ValueError('All64 original quantitative questions required')
    results=[]
    for q in questions:
        row=source[q['id']];original=data/row['path'];wave=Path(q['audio']);side=original.with_suffix('.json')
        if digest(original)!=row['wav_sha256']or digest(side)!=row['sidecar_sha256']or digest(wave)!=q['audio_sha256']:raise ValueError('Source or evaluation WAV changed')
        parents=[p for p in row['parents']if p['id']==q['id']+'-user']
        if len(parents)!=1:raise ValueError('One exact source user parent required')
        item=parents[0];a=audit[item['id']]
        if a['source_sha256']!=item['sha256']or a['source_words']!=q['question']:raise ValueError('Independent ASR source/question lineage mismatch')
        stereo,sr=sf.read(original,dtype='float32',always_2d=True);user,rate=sf.read(wave,dtype='float32');n=item['resampled_frames']
        if sr!=24000 or rate!=24000 or stereo.shape[1]!=2 or user.ndim!=1 or not np.array_equal(user[:n],stereo[:n,1])or np.any(user[n:]):raise ValueError('Evaluated whole user PCM must exactly match original source, plus only silence')
        results.append(dict(id=q['id'],question=q['question'],recognized=a['recognized'],WER=a['WER'],word_pass=a['word_pass'],source_audio_sha256=item['sha256'],evaluated_audio_sha256=q['audio_sha256'],exact_resampled_user_PCM_verified=True,words=a['words']))
    record=dict(complete64=True,reused_existing_recognition=True,recognizer=quality['recognizer'],protocol=quality['protocol'],quality_sha256=manifest['quality_sha256'],source_manifest_sha256=prep['source_manifest_sha256'],results=results,external_API_calls=0,new_model_inferences=0,optimizer_updates=0,native_internal_perception_proven=False)
    with out.open('x')as f:f.write(json.dumps(record,indent=2)+'\n')
    return record


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for key in ('data','parent','probe','out'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();r=run(a.data,a.parent,a.probe,a.out);print(json.dumps(dict(complete64=r['complete64'],passed=sum(x['word_pass']for x in r['results']),new_model_inferences=0,external_API_calls=0)))
