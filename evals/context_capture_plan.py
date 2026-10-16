"""Plan TRAIN captures from approved same-user teachers; never approve targets.

Derive scene kind and private world from manifests instead of assuming fold IDs
or reusing a convenient private value. No model/API call or optimizer runs here.
"""
import argparse
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def main():
    import soundfile as sf
    from evals.recovery_data import validate_cache, validate_conversion
    from evals.context_actual_targets import matching_user_stimulus
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data',type=Path,required=True)
    ap.add_argument('--cache',type=Path,action='append',required=True)
    ap.add_argument('--out',type=Path,required=True)
    args = ap.parse_args()
    actual = []
    for row in validate_conversion(args.data)['accepted']:
        if row['split'] != 'train': continue
        side = json.loads(Path(row['path']).with_suffix('.json').read_text())
        pcm, rate = sf.read(row['path'],dtype='float32',always_2d=True)
        if rate != 24000 or pcm.shape[1] != 2: raise RuntimeError('Invalid approved source PCM')
        actual.append((row,side,pcm[:,1].copy()))
    planned, rejected, seen = [], [], set()
    for cache in args.cache:
        receipt = validate_cache(cache)
        sources = {r['scenario_id']:r for r in validate_conversion(Path(receipt['conversion']))['accepted']}
        for cached in receipt['examples']:
            teacher = sources[cached['scenario_id']]
            if cached['split'] != 'train' or teacher['engine'] != 'gpt-live-1': continue
            if teacher['split'] != 'train' or teacher['wav_sha256'] in seen:
                raise RuntimeError('Duplicate or non-TRAIN teacher in capture plan')
            seen.add(teacher['wav_sha256'])
            side = json.loads(Path(teacher['path']).with_suffix('.json').read_text())
            meta = side['timeline']['meta']
            pcm,rate = sf.read(teacher['path'],dtype='float32',always_2d=True)
            if rate != 24000 or pcm.shape[1] != 2: raise RuntimeError('Invalid approved teacher PCM')
            matches = [(row,source) for row,source,user in actual
                if row['kind'] == teacher['kind']
                and all(source['timeline']['meta'].get(k) == meta.get(k) for k in ('project','field'))
                and matching_user_stimulus(user,pcm[:,1],source['user_words'],side['user_words'])]
            value = meta.get('phrase') or 'unused value'
            if len(matches) != 1 or not isinstance(value,str) or len(value.split()) != 2:
                rejected.append({'teacher':cached['scenario_id'],'matching_sources':len(matches),
                    'reason':'Need one exact same-user source and explicit two-word world'})
                continue
            row,_ = matches[0]
            planned.append({'scenario':row['scenario_id'],'expected_kind':row['kind'],
                'private_value':value,'teacher':cached['scenario_id'],'teacher_kind':teacher['kind'],
                'teacher_cache':str(cache),'teacher_cache_sha256':sha256(cache/'cache.json'),
                'teacher_codes_sha256':cached['sha256'],'teacher_wav_sha256':teacher['wav_sha256'],
                'teacher_sidecar_sha256':teacher['sidecar_sha256'],
                'source_wav_sha256':row['wav_sha256'],'source_sidecar_sha256':row['sidecar_sha256'],
                'project':meta.get('project'),'field':meta.get('field'),
                'identical_user_pcm_and_words':True,'teacher_source_model':'gpt-live-1',
                'conditioning_only':True,'training_approved':False})
    result = {'schema':'dl-context-approved-teacher-capture-plan-v1',
        'source_data':str(args.data),'source_ingestion_sha256':sha256(args.data/'ingestion.json'),
        'planned':planned,'rejected':rejected,'heldout_used':False,
        'labels_created':False,'training_approved':False}
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'planned':len(planned),'rejected':len(rejected),
        'kinds':{k:sum(r['expected_kind']==k for r in planned)
            for k in ('private','simple','clarify','cancel')},'training_approved':False}))


if __name__ == '__main__': main()
