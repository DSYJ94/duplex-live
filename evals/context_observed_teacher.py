"""Remove only a proved nonexistent final cache-padding frame; never crop speech."""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def observed_end(clean,samples,rate):
    import torch
    if rate!=24000 or samples%1920 or clean.ndim!=2 or clean.shape[0]!=17:
        raise RuntimeError('Observed teacher boundary needs complete native source frames')
    frames=samples//1920;cached=clean.shape[1]
    if (cached!=frames+1 or math.ceil((samples/rate)*12.5)!=cached
            or not 0<(samples/rate)*12.5-frames<1e-9
            or not torch.all(clean[1:,frames:]==-1)
            or not torch.all((clean[0,frames:]==0)|(clean[0,frames:]==3))
            or (clean[1:,:frames]<0).any() or (clean[1:,:frames]>=2048).any()):
        raise RuntimeError('Unavailable teacher acoustics are not exactly one floating-point end-padding frame')
    return frames


def canonical(example,spm):
    import torch
    import soundfile as sf
    from evals.recovery_data import validate_conversion
    from evals.recovery_encode import native_word_timing
    from contextspan.model.sequence_convention import assemble_training_sequence
    clean=example['clean_codes'];r=example['record']
    if not ((clean[1:]<0)|(clean[1:]>=2048)).any():return example
    if r['source_model']!='gpt-live-1' or r['source_split']!='train':
        raise RuntimeError('Observed native teacher repair is restricted to approved independent TRAIN data')
    cache=json.loads((Path(r['cache'])/'cache.json').read_text())
    source=next(x for x in validate_conversion(Path(cache['conversion']))['accepted'] if x['scenario_id']==r['id'])
    if sha256(Path(source['path']))!=r['wav_sha256']:
        raise RuntimeError('Teacher source wave changed')
    info=sf.info(source['path']);frames=observed_end(clean,info.frames,info.samplerate)
    side=json.loads(Path(source['path']).with_suffix('.json').read_text())
    if (max(a[1][1] for a in side['alignments'])>frames*.08
            or any(s['inject_frame']>=frames for s in r['spans'])):
        raise RuntimeError('Observed boundary would discard speech or a provided context')
    timing=native_word_timing(clean[0].tolist(),side['alignments'],spm.encode,12.5)
    trimmed=clean[:,:frames].clone()
    if native_word_timing(trimmed[0].tolist(),side['alignments'],spm.encode,12.5)!=timing:
        raise RuntimeError('Observed boundary changed a measured word target')
    codes,tmask,amask=assemble_training_sequence(trimmed,torch.ones(frames,dtype=torch.bool),r['spans'],spm)
    prefix=example['codes'][:,:r['prefix_frames']]
    result={**example,'clean_codes':trimmed,'codes':torch.cat([prefix,codes],dim=1),
        'text_mask':torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),tmask]),
        'audio_mask':torch.cat([torch.zeros(prefix.shape[1],dtype=torch.bool),amask])}
    result['user_target_mask']=result['audio_mask'].clone()
    proof=dict(policy='integer-observed-wave-boundary-v1',code_sha256=sha256(Path(__file__)),
        raw_cached_real_frames=clean.shape[1],observed_real_frames=frames,source_samples=info.frames,
        source_rate=info.samplerate,excluded_unobserved_padding_frames=1,
        excluded_text_tokens=clean[0,frames:].tolist(),all_excluded_audio_codes_unavailable=True,
        measured_words_and_controls_unchanged=True,source_audio_and_annotations_unchanged=True)
    result['record']={**r,'real_frames':frames,'frames':result['codes'].shape[1],
        'real_to_virtual':r['real_to_virtual'][:frames],'observed_boundary':proof}
    if (not torch.equal(result['codes'],example['codes'][:,:-1])
            or not torch.equal(result['text_mask'],example['text_mask'][:-1])):
        raise RuntimeError('Observed boundary changed more than the final nonexistent cache frame')
    return result
