"""Approve only the first measured TRAIN clarification question, before project arrival."""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def question_scope(actual, teacher, side):
    from evals.recovery_prepare_data import spoken_words
    if (actual['split']!='train' or actual['kind']!='clarify'
            or teacher['source_split']!='train' or teacher['kind']!='clarify'
            or teacher['source_model']!='gpt-live-1'):
        raise RuntimeError('Clarification questions require approved TRAIN sources')
    users=[e for e in side['timeline']['events'] if e['ch']=='user' and e['kind']=='utterance']
    words=[a for a in side['alignments'] if a[2]=='SPEAKER_MAIN']
    stop=next((i+1 for i,w in enumerate(words) if w[0].strip().endswith('?')),None)
    if len(users)<2 or stop is None:raise RuntimeError('Missing complete first clarification question')
    selected=words[:stop];text=spoken_words(' '.join(w[0] for w in selected));meta=side['timeline']['meta']
    def contains(phrase):
        p=spoken_words(phrase)
        return any(text[i:i+len(p)]==p for i in range(len(text)-len(p)+1))
    onset=min(w[1][0] for w in selected);end=max(w[1][1] for w in selected)
    project=spoken_words(meta['project'])
    first_user=spoken_words(users[0]['text']);second_user=spoken_words(users[1]['text'])
    identified=lambda tokens:any(tokens[i:i+len(project)]==project for i in range(len(tokens)-len(project)+1))
    if (not all(math.isfinite(t) for t in (onset,end)) or identified(first_user) or not identified(second_user)
            or onset<users[0]['actual_end_ms']/1000
            or end+.24>users[1]['actual_start_ms']/1000
            or not contains('which project') or not contains(meta['field'])
            or contains(meta['project']) or contains(meta['phrase'])
            or any(a['ready_at_s']<=end+.24 for a in side['reference_arrivals'])):
        raise RuntimeError('Question must ask for the missing project before any project/value/result arrives')
    return stop,end


def plan(history,data,examples,spm,teacher_id):
    import soundfile as sf
    from evals.recovery_data import validate_conversion
    from evals.context_actual_targets import matching_user_stimulus
    raw=json.loads((history/'student-history.json').read_text())
    report=json.loads((history/'report.json').read_text())
    events=json.loads((history/'worker-events.json').read_text())
    if (report.get('backend_execution_disabled') is not True or report.get('conditioning_only') is not True
            or report['native_foreground_backend_request'] or events or report['worker_events']):
        raise RuntimeError('Question conditioning must contain no backend spans or supplied answers')
    actual=next(r for r in validate_conversion(data)['accepted'] if r['scenario_id']==raw['source']['scenario'])
    source_side=json.loads(Path(actual['path']).with_suffix('.json').read_text())
    source_pcm,rate=sf.read(actual['path'],dtype='float32',always_2d=True)
    candidates=[]
    for e in examples:
        t=e['record']
        if t['id']!=teacher_id or t['kind']!='clarify' or t['source_model']!='gpt-live-1':continue
        cache=json.loads((Path(t['cache'])/'cache.json').read_text())
        approved=next(r for r in validate_conversion(Path(cache['conversion']))['accepted'] if r['scenario_id']==t['id'])
        side=json.loads(Path(approved['path']).with_suffix('.json').read_text())
        pcm,tr=sf.read(approved['path'],dtype='float32',always_2d=True)
        if tr!=rate or not matching_user_stimulus(source_pcm[:,1],pcm[:,1],source_side['user_words'],side['user_words']):
            continue
        if any(source_side['timeline']['meta'][k]!=side['timeline']['meta'][k] for k in ('project','field','phrase')):
            raise RuntimeError('Clarification teacher world differs from the approved source')
        count,end=question_scope(actual,t,side)
        clean=e['clean_codes'];positions=(clean[0]>=5).nonzero().flatten()
        word_count=sum(len(spm.encode(w[0].strip())) for w in side['alignments'] if w[2]=='SPEAKER_MAIN')
        first_words=[w for w in side['alignments'] if w[2]=='SPEAKER_MAIN'][:count]
        tokens=sum(len(spm.encode(w[0].strip())) for w in first_words)
        if len(positions)!=word_count:raise RuntimeError('Question token/word coverage changed')
        left=int(positions[0])-1;right=max(int(positions[tokens-1])+1,math.ceil(end*12.5))+2
        if (left<0 or right>clean.shape[1] or (clean[:9,left:right]<0).any()
                or (clean[0,:right]==4).any()):
            raise RuntimeError('Question-only scope contains a request or invalid native targets')
        # Include the complete measured question and its final padding, before the next user utterance.
        users=[x for x in side['timeline']['events'] if x['ch']=='user' and x['kind']=='utterance']
        if right*.08>users[1]['actual_start_ms']/1000:
            raise RuntimeError('Question padding reaches the future project response')
        candidates.append((t,clean,{'source_start_frame':left,'source_end_frame':right,
            'source_first_word_index':0,'source_end_word_index':count}))
    if len(candidates)!=1:raise RuntimeError('Need exactly one approved same-user clarification teacher')
    teacher,clean,selection=candidates[0];meta=source_side['timeline']['meta']
    world={meta['project']:{meta['field']:meta['phrase']}}
    if raw['source']['declared_counterfactual_world']!=world:raise RuntimeError('Changed clarification world')
    result={'schema':'dl-context-actual-future-plan-v1','target_kind':'clarification_question',
        'clarification_plan_code_sha256':sha256(Path(__file__)),
        'actual_history':str(history),'actual_history_sha256':raw['sha256'],
        'actual_history_manifest_sha256':sha256(history/'student-history.json'),
        'actual_report_sha256':sha256(history/'report.json'),'teacher':teacher,'selection':selection,
        'world':world,'actual_private_receipt':None,'all_past_labels_masked':True,
        'training_approved':False,'gold_targets_created':False}
    return result,clean[:9,selection['source_start_frame']:selection['source_end_frame']].clone()
