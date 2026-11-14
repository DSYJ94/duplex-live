"""Prepare complete derived targets with immutable original source lineage.

Verified public explanations retain their content with explicit spoken notation; defective
solutions use separately generated correction proposals, not automatic labels.
Nothing in this builder grants semantic, timing, speech or training approval.
"""
import argparse,hashlib,json,re
from pathlib import Path


def build(plan,review,repairs,audit):
    if not review['complete1072'] or len(review['rows'])!=1072 or not repairs['complete48'] or not audit['complete'] or len(audit['rows'])!=2144:raise ValueError('All complete parent cohorts required')
    semantic={x['id']:x for x in review['rows']};proposals={x['id']:x for x in repairs['rows']};audio={x['id']:x for x in audit['rows']}
    flags={x['id'] for x in review['rows'] if not x['correct'] or not x['explanation_consistent']}
    if flags!=set(proposals):raise ValueError('Exact diagnosed semantic membership required')
    utterances=[];scenes=[]
    for source in plan['exact_semantic_records']:
        ident=source['id'];original=source['response'];new=original;action='reuse_full_waveform';reason='unchanged authored target'
        if ident in proposals:
            new=proposals[ident]['corrected_response'];action='new_speech_required';reason='independently flagged original semantic target; correction proposal needs reapproval'
        elif source['verification']['method'].startswith('original-public'):
            from .spoken_math import verbalize
            new=verbalize(original,spell_numbers=False);action='new_speech_required';reason='preserve verified reasoning content with explicit spoken numbers and mathematical operators'
        if action=='reuse_full_waveform' and not audio[ident+'-assistant']['word_pass']:
            action='exact_original_words_repair_required';reason='unchanged short target failed prior word audit'
        scenes.append(dict(id=ident,fold=source['fold'],question=source['question'],original_response=original,proposed_response=new,original_numeric_target=source['numeric_target'],original_semantic_verdict=semantic[ident],correction=proposals.get(ident),training_approved=False))
        if action=='reuse_full_waveform' and not audio[ident+'-assistant']['word_pass']:
            action='exact_original_words_repair_required';reason='unchanged short target failed prior word audit'
        for role,text in [('user',source['question']),('assistant',new)]:
            key=ident+'-'+role;parent=audio[key]
            utterances.append(dict(id=key,fold=source['fold'],text=text,parent_source_sha256=parent['source_sha256'],parent_source_words=parent['source_words'],parent_word_pass=parent['word_pass'],action=action if role=='assistant' else ('reuse_full_waveform' if parent['word_pass'] else 'exact_original_words_repair_required'),reason=reason if role=='assistant' else 'preserve original question and all values',external_speech_approved=False))
    if len(utterances)!=2144 or len(scenes)!=1072:raise ValueError('Full cohort retained')
    return dict(schema='native-reasoning-audio1072-derived-plan-v1',complete_scenes=1072,TRAIN=sum(x['fold']=='TRAIN' for x in scenes),DEV=sum(x['fold']=='DEV' for x in scenes),utterances=utterances,scenes=scenes,original_words_and_waveforms_preserved_as_parents=True,all_original_spoken_words_unchanged=False,folds_unchanged=True,whole_cohort_required=True,owner_approved_external_payload=False,semantic_reapproval_required=True,source_audio_approval_required=True,native_training_approved=False,OPAL_unchanged=True)


def main():
    p=argparse.ArgumentParser()
    for n in ('plan','review','repairs','asr','out'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();paths=[a.plan,a.review/'report.json',a.repairs/'report.json',a.asr/'report.json'];value=build(*[json.loads(x.read_text()) for x in paths]);value['parent_files']=[dict(path=str(x),sha256=hashlib.sha256(x.read_bytes()).hexdigest()) for x in paths]
    with a.out.open('x') as f:json.dump(value,f,indent=2);f.write('\n')
    from collections import Counter
    print(json.dumps(dict(scenes=value['complete_scenes'],utterances=len(value['utterances']),actions=dict(Counter(x['action'] for x in value['utterances'])),native_training_approved=False)))


if __name__=='__main__':main()
