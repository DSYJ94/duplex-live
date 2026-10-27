"""Versioned causal teachers with measured native speech pacing and pause tails."""
import argparse,array,hashlib,json,math,wave
from pathlib import Path
from .timeline import compile_scene
from .pacing import units
from .repair import words

def paced_scene(scene,audio,receipt,assistant_words):
    timeline=compile_scene(scene,audio,receipt)
    for action in timeline['actions']:
        if action['kind']=='speak':
            audit=assistant_words[action['text']]
            assert audit['passed'] and audit['word_error_rate']<=.05 and audit['peak']<=.995
            action['native_speech_units']=units(action['text'],audit['words'])
            action['timing_source_sha256']=audit['source_sha256']
            action['original_teacher_text']=action['text']
    # A real interruption cuts the first phrase. Resume its unsaid tail before
    # proceeding, instead of silently dropping the rest of the first step.
    if scene['teacher_program']['kind']=='pause_resume':
        first,stop,resume=timeline['actions'];assert first['kind']=='speak' and stop['kind']=='end_turn' and resume['kind']=='speak'
        original_words=words(first['text']+' '+resume['text']);cut=stop['frame']-first['frame'];assert cut>0
        full=first['native_speech_units'];head,tail=full[:cut],full[cut:]
        if tail:
            assert head and any(u['text'] for u in head)
            first['text']=' '.join(u['text'] for u in head).strip();first['native_speech_units']=head;head[-1]['final']=True
            combined=tail+resume['native_speech_units']
            resume['text']=' '.join(u['text'] for u in combined).strip()
            for i,u in enumerate(combined):u['offset']=i;u['final']=i==len(combined)-1
            resume['native_speech_units']=combined
            resume['resumes_unspoken_first_step_without_restart']=True
            resume['resumed_parent_first_text']=first['original_teacher_text']
            resume['timing_sources_sha256']=[first['timing_source_sha256'],resume['timing_source_sha256']]
        assert words(first['text']+' '+resume['text'])==original_words
    occupied=-1
    for action in timeline['actions']:
        if action['kind']=='speak':
            # A completed worker receipt can wait while an ordinary follow-up
            # finishes. It must never be delivered before its actual receipt.
            action['frame']=max(action['frame'],occupied)
            occupied=action['frame']+len(action['native_speech_units'])
        elif action['kind']=='end_turn':occupied=action['frame']+1
        else:occupied=max(occupied,action['frame']+1)
    timeline['frames']=max(timeline['frames'],occupied+2)
    timeline['native_speech_timing_schema']='word-aligned-native-v1'
    timeline['all_assistant_words_preserved']=True;timeline['training_approved']=False
    return timeline

def main(a):
    source=Path(a.plan);plan=json.loads(source.read_text());audio_root=Path(a.audio_root)
    ar=json.loads((audio_root/'render-receipts.json').read_text());assert ar['complete'];audio={r['text']:r for r in ar['utterances']}
    q=Path(a.assistant_quality);quality=json.loads(q.read_text());assert quality['all_passed'] and quality['passed']==quality['requested']
    assistant={r['authored']:r for r in quality['rows']}
    out=Path(a.output);out.mkdir(exist_ok=False,parents=True);manifest=[]
    for scene in plan['scenes']:
        receipt=json.loads((Path(a.receipts)/(scene['id']+'.json')).read_text()) if scene['synthetic_private_world'] else None
        timeline=paced_scene(scene,audio,receipt,assistant);pcm=array.array('h',[0])*(timeline['frames']*16000)
        for turn in timeline['user_turns']:
            raw=audio_root/(turn['audio_id']+'.wav');assert hashlib.sha256(raw.read_bytes()).hexdigest()==turn['sha256']
            with wave.open(str(raw)) as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth())==(16000,1,2)
                values=array.array('h',w.readframes(w.getnframes()))
            start=round(turn['start_seconds']*16000);assert not any(pcm[start:start+len(values)]);pcm[start:start+len(values)]=values
        folder=out/scene['id'];folder.mkdir()
        with wave.open(str(folder/'user.wav'),'wb') as w:w.setnchannels(1);w.setsampwidth(2);w.setframerate(16000);w.writeframes(pcm.tobytes())
        timeline['user_wav_sha256']=hashlib.sha256((folder/'user.wav').read_bytes()).hexdigest();(folder/'timeline.json').write_text(json.dumps(timeline,indent=2)+'\n')
        if receipt:(folder/'real-worker.json').write_text(json.dumps(receipt,indent=2)+'\n')
        manifest.append(dict(id=scene['id'],fold=scene['fold'],family=scene['family'],frames=timeline['frames'],timeline_sha256=hashlib.sha256((folder/'timeline.json').read_bytes()).hexdigest(),user_wav_sha256=timeline['user_wav_sha256']))
    result=dict(schema='dl-venus-paced-control-timelines-v1',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),audio_receipts_sha256=hashlib.sha256((audio_root/'render-receipts.json').read_bytes()).hexdigest(),assistant_word_quality_sha256=hashlib.sha256(q.read_bytes()).hexdigest(),native_speech_timing_schema='word-aligned-native-v1',scenes=manifest,training_approved=False,independent_semantic_review='PENDING',old_artifacts_preserved=True)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();[p.add_argument('--'+x,required=True) for x in ('plan','audio-root','assistant-quality','receipts','output')];main(p.parse_args())
