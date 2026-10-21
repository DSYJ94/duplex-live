"""Author local synthetic cue probes with installed macOS speech, no speech API.

These diagnostic inputs are not new TRAIN/heldout approvals. No microphone is used.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import wave


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    texts = {'ready': 'Keep quiet and do not speak until I say Ready.',
             'geronimo': 'Keep quiet and do not speak until I say Geronimo.',
             'cue': 'Geronimo. What does a thermometer measure?'}
    clips = {}
    fragments = []
    for name, text in texts.items():
        source, dest = a.out/(name+'.aiff'), a.out/(name+'.wav')
        subprocess.run(['/usr/bin/say','-v','Samantha','-r','165','-o',str(source),text],check=True)
        subprocess.run(['/usr/bin/afconvert','-f','WAVE','-d','LEI16@24000','-c','1',str(source),str(dest)],check=True)
        with wave.open(str(dest)) as w:
            assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1,2,24000)
            clips[name] = w.readframes(w.getnframes())
        fragments.append({'id':name,'text':text,'sha256':hashlib.sha256(dest.read_bytes()).hexdigest()})
    silence = bytes(24000*2)
    cases = []
    for name, key, with_cue in [('ready-no-cue','ready',False),
                               ('geronimo-no-cue','geronimo',False),
                               ('geronimo-with-cue','geronimo',True)]:
        command_end = 1+len(clips[key])/48000
        cue_at = command_end+10 if with_cue else None
        payload = silence+clips[key]+silence*10
        if with_cue:
            payload += clips['cue']+silence*10
        target = a.out/(name+'.wav')
        with wave.open(str(target),'wb') as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(24000); w.writeframes(payload)
        cases.append({'id':name,'wav':target.name,'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
                      'keyword':'Ready' if key=='ready' else 'Geronimo',
                      'instruction':texts[key],'instruction_file_end_s':command_end,
                      'cue_file_start_s':cue_at,'duration_s':len(payload)/48000,
                      'expected':'No acknowledgment/question/speech after instruction until a subsequent cue utterance'})
    result = {'schema':'dl-native-local-synthetic-cue-diagnostic-v1',
              'speech_provider':'installed-macos-say','voice':'Samantha','rate_wpm':165,
              'owner_microphone_used':False,'external_speech_api_used':False,
              'gain_or_training_data_changed':False,'behavior_qualified':False,
              'fragments':fragments,'cases':cases}
    (a.out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
