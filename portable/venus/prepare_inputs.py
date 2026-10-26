"""Prepare public/synthetic development speech locally; no external TTS or ASR."""
import hashlib
import json
from pathlib import Path
import subprocess
import wave
import array


def main():
    root = Path('runs/venus-20261026/inputs')
    root.mkdir(parents=True, exist_ok=True)
    proposal = json.loads(Path('docs/NATIVE-FOUNDATION-COMPARISON-PROPOSAL-2026-10-25.json').read_text())
    previous = json.loads(Path('docs/GCP-NATIVE-SAVED8-COMPLETE20-REVIEW-2026-10-25.json').read_text())
    base = Path('runs/local-migration-20261024/gcp-native20-complete-copyback-v1/gcp-saved8-behavior-v1/paired-native20/baseline520')
    cases = []
    for index, case in enumerate(proposal['development_cases']):
        folder = root / case['id']
        folder.mkdir(exist_ok=True)
        turns = []
        texts = case.get('authored_user_turns', [None])
        for i, text in enumerate(texts):
            destination = folder / f'turn-{i}.wav'
            source = base / f'case-{index:02d}' / 'user_in.wav' if index < 4 else folder / f'turn-{i}.aiff'
            if destination.exists():
                with wave.open(str(destination)) as old:
                    empty = old.getnframes() == 0
                if empty:
                    destination.unlink()
                    if index >= 4 and source.exists():
                        source.unlink()
            if index < 4:
                expected = previous['actors']['baseline520']['cases'][index]['user_pcm_sha256']
                assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
            elif not source.exists():
                subprocess.run(['/usr/bin/say','-v','Samantha','-r','170','-o',str(source),text], check=True)
            if not destination.exists():
                subprocess.run(['/usr/bin/afconvert','-f','WAVE','-d','LEI16@16000','-c','1',str(source),str(destination)], check=True)
            with wave.open(str(destination)) as w:
                assert (w.getframerate(),w.getnchannels(),w.getsampwidth()) == (16000,1,2)
                frames = w.readframes(w.getnframes())
            samples = array.array('h',frames)
            if not samples:
                raise ValueError('Empty source audio; do not accept a silent passing case')
            peak = max(abs(x) for x in samples) / 32768
            if peak > .995:
                raise ValueError('Synthetic source clipping requires diagnosis; no gain reduction')
            turns.append(dict(path=str(destination), authored_text=text,
                sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                original_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                original_path=str(source), seconds=len(samples)/16000, peak=peak,
                independently_word_audited=index<4))
        cases.append(dict(id=case['id'],index=index,turns=turns,expected=case['expected']))
    result = dict(schema='dl-venus-native-development-inputs-v1',cases=cases,
        proposal_sha256=hashlib.sha256(Path('docs/NATIVE-FOUNDATION-COMPARISON-PROPOSAL-2026-10-25.json').read_bytes()).hexdigest(),
        new_voice='macOS Samantha at170wpm',gain_changes=False,external_api_calls=0,
        no_owner_recordings=True,no_sealed_evaluation=True,qualified=False,
        new_synthetic_word_audit_complete=False)
    (root/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'cases':len(cases),'turns':sum(len(c['turns']) for c in cases),'manifest':str(root/'manifest.json')}))


if __name__=='__main__':
    main()
