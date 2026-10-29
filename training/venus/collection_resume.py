"""Validate completed native cases before resuming; preserve every partial case."""
import hashlib,json,wave
from pathlib import Path


def completed_cases(root,cases,conditions):
    import numpy as np,soundfile as sf
    root=Path(root);path=root/'results.json'
    if not path.exists():return {}
    existing=json.loads(path.read_text());rows=list(existing['cases']);known={c['id']:c for c in cases};found={}
    indexed={(r['condition'],r['id']) for r in rows}
    for condition in conditions:
        for case in cases:
            orphan=root/condition/case['id']/'result.json'
            if (condition,case['id']) not in indexed and orphan.exists():
                rows.append(dict(**json.loads(orphan.read_text()),condition=condition,evaluation_set=case['evaluation_set']))
    if existing.get('required_cases')!=64:raise ValueError('Exact full paired collection required')
    for row in rows:
        key=(row['condition'],row['id'])
        if key in found or key[0] not in conditions or key[1] not in known:raise ValueError('Unknown or duplicate completed case')
        case=known[key[1]];folder=root/key[0]/key[1]
        stored=json.loads((folder/'result.json').read_text())
        if not row.get('complete') or row['input_sha256']!=case['input_sha256'] or row['input_frames']!=case['frames'] or any(row.get(k)!=v for k,v in stored.items()):raise ValueError('Completed case identity changed')
        traces=[json.loads(x) for x in (folder/'trace.jsonl').read_text().splitlines()]
        if len(traces)!=case['frames'] or [x['sequence'] for x in traces]!=list(range(case['frames'])):raise ValueError('Completed trace truncated or reordered')
        with wave.open(str(Path(case['folder'])/'user.wav')) as src:
            if src.getframerate()!=16000 or src.getnchannels()!=1 or src.getsampwidth()!=2:raise ValueError('Original PCM framing changed')
            pcm=src.readframes(src.getnframes())
        if len(pcm)!=case['frames']*32000:raise ValueError('Original input incomplete')
        for index,t in enumerate(traces):
            if t['input_sha256']!=hashlib.sha256(pcm[index*32000:(index+1)*32000]).hexdigest():raise ValueError('Original input hash differs')
            name=Path(t['chunk_file'])
            if name.name!=str(name) or name.is_absolute():raise ValueError('Unsafe chunk path')
            data,sr=sf.read(folder/name,dtype='float32')
            if sr!=24000 or data.ndim!=1 or len(data)!=t['audio_samples'] or not np.isfinite(data).all():raise ValueError('Completed waveform incomplete')
            peak=float(np.abs(data).max()) if len(data) else 0.
            if abs(peak-t['peak'])>1e-6:raise ValueError('Completed raw waveform changed')
        found[key]=row
    return found


def preserve_partial(root,condition,ident,attempt):
    folder=Path(root)/condition/ident
    if folder.exists():
        dest=Path(root)/('interrupted-before-'+attempt)/condition/ident
        dest.parent.mkdir(parents=True,exist_ok=True)
        if dest.exists():raise ValueError('Never overwrite an earlier partial attempt')
        folder.rename(dest)


def clone_original_collection(source,destination):
    import shutil,os
    source,destination=Path(source),Path(destination)
    if destination.exists():raise ValueError('New version required; never overwrite an original collection')
    def copy(src,dest):
        src=Path(src)
        if src.is_symlink() or src.suffix not in ('.json','.jsonl','.wav'):raise ValueError('Only original synthetic collection artifacts admitted')
        if src.name.startswith('chunk-') and src.suffix=='.wav':
            try:os.link(src,dest);return str(dest)
            except OSError:pass
        return shutil.copy2(src,dest)
    shutil.copytree(source,destination,copy_function=copy)


def reuse_parent(source, destination, cases, checkpoint_sha256, label):
    """Reuse all32 validated298 baselines, retaining every failed case/audit."""
    source, destination = Path(source), Path(destination)
    valid = completed_cases(source, cases, ('parent244','paced-continuation'))
    if len(valid) != 64:
        raise ValueError('Complete original64 collection required before baseline reuse')
    condition = 'paced-continuation'
    profile = json.loads((source/condition/'profile.json').read_text())
    if profile['checkpoint_sha256'] != checkpoint_sha256 or profile['optimizer_updates'] != 298:
        raise ValueError('Exact preserved298 baseline required')
    report = json.loads((source/'paired-native-report.json').read_text())
    old_audits = {(r['condition'],r['id']):r for r in report['cases']}
    if not report['complete'] or len(old_audits) != 64:
        raise ValueError('Every original ASR result required')
    selected, audits = [], []
    for case in cases:
        key = condition, case['id']
        row = valid[key]
        if row['teacher_actions_used'] or row['oracle_feedback_used']:
            raise ValueError('Only genuinely free-running baseline cases can be reused')
        audit = old_audits[key]
        folder = source/condition/case['id']
        if hashlib.sha256((folder/'joined-emitted.wav').read_bytes()).hexdigest() != audit['joined_wav_sha256']:
            raise ValueError('Baseline ASR waveform changed')
        for chunk in audit['source_chunks']:
            if hashlib.sha256((folder/chunk['file']).read_bytes()).hexdigest() != chunk['sha256']:
                raise ValueError('Baseline source chunk hash changed')
        selected.append(dict(row, condition=label))
        audits.append(dict(audit, condition=label))
    clone_original_collection(source/condition, destination/label)
    original_profile = (source/condition/'profile.json').read_bytes()
    (destination/label/'original-profile.json').write_bytes(original_profile)
    (destination/label/'profile.json').write_text(json.dumps(dict(profile,condition=label,
        baseline_reused_from=str(source/condition)),indent=2)+'\n')
    (destination/'results.json').write_text(json.dumps(dict(cases=selected,required_cases=64,qualified=False),indent=2)+'\n')
    (destination/'speech-audit-progress.json').write_text(json.dumps(dict(cases=audits,completed=32,required=64),indent=2)+'\n')
    (destination/'baseline-reuse.json').write_text(json.dumps(dict(source=str(source),
        original_condition=condition, derived_condition=label, cases=32,
        checkpoint_sha256=checkpoint_sha256, all_failures_preserved=True,
        original_profile_sha256=hashlib.sha256(original_profile).hexdigest(),
        original_results_sha256=hashlib.sha256((source/'results.json').read_bytes()).hexdigest(),
        original_audit_sha256=hashlib.sha256((source/'paired-native-report.json').read_bytes()).hexdigest()),indent=2)+'\n')
