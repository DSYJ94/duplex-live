"""Load the complete independently approved native teacher corpus, without delays twice."""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def load(approval_file,scope_file,assets,voice_file,examples,prefix,spm):
    from evals.context_retention_teacher import select_teachers
    from evals.context_teacher_native_approval import validate
    outer=json.loads(approval_file.read_text())
    if (outer['schema']!='dl-context-native-teacher-approval-set-v1' or not outer['passed']
            or outer.get('diagnostic_only') is not False or outer.get('complete_corpus') is not True
            or outer['optimizer_steps']!=0 or outer['training_approved'] is not False
            or outer['code_sha256']!=sha256(Path(__file__).with_name('context_teacher_native_approval.py'))):
        raise RuntimeError('Native teacher corpus lacks matching independent CPU approval')
    capture_scopes=[]
    for item in outer.get('capture_scopes',[]):
        path=Path(item['path'])
        if sha256(path)!=item['sha256']:raise RuntimeError('Changed declared historical teacher scope')
        capture_scopes.append(path)
    selected=select_teachers(examples,True);cases=outer['cases']
    if (len(cases)!=len(selected) or len({c['teacher']['id'] for c in cases})!=len(cases)
            or {c['teacher']['id'] for c in cases}!={e['record']['id'] for e in selected}):
        raise RuntimeError('Native teacher corpus must cover every approved independent TRAIN teacher exactly once')
    rows=[];records=[]
    for e in selected:
        case=next(c for c in cases if c['teacher']['id']==e['record']['id'])
        folder=Path(case['capture']);checked,tensors=validate(folder,examples,prefix,spm,scope_file,assets,voice_file,capture_scopes)
        if checked!=case or not checked['passed']:raise RuntimeError('Native teacher approval changed')
        trace=json.loads((folder/'teacher-engine-input-audit.json').read_text())
        rows.append(dict(inputs=tensors['delayed_inputs'],mask=tensors['text_mask'],targets=tensors['text_targets'],
            teacher=checked['teacher'],calls=trace['calls']))
        records.append(dict(teacher=checked['teacher'],capture=str(folder),
            capture_report_sha256=checked['capture_report_sha256'],
            capture_tensor_sha256=checked['capture_tensor_sha256'],
            native_input_sha256=checked['native_input_comparison']['actual_sha256']))
    manifest=dict(schema='dl-context-native-teacher-data-v1',approval_sha256=sha256(approval_file),
        scope_sha256=sha256(scope_file),assets_manifest_sha256=sha256(assets),voice_sha256=sha256(voice_file),
        data_loader_code_sha256=sha256(Path(__file__)),cases=records,
        conditioning='actual native streamed user codes; independently approved forced teacher text/assistant acoustics',
        targets='unchanged independently approved real assistant text only; prefix/spans excluded',
        prefix_loss_targets=0,span_loss_targets=0,user_loss_targets=0,audio_loss_targets=0)
    return manifest,rows
