"""Read-only full-corpus approval and exact prompt/action lineage validation."""
import json
from pathlib import Path
from training.control.curriculum import specifications,label_scene,xml_action
from training.control.teacher_data import sha,MODEL


def verified_data(folder,*,audit_preview=False):
    folder=Path(folder);manifest=json.loads((folder/'manifest.json').read_text())
    if (manifest['schema']!='dl-verified-controller-text-data-v1' or not manifest['all_label_quality_passed']
            or not manifest['synthetic_text_only'] or manifest['audio_generated'] or manifest['prior_evaluation_read']
            or manifest['provenance_class']!='closed-content' or manifest['teacher_model']!=MODEL):raise ValueError('Text corpus lacks complete quality/provenance approval')
    if manifest['preview']:
        if not audit_preview or manifest['training_approved']:raise ValueError('Teacher preview is not approved for training')
    elif not manifest['training_approved']:raise ValueError('Text corpus has not been approved for training')
    if not manifest['producer_commit'] or len(manifest['producer_commit'])!=40:raise ValueError('Missing immutable text-data producer')
    required={'training/control/curriculum.py','training/control/teacher_data.py','evals/qwen_response_executor.py','evals/qwen35_controller.py','evals/duplex_reasoning_worker.py','evals/duplex_speech_context.py'}
    if set(manifest['source_files'])!=required:raise ValueError('Incomplete data compiler inventory')
    for filename,digest in manifest['source_files'].items():
        if sha(filename)!=digest:raise ValueError('Changed data compiler: '+filename)
    if sha('decisions/0003-provenance-classes.md')!=manifest['authorization_sha256']:raise ValueError('Changed data authorization source')
    for file,key in [('scenes.json','scenes_sha256'),('quality.json','quality_sha256'),('train.jsonl','train_sha256')]:
        if sha(folder/file)!=manifest[key]:raise ValueError('Changed text corpus artifact: '+file)
    scenes=json.loads((folder/'scenes.json').read_text());quality=json.loads((folder/'quality.json').read_text())
    if scenes!=specifications(manifest['preview']):raise ValueError('Source scenes differ from predeclared families and values')
    train=[s for s in scenes if s['split']=='train'];heldout=[s for s in scenes if s['split']=='heldout']
    if not quality['all_passed'] or quality['failed'] or quality['requested']!=len(train) or quality['passed']!=len(train):raise ValueError('Incomplete text label quality inventory')
    if len(quality['rows'])!=len(train) or len({q['id'] for q in quality['rows']})!=len(train):raise ValueError('Duplicate text quality records')
    records=[json.loads(line) for line in (folder/'train.jsonl').read_text().splitlines()]
    if (len(records)!=len(train) or len({r['id'] for r in records})!=len(records)
            or {r['id'] for r in records}!={s['id'] for s in train}
            or {q['id'] for q in quality['rows']}!={s['id'] for s in train}):raise ValueError('Changed/duplicate TRAIN identity inventory')
    by_id={s['id']:s for s in train};checked={q['id']:q for q in quality['rows']}
    class RecordedGenerator:
        model_id=MODEL
        def __init__(self,row):self.row=row
        def request(self,messages,tools,assistant_prefill=''):
            if messages!=self.row['student_messages'] or tools!=self.row['student_tools'] or assistant_prefill!=self.row['student_assistant_prefill']:raise ValueError('Student prefix differs from exact runtime compiler')
            text=self.row['target_native_text']
            return dict(raw_text=text,cleaned_text=text,assistant_prefill=assistant_prefill,input_tokens=0,generated_tokens=0,verification_replay_not_inference=True)
    for row in records:
        scene=by_id[row['id']];path=folder/'rows'/row['id'];q=checked[row['id']]
        if (row['split']!='train' or row['family']!=scene['family'] or not row['quality_passed'] or not q['passed']
                or json.loads((path/'scene.json').read_text())!=scene or sha(path/'scene.json')!=row['source_scene_sha256']):raise ValueError('TRAIN row lacks unchanged source and quality')
        if json.loads((path/'quality.json').read_text())!=dict(id=q['id'],attempts=q['attempts'],passed=q['passed']):raise ValueError('Row and batch quality records differ')
        attempt=row['quality_attempt'];entry=q['attempts'][attempt];provenance=row['teacher_provenance'];raw=path/(f'attempt-{attempt}-teacher.json')
        if not entry['passed'] or entry['teacher']!=provenance or sha(raw)!=provenance['response_sha256']:raise ValueError('Teacher target lineage differs')
        response=json.loads(raw.read_text());calls=[c for c in response['output'] if c['type']=='function_call']
        if (response['status']!='completed' or len(calls)!=1 or response['id']!=provenance['teacher_response_id']
                or response['model']!=provenance['actual_model'] or not provenance['source_date_utc']):raise ValueError('Missing actual teacher execution provenance')
        action=dict(name=calls[0]['name'],arguments=json.loads(calls[0]['arguments']))
        if action!=row['target_native_action'] or xml_action(action)!=row['target_native_text']:raise ValueError('Training label was rewritten after actual teacher response')
        if scene['kind']in ('cancel-preserve','cancel-one','clarify'):
            review=entry['semantic_review'];review_path=path/(f'attempt-{attempt}-review.json')
            if not review['passed'] or sha(review_path)!=review['response_sha256']:raise ValueError('Missing passed semantic surface check')
            reviewed=json.loads(review_path.read_text());votes=[c for c in reviewed['output'] if c['type']=='function_call']
            if (reviewed['status']!='completed' or len(votes)!=1 or votes[0]['name']!='judge_control_utterance' or reviewed['id']!=review['response_id']
                    or json.loads(votes[0]['arguments'])!=dict(verdict='pass',reason=review['reason'])):raise ValueError('Semantic review is not an actual passed response')
        result,validation=label_scene(scene,RecordedGenerator(row))
        if not validation['passed'] or {k:result[k] for k in ('route','content','consume_pending_speech_ids')}!=row['target_canonical_result']:raise ValueError('Full worker action/answer verification failed')
    return records,heldout,manifest
