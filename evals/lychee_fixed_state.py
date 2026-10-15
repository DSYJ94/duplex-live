"""Free native continuation from an explicitly teacher-started TRAIN state.

This isolates input/deployment mismatch from on-policy history changes. The
future S-S control is supplied as in the approved training layout. No future
word or acoustic code is supplied. This cannot qualify timing or a full task.
"""
import json
from pathlib import Path
from evals.moshi_model_download import sha256


def run(framework,args,loading):
    import numpy as np
    import torch
    from data.synthetic.filter import norm_words
    from evals.recovery_data import validate_conversion
    from evals.lychee_context import create_session
    from lychee_fd.vllm_integration.model_lychee import LycheeDuplexState as state
    data=json.loads((args.fixed_state_data/'semantic-data.json').read_text())
    if not data['training_approved'] or data['schema']!='dl-lychee-native-semantic-data-v1':
        raise RuntimeError('Approved native TRAIN layout required for fixed-state diagnosis')
    for row in data['source_ingestions']:
        folder=Path(row['path']);validate_conversion(folder)
        if sha256(folder/'ingestion.json')!=row['sha256']:raise RuntimeError('Fixed-state source changed')
    histories={}
    for folder in args.fixed_state_histories:
        receipt=json.loads((folder/'actual-idle-state.json').read_text())
        if receipt['source_split']!='train' or sha256(folder/receipt['file'])!=receipt['sha256']:
            raise RuntimeError('Unapproved/changed fixed TRAIN history')
        histories[receipt['sha256']]=(folder,receipt)
    if framework.adding_text_hiddenstates:raise RuntimeError('This fixed-state mapping requires the pinned false configuration')
    selected=[r for r in data['examples'] if r['source_scenario'] in args.fixed_state_scenarios]
    if len(selected)!=len(args.fixed_state_scenarios):raise RuntimeError('Missing/duplicate requested fixed TRAIN scenes')
    model=framework.vllm_engine.engine.model_executor.driver_worker.model_runner.model
    records=[]
    with torch.inference_mode():
        for row in selected:
            if row['source_split']!='train' or sha256(args.fixed_state_data/row['file'])!=row['sha256']:
                raise RuntimeError('Fixed TRAIN data changed')
            item=torch.load(args.fixed_state_data/row['file'],map_location='cpu',weights_only=True)
            folder,receipt=histories[row['student_history_sha256']]
            saved=torch.load(folder/receipt['file'],map_location='cpu',weights_only=True)
            first=int((item['labels']!=-100).nonzero()[0]);system=item['prefix_len'];body=first-system
            prefix={'input_ids':item['input_ids'][system:first].tolist(),
                'stoken_ids':item['stoken_ids'][:body].tolist(),'control_input_ids':item['control_ids'][:body].tolist()}
            if prefix['control_input_ids'][-1]!=framework.ss_token_id:
                raise RuntimeError('Approved fixed target does not begin after a future S-S event')
            instructions=framework.tokenizer.decode(item['input_ids'][:system].tolist(),skip_special_tokens=False,
                clean_up_tokenization_spaces=False)
            session=create_session(framework,instructions,prefix=prefix,initial_listening_state='s',
                start_speak_token_factor=1.,start_listen_token_factor=1.,bc_speak_token_factor=1.,
                end_speak_token_factor=1.,audio_incremental_mode=True)
            session.restore_incremental_audio_cache(saved['audio_cache'])
            session._audio_tail=saved['audio_tail'].numpy().copy()
            if (session.get_prefix()!=prefix or session.system_input_length!=system
                    or not torch.equal(session.prefix_input_ids.cpu()[0],item['input_ids'][:system])):
                raise RuntimeError('Fixed-state prefix round trip failed')
            while len(session._audio_input_id_list)<body:
                session._append_incremental_audio(np.zeros(6400,dtype=np.float32),flush_audio_tail=False)
            comparison=[]
            def compare(module,inputs):
                if comparison:return
                positions,embeds=inputs[:2];expected=item['embeddings'][:first].to(embeds.device)
                full=torch.equal(positions,torch.arange(first,device=positions.device))
                same_shape=embeds.shape==expected.shape
                comparison.append({'full_prefix_positions':full,'actual_shape':list(embeds.shape),
                    'expected_shape':list(expected.shape),
                    'input_ids_exact':torch.equal(state.text_input_ids.cpu()[0],item['input_ids'][:first]),
                    'max_abs_input_embedding_error':float((embeds-expected).abs().max()) if same_shape else None})
            hook=model.main_layers[0].register_forward_pre_hook(compare)
            events=[]
            try:
                for _ in range(20):
                    events.extend(session.advance_stream(np.zeros(6400,dtype=np.float32),audio_is_incremental=True,
                        emit_generation_complete=False))
                result=session.get_prefix()
            finally:hook.remove();session.close()
            decoded=framework.decoder(result['input_ids'],result['stoken_ids'],result['control_input_ids'])
            future=[e for e in decoded if e['start_pos']>=body-1]
            text=' '.join(e['text'] for e in future)
            value=norm_words(row['actual_receipt']['value']);words=norm_words(text)
            case={'source_scenario':row['source_scenario'],'source_split':'train','teacher_start_control_supplied':True,
                'future_words_or_acoustic_codes_supplied':False,'starting_body_tokens':body,
                'expected_actual_value':row['actual_receipt']['value'],'free_native_text':text,
                'literal_value_mentioned':any(words[i:i+len(value)]==value for i in range(len(words))),
                'actual_input_comparison':comparison,'input_data_sha256':row['sha256'],
                'realtime_tested':False,'full_task_qualified':False}
            sub=args.out/row['source_scenario'];sub.mkdir()
            (sub/'native-events.json').write_text(json.dumps(events)+'\n')
            (sub/'decoded.json').write_text(json.dumps(future,indent=2)+'\n')
            (sub/'report.json').write_text(json.dumps(case,indent=2)+'\n')
            records.append(case);print(json.dumps(case),flush=True)
    report={'schema':'dl-lychee-fixed-train-state-v1','strict_loading':loading,
        'data_sha256':sha256(args.fixed_state_data/'semantic-data.json'),'cases':records,
        'scope':'teacher-started fixed TRAIN state; no word forcing; not live task success',
        'quality_qualified':False,'training_approved':False}
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
