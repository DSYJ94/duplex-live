"""Evaluation-only native input guard and paced observation without a TRAIN recorder."""
import json
from pathlib import Path
import time
from evals.moshi_model_download import sha256


def validate_options(args):
    if (args.train_history or args.backend_mode not in ('native-asr','observe-only')
            or args.private_value is not None or args.ret_logit_bias or args.warm_audio
            or args.pending_status or args.adapter_disabled or args.adapter_zero
            or args.layout_audit or args.event_log_mode!='immediate'
            or args.backend_reference_profile!='answer-only' or args.persona_profile!='recorded-records'
            or args.math_attention):
        raise ValueError('Fresh qualification requires unforced fixed native inference, no TRAIN labels or diagnostic patches')


def load(args):
    if args.scenario.startswith('native-fresh2-'):
        from evals.native_fresh_inputs_v2 import inference_input
        from evals.native_fresh_review import validate
        validate(args.native_fresh_behavior_review,args.native_fresh_candidate_proof,args.adapter_proof)
    else:
        from evals.native_fresh_inputs import inference_input
    validate_options(args)
    source=inference_input(args.data,args.native_fresh_quality,args.scenario)
    kind={'ordinary':'simple','quiet':'simple','pause':'simple'}.get(source['kind'],source['kind'])
    if (args.backend_mode!=source['backend'] or args.expected_kind!=kind
            or args.lookup_delivery_delay_s!=source['minimum_lookup_delay_s']):
        raise ValueError('Native fresh routing/kind/delay differs from the predeclared plan')
    return source,dict(scenario_id=args.scenario,kind=kind,split='fresh-evaluation',
                       path=source['path'],wav_sha256=source['wav_sha256'])


def user_stream(audio, rate, source):
    """V2 contains its declared twenty-second tail already; retain the v1 clock."""
    import numpy as np
    from evals.native_fresh_inputs_v2 import PLAN_SHA256
    if source['scenario'].startswith('native-fresh2-'):
        if source['plan_sha256'] != PLAN_SHA256:
            raise ValueError('Second collection requires its exact sealed plan')
        return audio
    return np.pad(audio, (0, 20*rate))


def finish(args,source):
    p=args.out/'report.json';r=json.loads(p.read_text())
    if (r['scheduled_dispatch'] or r['word_or_acoustic_tokens_forced'] or r['response_start_forced']
            or r.get('student_history') or r.get('training_approved')
            or r['real_user_samples_processed']!=r['real_user_samples_expected']):
        raise RuntimeError('Fresh inference lost native causal/input separation')
    r.update(source_split='fresh-evaluation',fresh_holdout=True,fresh_evaluation_input=source,
             evaluation_only=True,training_approved=False,quality_qualified=False,
             full20_task_qualification_passed=False)
    p.write_text(json.dumps(r,indent=2)+'\n')


def observe(engine,user,args,loading):
    import numpy as np
    import soundfile as sf
    import torch
    from evals.audio_playout import render_packets
    if (args.source_split!='fresh-evaluation' or not loading.get('fresh_evaluation_input')
            or args.backend_mode!='observe-only' or getattr(engine,'_dl_student_history',None)):
        raise RuntimeError('Evaluation observer cannot collect TRAIN history or execute a backend')
    rows=[];tokens=[];packets=[];epoch=time.monotonic()
    with (args.out/'events.jsonl').open('w') as log:
        for offset in range(0,len(user),engine.frame_size):
            end=min(len(user),offset+engine.frame_size);due=end/24000
            time.sleep(max(0,epoch+due-time.monotonic()));before=time.monotonic()-epoch
            chunk=user[offset:end]
            if len(chunk)<engine.frame_size:chunk=np.pad(chunk,(0,engine.frame_size-len(chunk)))
            result=engine.step(chunk);torch.cuda.synchronize();available=time.monotonic()-epoch
            tokens.append(result['text_token']);pcm=result['agent_pcm']
            if pcm is not None:
                pcm=np.asarray(pcm,dtype=np.float32)
                if pcm.shape!=(1920,) or not np.isfinite(pcm).all():raise RuntimeError('Invalid fresh native PCM')
                packets.append((available,pcm))
            row=dict(input_start_s=offset/24000,input_available_s=due,started_s=before,available_s=available,
                     compute_s=available-before,input_backlog_s=before-due,input_samples=end-offset,
                     output_samples=0 if pcm is None else len(pcm),text_token=result['text_token'],
                     native_retrieval_marker=result['is_ret'])
            rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
    if not packets or sum(r['input_samples'] for r in rows)!=len(user):raise RuntimeError('Fresh input or PCM missing')
    output,playout=render_packets(packets,24000,.16);sf.write(args.out/'model.wav',output,24000)
    sf.write(args.out/'packet-stream.wav',np.concatenate([p for _,p in packets]),24000)
    (args.out/'worker-events.json').write_text('[]\n')
    report=dict(schema='dl-native-fresh-observation-v1',scenario=args.scenario,text=engine.decode_text(tokens),
        loading=loading,source_split=args.source_split,fresh_holdout=True,worker_events=[],student_history=None,
        scheduled_dispatch=False,native_foreground_backend_request=False,native_foreground_tool_choice=False,
        native_request_markers_observed_not_executed=sum(r['native_retrieval_marker'] for r in rows),
        backend_execution_disabled=True,backend_context='none; evaluation without external requests',
        input_seconds=len(user)/24000,wall_seconds=time.monotonic()-epoch,
        compute_median_s=float(np.median([r['compute_s'] for r in rows])),
        compute_p95_s=float(np.percentile([r['compute_s'] for r in rows],95)),
        maximum_input_backlog_s=max(r['input_backlog_s'] for r in rows),
        real_user_samples_processed=sum(r['input_samples'] for r in rows),real_user_samples_expected=len(user),
        playout=playout,audio_sha256=sha256(args.out/'model.wav'),word_or_acoustic_tokens_forced=False,
        response_start_forced=False,actual_lm_and_pcm_executed_together=True,playout_is_causal_simulation=True,
        microphone_acceptance=False,quality_qualified=False,training_approved=False,optimizer_steps=0)
    (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n')
