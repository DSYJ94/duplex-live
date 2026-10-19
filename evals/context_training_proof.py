"""Eight-update native Context Spanning adapter proof, never task qualification."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import subprocess
import sys
import time
from evals.moshi_model_download import sha256


def logits_digests(out):
    import torch
    result = {}
    for name, logits, mask in [('text', out.text_logits, out.text_mask),
                               ('audio', out.logits, out.mask)]:
        value = logits[mask].detach()
        if not torch.isfinite(value).all():
            raise RuntimeError('Nonfinite native valid logits')
        result[name] = hashlib.sha256(value.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    for name in ('assets', 'base', 'voice_reference', 'voice_provenance', 'out'):
        ap.add_argument('--' + name.replace('_', '-'), type=Path, required=True)
    ap.add_argument('--cache', type=Path, action='append', required=True)
    ap.add_argument('--steps', type=int, choices=(8, 128), default=8)
    ap.add_argument('--pilot-preflight', type=Path)
    ap.add_argument('--layout-audit', type=Path)
    ap.add_argument('--teacher-audit', type=Path, action='append')
    ap.add_argument('--teacher-engine-input-audit',type=Path)
    ap.add_argument('--teacher-engine-input-id')
    ap.add_argument('--teacher-engine-input-voice',type=Path)
    ap.add_argument('--teacher-engine-native-audio',action='store_true',
        help='Diagnostic forced teacher words with sampled agent audio; no training approval')
    ap.add_argument('--teacher-native-math-audit',type=Path)
    ap.add_argument('--teacher-native-math-diagnostic')
    ap.add_argument('--teacher-native-math-diagnostic-mode',choices=('attention','linear'),default='attention')
    ap.add_argument('--teacher-native-fp32-preflight',type=Path)
    ap.add_argument('--teacher-native-fp32-source-commit')
    ap.add_argument('--actual-replay', type=Path, action='append')
    ap.add_argument('--actual-source-data', type=Path)
    ap.add_argument('--actual-head-data', type=Path, action='append')
    ap.add_argument('--actual-early-data', type=Path, action='append')
    ap.add_argument('--actual-mixed-data', type=Path, action='append')
    ap.add_argument('--actual-sampling',choices=('component-order','balanced-kind'),default='component-order')
    ap.add_argument('--actual-retention-sampling',choices=('legacy-modulo','balanced-kind'),default='legacy-modulo')
    ap.add_argument('--actual-retention-teacher-weight',type=float,default=0.)
    ap.add_argument('--actual-base-retention-weight',type=float,choices=(0.,1.),default=1.)
    ap.add_argument('--actual-native-teacher-loss',choices=('legacy','balanced-spoken'),default='legacy')
    ap.add_argument('--actual-retention-teacher-audit',type=Path)
    ap.add_argument('--actual-native-teacher-approval',type=Path)
    ap.add_argument('--actual-native-teacher-scope',type=Path)
    ap.add_argument('--actual-native-teacher-preflight',type=Path)
    ap.add_argument('--actual-native-teacher-coverage',choices=('four-longest','all-approved'),default='four-longest')
    for name in ('cache', 'approval', 'scope', 'preflight'):
        ap.add_argument('--native-v2-' + name, type=Path)
    ap.add_argument('--actual-response-first-word-weight',type=float,default=1.)
    ap.add_argument('--actual-temporal-audit', action='store_true')
    ap.add_argument('--actual-temporal-fit', action='store_true')
    ap.add_argument('--actual-temporal-preflight', type=Path)
    ap.add_argument('--resume-training',type=Path,help='Exact adapter/optimizer/RNG state for the same approved recipe/data')
    ap.add_argument('--warm-start-proof',type=Path,
        help='Verified adapter weights for a new corrective stage; optimizer and step count restart explicitly')
    ap.add_argument('--actual-fit-updates',type=int,choices=(8,32,128,512),default=8)
    ap.add_argument('--actual-source-commit')
    ap.add_argument('--actual-teacher-id')
    ap.add_argument('--actual-future-component', action='store_true')
    ap.add_argument('--actual-early-window', action='store_true')
    ap.add_argument('--actual-native-response-boundary',action='store_true')
    ap.add_argument('--actual-native-audio-future',action='store_true')
    ap.add_argument('--actual-engine-replay', action='store_true')
    ap.add_argument('--actual-trained-collector',action='store_true')
    ap.add_argument('--actual-cancel-ack',action='store_true')
    ap.add_argument('--actual-trained-response-boundary',action='store_true')
    ap.add_argument('--actual-trained-private-future',action='store_true')
    ap.add_argument('--actual-control-rollin',choices=('cancel_request','clarification_lookup'))
    ap.add_argument('--actual-recorded-voice', action='store_true')
    ap.add_argument('--actual-native-priming', action='store_true')
    ap.add_argument('--actual-cold-graphs', action='store_true')
    ap.add_argument('--reload', action='store_true')
    ap.add_argument('--history-audit-only', action='store_true')
    ap.add_argument('--history-audit', action='store_true')
    ap.add_argument('--history-logits', action='store_true',
        help='Also compare canonical streamed and batched temporal outputs in the history audit')
    ap.add_argument('--text-padding', choices=('legacy-epad0','context-pad3'), default='legacy-epad0')
    ap.add_argument('--request-policy', choices=('source-dispatch','native-prelead','native-prelead-epad'), default='source-dispatch')
    args = ap.parse_args()
    from evals.context_native_v2_teacher_objective import validate_configuration as validate_native_v2
    validate_native_v2(args)
    if args.teacher_engine_native_audio and (not args.teacher_engine_input_audit
            or not args.teacher_engine_input_id or args.teacher_native_math_audit):
        ap.error('Native teacher audio rendering requires one independently scoped teacher and no math/optimizer path')
    if bool(args.teacher_native_fp32_preflight)!=bool(args.teacher_native_fp32_source_commit) or (
            args.teacher_native_fp32_preflight and (not args.teacher_native_math_audit or args.teacher_native_math_diagnostic)):
        ap.error('Higher precision reference requires the original FP32 audit and source in a separate zero-update run')
    if args.teacher_native_math_diagnostic and not args.teacher_native_math_audit:
        ap.error('Numerical diagnostic requires the independently approved teacher corpus')
    if any((args.teacher_engine_input_audit,args.teacher_engine_input_id,args.teacher_engine_input_voice,args.teacher_native_math_audit)):
        if (not args.teacher_engine_input_audit or not args.teacher_engine_input_voice
                or bool(args.teacher_engine_input_id)==bool(args.teacher_native_math_audit)):
            ap.error('Teacher Engine input audit needs an independent scope, explicit TRAIN identity and exact voice')
        if (args.reload or args.actual_replay or args.actual_head_data or args.actual_early_data
                or args.actual_mixed_data or args.actual_temporal_fit or args.actual_temporal_audit
                or args.teacher_audit or args.pilot_preflight or args.history_audit or args.history_audit_only
                or args.resume_training or args.warm_start_proof or args.actual_retention_teacher_weight
                or args.steps!=8 or args.request_policy!='native-prelead-epad'):
            ap.error('Teacher Engine input audit is a separate zero-update TRAIN-only path')
    from evals.context_decision_loss import validate as validate_decision_weight
    validate_decision_weight(args.actual_response_first_word_weight)
    from evals.context_retention_teacher import validate_weight as validate_teacher_weight
    validate_teacher_weight(args.actual_retention_teacher_weight)
    native_teacher=any((args.actual_native_teacher_approval,args.actual_native_teacher_scope,args.actual_native_teacher_preflight))
    from evals.context_native_teacher_objective import validate_base_retention
    validate_base_retention(args.actual_base_retention_weight,native_teacher)
    from evals.context_teacher_text_balance import validate as validate_teacher_balance
    validate_teacher_balance(args.actual_native_teacher_loss,native_teacher,args.actual_native_teacher_coverage)
    if native_teacher and not (all((args.actual_native_teacher_approval,args.actual_native_teacher_scope,args.actual_native_teacher_preflight))
            and args.actual_retention_teacher_weight>0 and args.actual_temporal_fit and args.actual_mixed_data
            and args.actual_retention_sampling=='balanced-kind' and not args.actual_retention_teacher_audit):
        ap.error('Native teacher supervision requires complete CPU/math approvals, positive weight and balanced mixed training')
    if args.actual_native_teacher_coverage!='four-longest' and not native_teacher:
        ap.error('Expanded native teacher coverage requires its explicit approvals')
    if args.actual_retention_teacher_weight and not (args.actual_temporal_fit and args.actual_mixed_data
            and (args.actual_retention_teacher_audit or native_teacher)):
        ap.error('Teacher retention needs explicit mixed temporal adaptation and an independent scope audit')
    if args.actual_retention_teacher_audit and not args.actual_retention_teacher_weight:
        ap.error('Teacher retention scope audit requires an explicit positive teacher weight')
    if sum((args.actual_cancel_ack,args.actual_trained_response_boundary,args.actual_trained_private_future,bool(args.actual_control_rollin)))>1:
        ap.error('Choose one independently scoped trained future kind')
    if args.actual_retention_sampling!='legacy-modulo' and not (args.actual_temporal_fit and args.actual_sampling=='balanced-kind'):
        ap.error('Per-kind retention coverage requires explicit balanced-kind temporal adaptation')
    if args.actual_trained_collector and not (args.actual_engine_replay and args.actual_recorded_voice
            and args.actual_replay and len(args.actual_replay)==1 and args.actual_source_data and args.actual_source_commit):
        ap.error('Trained replay requires one explicit TRAIN history, source provenance and its exact recorded voice')
    if args.actual_cancel_ack and not (args.actual_trained_collector and args.actual_future_component
            and args.actual_teacher_id and not args.actual_early_window and not args.actual_native_audio_future):
        ap.error('Cancellation acknowledgment needs an explicit verified trained collector and late teacher selection')
    if args.actual_trained_response_boundary and not (args.actual_trained_collector and args.actual_future_component
            and args.actual_early_window and args.actual_teacher_id and not args.actual_native_response_boundary
            and not args.actual_cancel_ack):
        ap.error('Trained response correction needs explicit early TRAIN teacher selection and its exact trained collector')
    if args.actual_trained_private_future and not (args.actual_trained_collector and args.actual_future_component
            and args.actual_teacher_id and not args.actual_early_window):
        ap.error('Trained private correction requires explicit late teacher selection and a verified trained collector')
    if args.actual_control_rollin and not (args.actual_trained_collector and args.actual_future_component
            and args.actual_teacher_id and not args.actual_early_window and args.actual_native_audio_future):
        ap.error('Control roll-in needs one explicit trained TRAIN history, teacher and own native audio')
    if args.actual_trained_collector and args.actual_future_component and not (args.actual_cancel_ack
            or args.actual_trained_response_boundary or args.actual_trained_private_future or args.actual_control_rollin):
        ap.error('Trained prefix replay is read-only until separately audited future supervision is implemented')
    if args.actual_response_first_word_weight!=1. and not (args.actual_temporal_fit and args.actual_mixed_data):
        ap.error('Decision weighting requires explicit audited mixed-component temporal training')
    if args.actual_teacher_id and not args.actual_future_component:
        ap.error('Explicit teacher selection requires an independently planned future component')
    if args.actual_native_audio_future and not (args.actual_early_window or args.actual_control_rollin or args.actual_trained_private_future):
        ap.error('Native audio conditioning requires the independently gated early future path')
    if (args.resume_training or args.actual_fit_updates != 8) and not args.actual_temporal_fit:
        ap.error('Resume/update budget is specific to the actual temporal trainer')
    if args.warm_start_proof and (not args.actual_temporal_fit or
            (args.actual_fit_updates!=8 and not args.resume_training)):
        ap.error('A new warm-start stage requires a separate eight-update temporal proof before continuation')
    if args.actual_fit_updates != 8 and not args.resume_training:
        ap.error('Longer adaptation requires a verified resumed eight-update proof')
    if args.actual_sampling!='component-order' and not args.actual_temporal_fit:
        ap.error('Balanced sampling is specific to independently audited temporal adaptation')
    if args.actual_early_data or args.actual_mixed_data:
        if (args.actual_head_data or (args.actual_early_data and args.actual_mixed_data)
                or not (args.actual_temporal_audit or args.actual_temporal_fit)):
            ap.error('Early/mixed components require a single separate temporal audit/fit path')
        args.actual_head_data = args.actual_early_data or args.actual_mixed_data
    if args.actual_early_window and not args.actual_future_component:
        ap.error('Early future requires the complete strict native engine component path')
    if args.actual_native_response_boundary and not (args.actual_early_window and args.actual_future_component and args.actual_teacher_id):
        ap.error('Response-boundary composition requires explicit early-future teacher selection')
    if args.actual_temporal_fit and (not args.actual_head_data or not args.actual_temporal_preflight
            or args.actual_temporal_audit or args.steps != 8):
        ap.error('Actual temporal fit requires audited components and a passed matching temporal audit')
    if args.actual_temporal_audit and (not args.actual_head_data or args.reload):
        ap.error('Actual temporal audit needs separately validated components and no reload')
    if args.actual_head_data and (not args.actual_source_data or args.actual_replay
            or args.pilot_preflight or args.teacher_audit or args.history_audit or args.history_audit_only):
        ap.error('Head-only proof requires independent future components and original source data')
    if args.actual_replay and (not args.actual_source_data or not args.actual_source_commit
            or args.reload or args.pilot_preflight or args.teacher_audit):
        ap.error('Actual restart needs explicit source data/commit and a separate diagnostic run')
    if args.actual_future_component and not args.actual_engine_replay:
        ap.error('Future component requires strict full engine replay')
    if args.actual_engine_replay and (not args.actual_replay or not args.actual_recorded_voice):
        ap.error('Full engine replay requires an actual history and its exact recorded voice')
    if args.actual_recorded_voice and not args.actual_replay:
        ap.error('Recorded actual voice is a separate restart diagnostic only')
    if args.history_logits and not (args.history_audit or args.history_audit_only):
        ap.error('Temporal history logits require a history audit')
    if args.steps == 128 and (not args.pilot_preflight or not args.layout_audit):
        ap.error('Bounded pilot requires the passed eight-update proof and native history audit')
    parent = None
    if args.pilot_preflight:
        parent = json.loads((args.pilot_preflight / 'report.json').read_text())
        reload = json.loads((args.pilot_preflight / 'reload-report.json').read_text())
        layout = json.loads(args.layout_audit.read_text())
        if (not parent['passed'] or parent['optimizer_steps'] != 8 or not parent['frozen_weights_unchanged']
                or not reload['passed'] or not reload['all_valid_logits_exact']
                or parent['checkpoint_sha256'] != sha256(args.pilot_preflight / 'adapters.safetensors')
                or reload['checkpoint_sha256'] != parent['checkpoint_sha256']
                or not layout['candidate_passed']
                or layout['data_sha256'] != parent['data_sha256']
                or layout['layout_code_sha256'] != sha256(Path(__file__).with_name('context_native_layout.py'))):
            raise RuntimeError('Unverified pilot preflight or layout audit')
    import numpy as np
    import soundfile as sf
    import torch
    from safetensors.torch import save_file, load_file
    from evals.context_spanning_download import SOURCE_COMMIT, MODEL_REVISION, WEIGHT_SHA256, VOICE_SHA256
    from evals.context_spanning_probe import audited_loads
    from evals.personaplex_train import MODEL_HASHES
    from evals.personaplex_adapters import install_adapters, adapter_parameters, checkpoint_language_layers
    from evals.moshi_training_smoke import tensor_hashes
    from evals.context_training_data import load_training
    from evals.context_training_loss import loss_fn
    from evals.context_training_parity import precise_native, depth_parity
    from evals.personaplex_depth import native_depth_context
    from evals.lychee_context import BASE_INSTRUCTIONS
    torch.backends.cudnn.enabled = False
    torch.backends.cuda.enable_cudnn_sdp(False)
    assets = json.loads(args.assets.read_text()); source = Path(assets['source']); weights = Path(assets['path'])
    if (assets['source_commit'] != SOURCE_COMMIT or assets['revision'] != MODEL_REVISION
            or subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip() != SOURCE_COMMIT
            or subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain'], text=True)):
        raise RuntimeError('Dirty or changed upstream source')
    if (assets['files']['context_spanning_7b.pt']['sha256'] != WEIGHT_SHA256
            or assets['files']['voices/main.pt']['sha256'] != VOICE_SHA256):
        raise RuntimeError('Wrong published assets')
    for name, record in assets['files'].items():
        if sha256(weights / name) != record['sha256']:
            raise RuntimeError('Changed public asset')
    for name, digest in MODEL_HASHES.items():
        if sha256(args.base / name) != digest:
            raise RuntimeError('Changed base asset')
    args.out.mkdir(exist_ok=args.reload)
    os.environ.update(CS_BASE_DIR=str(args.base), CS_WEIGHTS_DIR=str(weights), CS_VOICES_DIR=str(weights / 'voices'))
    os.environ['TRITON_LIBCUDA_PATH'] = subprocess.check_output([sys.executable, '-m', 'evals.cuda_linker_path',
        '--out', str(args.out / ('driver-reload' if args.reload else 'driver'))], text=True).strip()
    sys.path.insert(0, str(source))
    from contextspan.model.weights import load_model
    from contextspan.model.sequence_convention import persona_prefix
    from contextspan.moshi.models.lm import normalize_audio
    checkpoint = weights / 'context_spanning_7b.pt'
    if torch.serialization.get_unsafe_globals_in_checkpoint(checkpoint):
        raise RuntimeError('Unsupported checkpoint pickle globals')
    loads = []
    with audited_loads(loads):
        lm, mimi, spm = load_model(str(checkpoint), 'cuda')
    final = [r for r in loads if r['module_identity'] == id(lm)][-1]
    if not final['strict'] or final['missing'] or final['unexpected'] or lm.dep_q != 16:
        raise RuntimeError('Incomplete native 16-head checkpoint')
    lm.eval().requires_grad_(False)
    provenance = json.loads(args.voice_provenance.read_text())
    if (provenance['schema'] != 'dl-voice-codec-probe-v1' or provenance['source_model'] != 'gpt-live-1'
            or provenance['reference_sha256'] != sha256(args.voice_reference)):
        raise RuntimeError('Changed accepted synthetic voice provenance')
    if args.teacher_engine_input_audit:
        import shutil
        scope_report=json.loads(args.teacher_engine_input_audit.read_text())
        if sha256(args.teacher_engine_input_voice)!=scope_report['voice_sha256']:
            raise RuntimeError('Teacher Engine audit voice differs from its independent scope')
        voice=torch.load(args.teacher_engine_input_voice,map_location='cpu',weights_only=True)['codes']
        shutil.copyfile(args.teacher_engine_input_voice,args.out/'voice.pt')
    elif args.actual_recorded_voice:
        history = args.actual_replay[0]
        record = json.loads((history / 'student-history.json').read_text())
        voice_file = history / 'student-voice.pt'
        if sha256(voice_file) != record['source']['voice_codes_sha256']:
            raise RuntimeError('Actual recorded voice provenance changed')
        voice = torch.load(voice_file, map_location='cpu', weights_only=True)['codes']
        torch.save({'codes': voice}, args.out / 'voice.pt')
    elif args.actual_head_data:
        voice_source = args.out if args.reload else args.actual_head_data[0]
        voice = torch.load(voice_source/'voice.pt',map_location='cpu',weights_only=True)['codes']
        if not args.reload: torch.save({'codes':voice},args.out/'voice.pt')
    elif args.reload or parent or args.teacher_audit:
        voice_source = args.out if args.reload else (args.pilot_preflight if parent else args.teacher_audit[0])
        if parent and not args.reload and sha256(voice_source / 'voice.pt') != parent['voice_sha256']:
            raise RuntimeError('Changed preflight voice codes')
        voice = torch.load(voice_source / 'voice.pt', map_location='cpu', weights_only=True)['codes']
        if not args.reload: torch.save({'codes': voice}, args.out / 'voice.pt')
    else:
        pcm, rate = sf.read(args.voice_reference, dtype='float32')
        if rate != mimi.sample_rate or pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise RuntimeError('Expected accepted mono native-rate voice reference')
        x = normalize_audio(pcm[None, :], rate, -24.)
        with torch.no_grad():
            voice = mimi.encode(torch.as_tensor(x, dtype=torch.float32, device='cuda').reshape(1, 1, -1))[0].long().cpu()
        torch.save({'codes': voice}, args.out / 'voice.pt')
    if args.teacher_engine_input_audit:
        prefix=persona_prefix(BASE_INSTRUCTIONS,voice,spm)
        examples,_=load_training(args.cache,spm,prefix,text_padding=args.text_padding,request_policy=args.request_policy)
        if args.teacher_native_math_audit:
            if args.teacher_native_math_diagnostic:
                from evals.context_teacher_numeric_diagnostic import run as diagnose_teacher_math
                diagnose_teacher_math(lm,spm,voice,examples,args)
                return
            from evals.context_native_teacher_math import run as audit_teacher_math
            audit_teacher_math(lm,spm,voice,examples,args)
            return
        from evals.context_teacher_engine_inputs import run as audit_teacher_engine
        audit_teacher_engine(lm,mimi,spm,voice,examples,args)
        return
    if args.actual_engine_replay:
        from evals.context_engine_replay import run
        cut_index = None; callback = None
        if args.actual_future_component:
            from evals.context_future_component import plan, generate
            prefix = persona_prefix(BASE_INSTRUCTIONS, voice, spm)
            examples, records = load_training(args.cache, spm, prefix, text_padding=args.text_padding,
                request_policy=args.request_policy)
            if args.actual_control_rollin:
                from evals.context_control_rollin import plan as control_plan
                manifest=control_plan(args.actual_replay[0],args.actual_source_data,args.assets,examples,spm,voice,
                    args.actual_teacher_id,args.actual_control_rollin);gold=None
            elif args.actual_cancel_ack:
                from evals.context_cancel_ack import plan as cancel_plan
                manifest,gold=cancel_plan(args.actual_replay[0],args.actual_source_data,examples,spm,args.actual_teacher_id)
            else:
                manifest, gold = plan(args.actual_replay[0], args.actual_source_data, examples, spm,
                    early=args.actual_early_window,teacher_id=args.actual_teacher_id)
                if args.actual_trained_private_future and (manifest.get('target_kind')!='private_result'
                        or manifest.get('conditioning_policy')!='verified-trained-rollout'):
                    raise RuntimeError('Trained private correction did not select its independently verified receipt and collector')
                if args.actual_trained_private_future and args.actual_native_audio_future:
                    from evals.context_native_private import plan as native_private_plan
                    manifest,gold=native_private_plan(manifest,examples)
            if args.actual_early_window:
                if args.actual_trained_response_boundary:
                    from evals.context_trained_boundary import plan as early_plan
                elif args.actual_native_response_boundary:
                    from evals.context_response_boundary import plan as early_plan
                else:
                    from evals.context_early_plan import plan as early_plan
                manifest,gold = early_plan(manifest,examples)
            (args.out/'actual-future-plan.json').write_text(json.dumps(manifest,indent=2)+'\n')
            cut_index = manifest['replay_cut_index']
            def callback(engine, saved, index, rows, pcm, gate):
                (args.out/'engine-replay-before-future.json').write_text(json.dumps(gate,indent=2)+'\n')
                if args.actual_control_rollin:
                    from evals.context_control_rollin import generate as control_generate
                    return control_generate(engine,saved,index,rows,pcm,manifest,args.out)
                if args.actual_early_window or manifest.get('native_late_private_version')==1:
                    from evals.context_early_future import generate as early_generate
                    return early_generate(engine,saved,index,rows,pcm,gold,manifest,args.out,
                        native_audio=args.actual_native_audio_future)
                return generate(engine,saved,index,rows,pcm,gold,manifest,args.out)
        result = run(lm, mimi, spm, voice, args.actual_replay, args.actual_source_data,
            args.assets, args.actual_source_commit, cut_index=cut_index, on_verified=callback,
            before_reply=args.actual_early_window,allow_trained=args.actual_trained_collector)
        (args.out / 'engine-replay.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        if not result['passed']: raise RuntimeError('Full native engine restart failed')
        return
    del mimi
    if args.actual_replay:
        from evals.context_actual_replay import run
        result = run(lm, voice, args.actual_replay, args.actual_source_data,
            args.assets, args.actual_source_commit, warm_graphs=not args.actual_cold_graphs, native_priming=args.actual_native_priming)
        (args.out / 'actual-replay.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        if not result['passed']: raise RuntimeError('Actual unforced token restart failed')
        return
    prefix = persona_prefix(BASE_INSTRUCTIONS, voice, spm)
    examples, records = load_training(args.cache, spm, prefix, text_padding=args.text_padding,
        request_policy=args.request_policy)
    if args.actual_head_data:
        from evals.context_head_proof import run
        run(lm,spm,voice,examples,args)
        return
    (args.out / 'data.json').write_text(json.dumps(records, indent=2) + '\n')
    if parent and (parent['data_sha256'] != sha256(args.out / 'data.json')
            or parent['voice_reference_sha256'] != sha256(args.voice_reference)):
        raise RuntimeError('Pilot corpus/voice differs from verified preflight')
    if args.teacher_audit:
        for path in args.teacher_audit:
            original = json.loads((path / 'report.json').read_text())
            if (original['data_sha256'] != sha256(args.out / 'data.json')
                    or original['voice_sha256'] != sha256(voice_source / 'voice.pt')):
                raise RuntimeError('Diagnostic corpus/voice differs from training')
        from evals.context_teacher_audit import audit
        result = audit(lm, spm, examples, args.teacher_audit)
        (args.out / 'teacher-audit.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        return
    # Fixed complete TRAIN private case; no random window or validation-derived labels.
    first = next(e for e in examples if e['record']['kind'] == 'private')
    if args.history_audit_only or args.history_audit:
        from evals.context_history_audit import audit
        sample = torch.load(Path(first['record']['cache']) / (first['record']['id'] + '.pt'),
            map_location='cpu', weights_only=True)
        sample['codes'] = first['clean_codes'][None]
        if args.history_logits:
            with precise_native(lm):
                result = audit(lm, spm, prefix, first, sample, compare_logits=True)
        else:
            result = audit(lm, spm, prefix, first, sample)
        result.update(source_commit=SOURCE_COMMIT, data_sha256=sha256(args.out / 'data.json'),
            layout_code_sha256=sha256(Path(__file__).with_name('context_native_layout.py')),
            repo_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip())
        (args.out / 'history-audit.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result), flush=True)
        if args.history_audit_only: return
        if not result['candidate_passed']: raise RuntimeError('Integrated actual history audit failed')
    def tensors(example):
        return {k: example[k].cuda()[None] for k in ('codes', 'text_mask', 'audio_mask', 'user_target_mask')}
    fixed = tensors(first)
    depth_check = None
    if not args.reload:
        span = first['record']['spans'][0]
        boundary = first['record']['real_to_virtual'][span['inject_frame']]
        depth_check = depth_parity(lm, fixed['codes'], [boundary-2, boundary-1, boundary])
        (args.out / 'native-parity.json').write_text(json.dumps(depth_check, indent=2) + '\n')
    def precise(batch):
        with torch.no_grad(), precise_native(lm), native_depth_context(lm):
            return logits_digests(lm.forward_train(batch['codes']))
    frozen = tensor_hashes(lm.named_parameters())
    baseline = precise(fixed) if not args.reload else None
    torch.manual_seed(17); torch.cuda.manual_seed_all(17)
    rng_before = torch.cuda.get_rng_state().clone()
    with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        modules = install_adapters(lm, last_layers=4)
    if not torch.equal(rng_before, torch.cuda.get_rng_state()):
        raise RuntimeError('Adapter construction advanced sampling RNG')
    learned = adapter_parameters(lm)
    if not learned or {n for n, p in lm.named_parameters() if p.requires_grad} != set(learned):
        raise RuntimeError('Unexpected trainable set')
    if args.reload:
        report = json.loads((args.out / 'report.json').read_text())
        if (report['checkpoint_sha256'] != sha256(args.out / 'adapters.safetensors')
                or report['data_sha256'] != sha256(args.out / 'data.json')
                or report['voice_sha256'] != sha256(args.out / 'voice.pt')
                or report['frozen_hashes'] != frozen
                or report['assets_manifest_sha256'] != sha256(args.assets)):
            raise RuntimeError('Changed proof artifacts')
        state = load_file(args.out / 'adapters.safetensors', device='cuda')
        if set(state) != set(learned):
            raise RuntimeError('Adapter key coverage mismatch')
        with torch.no_grad():
            for n, p in learned.items(): p.copy_(state[n])
        actual = precise(fixed)
        passed = actual == report['expected_logits']
        reload_report = {'fresh_process': True, 'all_valid_logits_exact': passed, 'actual': actual,
            'checkpoint_sha256': report['checkpoint_sha256'], 'passed': passed}
        (args.out / 'reload-report.json').write_text(json.dumps(reload_report, indent=2) + '\n')
        print(json.dumps(reload_report), flush=True)
        if not passed: raise RuntimeError('Fresh-process native logits differ')
        return
    if precise(fixed) != baseline:
        raise RuntimeError('Zero adapters changed full valid native logits')
    print(json.dumps({'zero_adapter_all_valid_logits_exact': True, 'train_examples': len(examples),
        'maximum_frames': max(e['codes'].shape[1] for e in examples), 'adapter_modules': modules}), flush=True)
    optimizer = torch.optim.AdamW(list(learned.values()), lr=2e-5, weight_decay=0.)
    # Deliberate retention-balanced feasibility order; this is not convergence training.
    kinds = ['private', 'simple', 'clarify', 'cancel']
    if args.steps == 8:
        order = [max((e for e in examples if e['record']['kind'] == kind), key=lambda e: e['codes'].shape[1])
                 for kind in kinds]
    else:
        groups = {kind: [e for e in examples if e['record']['kind'] == kind] for kind in kinds}
        rng = random.Random(17)
        for group in groups.values(): rng.shuffle(group)
        order = [groups[kinds[s % 4]][(s // 4) % len(groups[kinds[s % 4]])] for s in range(args.steps)]
    steps = []; start = time.monotonic(); seen_a = seen_b = False
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        example = order[step % len(order)]; batch = tensors(example)
        optimizer.zero_grad(set_to_none=True)
        with checkpoint_language_layers(lm), native_depth_context(lm), torch.autocast('cuda', dtype=torch.bfloat16):
            out = lm.forward_train(batch['codes'])
            loss, detail = loss_fn(out, **batch, lm=lm)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite loss before backward')
            loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in learned.values()):
            raise RuntimeError('Missing or nonfinite intended adapter gradient')
        a = sum(int(torch.count_nonzero(p.grad)) for n, p in learned.items() if '.lora_A.' in n)
        b = sum(int(torch.count_nonzero(p.grad)) for n, p in learned.items() if '.lora_B.' in n)
        seen_a |= a > 0; seen_b |= b > 0
        norm = float(torch.nn.utils.clip_grad_norm_(list(learned.values()), 1.))
        if not math_isfinite(norm): raise RuntimeError('Nonfinite gradient norm')
        optimizer.step()
        row = {'step': step + 1, 'id': example['record']['id'], 'kind': example['record']['kind'],
            'loss': float(loss.detach()), 'nonzero_A_gradients': a, 'nonzero_B_gradients': b,
            'gradient_norm_before_clip': norm, **detail}
        steps.append(row); print(json.dumps(row), flush=True)
        (args.out / 'steps.json').write_text(json.dumps(steps, indent=2) + '\n')
        del out, loss, batch
    if not seen_a or not seen_b: raise RuntimeError('Dead low-rank update')
    if frozen != tensor_hashes((n, p) for n, p in lm.named_parameters() if not p.requires_grad):
        raise RuntimeError('Frozen native parameters changed')
    expected = precise(fixed)
    if expected == baseline: raise RuntimeError('Eight updates did not affect native outputs')
    save_file({n: p.detach().cpu().contiguous() for n, p in learned.items()}, args.out / 'adapters.safetensors')
    report = {'schema': 'dl-context-adapter-proof-v1', 'source_commit': SOURCE_COMMIT,
        'repo_commit': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'assets_manifest_sha256': sha256(args.assets), 'base_hashes': MODEL_HASHES,
        'voice_reference_sha256': sha256(args.voice_reference), 'voice_sha256': sha256(args.out / 'voice.pt'),
        'data_sha256': sha256(args.out / 'data.json'), 'checkpoint_sha256': sha256(args.out / 'adapters.safetensors'),
        'adapter_modules': modules, 'last_layers': 4, 'rank': 8, 'scaling': 2.,
        'raw_QKV_adapted': False, 'trainable_parameters': sum(p.numel() for p in learned.values()),
        'zero_adapter_all_valid_logits_exact': True, 'adapter_construction_rng_preserved': True,
        'canonical_logits_dtype': 'float32', 'native_depth_parity': depth_check,
        'cudnn_attention_enabled': torch.backends.cuda.cudnn_sdp_enabled(),
        'frozen_weights_unchanged': True, 'frozen_hashes': frozen, 'optimizer_steps': len(steps),
        'expected_logits': expected, 'baseline_logits': baseline, 'steps': steps,
        'peak_allocated_bytes': torch.cuda.max_memory_allocated(), 'peak_reserved_bytes': torch.cuda.max_memory_reserved(),
        'training_seconds': time.monotonic() - start, 'train_examples': len(examples),
        'maximum_frames': max(e['codes'].shape[1] for e in examples),
        'training_examples_seen': len({s['id'] for s in steps}),
        'pilot_preflight': str(args.pilot_preflight) if parent else None,
        'initialization': 'released checkpoint, fresh zero-B adapters; preflight is a gate, not warm start',
        'user_loss': 'clean separate user stem, assistant semantic weight 1, other 15 heads .02',
        'text_padding_convention': args.text_padding,
        'request_policy': args.request_policy,
        'waveform_AEC_tested': False, 'base_audio_parameters_frozen': True,
        'audio_behavior_preservation_proven': False, 'quality_qualified': False, 'passed': True}
    (args.out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('passed', 'optimizer_steps', 'trainable_parameters',
        'peak_allocated_bytes', 'checkpoint_sha256')}), flush=True)


def math_isfinite(value):
    import math
    return math.isfinite(value)


if __name__ == '__main__': main()
