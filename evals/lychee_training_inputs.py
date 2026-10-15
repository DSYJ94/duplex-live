"""Causal semantic targets after real TRAIN student history and worker results.

Capture the actual native prefill on the next real arrived audio chunk. Append
verified native gold speech during an explicitly quiet continuation. Only future
text targets have loss. This is data assembly, not an inference/task result.
"""
import json
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def run(framework,args,loading):
    import numpy as np
    import soundfile as sf
    import torch
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words
    from evals.recovery_data import validate_conversion
    from evals.lychee_context import create_session
    from evals.lychee_semantic_layout import continuation
    from lychee_fd.vllm_integration.model_lychee import LycheeDuplexState as state
    approved={};source_ingestions=[]
    for path in [args.data,*args.training_source_data]:
        source=validate_conversion(path)
        source_ingestions.append({'path':str(path),'sha256':sha256(path/'ingestion.json')})
        for row in source['accepted']:
            if row['split']!='train' or row['kind']!='private':continue
            if row['scenario_id'] in approved:raise RuntimeError('Duplicate approved TRAIN scenario')
            approved[row['scenario_id']]=row
    manifest=json.loads((args.codec_data/'manifest.json').read_text())
    audit=json.loads((args.codec_audit/'report.json').read_text())
    checks=json.loads((args.codec_audit/'checks.json').read_text())
    if (checks['source_manifest_sha256']!=sha256(args.codec_data/'manifest.json')
            or checks['speech_report_sha256']!=sha256(args.codec_audit/'report.json') or checks['wer_bar']!=.08):
        raise RuntimeError('Independent native codec evidence changed')
    speech={r['run']:r for r in audit['results']}
    config=json.loads((Path(json.loads(args.assets.read_text())['models']['duplex']['path'])/'config.json').read_text())
    if config['adding_text_hiddenstates'] or config['control_token_chunk_size']!=10:
        raise RuntimeError('Unqualified native grammar configuration')
    model=framework.vllm_engine.engine.model_executor.driver_worker.model_runner.model
    tokens={'eot':framework.eos_token_id,'tts_start':framework.tts_start_id,
        'tts_end':framework.tts_end_id,'tts_pad':framework.tts_pad_id}
    examples=[];rejected=[]
    with torch.inference_mode():
        zeros=framework.stepaudio_audio_prepreocess(np.zeros(6400,dtype=np.float32))
        feats=torch.nn.utils.rnn.pad_sequence(zeros['feats'],batch_first=True,padding_value=0).transpose(1,2)
        lens=torch.tensor(zeros['feats_lengths'],dtype=torch.int32,device=framework.device)
        zfeatures,zlens=framework.vllm_engine.precompute_audio(feats,lens)
        zids=torch.tensor(zeros['input_ids'],device=framework.device,dtype=torch.long)
        zero_embeds=model.embed_tokens(zids)
        zmask=zids==framework.audio_token_id
        values=zfeatures[0,:int(zlens[0])]
        if len(zids)!=10 or int(zmask.sum())!=len(values):raise RuntimeError('Native quiet audio clock changed')
        zero_embeds[zmask]=values.to(zero_embeds.dtype)
        for folder in args.training_states:
            record=json.loads((folder/'actual-idle-state.json').read_text())
            if record['source_split']!='train' or record['source_scenario'] not in approved:
                raise RuntimeError('Held-out or unapproved history cannot enter semantic data')
            row=approved[record['source_scenario']]
            if (record['source_wav_sha256']!=row['wav_sha256'] or record['source_sidecar_sha256']!=row['sidecar_sha256']
                    or sha256(folder/record['file'])!=record['sha256']):
                raise RuntimeError('Native TRAIN history provenance changed')
            saved=torch.load(folder/record['file'],map_location='cpu',weights_only=True)
            receipt=saved['actual_receipt'];phrase=norm_words(receipt['value'])
            candidates=[]
            for item in manifest['examples']:
                if item['source_scenario']!=row['scenario_id']:continue
                gold_folder=Path(item['path']);gold=json.loads((gold_folder/'report.json').read_text())
                spoken=speech[str(gold_folder)];words=norm_words(gold['text'])
                contains=any(words[i:i+len(phrase)]==phrase for i in range(len(words)))
                if not contains:continue
                if (sha256(gold_folder/'report.json')!=item['report_sha256']
                        or gold['source_conversation_sha256']!=row['wav_sha256']
                        or gold['source_sidecar_sha256']!=row['sidecar_sha256']
                        or gold['source_split']!='train' or gold['source_kind']!='private'
                        or sha256(gold_folder/'codes.pt')!=gold['codes_sha256']
                        or sha256(gold_folder/'model.wav')!=spoken['wav_sha256']):
                    raise RuntimeError('Native gold source changed')
                if (spoken['wer_against_model_text']==0 and spoken['asr_word_bounds_valid']
                        and spoken['asr_time_bounds_valid'] and not spoken.get('audit_failure')):
                    candidates.append((gold_folder,gold,spoken))
            if not candidates:
                rejected.append({'source_scenario':row['scenario_id'],'reason':'No exact independently transcribed native gold reply with the actual value'});continue
            gold_folder,gold,spoken=min(candidates,key=lambda x:len(x[1]['text']))
            pcm,rate=sf.read(row['path'],dtype='float32',always_2d=True)
            if rate!=24000 or pcm.shape[1]!=2:raise RuntimeError('Source audio layout changed')
            user=np.pad(resample_poly(pcm[:,1],2,3).astype(np.float32),(0,20*16000))
            begin=round(saved['frame']['input_start_s']*16000)
            future=user[begin:]
            if any(float(np.sqrt(np.mean(future[i:i+320]**2)))>=.0015 for i in range(0,len(future),320)):
                rejected.append({'source_scenario':row['scenario_id'],'reason':'Later real user speech prevents the declared quiet continuation'});continue
            session=create_session(framework,saved['instructions'],prefix=saved['prefix'],initial_listening_state='l',
                start_speak_token_factor=1.,start_listen_token_factor=1.,bc_speak_token_factor=1.,
                end_speak_token_factor=1.,audio_incremental_mode=True)
            session.restore_incremental_audio_cache(saved['audio_cache'])
            session._audio_tail=saved['audio_tail'].numpy().copy()
            session.stoken_mapping=saved['stoken_mapping'].to(framework.device)
            session.next_chunk_pos=saved['next_chunk_pos'];session.last_ss_pos=saved['last_ss_pos']
            framework.last_ss_pos=saved['last_ss_pos']
            if session.get_prefix()!=saved['prefix'] or not torch.equal(session.prefix_input_ids.cpu(),saved['system_ids']):
                raise RuntimeError('Native replay changed actual student history or completed instructions')
            captured=[]
            def capture(module,inputs):
                if captured:return
                positions,embeds=inputs[:2]
                if not torch.equal(positions,torch.arange(len(positions),device=positions.device)):
                    raise RuntimeError('Training prefix is not a fresh complete native prefill')
                if (len(embeds)!=state.text_input_ids.numel()
                        or state.stoken_ids.numel()!=state.text_input_ids.numel()-int(state.prefix_input_len)):
                    raise RuntimeError('Native training prefill has incomplete or misaligned channels')
                captured.append({'embeddings':embeds.detach().clone(),
                    'input_ids':state.text_input_ids.detach().clone(),
                    'stoken_ids':state.stoken_ids.detach().clone(),'control_ids':state.control_input_ids.detach().clone(),
                    'prefix_len':int(state.prefix_input_len)})
            hook=model.main_layers[0].register_forward_pre_hook(capture)
            try:
                chunk=user[begin:begin+6400]
                if len(chunk)<6400:chunk=np.pad(chunk,(0,6400-len(chunk)))
                list(session.advance_stream(chunk,audio_is_incremental=True,emit_generation_complete=False))
            finally:hook.remove();session.close()
            if len(captured)!=1:raise RuntimeError('Native replay did not capture actual full prefix')
            prefix=captured[0];system=prefix['prefix_len'];history=prefix['input_ids'].shape[1]-system
            earliest=math.ceil(max(receipt['received_at_s'],saved['frame']['snapshot_elapsed_s']+.4)/.04)
            codes=torch.load(gold_folder/'codes.pt',map_location='cpu',weights_only=True)['s3_tokens'].tolist()
            target=continuation(history,framework.tokenizer.encode(gold['text'],add_special_tokens=False),codes,config,tokens,earliest)
            new_text=torch.tensor(target['text_ids'],device=framework.device)
            new_speech=torch.tensor(target['stoken_ids'],device=framework.device)
            new_control=torch.tensor(target['control_ids'],device=framework.device)
            extra=model.embed_tokens(new_text)+model.embed_tokens(new_speech)
            extra=extra+model.embed_tokens(new_control)
            extra=extra+zero_embeds[torch.arange(history,history+len(new_text),device=framework.device)%10]
            embeddings=torch.cat([prefix['embeddings'],extra],dim=0).cpu()
            ids=torch.cat([prefix['input_ids'][0],new_text]).cpu()
            stokens=torch.cat([prefix['stoken_ids'][0],new_speech]).cpu()
            controls=torch.cat([prefix['control_ids'][0],new_control]).cpu()
            labels=torch.full_like(ids,-100)
            for slot,token in zip(target['supervised_slots'],target['supervised_ids']):labels[system+slot]=token
            decoded=framework.decoder(ids[system:].tolist(),stokens.tolist(),controls.tolist())
            if (not decoded or norm_words(decoded[-1]['text'])!=norm_words(gold['text']) or decoded[-1]['audio']!=codes
                    or target['first_word_slot']*.04<receipt['received_at_s']):
                raise RuntimeError('Native continuation lost gold words/units or precedes the actual receipt')
            name=row['scenario_id']+'.pt'
            torch.save({'embeddings':embeddings,'input_ids':ids,'stoken_ids':stokens,'control_ids':controls,
                'labels':labels,'prefix_len':system},args.out/name)
            examples.append({'file':name,'sha256':sha256(args.out/name),'source_scenario':row['scenario_id'],
                'source_split':'train','student_history_sha256':record['sha256'],
                'native_past_prefill_tokens':len(prefix['embeddings']),'native_history_replayed_exactly':True,
                'gold_report_sha256':sha256(gold_folder/'report.json'),'gold_text':gold['text'],
                'gold_measured_asr_words':[w for segment in spoken['asr_segments'] for w in segment.get('words',[])],
                'actual_receipt':receipt,'first_supervised_token_time_s':target['first_word_slot']*.04,
                'loss_positions':int((labels!=-100).sum()),'future_input':'declared quiet continuation; real later user speech excluded',
                'target_control':'gold future native grammar; not a sampled foreground decision',
                'loss_scope':'future semantic text and EOT only; actual history and all speech/control labels masked'})
            print(json.dumps({k:v for k,v in examples[-1].items() if k!='gold_measured_asr_words'}),flush=True)
            del captured,prefix,embeddings,extra
    if not examples:raise RuntimeError('No causal native semantic training examples passed')
    result={'schema':'dl-lychee-native-semantic-data-v1','source_ingestion_sha256':sha256(args.data/'ingestion.json'),
        'source_ingestions':source_ingestions,
        'codec_manifest_sha256':sha256(args.codec_data/'manifest.json'),'codec_checks_sha256':sha256(args.codec_audit/'checks.json'),
        'native_loading':loading,'examples':examples,'rejected':rejected,'training_approved':True,
        'approval_scope':'bounded semantic implementation and behavior pilot; not corpus scale, native routing or full-task success'}
    (args.out/'semantic-data.json').write_text(json.dumps(result,indent=2)+'\n')
