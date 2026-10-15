"""Rebuild only the nonspoken instruction prefix using arrived worker content.

Every native conversation token, already-arrived audio cache, and next audio
clock is preserved. No assistant output token is forced. Runtime tests must
establish actual result use; preservation alone does not qualify this interface.
"""
import hashlib

BASE_INSTRUCTIONS = (
    'You are a helpful English voice assistant. Answer general questions clearly. '
    'Private project records require a real lookup. Never invent their values. '
    'A pending lookup has no result yet. Keep listening and answer independent questions. '
    'When an actual completed lookup is supplied below, use that exact returned value. '
    'Do not claim a result before it has arrived. Ask which project if it is missing.'
)


def role_text(receipts=()):
    content=BASE_INSTRUCTIONS
    for receipt in receipts:
        if receipt['status']=='pending':
            content+=f"\nCURRENT CLIENT STATE: a lookup of {receipt['field']} for {receipt['project']} is running. The value has not returned yet."
        elif receipt['status']=='completed':
            content+=(f"\nCURRENT CLIENT STATE: the lookup has now completed successfully. "
                f"The actual {receipt['field']} for {receipt['project']} is {receipt['value']}. "
                "This authoritative value is available to you now. Tell the user this result, "
                "as well as answering their independent question. Earlier statements that you "
                "lack access or are waiting are now outdated.")
        else:
            raise ValueError('Unknown actual client receipt state')
    return '<|BOT|>system\n'+content+'<|EOT|>'


def create_session(framework, instructions, **kwargs):
    from lychee_fd.runtime import vllm_generation as native
    old=native.SYSTEM_MESSAGE_PREFIX
    try:
        native.SYSTEM_MESSAGE_PREFIX=instructions
        return framework.create_incremental_stream_session(**kwargs)
    finally:
        native.SYSTEM_MESSAGE_PREFIX=old


def refresh_session(session, instructions):
    import torch
    framework=session.fw
    if framework.vllm_engine._effective_prefix_caching:
        raise RuntimeError('Audio-dependent KV cannot use the text-only prefix cache')
    prefix=session.get_prefix();cache=session.export_incremental_audio_cache()
    old_system=session.system_input_length
    mapping=session.stoken_mapping.detach().clone()
    tail=session._audio_tail.copy()
    fields=('next_chunk_pos','chunk_idx','last_ss_pos','speaking_text_processor','speaking_stoken_processor',
        'latest_sl_prob','latest_ss_prob','latest_ks_prob','latest_kl_prob','latest_bc_prob')
    state={name:getattr(session,name) for name in fields}
    kwargs={'prefix':prefix,'initial_listening_state':session.get_listening_state(),
        'start_speak_token_factor':session.start_speak_token_factor,
        'start_listen_token_factor':session.start_listen_token_factor,
        'bc_speak_token_factor':session.bc_speak_token_factor,
        'end_speak_token_factor':session.end_speak_token_factor,'audio_incremental_mode':True}
    cpu_rng,gpu_rng=torch.get_rng_state(),torch.cuda.get_rng_state()
    session.close()
    fresh=create_session(framework,instructions,**kwargs)
    restored=fresh.restore_incremental_audio_cache(cache)
    fresh._audio_tail=tail
    delta=fresh.system_input_length-old_system
    body_mapping=mapping[:,old_system:]
    shifted=torch.where(body_mapping>=0,body_mapping+delta,body_mapping)
    fresh.stoken_mapping=torch.cat([torch.full((1,fresh.system_input_length),-1,
        dtype=mapping.dtype,device=mapping.device),shifted],dim=1)
    for name,value in state.items():
        if name=='last_ss_pos' and value is not None:value+=delta
        setattr(fresh,name,value)
    framework.last_ss_pos=fresh.last_ss_pos
    if (fresh.get_prefix()!=prefix or fresh.stoken_mapping.shape[1]!=fresh.generated_input_ids.shape[1]
            or restored['audio_input_token_count']!=cache['audio_input_token_count']
            or not (fresh._audio_tail==tail).all()
            or not torch.equal(cpu_rng,torch.get_rng_state()) or not torch.equal(gpu_rng,torch.cuda.get_rng_state())):
        fresh.close()
        raise RuntimeError('Instruction refresh changed native history, audio clock or sampling RNG')
    return fresh,{'schema':'dl-lychee-context-preservation-v1','body_tokens_preserved':True,
        'audio_input_tokens_preserved':restored['audio_input_token_count'],
        'actual_partial_audio_samples_preserved':len(tail),'next_audio_clock_preserved':state['next_chunk_pos'],
        'sampling_rng_preserved':True,'speech_mapping_indices_shifted_with_prefix':True,'old_system_tokens':old_system,'new_system_tokens':fresh.system_input_length,
        'instructions_sha256':hashlib.sha256(instructions.encode()).hexdigest(),
        'assistant_tokens_forced':False,'full_task_qualified':False}
