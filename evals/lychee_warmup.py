"""Warm only the audio front end on zeros before any real user samples arrive."""
import time


def warm_input(framework):
    import numpy as np
    import torch
    engine=framework.vllm_engine
    if engine._active_request_id is not None:
        raise RuntimeError('Input warmup must precede the conversation')
    cpu,gpu=torch.get_rng_state(),torch.cuda.get_rng_state()
    began=time.monotonic()
    with torch.inference_mode(),torch.random.fork_rng(devices=[torch.cuda.current_device()]):
        data=framework.stepaudio_audio_prepreocess(np.zeros(6400,dtype=np.float32))
        features=torch.nn.utils.rnn.pad_sequence(data['feats'],batch_first=True,padding_value=0).transpose(1,2)
        lengths=torch.tensor(data['feats_lengths'],dtype=torch.int32,device=framework.device)
        output,output_lengths=engine.precompute_audio(features,lengths)
        if not torch.isfinite(output).all() or len(data['input_ids'])!=10:
            raise RuntimeError('Unexpected native warmup feature clock')
        torch.cuda.synchronize()
    if (engine._active_request_id is not None or not torch.equal(cpu,torch.get_rng_state())
            or not torch.equal(gpu,torch.cuda.get_rng_state())):
        raise RuntimeError('Input warmup changed the native conversation or sampling state')
    return {'mode':'400ms zeros through only mel preprocessing and frozen audio encoder',
        'seconds':time.monotonic()-began,'user_samples_previewed':False,'lm_tokens_generated':0,
        'sampling_rng_preserved':True,'conversation_started':False}
