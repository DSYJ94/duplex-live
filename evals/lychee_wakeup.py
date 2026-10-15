"""Disclosed client response-create diagnostic after an actual tool receipt.

Force only one future allowed start-speaking control decision, while already
arrived user audio is quiet. All words/audio remain sampled. This is not a
learned model timing or foreground delegation-choice qualification.
"""
from contextlib import contextmanager


@contextmanager
def result_response_event(holder,events):
    import time
    import torch
    from lychee_fd.runtime import vllm_generation as native
    original=native.ListeningControlLogitsProcessor
    class ClientEventProcessor(original):
        def __call__(self,input_ids,scores):
            scores=super().__call__(input_ids,scores)
            if (holder.get('result_wakeup_pending') and holder.get('user_chunk_quiet')
                    and torch.isfinite(scores[:,self.ss_token_id]).all()):
                keep=scores[:,self.ss_token_id].clone()
                scores.fill_(float('-inf'));scores[:,self.ss_token_id]=keep
                holder['result_wakeup_pending']=False
                events.append({'event':'application_response_start_control','at_s':time.monotonic()-holder['start'],
                    'control_id':int(self.ss_token_id),'native_control_choice':False,
                    'reason':'actual completed result; arrived user chunk quiet; native start-speaking slot',
                    'text_or_acoustic_output_forced':False})
            return scores
    native.ListeningControlLogitsProcessor=ClientEventProcessor
    try:
        yield
    finally:
        native.ListeningControlLogitsProcessor=original
