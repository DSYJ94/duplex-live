"""Explicit inference-only calibration of the native retrieval-token logit."""
import math
from pathlib import Path
from evals.moshi_model_download import sha256


def validate(bias):
    if not math.isfinite(bias) or not -8.<=bias<=0.:
        raise ValueError('RET calibration must be finite and between -8 and zero')


def adjust(logits,bias):
    validate(bias)
    if bias==0.:return logits
    if logits.ndim!=4 or logits.shape[1]!=1 or logits.shape[-1]<=4:
        raise RuntimeError('RET calibration requires the native temporal text-logit layout')
    calibrated=logits.clone()
    calibrated[...,4]+=bias
    return calibrated


def install(engine,bias):
    validate(bias)
    if bias==0.:return None
    if hasattr(engine,'_dl_ret_bias_calls'):
        raise RuntimeError('RET calibration is already installed')
    state=engine.lm_gen._streaming_state;original=state.graphed_main
    engine._dl_ret_bias_calls=0
    def calibrated(value):
        hidden,logits=original(value)
        engine._dl_ret_bias_calls+=1
        return hidden,adjust(logits,bias)
    state.graphed_main=calibrated
    return {'token':4,'logit_bias':bias,'policy':'constant live temporal RET-logit adjustment',
        'code_sha256':sha256(Path(__file__)),'scenario_dependent':False,
        'text_or_audio_tokens_forced':False,'inference_only':True}
