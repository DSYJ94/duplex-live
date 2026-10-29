from types import SimpleNamespace
import pytest
from portable.venus.terminal_speech import install


def fixture():
    calls=[];cache=object()
    def generate(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(shape=(1,min(75,kwargs['max_new_token']-1),1)),cache
    d=SimpleNamespace(current_turn_ended=True,model=SimpleNamespace(tts=SimpleNamespace(generate_chunk=generate)))
    return d,calls,cache


def test_final_chunk_allows_natural_finish_instead_of_capping_then_resetting():
    d,calls,cache=fixture();before,_=d.model.tts.generate_chunk(max_new_token=26,min_new_tokens=0)
    assert before.shape[1]==25
    records=install(d,128)
    tokens,actual=d.model.tts.generate_chunk(max_new_token=26,min_new_tokens=0,force_no_stop=False,past_key_values=cache)
    assert tokens.shape[1]==75 and actual is cache
    assert calls[-1]['max_new_token']==128 and calls[-1]['min_new_tokens']==0
    assert calls[-1]['force_no_stop'] is False
    assert records[-1]['changed'] and not records[-1]['possibly_budget_exhausted']


def test_first_nonfinal_chunk_and_existing_nonstandard_budget_stay_unchanged():
    d,calls,_=fixture();d.current_turn_ended=False;records=install(d)
    d.model.tts.generate_chunk(max_new_token=26,min_new_tokens=0)
    assert calls[-1]['max_new_token']==26 and not records[-1]['changed']
    d.current_turn_ended=True
    d.model.tts.generate_chunk(max_new_token=64,min_new_tokens=0)
    assert calls[-1]['max_new_token']==64 and not records[-1]['changed']


@pytest.mark.parametrize('budget',[26,201,1.5,True])
def test_budget_cannot_be_unbounded_or_inferred(budget):
    d,_,_=fixture()
    with pytest.raises(ValueError):install(d,budget)


def test_probe_calls_the_actual_native_frame_interface_without_teacher_feedback():
    import inspect
    from portable.venus.native_app import NativeEngine
    from training.venus.terminal_speech_probe import native_frame
    calls=[]
    class Engine:
        def step(self,pcm,feedback):
            inspect.signature(NativeEngine.step).bind(self,pcm,feedback)
            calls.append((pcm,feedback))
            return {'actual_interface':True}
    assert native_frame(Engine(),b'one-native-frame')=={'actual_interface':True}
    assert calls==[(b'one-native-frame',None)]
