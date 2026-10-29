"""Content-free wall-time diagnostics; never fabricate missing native timings."""
import math

NATIVE_FIELDS=('cost_llm','cost_tts_prep','cost_tts','cost_token2wav','cost_all')


def stage_timings(result,*,prefill_seconds,generation_seconds):
    native={};invalid=[]
    for name in NATIVE_FIELDS:
        value=result.get(name)
        if type(value) not in (int,float) or not math.isfinite(value) or value<0:
            native[name]=None
            if name in result:invalid.append(name)
        else:native[name]=float(value)
    return dict(prefill_wall_seconds=prefill_seconds,generation_wall_seconds=generation_seconds,
        native_reported_wall_seconds=native,invalid_native_timing_fields=invalid,
        gpu_kernel_profiler=False,stages_not_required_to_sum_to_elapsed=True)
