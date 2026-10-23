"""Explicit resource placement; recognition and native inference stay unchanged."""


def assign(visible,profile,asr_python):
    devices=visible.split(',')
    if not asr_python or not all(x.strip() and x==x.strip() for x in devices):
        raise ValueError('Actual assigned CUDA devices and isolated recognition Python required')
    if profile=='separate-gpu':
        if len(devices)!=2 or len(set(devices))!=2:
            raise ValueError('Native-ASR consultation requires two distinct assigned GPUs')
        return devices[0],devices[1]
    if profile=='shared-gpu-evaluation':
        if len(devices)!=1:raise ValueError('Shared evaluation requires exactly one assigned GPU')
        return devices[0],devices[0]
    raise ValueError('Unknown native-ASR resource profile')


def report(args):
    profile=getattr(args,'native_asr_placement','separate-gpu')
    if profile not in ('separate-gpu','shared-gpu-evaluation'):raise ValueError('Unknown ASR placement')
    return dict(profile=profile,separate_native_and_asr_processes=True,
                distinct_physical_gpus=profile=='separate-gpu',
                recognition_model_beam_timestamps_and_arrived_audio_bounds_unchanged=True,
                resource_timing_cohort_differs_from_two_gpu=profile=='shared-gpu-evaluation')
