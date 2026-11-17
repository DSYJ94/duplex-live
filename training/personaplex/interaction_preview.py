"""Single-play native interaction on the actual frame clock, never live latency.

Raw collector PCM omits frames with no output. Reinsert those gaps from the
verified trace; do not line up compact output with the start of user audio.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reconstruct(user,raw,trace,*,user_frames,sample_rate=24000,frame_samples=1920):
    if (sample_rate!=24000 or frame_samples!=1920 or user.ndim!=1 or raw.ndim!=1
            or not np.isfinite(user).all() or not np.isfinite(raw).all()
            or type(user_frames) is not int or user_frames<1
            or len(user)!=user_frames+60*sample_rate or np.any(user[user_frames:])
            or len(trace)!=(len(user)+frame_samples-1)//frame_samples):
        raise ValueError('Whole verified user input and native trace geometry required')
    timeline=np.zeros((len(trace)*frame_samples,2),dtype=np.float32)
    timeline[:len(user),1]=user
    offset=0
    for index,entry in enumerate(trace):
        count=entry['audio_samples']
        if (entry['sequence']!=index or entry['raw_sample_offset']!=offset
                or count not in (0,frame_samples) or offset+count>len(raw)):
            raise ValueError('Every native output frame/offset must remain exact')
        pcm=raw[offset:offset+count]
        if hashlib.sha256(pcm.tobytes()).hexdigest()!=entry['pcm_sha256']:
            raise ValueError('Native per-frame output hash differs')
        timeline[index*frame_samples:index*frame_samples+count,0]=pcm
        offset+=count
    if offset!=len(raw):raise ValueError('No missing/extra raw output samples allowed')
    return timeline


def render(case,user_row,output,*,response_seconds=15):
    case=Path(case);output=Path(output)
    if type(response_seconds) is not int or not 1<=response_seconds<=60:
        raise ValueError('Explicit bounded response excerpt required')
    result=json.loads((case/'result.json').read_text())
    user_path=Path(user_row['folder'])/'user.wav';raw_path=case/'agent-raw.wav';trace_path=case/'trace.jsonl'
    if (result.get('complete') is not True or result['id']!=user_row['id']
            or result['input_sha256']!=user_row['input_sha256'] or sha(user_path)!=result['input_sha256']
            or sha(raw_path)!=result['raw_wav_sha256'] or sha(trace_path)!=result['trace_sha256']
            or result.get('paced_replay') is not False or result.get('live_latency_measured') is not False):
        raise ValueError('Exact completed offline native source case required')
    user,rate=sf.read(user_path,dtype='float32');raw,raw_rate=sf.read(raw_path,dtype='float32')
    if rate!=raw_rate or rate!=24000:raise ValueError('Exact native sample rate required')
    trace=[json.loads(line)for line in trace_path.read_text().splitlines()]
    full=reconstruct(user,raw,trace,user_frames=user_row['user_frames'])
    end=min(len(full),user_row['user_frames']+response_seconds*rate)
    output.mkdir(exist_ok=False)
    # Separate stereo channels preserve original amplitudes/overlap. No gain,
    # time stretching, silence deletion, answer editing or lip-sync correction.
    for name,wave in [('complete-interaction.wav',full),('preview.wav',full[:end])]:
        sf.write(output/name,wave,rate,subtype='FLOAT')
        replay,replay_rate=sf.read(output/name,dtype='float32',always_2d=True)
        if replay_rate!=rate or not np.array_equal(replay,wave):raise ValueError('Presentation PCM round trip differs')
    report=dict(id=result['id'],source_result_sha256=sha(case/'result.json'),user_sha256=sha(user_path),
                raw_output_sha256=sha(raw_path),trace_sha256=sha(trace_path),
                channels=['native_assistant','synthetic_user'],sample_rate=rate,
                complete_samples=len(full),preview_samples=end,response_excerpt_seconds=response_seconds,
                preview_is_prefix_excerpt=end<len(full),zero_output_gaps_restored_from_trace=True,
                original_PCM_amplitude_preserved=True,offline_native_frame_clock=True,
                live_latency_measured=False,correctness_or_quality_qualified=False,
                checkpoint_changed=False,optimizer_updates=0,
                generated_text=result.get('generated_text'),preview_sha256=sha(output/'preview.wav'))
    (output/'preview.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--case',type=Path,required=True)
    p.add_argument('--input-manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--response-seconds',type=int,default=15);a=p.parse_args()
    source=json.loads(a.input_manifest.read_text())
    selected=[row for row in source['cases'] if row['id']==a.case.name]
    if len(selected)!=1:raise ValueError('One exact source case required')
    print(json.dumps(render(a.case,selected[0],a.output,response_seconds=a.response_seconds)),flush=True)
