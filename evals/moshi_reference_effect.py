"""Repeat one native LMGen slot to isolate captured reference influence.

Three passes share identical approved text/audio histories. Two controls must
repeat exactly; only the third receives real frozen ARC rows. This is a counterfactual,
not an audible reference-use or task-success qualification.
"""
import argparse
import json
import math
import os
from pathlib import Path
from evals.recovery_data import validate_cache
from evals.recovery_train import load_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    import torch
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    from moshi.models import LMGen
    from moshi.inference_utils.utils import get_condition_tensors
    receipt = validate_cache(args.cache)
    row = next(r for r in receipt["examples"] if r["scenario_id"] == "recovery-validation-000")
    state = torch.load(args.cache / row["file"],map_location="cpu",weights_only=True)
    arrival = state["arrivals"][0]
    injection_frame = math.ceil(arrival["ready_at_s"]/.08)
    lm = load_base(json.loads(args.manifest.read_text())).eval()
    logits_history = []
    def hook(logits):
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite captured inference logits")
        logits_history.append(logits.detach().float().cpu().clone())
    cursor = [0]
    def same_text_history(tokens):
        tokens.fill_(int(state["codes"][0,0,cursor[0]]))
    gen = LMGen(lm,use_sampling=False,force_streaming_sum=True,on_text_logits_hook=hook,on_text_hook=same_text_history,
        condition_tensors=get_condition_tensors("moshi",lm,1,1.,first_speaker="user"))
    source = state["codes"][:,lm.dep_q+1:].cuda()
    assistant = state["codes"][:,1:lm.dep_q+1].cuda()
    histories = []
    with gen.streaming(1),torch.inference_mode():
        for _ in range(16):
            gen.step(source[...,:1],depformer_replace_tokens=assistant[...,:1])
        for pass_index in range(3):
            gen.reset_streaming()
            # Native reset handles model/history state, but not the reference
            # queue. Clear both explicitly between controlled experiments.
            gen._streaming_state.pending_streaming_sums = [None]
            gen._streaming_state.condition_streaming_sum.zero_()
            logits_history.clear()
            for frame in range(source.shape[-1]):
                cursor[0] = frame
                if pass_index == 2 and frame == injection_frame:
                    gen.update_streaming_sum_tensors([arrival["rows"]])
                gen.apply_pending_streaming_sum_condition(torch.ones(1,dtype=torch.bool))
                active = gen._streaming_state.condition_streaming_sum
                if (pass_index < 2 or frame < injection_frame) and active.count_nonzero():
                    raise RuntimeError("Reference leaked into a control or causal prefix")
                gen.step(source[...,frame:frame+1],depformer_replace_tokens=assistant[...,frame:frame+1])
            histories.append(list(logits_history))
    records = [{"frame":i,"max_abs_logit_difference":float((a-b).abs().max()),
        "control_repeat_max_abs_error":float((a-c).abs().max()),
        "control_top_token":int(a.flatten().argmax()),"reference_top_token":int(b.flatten().argmax())}
        for i,(a,c,b) in enumerate(zip(*histories))]
    repeat = max(r["control_repeat_max_abs_error"] for r in records)
    before = max(r["max_abs_logit_difference"] for r in records[:injection_frame])
    after = max(r["max_abs_logit_difference"] for r in records[injection_frame:])
    args.out.mkdir(parents=True,exist_ok=False)
    report = {"control_repeat_max_abs_error":repeat,"same_slot_prefix_max_abs_error":before,"post_reference_max_abs_difference":after,
        "injection_frame":injection_frame,"injection_time_s":injection_frame*.08,
        "reference_rows":len(arrival["rows"]),"cuda_graphs":True,"greedy_sampling":True,
        "attention_backend":"math SDPA, deterministic algorithms required; diagnostic only, no production latency claim",
        "assistant_history":"identical approved teacher-forced text and audio in three single-slot passes; no free-run quality claim",
        "complete_task_success_tested":False,"records":records,"passed":repeat==0. and before==0. and after>.001}
    (args.out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "records"}),flush=True)
    if not report["passed"]:
        raise RuntimeError("Captured inference failed paired reference-effect proof")


if __name__ == "__main__":
    main()
