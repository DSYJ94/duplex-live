"""Paired native LMGen slots verify references reach captured inference logits.

Both slots share identical input and greedy generated histories until only the
second receives real frozen ARC rows. This is an engineering counterfactual,
not an audible reference-use or task-success qualification.
"""
import argparse
import json
import math
from pathlib import Path
from evals.recovery_data import validate_cache
from evals.recovery_train import load_base


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    import torch
    from moshi.models import LMGen
    from moshi.inference_utils.utils import get_condition_tensors
    from evals.moshi_rag_reference_step import apply_reference_frame
    receipt = validate_cache(args.cache)
    row = next(r for r in receipt["examples"] if r["scenario_id"] == "recovery-validation-000")
    state = torch.load(args.cache / row["file"],map_location="cpu",weights_only=True)
    arrival = state["arrivals"][0]
    injection_frame = math.ceil(arrival["ready_at_s"]/.08)
    lm = load_base(json.loads(args.manifest.read_text())).eval()
    records = []
    def hook(logits):
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite captured inference logits")
        records.append({"max_abs_logit_difference":float((logits[0]-logits[1]).abs().max()),
            "slot_top_tokens":[int(v) for v in logits[:,0,0].argmax(-1)]})
    gen = LMGen(lm,use_sampling=False,force_streaming_sum=True,on_text_logits_hook=hook,
        condition_tensors=get_condition_tensors("moshi",lm,2,1.,first_speaker="user"))
    source = state["codes"][:,lm.dep_q+1:].cuda().expand(2,-1,-1)
    with gen.streaming(2),torch.inference_mode():
        for _ in range(16):
            gen.step(source[...,:1])
        gen.reset_streaming()
        records.clear()
        for frame in range(source.shape[-1]):
            if frame == injection_frame:
                gen.update_streaming_sum_tensors([None,arrival["rows"]])
            gen.apply_pending_streaming_sum_condition(torch.ones(2,dtype=torch.bool))
            active = gen._streaming_state.condition_streaming_sum
            if active[0].count_nonzero():
                raise RuntimeError("Reference leaked into the control slot")
            gen.step(source[...,frame:frame+1])
            records[-1].update(frame=frame,reference_l1_slot1=float(active[1].float().abs().sum()))
    before = max(r["max_abs_logit_difference"] for r in records[:injection_frame])
    after = max(r["max_abs_logit_difference"] for r in records[injection_frame:])
    args.out.mkdir(parents=True,exist_ok=False)
    report = {"same_slot_prefix_max_abs_error":before,"post_reference_max_abs_difference":after,
        "injection_frame":injection_frame,"injection_time_s":injection_frame*.08,
        "reference_rows":len(arrival["rows"]),"cuda_graphs":True,"greedy_sampling":True,
        "complete_task_success_tested":False,"records":records,"passed":before==0. and after>.001}
    (args.out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({k:v for k,v in report.items() if k != "records"}),flush=True)
    if not report["passed"]:
        raise RuntimeError("Captured inference failed paired reference-effect proof")


if __name__ == "__main__":
    main()
