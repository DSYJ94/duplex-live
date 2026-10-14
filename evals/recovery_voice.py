"""Native model-triggered delegation over fresh audio and actual record tools.

Backend context uses measured completed user words, a causal transcript oracle.
This is not microphone/live-ASR acceptance. No ordered answer bank is used.
"""
import argparse
import asyncio
import concurrent.futures
import json
import math
import os
import time
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_data import validate_conversion
from evals.recovery_worker import RecordWorld,run_worker,WorkerSuperseded
from evals.recovery_context import causal_dialogue


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    parser.add_argument("--scenario",required=True)
    parser.add_argument("--adapter",type=Path)
    args = parser.parse_args(argv)
    import numpy as np
    import soundfile as sf
    import torch
    from moshi.models import loaders,LMGen
    from moshi.inference_utils.utils import get_condition_tensors,get_conditioning_remote_async
    from evals.recovery_train import load_base
    from evals.moshi_rag_adapter_pilot import load_adapter_exact
    from evals.moshi_rag_reference_step import apply_reference_frame
    conversion = validate_conversion(args.data)
    row = next(r for r in conversion["accepted"] if r["scenario_id"] == args.scenario)
    if row["split"] != "validation":
        raise RuntimeError("Development replay must use the sealed validation fold")
    sidecar = json.loads(Path(row["path"]).with_suffix(".json").read_text())
    metadata = sidecar["timeline"]["meta"]
    manifest = json.loads(args.manifest.read_text())
    lm = load_base(manifest)
    if args.adapter:
        audit = json.loads((args.adapter.parent / "report.json").read_text())
        reload = json.loads((args.adapter.parent / "reload-report.json").read_text())
        if not reload.get("passed") or sha256(args.adapter) != audit["saved_adapter_sha256"]:
            raise RuntimeError("Unverified adapter cannot enter speech evaluation")
        lm = load_adapter_exact(lm,args.adapter)
    lm.eval().requires_grad_(False)
    rag = Path(manifest["rag"]["path"])
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo=None,moshi_weights=rag / "model.safetensors",
        mimi_weights=rag / "tokenizer-e351c8d8-checkpoint125.safetensors",tokenizer=rag / "tokenizer_spm_32k_3.model",
        config_path=rag / "config.json")
    mimi = info.get_mimi(device="cuda").eval().requires_grad_(False)
    tokenizer = info.get_text_tokenizer()
    gen = LMGen(lm,condition_tensors=get_condition_tensors("moshi",lm,1,1.,first_speaker="user"),
        force_streaming_sum=True,**info.lm_gen_config)
    audio,rate = sf.read(row["path"],dtype="float32",always_2d=True)
    if rate != mimi.sample_rate or audio.shape[1] != 2:
        raise RuntimeError("Invalid approved native input")
    pcm = np.concatenate([audio[:,1],np.zeros(20*rate,dtype=np.float32)])
    frame = round(rate/mimi.frame_rate)
    args.out.mkdir(parents=True,exist_ok=False)
    sf.write(args.out / "stem_user.wav",pcm,rate)
    world = RecordWorld({metadata["project"]:{metadata["field"]:metadata["phrase"]}}) if metadata["phrase"] else RecordWorld({})
    events,packets,output = [],[],np.zeros(0,dtype=np.float32)
    futures,completed,active = {},[],None
    delay = max(0.,(metadata.get("planned_lookup_ready_s") or 0)-(metadata.get("lookup_trigger_s") or 0))
    start = None
    def worker(context,task_id,revision,triggered):
        result = run_worker(context,world,task_id,expected_revision=revision)
        receipt = next((r for r in result["receipts"] if r["tool"] == "lookup_record"),None)
        if receipt:
            time.sleep(max(0.,start+triggered+delay-time.monotonic()))
            if not world.deliverable(receipt):
                raise WorkerSuperseded("Completed lookup was cancelled before reference delivery")
            world.mark_ready(receipt)
        rows = asyncio.run(get_conditioning_remote_async(result["content"],os.environ.get("RECOVERY_REFERENCE_URL","http://127.0.0.1:18001"),timeout=60))
        return result,rows,receipt,time.monotonic()-start
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        with mimi.streaming(1),gen.streaming(1),torch.inference_mode():
            for _ in range(16):
                tokens = gen.step(mimi.encode(torch.zeros(1,1,frame,device="cuda")))
                if tokens is not None:
                    mimi.decode(tokens[:,1:])
            torch.cuda.synchronize()
            mimi.reset_streaming()
            gen.reset_streaming()
            torch.manual_seed(424242)
            start = time.monotonic()
            with (args.out / "packets.jsonl").open("w") as log:
                for index in range(len(pcm)//frame):
                    due = (index+1)/mimi.frame_rate
                    time.sleep(max(0.,start+due-time.monotonic()))
                    before = time.monotonic()-start
                    for task_id,(future,revision) in list(futures.items()):
                        if not future.done():
                            continue
                        del futures[task_id]
                        try:
                            result,rows,receipt,ready = future.result()
                            completed.append((task_id,result,rows,receipt,ready))
                            events.append({"event":"worker_completed","task_id":task_id,"ready_at_s":ready,"result":result})
                        except WorkerSuperseded as error:
                            events.append({"event":"result_rejected","task_id":task_id,"reason":str(error),"at_s":before})
                        except Exception as error:
                            events.append({"event":"worker_failed","task_id":task_id,"error_type":type(error).__name__,"at_s":before})
                    if active is not None and active[1] and not world.deliverable(active[1]):
                        gen._streaming_state.pending_streaming_sums[0] = None
                        events.append({"event":"active_reference_cancelled","task_id":active[0],"at_s":before})
                        active = None
                    # Hold fact delivery while any new semantic request is being
                    # resolved, so a cancellation can invalidate a racing result.
                    pending_rows = gen._streaming_state.pending_streaming_sums[0]
                    if completed and not futures and (pending_rows is None or not len(pending_rows)):
                        task_id,result,rows,receipt,ready = completed.pop(0)
                        if receipt and not world.deliverable(receipt):
                            events.append({"event":"result_rejected","task_id":task_id,"reason":"stale completed receipt","at_s":before})
                        else:
                            gen.update_streaming_sum_tensors([rows[0]])
                            active = (task_id,receipt)
                            events.append({"event":"reference_queued","task_id":task_id,"at_s":before,"rows":rows.shape[1],
                                "route":result["route"],"receipt":receipt,"ready_at_s":ready})
                    codes = mimi.encode(torch.from_numpy(pcm[index*frame:(index+1)*frame]).cuda()[None,None])
                    if index == 0:
                        gen.step(codes)
                    applied = apply_reference_frame(gen)
                    tokens = gen.step(codes)
                    text,token = "",None
                    if tokens is not None:
                        token = int(tokens[0,0,0])
                        if token >= 5:
                            text = tokenizer.id_to_piece(token).replace("▁"," ")
                        wave = mimi.decode(tokens[:,1:]).detach().float().cpu().numpy().reshape(-1)
                        if not np.isfinite(wave).all():
                            raise RuntimeError("Nonfinite generated speech")
                        offset = max(len(output),math.ceil((time.monotonic()-start)*rate))
                        output = np.concatenate([np.pad(output,(0,offset-len(output))),wave])
                    torch.cuda.synchronize()
                    elapsed = time.monotonic()-start
                    if token == 4:
                        if len([e for e in events if e["event"] == "worker_started"]) >= 8:
                            events.append({"event":"request_limit_reached","at_s":elapsed})
                        else:
                            words = [w for w in sidecar["user_words"] if w["end_s"] <= due]
                            context = causal_dialogue(sidecar,due,packets,elapsed)
                            task_id = "request-"+str(index)
                            revision = world.reserve(task_id,context=context)
                            futures[task_id] = (pool.submit(worker,context,task_id,revision,elapsed),revision)
                            events.append({"event":"worker_started","task_id":task_id,"at_s":elapsed,"input_available_s":due,
                                "latest_user_word_end_s":max((w["end_s"] for w in words),default=None),"context":context,
                                "trigger":"native model RAG token"})
                    packet = {"frame":index,"input_available_s":due,"input_consumed_s":before,"packet_available_s":elapsed,
                        "compute_s":elapsed-before,"input_backlog_s":before-due,"text":text,"token":token,"reference_l1":applied}
                    packets.append(packet)
                    log.write(json.dumps(packet)+"\n")
                    log.flush()
    sf.write(args.out / "model.wav",output,rate)
    report = {"scenario":args.scenario,"split":"validation","adapter":str(args.adapter) if args.adapter else None,
        "text":"".join(p["text"] for p in packets),"events":events,"tool_receipts":world.receipts,
        "reference_applied_frames":sum(p["reference_l1"]>0 for p in packets),
        "compute_p50_s":float(np.median([p["compute_s"] for p in packets])),
        "compute_p95_s":float(np.percentile([p["compute_s"] for p in packets],95)),
        "max_input_backlog_s":max(p["input_backlog_s"] for p in packets),
        "context_source":"measured completed user words: causal transcript oracle, not live ASR",
        "context_format":"separate chronological user utterances and generated assistant chunks",
        "routing":"native speech model requests work; API worker selects semantic operation and arguments",
        "complete_task_success":"pending grading","human_listening":"pending","microphone_acceptance":False}
    (args.out / "report.json").write_text(json.dumps(report,indent=2)+"\n")
    (args.out.parent / "manifest.jsonl").write_text(json.dumps({"scenario_id":args.scenario,"dir":args.out.name})+"\n")
    print(json.dumps({k:v for k,v in report.items() if k not in ("events","tool_receipts")}),flush=True)


if __name__ == "__main__":
    main()
