"""Encode approved audio with native Mimi/interleaver and frozen ARC results."""
import argparse
import asyncio
import json
import math
import sys
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.recovery_data import validate_conversion


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",type=Path,required=True)
    parser.add_argument("--manifest",type=Path,required=True)
    parser.add_argument("--out",type=Path,required=True)
    args = parser.parse_args()
    import torch
    import soundfile as sf
    from moshi.models import loaders
    from moshi.inference_utils.utils import get_conditioning_remote_async
    from evals.moshi_timed_reference import reference_schedule
    sys.path.insert(0,"/projects/u6si/duplex-live/vendor/moshi-finetune-2acc879fe7c48f885a18f6cc9548bccb2674d87b")
    from finetune.data.interleaver import Interleaver,InterleavedTokenizer
    conversion = validate_conversion(args.data)
    manifest = json.loads(args.manifest.read_text())
    rag = Path(manifest["rag"]["path"])
    for name in ("config.json","tokenizer_spm_32k_3.model","tokenizer-e351c8d8-checkpoint125.safetensors"):
        if sha256(rag / name) != manifest["rag"]["files"][name]["sha256"]:
            raise RuntimeError("Pinned codec/vocabulary/config changed")
    info = loaders.CheckpointInfo.from_hf_repo(hf_repo=None,moshi_weights=rag / "model.safetensors",
        mimi_weights=rag / "tokenizer-e351c8d8-checkpoint125.safetensors",
        tokenizer=rag / "tokenizer_spm_32k_3.model",config_path=rag / "config.json")
    mimi = info.get_mimi(device="cuda").eval().requires_grad_(False)
    tokenizer = info.get_text_tokenizer()
    interleaver = Interleaver(tokenizer,mimi.frame_rate,text_padding=3,end_of_text_padding=0,
        zero_padding=-1,keep_main_only=True,keep_and_shift=True,device="cuda")
    args.out.mkdir(parents=True,exist_ok=False)
    receipt = {"conversion":str(args.data),"ingestion_sha256":sha256(args.data / "ingestion.json"),
        "model":manifest["rag"],"examples":[],"frame_rate":mimi.frame_rate,
        "method":"pinned native InterleavedTokenizer, assistant left/user right, measured alignments; frozen HTTP ARC rows",
        "human_listening":"pending"}
    for row in conversion["accepted"]:
        audio,rate = sf.read(row["path"],dtype="float32",always_2d=True)
        if rate != mimi.sample_rate or audio.shape[1] != 2:
            raise RuntimeError("Native cache requires approved stereo codec-rate audio")
        sidecar = json.loads(Path(row["path"]).with_suffix(".json").read_text())
        codes = InterleavedTokenizer(mimi,interleaver,row["duration"])(audio.T,0.,row["path"]).codes
        if codes.shape[:2] != (1,17):
            raise RuntimeError("Unexpected native codebook layout")
        expected = [token for word,_,speaker in sidecar["alignments"] if speaker == "SPEAKER_MAIN"
                    for token in tokenizer.encode(word.strip())]
        actual = [token for token in codes[0,0].tolist() if token not in (-1,0,3)]
        if actual != expected:
            raise RuntimeError("Native interleaver dropped or reordered measured assistant tokens")
        markers = []
        for event in sidecar["timeline"]["events"]:
            if event["kind"] != "marker":
                continue
            index = round(event["actual_start_ms"] / 1000 * mimi.frame_rate)
            if not 0 <= index < codes.shape[-1] or int(codes[0,0,index]) not in (0,3):
                raise RuntimeError("Native delegation marker collides with a measured word")
            codes[0,0,index] = 4
            markers.append(index)
        arrivals = []
        for item in sidecar["reference_arrivals"]:
            rows = asyncio.run(get_conditioning_remote_async(item["text"],"http://127.0.0.1:18001",timeout=60))
            arrivals.append({**item,"rows":rows[0].detach().cpu()})
        reference = reference_schedule(torch.arange(codes.shape[-1],dtype=torch.float64)*.08,arrivals,4096)[None]
        name = row["scenario_id"] + ".pt"
        torch.save({"codes":codes.cpu(),"reference":reference,"arrivals":arrivals,"markers":markers},args.out / name)
        receipt["examples"].append({"scenario_id":row["scenario_id"],"split":row["split"],"file":name,
            "sha256":sha256(args.out / name),"frames":codes.shape[-1],"markers":markers,
            "reference_rows":int((reference.abs().sum(-1)>0).sum()),"kind":sidecar["timeline"]["meta"]["kind"]})
        (args.out / "cache.json").write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt["examples"][-1]),flush=True)


if __name__ == "__main__":
    main()
