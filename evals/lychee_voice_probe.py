"""Strict native Lychee-FD development audition using the released duplex heads.

Offline quality evidence only: the official offline generator is not a paced
microphone test. No delegation adapter or tool-selection claim is made here.
"""
import argparse
import importlib
import io
import json
import math
import subprocess
import sys
import time
import types
from pathlib import Path
from evals.moshi_model_download import sha256
from evals.lychee_model_download import SOURCE_COMMIT


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--assets", type=Path, required=True)
    ap.add_argument("--batch", type=Path, required=True)
    ap.add_argument("--voice", type=Path, required=True)
    ap.add_argument("--voice-provenance", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    provenance = json.loads(args.voice_provenance.read_text())
    if (provenance.get("schema") != "dl-voice-codec-probe-v1"
            or provenance.get("reference_sha256") != sha256(args.voice)
            or provenance.get("source_model") != "gpt-live-1"):
        raise RuntimeError("Synthetic voice prompt does not match its receipt")
    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio
    import transformers
    from evals.lychee_environment_check import configure_bf16
    compatibility = configure_bf16()
    from scipy.signal import resample_poly
    from data.synthetic.quality import checked_rows
    from evals.minicpm_recovery_probe import read_reference_wav

    checked_rows(args.batch, [{"scenario_id": "dense_dialog-001"}], role="Lychee native voice audition")
    assets = json.loads(args.assets.read_text())
    source = Path(assets["source"])
    if (assets["source_commit"] != SOURCE_COMMIT or subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != SOURCE_COMMIT
            or subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)):
        raise RuntimeError("Lychee source changed")
    for model in assets["models"].values():
        for name, record in model["files"].items():
            if sha256(Path(model["path"]) / name) != record["sha256"]:
                raise RuntimeError("Pinned Lychee asset changed: " + name)
    model_path = Path(assets["models"]["duplex"]["path"])
    vocoder_path = Path(assets["models"]["vocoder"]["path"]) / "token2wav"
    sys.path[:0] = [str(source), str(source / "third_party" / "Step-Audio2")]
    args.out.mkdir(parents=True, exist_ok=False)

    # The published HF auto_map names StepAudio2ForCausalLM (a simplex class
    # without the acoustic/control/merge branches). The published offline runtime
    # imports an absent lychee_fd.models.lychee_duplex module. Select the actual
    # released training implementation explicitly; never discard unmatched heads.
    native = importlib.import_module("Training_package.Code.Models.LycheeFD")
    loading_receipt = {}

    class StrictLycheeFD(native.LycheeFD):
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            if Path(path).resolve() != model_path.resolve():
                raise RuntimeError("Unexpected checkpoint path")
            # One allocated GH200 fits the checkpoint. Explicit placement avoids
            # automatic splitting/offload of the custom control/acoustic heads.
            kwargs["device_map"] = {"": 0}
            model, info = super().from_pretrained(path, **kwargs,
                local_files_only=True, output_loading_info=True)
            loading_receipt.update(info)
            if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
                raise RuntimeError("Duplex checkpoint does not load exactly: " + repr(info))
            if any(p.device.type != "cuda" for p in model.parameters()):
                raise RuntimeError("Native voice probe must not silently offload parameters")
            return model

    bridge = types.ModuleType("lychee_fd.models.lychee_duplex")
    bridge.LycheeDuplexConfig = native.LycheeFDConfig
    bridge.LycheeDuplexForCausalLM = StrictLycheeFD
    sys.modules[bridge.__name__] = bridge
    from lychee_fd.runtime.hf_v9_generation import SingleTurnGenerationFramework

    def wav_load(*pos, **kw):
        pcm, rate = read_reference_wav(*pos, **kw)
        return torch.from_numpy(pcm), rate

    torchaudio.load = wav_load  # explicit PCM-only adapter already exercised by MiniCPM
    torch.manual_seed(424242)
    environment = {
        "schema": "dl-lychee-voice-probe-v1", "assets_manifest_sha256": sha256(args.assets),
        "source_commit": SOURCE_COMMIT, "repo_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "torch": torch.__version__, "transformers": transformers.__version__,
        "mode": "official offline multi-head generation; no paced or microphone claim",
        "model_class": "Training_package.Code.Models.LycheeFD.LycheeFD with strict loading wrapper",
        "compatibility": "explicit alias for absent official runtime module; PCM-only TorchAudio loader",
        "bf16_environment_rule": compatibility,
        "training_approved": False, "learned_delegation_tested": False,
        "voice_provenance": provenance,
    }
    (args.out / "environment.json").write_text(json.dumps(environment, indent=2)+"\n")
    framework = SingleTurnGenerationFramework("V9", str(model_path),
        device="cuda", attn_implementation="sdpa", torch_dtype=torch.bfloat16)
    (args.out / "loading.json").write_text(json.dumps(loading_receipt, indent=2)+"\n")
    pcm, rate = sf.read(args.batch / "dense_dialog-001" / "stem_user.wav", dtype="float32")
    if pcm.ndim != 1 or len(pcm) < 60*rate or not np.isfinite(pcm).all():
        raise RuntimeError("Invalid approved user audio")
    divisor = math.gcd(rate, 16000)
    audio = resample_poly(pcm[:60*rate], 16000//divisor, rate//divisor).astype(np.float32)
    audio = np.pad(audio, (0, 10*16000))
    sf.write(args.out / "user_in.wav", audio, 16000)
    started = time.monotonic()
    with torch.inference_mode():
        events = framework.full_chunk_stream_offline_generation(audio)
    (args.out / "events.json").write_text(json.dumps(events, indent=2)+"\n")
    # Release the large speech LM before loading the separate published vocoder.
    del framework
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    from token2wav import Token2wav
    vocoder = Token2wav(str(vocoder_path))
    reference = args.voice
    output = np.zeros(70*24000, dtype=np.float32)
    for index, event in enumerate(events):
        if not event.get("audio"):
            continue
        blob = vocoder(event["audio"], str(reference))
        speech, sr = sf.read(io.BytesIO(blob), dtype="float32")
        if sr != 24000 or speech.ndim != 1 or not np.isfinite(speech).all():
            raise RuntimeError("Invalid vocoder output")
        sf.write(args.out / f"utterance-{index:03d}.wav", speech, sr)
        start = round(event["start_time"]*sr)
        end = start+len(speech)
        output = np.pad(output, (0, max(0, end-len(output))))
        output[start:end] += speech
    if not np.any(np.abs(output) > .001) or float(np.max(np.abs(output))) > 1:
        raise RuntimeError("Missing or clipping assembled speech; inspect separate utterances")
    sf.write(args.out / "model.wav", output, 24000)
    report = {"text": " ".join(e["text"] for e in events),
        "wall_seconds_including_vocoder": time.monotonic()-started,
        "voice_reference": str(reference), "voice_reference_sha256": sha256(reference),
        "loading": loading_receipt, "human_voice_verdict": "pending",
        "paced": False, "live_acceptance": False}
    (args.out / "report.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
