"""Runbook 02 bake-off: MiniCPM-o 4.5 duplex T1–T6 protocol (audio-only).

Runs on Isambard via sbatch. Phase A synthesises user-side utterances with the
model's own TTS (voice_cloning, user voice = system_ref_audio_2.wav); Phase B
drives MiniCPMODuplex with a scripted per-second user timeline and records
everything. Outputs go to $DL/runs/bakeoff/<test>/ (Lustre — NOT git);
summaries are committed to evals/baselines/minicpm-o/ separately.

Tests (Runbook 02 §3):
  T1 baseline turn-taking  T2 barge-in  T3 backchannel
  T4 thinking pause        T5 background speech  T6 long session (optional)
"""

import argparse
import json
import time
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModel, AutoTokenizer

REPO = "openbmb/MiniCPM-o-4_5"
SNAP_ASSETS = None  # resolved from HF cache at runtime
SR_IN = 16000      # duplex input rate (chunk = 1000 ms = 16000 samples)
SR_OUT = 24000     # token2wav output rate

ASSISTANT_VOICE = "HT_ref_audio.wav"
USER_VOICE = "system_ref_audio_2.wav"

# ---------------------------------------------------------------- Phase A ---
# User-side utterances to synthesise (text -> filename stem).
PROMPTS = {
    # T1: ten short questions
    **{f"t1_q{i}": q for i, q in enumerate([
        "What is the capital of France?",
        "How many planets are in the solar system?",
        "What is seventeen times six?",
        "Who wrote Romeo and Juliet?",
        "What colour do you get mixing blue and yellow?",
        "How do you say thank you in Japanese?",
        "What is the boiling point of water in Fahrenheit?",
        "Name three primary colours.",
        "What year did the Apollo eleven mission land on the Moon?",
        "What is the square root of one hundred and forty four?",
    ], start=1)},
    # T2: long-answer bait + interruption
    "t2_bait": "Tell me a long, detailed story about the history of the railway, from the very beginning.",
    "t2_interrupt": "Stop — actually, just tell me who built the first locomotive, one sentence.",
    # T3: backchannels (played while model speaks)
    "t3_bc1": "Mm-hmm.",
    "t3_bc2": "Right.",
    "t3_bc3": "Uh-huh, okay.",
    # T4: thinking pause (two halves of one sentence)
    "t4_part1": "So I was wondering if you could explain, um...",
    "t4_part2": "...the difference between TCP and UDP, in one sentence.",
    # T5: background chatter (user never addresses the model)
    "t5_bg1": "Hey Sam, did you see the match last night? Unbelievable finish.",
    "t5_bg2": "I know, right? That last goal was something else entirely.",
    # T6: long-session dialogue lines
    **{f"t6_l{i}": q for i, q in enumerate([
        "Let's play twenty questions. I'm thinking of an animal.",
        "It lives in the ocean.",
        "No, bigger than that.",
        "It has teeth.",
        "No — it's a mammal.",
        "Yes! A sperm whale. Your turn — think of something.",
        "Is it bigger than a breadbox?",
        "Can you eat it?",
        "Hmm, is it found in a kitchen?",
        "I give up — what was it?",
        "Nice. Let's switch: tell me a two-sentence story about a lighthouse.",
        "Now make it spooky.",
        "Okay, one more round of twenty questions. I've picked one.",
        "It is alive.",
        "It can fly.",
        "It's not an insect.",
        "Yes, it's a bat. Thanks for playing!",
    ], start=1)},
}


def find_assets():
    import glob
    pat = "/projects/u6si/duplex-live/models/hf-cache/hub/models--openbmb--MiniCPM-o-4_5/snapshots/*/assets"
    hits = glob.glob(pat)
    assert hits, "assets dir not found in HF cache"
    return Path(hits[0])


def synth_user_prompts(model, tokenizer, out_dir: Path, only=None):
    """Phase A: TTS all user utterances with the user voice."""
    assets = find_assets()
    ref, _ = librosa.load(assets / USER_VOICE, sr=16000, mono=True)
    sys_prompt = model.get_sys_prompt(ref_audio=ref, mode="voice_cloning", language="en")
    out_dir.mkdir(parents=True, exist_ok=True)
    for stem, text in PROMPTS.items():
        if only and stem not in only:
            continue
        wav_path = out_dir / f"{stem}.wav"
        if wav_path.exists():
            continue
        msgs = [sys_prompt, {"role": "user", "content": [f"Please repeat the following content without any dialogue or explanation: <text>{text}</text>"]}]
        t0 = time.time()
        model.chat(
            msgs=msgs, tokenizer=tokenizer, use_tts_template=True,
            generate_audio=True, output_audio_path=str(wav_path),
        )
        print(f"[tts] {stem}: {time.time() - t0:.1f}s -> {wav_path.name}", flush=True)


# ---------------------------------------------------------------- Phase B ---
def load_timeline(spec, prompts_dir: Path, total_s: int) -> np.ndarray:
    """spec: list of (start_s, stem). Returns float32 16k mono waveform."""
    tl = np.zeros(total_s * SR_IN, dtype=np.float32)
    for start_s, stem in spec:
        wav, _ = librosa.load(prompts_dir / f"{stem}.wav", sr=SR_IN, mono=True)
        i0 = start_s * SR_IN
        i1 = min(i0 + len(wav), len(tl))
        tl[i0:i1] += wav[: i1 - i0]
    return np.clip(tl, -1.0, 1.0)


def run_duplex_test(duplex, name: str, timeline: np.ndarray, out_dir: Path,
                    prompts_dir: Path, ref_audio):
    """Drive the duplex session chunk-by-chunk, record log + audio."""
    out_dir.mkdir(parents=True, exist_ok=True)
    assets = find_assets()
    duplex.prepare(
        prefix_system_prompt="Streaming Omni Conversation.",
        ref_audio=ref_audio,
        prompt_wav_path=str(assets / ASSISTANT_VOICE),
    )
    n_chunks = int(np.ceil(len(timeline) / SR_IN))
    log_path = out_dir / "log.jsonl"
    out_audio = []
    with open(log_path, "w") as logf:
        for c in range(n_chunks):
            chunk = timeline[c * SR_IN:(c + 1) * SR_IN]
            if len(chunk) < SR_IN:
                chunk = np.concatenate([chunk, np.zeros(SR_IN - len(chunk), np.float32)])
            t0 = time.time()
            pre = duplex.streaming_prefill(audio_waveform=chunk)
            res = duplex.streaming_generate(prompt_wav_path=str(assets / ASSISTANT_VOICE))
            wall = time.time() - t0
            wav = res.get("audio_waveform")
            if wav is not None:
                wav = np.asarray(wav if isinstance(wav, np.ndarray) else wav.cpu().numpy(),
                                 dtype=np.float32).reshape(-1)
                out_audio.append(wav)
            rec = {
                "chunk": c, "t_s": c, "wall_s": round(wall, 3),
                "prefill_ok": pre.get("success"),
                "is_listen": res.get("is_listen"),
                "text": res.get("text", ""),
                "end_of_turn": res.get("end_of_turn"),
                "n_tokens": res.get("n_tokens"), "n_tts_tokens": res.get("n_tts_tokens"),
                "cost_llm": round(res.get("cost_llm", 0.0), 3),
                "cost_tts": round(res.get("cost_tts", 0.0), 3),
                "cost_token2wav": round(res.get("cost_token2wav", 0.0), 3),
                "out_samples": int(len(wav)) if wav is not None else 0,
            }
            logf.write(json.dumps(rec) + "\n")
            if c % 10 == 0 or (res.get("text") or "").strip():
                print(f"[{name}] chunk {c:4d} listen={rec['is_listen']} "
                      f"text={rec['text'][:60]!r} wall={wall:.2f}s", flush=True)
    sf.write(out_dir / "user_in.wav", timeline, SR_IN)
    if out_audio:
        sf.write(out_dir / "model_out.wav", np.concatenate(out_audio), SR_OUT)
    print(f"[{name}] done -> {out_dir}", flush=True)


TESTS = {
    # T1: ten questions with 15 s gaps (8 s was too tight — answers overlapped)
    "t1": ([(2 + i * 15, f"t1_q{i}") for i in range(1, 11)], 2 + 9 * 15 + 15),
    # T2: bait question, interrupt ~12 s into the answer
    "t2": ([(2, "t2_bait"), (16, "t2_interrupt")], 40),
    # T3: long-answer bait, three backchannels squarely mid-answer
    "t3": ([(2, "t2_bait"), (10, "t3_bc1"), (16, "t3_bc2"), (22, "t3_bc3")], 45),
    # T4: sentence part 1, 3 s silence, part 2
    "t4": ([(2, "t4_part1"), (9, "t4_part2")], 28),
    # T5: background-only chatter, user never addresses the model
    "t5": ([(2, "t5_bg1"), (8, "t5_bg2")], 22),
    # T6: ~15 min dialogue, one line every ~30 s
    "t6": ([(2 + i * 30, f"t6_l{i}") for i in range(1, 18)], 2 + 16 * 30 + 45),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/projects/u6si/duplex-live/runs/bakeoff")
    ap.add_argument("--tests", default="t1,t2,t3,t4,t5")
    args = ap.parse_args()
    out_root = Path(args.out)
    prompts_dir = out_root / "prompts"
    tests = [t.strip() for t in args.tests.split(",") if t.strip()]

    model = AutoModel.from_pretrained(
        REPO, trust_remote_code=True, attn_implementation="sdpa",
        torch_dtype=torch.bfloat16, init_vision=True, init_audio=True, init_tts=True,
    )
    model.eval().cuda()
    tokenizer = AutoTokenizer.from_pretrained(REPO, trust_remote_code=True)
    model.init_tts()
    print("[load] model ready", flush=True)

    synth_user_prompts(model, tokenizer, prompts_dir)

    assets = find_assets()
    ref_audio, _ = librosa.load(assets / ASSISTANT_VOICE, sr=16000, mono=True)
    duplex = model.as_duplex()
    print("[duplex] session wrapper ready", flush=True)

    for name in tests:
        spec, total_s = TESTS[name]
        timeline = load_timeline(spec, prompts_dir, total_s)
        run_duplex_test(duplex, name, timeline, out_root / name, prompts_dir, ref_audio)

    print("BAKEOFF DONE", flush=True)


if __name__ == "__main__":
    main()
