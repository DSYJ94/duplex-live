"""Build the local listening comparison from existing, completed probe artifacts."""
import argparse
import html
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", type=Path, default=Path("runs"))
    ap.add_argument("--quality-job", required=True, type=int)
    ap.add_argument("--minicpm-job", type=int)
    ap.add_argument("--custom-persona-job", type=int)
    ap.add_argument("--lychee-job", type=int)
    ap.add_argument("--gpt-live-job", type=int)
    ap.add_argument("--codec-cpu-job", type=int)
    ap.add_argument("--codec-cuda-job", type=int)
    args = ap.parse_args()
    cards = []

    def card(title, relative, note, completed=None):
        if not (args.runs / relative).is_file() or (completed and not (args.runs / completed).is_file()):
            return
        cards.append(f'<article><h3>{html.escape(title)}</h3><p>{html.escape(note)}</p>'
                     f'<audio controls preload="none" src="/audio/{html.escape(relative, quote=True)}"></audio></article>')

    prefix = f"voice-quality-{args.quality_job}"
    card("GPT-Live original excerpt", f"{prefix}/codec/reference.wav",
         "10 seconds of the synthetic marin voice. Continuous original audio, without resynthesis.")
    card("Same excerpt through Mimi", f"{prefix}/codec/mimi-roundtrip.wav",
         "Exactly the same words, encoded and decoded through the native eight-codebook codec. No speech model generation.")
    for device, job in (("CPU", args.codec_cpu_job), ("GPU", args.codec_cuda_job)):
        if job:
            card("Same codec tokens, uncompiled " + device, f"voice-codec-decode-{job}/decoded.wav",
                 "The same captured codec tokens decoded without compilation; changes no words or model weights.",
                 f"voice-codec-decode-{job}/report.json")
    controls = "".join(cards)
    cards.clear()
    card("Moshi original capture", "moshi-rag-speech-7387777-delayed-reference/model.wav",
         "Existing capture inserts two buffer underruns totaling 123 ms. Model quality remains unaccepted.")
    card("Same Moshi PCM, 160 ms startup buffer", "moshi-playout-diagnostic-20261014/buffered.wav",
         "The exact original generated samples, with a causal startup buffer. Removes both recorded underruns; not proof all clicks are fixed.")
    if args.gpt_live_job:
        root = f"gpt-live-replay-{args.gpt_live_job}/t1"
        card("Fresh GPT-Live voice only", root+"/packet-stream.wav",
             "Every original generated sample joined continuously to isolate voice quality from network gaps. Not a live timing demonstration.", root+"/report.json")
        card("Fresh GPT-Live, 160 ms startup buffer", root+"/model_out-buffered.wav",
             "Fresh actual API replay. Buffering still leaves underruns here; inspect the voice-only player to judge the model sound.", root+"/report.json")
    playback = "".join(cards)
    cards.clear()
    card("GPT-Live reference", "gpt-live-replay-7400991/t1/model_out.wav",
         "The original four-question development replay.")
    card("PersonaPlex NATF2 — rejected", "personaplex-recovery-7402673/t1/model_out.wav",
         "The voice already judged below the target. Includes an incorrect bread-staling answer.")
    for voice in ("NATF1", "NATM1", "synthetic-marin"):
        root = f"voice-quality-{args.custom_persona_job}" if voice == "synthetic-marin" and args.custom_persona_job else prefix
        card("PersonaPlex " + voice, f"{root}/{voice}/t1/model_out.wav",
             "Same model, input, prompt and sampling as NATF2; only voice conditioning changes. Synthetic-marin voice accepted by owner on 14 October; other voices unqualified.",
             f"{root}/{voice}/summary.json")
    card("MiniCPM original voice", "minicpm-recovery-7377673/model.wav",
         "Existing native duplex replay. Previous latency and factual failures remain; voice audition only.")
    if args.minicpm_job:
        root = f"minicpm-recovery-{args.minicpm_job}"
        card("MiniCPM with synthetic marin reference", root+"/model.wav",
             "Same development user audio. Voice conditioning changed; this does not turn it into GPT-Live.", root+"/report.json")
    if args.lychee_job:
        root = f"lychee-voice-{args.lychee_job}"
        card("Lychee-FD with synthetic marin reference", root+"/model.wav",
             "Strictly loaded native model and published vocoder, offline audition. No paced or delegation qualification.", root+"/report.json")
    alternatives = "".join(cards)
    cards.clear()
    if args.gpt_live_job:
        root = f"gpt-live-replay-{args.gpt_live_job}/t1"
        card("GPT-Live voice only", root+"/packet-stream.wav",
             "Start here. Original generated speech without inserted network gaps. Voice-quality reference, not a latency demonstration.", root+"/report.json")
    card("PersonaPlex original NATF2", "personaplex-recovery-7402673/t1/model_out.wav",
         "Rejected by owner as muffled, like radio speech. This exporter already joins speech frames continuously.")
    if args.custom_persona_job:
        root = f"voice-quality-{args.custom_persona_job}/synthetic-marin"
        card("PersonaPlex with synthetic voice reference", root+"/t1/model_out.wav",
             "Voice quality accepted by owner as comparable to GPT-Live. Factual answers still fail; delegation and real-time use remain unqualified.", root+"/summary.json")
    if args.lychee_job:
        root = f"lychee-voice-{args.lychee_job}"
        card("Lychee-FD — accepted voice", root+"/utterance-000.wav",
             "Voice quality accepted by owner as comparable to GPT-Live. First response excerpt; full recording below. Offline audition, no delegation or live acceptance.", root+"/report.json")
    card("PersonaPlex paced native replay", "personaplex-paced-7415855/t1/model-buffered.wav",
         "Same accepted voice setup paced at microphone speed. 32.46 ms p95 compute per 80 ms frame; one 50.71 ms playback gap remains. No delegation qualification.",
         "personaplex-paced-7415855/t1/paced-report.json")
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Voice diagnostics</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#10141c;color:#e7ecf5;font:16px/1.5 system-ui}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}article{padding:20px;background:#1a2230;border:1px solid #354157;border-radius:12px}audio{width:100%}p{color:#b9c5d8}a{color:#9dc3ff}</style>'
    page += '<h1>Compare the audio artifacts</h1><p>The owner accepts the custom PersonaPlex and Lychee voices as comparable to GPT-Live. Original NATF2 is muffled and rejected. Voice acceptance is separate from intelligence, delegation and real-time performance.</p>'
    page += '<section>'+''.join(cards)+'</section>'
    page += '<details><summary>Codec controls: same words, different decoder execution</summary><section>'+controls+'</section></details>'
    page += '<details><summary>Playback timing controls</summary><section>'+playback+'</section></details>'
    page += '<details><summary>Other voices on the same four science questions</summary><p>Different models may answer differently. Only completed local recordings appear below. These are development auditions, not a full-task ranking.</p><section>'+alternatives+'</section></details>'
    page += '<p><a href="/audio/voice-comparison/index.html">Earlier task comparisons</a> · <a href="/">Full gallery</a></p></html>'
    out = args.runs / "voice-diagnostics" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()
