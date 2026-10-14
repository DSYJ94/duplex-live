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
    controls = "".join(cards)
    cards.clear()
    card("GPT-Live reference", "gpt-live-replay-7400991/t1/model_out.wav",
         "The original four-question development replay.")
    card("PersonaPlex NATF2 — rejected", "personaplex-recovery-7402673/t1/model_out.wav",
         "The voice already judged below the target. Includes an incorrect bread-staling answer.")
    for voice in ("NATF1", "NATM1", "synthetic-marin"):
        root = f"voice-quality-{args.custom_persona_job}" if voice == "synthetic-marin" and args.custom_persona_job else prefix
        card("PersonaPlex " + voice, f"{root}/{voice}/t1/model_out.wav",
             "Same model, input, prompt and sampling as NATF2; only voice conditioning changes. Not yet human accepted.",
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
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    page += '<title>Voice diagnostics</title><style>body{max-width:1100px;margin:32px auto;padding:0 20px;background:#10141c;color:#e7ecf5;font:16px/1.5 system-ui}section{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:16px}article{padding:20px;background:#1a2230;border:1px solid #354157;border-radius:12px}audio{width:100%}p{color:#b9c5d8}a{color:#9dc3ff}</style>'
    page += '<h1>Voice diagnostics</h1><p>The owner rejected both current PersonaPlex and MoshiRAG voices. None of these new probes has been accepted. Compare naturalness, clarity, rhythm and artifacts separately from answer correctness.</p>'
    page += '<h2>Does the codec preserve an acceptable voice?</h2><section>'+controls+'</section>'
    page += '<h2>Generated speech on the same four science questions</h2><p>Different models may answer differently. Only completed local recordings appear below. These are development auditions, not a controlled full-task ranking.</p><section>'+''.join(cards)+'</section>'
    page += '<p><a href="/audio/voice-comparison/index.html">Earlier task comparisons</a> · <a href="/">Full gallery</a></p></html>'
    out = args.runs / "voice-diagnostics" / "index.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    print(out)


if __name__ == "__main__":
    main()
