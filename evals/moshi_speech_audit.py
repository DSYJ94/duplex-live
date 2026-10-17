"""Plain words: transcribe generated native speech and retain the actual audio evidence.

ASR agreement is an intelligibility diagnostic, not a semantic or human rating.
This does not grant a training-data approval stamp to generated model output.
"""
import argparse
import hashlib
import importlib.metadata
import json
import math
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runs", nargs="+", type=Path, default=[])
    ap.add_argument("--live-runs", nargs="+", type=Path, default=[])
    ap.add_argument('--future-runs',nargs='+',type=Path,default=[],
                    help='Native sampled audio under approved forced-text conditioning; never a native reasoning pass')
    ap.add_argument('--teacher-render-runs',nargs='+',type=Path,default=[],
                    help='Complete forced teacher words with native sampled acoustics; diagnostic only')
    ap.add_argument('--tts-runs',nargs='+',type=Path,default=[],
                    help='External-controller text rendered by streaming TTS; never native duplex qualification')
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--trim-trailing-silence", action="store_true")
    ap.add_argument("--keep-silent-failures", action="store_true",
                    help="Retain silent clips as explicit failed records and audit later clips")
    ap.add_argument("--device",choices=("cuda","cpu"),default="cuda")
    args = ap.parse_args()
    if not any((args.runs,args.live_runs,args.future_runs,args.teacher_render_runs,args.tts_runs)):
        ap.error("At least one native or GPT-Live run is required")
    import numpy as np
    import soundfile as sf
    import whisper_timestamped as whisper
    from scipy.signal import resample_poly
    from data.synthetic.filter import norm_words, wer
    args.out.mkdir(parents=True, exist_ok=False)
    model = whisper.load_model("turbo", device=args.device)
    results = []
    for run,kind in ([(r,"native") for r in args.runs]+[(r,"GPT-Live") for r in args.live_runs]
                     +[(r,'native_forced_text_future') for r in args.future_runs]
                     +[(r,'forced_teacher_text_native_audio') for r in args.teacher_render_runs]
                     +[(r,'external_controller_streaming_tts') for r in args.tts_runs]):
        wav = run / ('teacher-text-native-audio.wav' if kind=='forced_teacher_text_native_audio'
                     else 'early-native-audio.wav' if kind=='native_forced_text_future'
                     else "model.wav" if kind in ('native','external_controller_streaming_tts') else "model_out.wav")
        pcm, sr = sf.read(wav, dtype="float32")
        if pcm.ndim != 1 or not np.isfinite(pcm).all():
            raise ValueError("Invalid generated mono speech")
        g = math.gcd(sr, 16000)
        audio = resample_poly(pcm, 16000//g, sr//g).astype(np.float32)
        original_samples = len(audio)
        silent_failure = False
        if args.trim_trailing_silence:
            # Keep the original WAV intact. Whisper can hallucinate repeated
            # words over a long quiet tail; trim only that tail for this
            # diagnostic, retaining 250 ms after the last active 20 ms window.
            frame = 320
            active = [i for i in range(0, len(audio), frame)
                      if float(np.sqrt(np.mean(audio[i:i+frame] ** 2))) > 0.001]
            if not active:
                if not args.keep_silent_failures:
                    raise RuntimeError("Generated audio has no signal above the diagnostic silence floor")
                silent_failure = True
            else:
                audio = audio[:min(len(audio), active[-1] + frame + 4000)]
        result = {"text": "", "segments": []} if silent_failure else whisper.transcribe(
            model, audio, language="en", vad=False, beam_size=5, best_of=5, temperature=0., verbose=None)
        if kind=='native_forced_text_future':
            from evals.recovery_data import validate_conversion
            manifest=json.loads((run/'early-future.json').read_text())
            if (manifest['future_audio_conditioning']!='native_sampled'
                    or manifest['future_assistant_audio_tokens_forced'] is not False
                    or manifest['future_text_tokens_forced'] is not True
                    or manifest['native_audio_sha256']!=hashlib.sha256(wav.read_bytes()).hexdigest()):
                raise RuntimeError('Future speech audit requires native audio with explicit forced-text provenance')
            teacher=manifest['teacher'];cache=Path(teacher['cache'])
            if hashlib.sha256((cache/'cache.json').read_bytes()).hexdigest()!=teacher['cache_sha256']:
                raise RuntimeError('Changed teacher cache')
            receipt=json.loads((cache/'cache.json').read_text())
            origin=next(r for r in validate_conversion(Path(receipt['conversion']))['accepted']
                if r['scenario_id']==teacher['id'])
            side=Path(origin['path']).with_suffix('.json')
            if hashlib.sha256(side.read_bytes()).hexdigest()!=teacher['sidecar_sha256']:
                raise RuntimeError('Changed measured teacher words')
            from evals.context_future_speech_scope import approved_words
            words,text_scope=approved_words(manifest,json.loads(side.read_text()),origin)
            predicted=' '.join(words)
        elif kind=='forced_teacher_text_native_audio':
            rendered=json.loads((run/'teacher-text-render-audit.json').read_text())
            if (rendered['schema']!='dl-context-teacher-text-native-audio-probe-v2'
                    or not rendered['passed'] or not rendered['teacher_text_forced']
                    or rendered['agent_acoustic_tokens_forced'] is not False
                    or rendered['audio_sha256']!=hashlib.sha256(wav.read_bytes()).hexdigest()
                    or rendered['teacher']['source_split']!='train'
                    or rendered['teacher']['source_model']!='gpt-live-1'
                    or rendered['generated_acoustic_frames']!=rendered['teacher']['real_frames']
                    or not rendered['provided_text_user_and_initial_prefix_only']
                    or not rendered['initial_pending_agent_codes_verified']
                    or not rendered['non_agent_native_input_comparison']['exact']
                    or rendered['training_approved'] is not False):
                raise RuntimeError('Teacher render lacks its diagnostic-only text/acoustic provenance')
            predicted=rendered['expected_text']
        elif kind=='external_controller_streaming_tts':
            spoken=json.loads((run/'report.json').read_text())
            if (spoken['schema']!='dl-external-controller-tts-probe-v1' or not spoken['passed']
                    or spoken['audio_sha256']!=hashlib.sha256(wav.read_bytes()).hexdigest()
                    or spoken['text_chosen_by']!='external gpt-6-luna controller'
                    or spoken['native_duplex_qualified'] is not False
                    or spoken['training_approved'] is not False):
                raise RuntimeError('TTS speech audit requires explicit external-controller provenance')
            predicted=spoken['text']
        elif kind == "native":
            predicted = json.loads((run / "report.json").read_text())["text"]
        else:
            predicted = "".join(p["text"] for p in json.loads((run / "transcripts.json").read_text()) if p["speaker"] == "assistant")
        record = {"run": str(run), "wav_sha256": hashlib.sha256(wav.read_bytes()).hexdigest(),
                  "audio_duration_s": len(pcm)/sr, "model_text": predicted,
                  "asr_input_duration_s": len(audio)/16000,
                  "trailing_silence_trimmed_s": (original_samples-len(audio))/16000,
                  "silence_trim_enabled": args.trim_trailing_silence,
                  "asr_text": result["text"], "asr_segments": result["segments"],
                  "wer_against_model_text": wer(norm_words(predicted), norm_words(result["text"])),
                  "semantic_review": "pending", "human_listening": "pending"}
        record["source_kind"] = kind
        if kind=='native_forced_text_future':
            record.update(text_was_forced=True,assistant_audio_tokens_forced=False,
                native_reasoning_qualified=False,training_approved=False,text_scope=text_scope)
        if kind=='forced_teacher_text_native_audio':
            record.update(text_was_forced=True,assistant_audio_tokens_forced=False,
                native_reasoning_qualified=False,training_approved=False)
        if kind=='external_controller_streaming_tts':
            record.update(text_chosen_by='external gpt-6-luna controller',native_reasoning_qualified=False,
                hybrid_system_qualified=False,training_approved=False)
        if silent_failure:
            record["audit_failure"] = "Generated audio has no signal above the diagnostic silence floor"
        record["asr_time_bounds_valid"] = all(
            0 <= segment["start"] < segment["end"] <= len(audio)/16000 + 0.04
            for segment in result["segments"])
        record["asr_word_bounds_valid"] = all(0 <= word["start"] < word["end"] <= len(audio)/16000+.04
            for segment in result["segments"] for word in segment.get("words",[]))
        results.append(record)
        print(json.dumps({k: v for k, v in record.items() if k != "asr_segments"}), flush=True)
    (args.out / "report.json").write_text(json.dumps({"schema": "dl-native-speech-audit-v1",
        "whisper": importlib.metadata.version("openai-whisper"),
        "timestamped": importlib.metadata.version("whisper-timestamped"), "device":args.device,"results": results}, indent=2)+"\n")


if __name__ == "__main__":
    main()
