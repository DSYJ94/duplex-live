# Duplex alternatives from the owner's references — 15 October 2026

Plain words: these references support two useful changes to our work: train and
measure interaction decisions separately from factual speech, and test leakage
from the loudspeaker without suppressing genuine user speech. They do not justify
replacing the now-smooth Lychee path before a bounded comparison. DuplexCascade
is a real released cascade, while FLM-Audio is a real released native multistream
model. Neither is a drop-in Lychee checkpoint or an established delegation model.

This note checks four supplied resources and their first-party releases. No jobs,
downloads of weights, gate acceptance, training or model qualification took place.
Sources were read on 15 October 2026; upstream `main` references can change. The
adoption decisions below are engineering judgments, not results from the papers.

## Verified resources

| Supplied resource | Verified identity and practical disposition |
|---|---|
| [2603.09180](https://arxiv.org/abs/2603.09180) | *DuplexCascade: Full-Duplex Speech-to-Speech Dialogue with VAD-Free Cascaded ASR–LLM–TTS Pipeline and Micro-Turn Optimization*, v1, 10 March 2026. Read the architecture and training recipe. |
| [sbintuitions/DuplexCascade](https://github.com/sbintuitions/DuplexCascade) | Official inference server and model wrapper, plus linked gated weights. Read actual loading and runtime code. |
| [2509.02521](https://arxiv.org/abs/2509.02521) | *FLM-Audio: Natural Monologues Improves Native Full-Duplex Chatbots via Dual Training*, current v3, 30 January 2026. The HTML title uses “Improve”. Official code and weights exist despite the paper's old “URLs will be available” wording. |
| [2606.19453v1](https://arxiv.org/html/2606.19453v1) | *A Survey of Full-Duplex Spoken Dialogue Systems: Architectural Hierarchy, Interaction Ontology, and Decision State Machine*, 17 June 2026. Useful evaluation framework; not a new trainable model release. |

## DuplexCascade: useful control recipe, separate architecture

Plain words: this keeps speech recognition, text reasoning and speech synthesis
as separate streaming services. It preserves access to a conventional text LLM,
at the cost of converting user audio into text before reasoning.

The paper sends partial ASR text to Qwen2-7B-Instruct every 0.6 seconds. The LLM
emits text and control decisions for waiting, answering, interruption,
backchannels and thinking pauses. Streaming TTS renders the response. This is
text micro-turn interleaving, rather than simultaneous user/assistant acoustic
token prediction in one native model. The authors train on 50,000 UltraChat
dialogues converted into simulated pauses, interruptions and backchannels, with
loss only on system micro-turns. Adaptation uses rank-16 query/value LoRA plus
trainable embeddings and prediction head, weighted control-token losses and
5,000 updates. Their reported training cost is eight H100s for five hours per
variant; it is not an Isambard runtime estimate. The paper reports 1.724-second
smooth-turn latency for its main model, so “full duplex” here does not imply
200-millisecond response onset. [Paper, sections 3–4 and table 1](https://arxiv.org/html/2603.09180v1)

The public repository contains inference instructions, `model.py`, `server.py`,
requirements and a browser client; the displayed root has no standalone data
construction or training script. Its model wrapper implements generic
cross-entropy and PEFT loading, which does not establish reproduction of the
paper's weighted training pipeline. [Repository](https://github.com/sbintuitions/DuplexCascade),
[model wrapper](https://raw.githubusercontent.com/sbintuitions/DuplexCascade/main/model.py)

The server requires separate Kyutai ASR and TTS services. It downloads
`sbintuitions/DuplexCascade`, builds the base model and LoRA, then loads the saved
state with `strict=False`. For our reproduction, missing/unexpected parameters
must be checked explicitly before a valid audition. The supplied server defaults
to a 0.6-second tick; changing it changes the policy's operating conditions.
[Server](https://raw.githubusercontent.com/sbintuitions/DuplexCascade/main/server.py),
[upstream launch instructions](https://github.com/sbintuitions/DuplexCascade#inference)

Real weights are listed: a 17.4 GB `model_state.safetensors`, tokenizer and
`train_cfg.json`. Access is gated; the model-card metadata also sets
`extra_gated_eu_disallowed: true`. Public source availability does not establish
that our account is eligible or authorised to fetch those weights. Preserve the
gate; do not route around it. [Weight listing](https://huggingface.co/sbintuitions/DuplexCascade/tree/main),
[model-card metadata](https://huggingface.co/sbintuitions/DuplexCascade/blob/main/README.md)

**Adoption judgment:** borrow the distinction between interruption, backchannel,
unfinished user thought and true turn completion. Build those as separate
retention examples and evaluation cases in the existing native stream. Do not
add its literal control tokens to Lychee without defining compatible targets and
training the appropriate native control path. If eligible checkpoint access is
available, an isolated cascade audition is a useful reasoning-oriented comparator;
it should not replace the accepted Lychee audio baseline on the strength of
published benchmark scores.

## FLM-Audio: native streams with contiguous text

Plain words: this is closer to Moshi's stream organization, but uses continuous
sentences in its text channel rather than spacing every word to match speech.

FLM-Audio uses a roughly 7B Qwen2.5-VL language backbone, Mimi at 12.5 frames/s,
eight audio codebooks per speaker, and a depth transformer for generated audio.
Each step combines text, listening and speaking representations. Its text can
run ahead of corresponding speech, then emit waiting tokens. The dual curriculum
first combines ASR-style and TTS-style examples, then progresses through
transcription/answer/speech supervision to response-only native duplex. The
reported alignment corpus is approximately one million hours and SFT uses
200,000 synthetic dialogues; recreating that foundation training is outside
this week's practical scope. Importantly for echo work, its speaking/text loss
does not train a listening-stream next-token target. Appendix B adds speaker
leakage into listening audio with probability 0.3, gain 0–0.2 and delay
0.1–0.5 seconds. This is augmentation, not proof of a learned AEC cancellation
mechanism. [Paper, sections 2–4 and appendix B](https://arxiv.org/html/2509.02521v3)

The authors release actual inference/server/client code and Hugging Face
weights. The current weight listing totals 16.8 GB, including four model shards,
custom model/configuration code and a Mimi tokenizer checkpoint. Its released
configuration is 28 backbone layers, width 3584, an eight-channel audio vocabulary
and a six-layer depth model. Those facts do not make Lychee adapters compatible.
[Official repository](https://github.com/cofe-ai/flm-audio),
[actual files](https://huggingface.co/CofeAI/FLM-Audio/tree/main),
[released configuration](https://huggingface.co/CofeAI/FLM-Audio/blob/main/config.json)

The documented deployment is a server plus CLI/Gradio clients, with Linux and at
least 20 GB VRAM recommended. The README identifies Apache-2.0 code, MIT vendored
Moshi code, and an additional research-use/default-voice statement; commercial
voice permissions cannot be inferred from the code license alone. The inspected
documentation exposes inference, not a validated matching fine-tuning recipe.
[Official deployment and licensing notes](https://github.com/cofe-ai/flm-audio)

Its server requirements pin Transformers 4.51.3 and `sphn==0.1.4`. This differs
from the accepted Lychee overlays; an isolated environment and ARM import/CUDA
check are necessary. Do not change the working container Torch to satisfy an
unverified installation. [Server requirements](https://raw.githubusercontent.com/cofe-ai/flm-audio/main/requirements-server.txt)

**Adoption judgment:** an unchanged FLM-Audio audition is feasible as a bounded
alternative, conditional on ARM runtime compatibility and English speech quality.
Prove timed result injection and a trainable forward/reload before scheduling
fine-tuning. Its useful immediate lesson is to preserve coherent text and measure
text/audio causality explicitly. Lychee already permits front-loaded text, so
FLM-Audio does not justify re-spacing Lychee targets into Moshi's word-aligned
layout. A different codec and depth model require separate native data conversion,
decoder audits, state collection and training verification; current Lychee
checkpoints and native S3 units cannot simply transfer.

## Survey: turn “duplex” into separate acceptance tests

Plain words: listening during speech, choosing a sensible reaction and producing
correct speech are separate requirements.

The survey organizes architectures by where decisions occur (module,
hidden-state, token and representation levels), classifies interactions by
timing, user intent and required response, and proposes Idle/Listen/Speak/Wait/
Dual states. It distinguishes genuine interruption from a supportive backchannel,
irrelevant background speech and an unfinished thought followed by silence. It
also warns of a realization gap between an architecture's possible concurrency
and behaviors demonstrated by data/evaluation. These are the survey authors'
analytical categories, not measurements of our model. [Survey, sections 4–8](https://arxiv.org/html/2606.19453v1)

**Adoption judgment:** report an explicit matrix beside factual/delegation gates:

| Condition | Required observation in our actual paced run |
|---|---|
| User interrupts with a correction | Stop the old spoken answer, preserve the correction and answer the new request. |
| User says “yes” while listening | Continue the intended reply without treating this as a new task. |
| User hesitates inside a sentence | Wait through the pause without inventing the missing request. |
| Background/echo speech overlaps output | Ignore irrelevant playback while retaining real overlapping user words. |
| A worker returns during speech | Keep the current interaction responsive and incorporate only the actual, current result. |
| Sustained user/assistant overlap | Retain user content; do not count simultaneous sound alone as understanding. |

For each cell retain audio, actual token/control events, arrived-input clocks,
worker receipts where relevant, and independent semantic/speech grading. A
synthetic waveform fixture tests plumbing; an actual microphone session tests a
different gate. This is our proposed evaluation extension, not a claim that the
survey's state machine should replace Lychee's native learned controls.

## What to implement within the remaining allocation

Plain words: complete the native result-return experiment first, then spend
compute on a small comparison that can change a decision.

The current repository reports strict native loading, a smooth concurrent
Lychee capture and conditional-adapter counterfactual development successes,
with remaining initial access denial, response delay, speech audit and live
microphone gaps. Treat those as the current local evidence, not as a completed
product. [Current execution record](../LYCHEE_NEXT_STAGE.md#current-execution--15-october-native-lychee-and-actual-audio)

1. Preserve the accepted audio path and complete concurrent actual-worker speech
   auditing before another backbone change.
2. Extend native training/evaluation with ordinary answers, no-result/pending
   turns, corrections, backchannels and leakage/overlap controls. Current
   result-only adaptation cannot establish preservation of all those behaviors.
3. Make echo attribution depend on clean speaker stems and actual playout
   reference timing. Never mask all user labels merely because the assistant is
   speaking: that would remove the barge-in supervision we need. Loss masking
   alone cannot remove echo from model input. The AEC design needs a separate
   clean-user target/input contamination check in the parent architecture note.
4. Bound any new-backbone work to unchanged baseline auditions first. FLM-Audio
   is the directly native alternative from these resources. DuplexCascade is a
   separately labelled text cascade and has a verified access gate. Neither has
   demonstrated our asynchronous delegation contract.
5. Allocate larger training only after the candidate passes cold native
   continuation, ordinary-answer retention, actual worker-result use and the
   relevant speech/interaction cells. Published training costs and scores do
   not predict this project's outcome or justify spending the allocation blindly.

This file records research and implementation recommendations only. No candidate
was adopted or qualified by this review.
