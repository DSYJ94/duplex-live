# Research map for native duplex voice and asynchronous reasoning

Reviewed 21 October 2026 against the authors' papers, repositories and model cards.
This is a research disposition, not a new training approval or a live job report.
Branch URLs describe what was inspected today; they are not dependency pins. Any
code adoption needs its own immutable commit, license inventory and validation.

## Target and current implementation

The target is a strong native voice model that listens while speaking, handles
interruptions and intentional silence, answers basic questions, and learns when
to ask a stronger asynchronous backend for reasoning or tools. A useful backend
answer must return into the ongoing conversation without blocking listening,
reviving cancelled work or overwriting newer user intent.

Our existing DuetaSpan-7B / PersonaPlex / Moshi lineage already provides the
appropriate native speech streams and Context Span interface. The saved
[cookbook](NATIVE-TRAINING-COOKBOOK-2026-10-21.md) describes the latest eight-update
research stage, not a qualified final model. The
[control repair](NATIVE-CONTROL3-REPAIR-2026-10-21.md) targets observed silence and
pause/resume faults. Its 40 proposed TRAIN scenes are a focused control repair;
they do not constitute a complete curriculum for learning reasoning delegation.
All 28 original actual histories, 49 approved teacher histories, KL/hidden
retention and held-out hashes remain required.

## What the supplied sources contribute

### Context Spanning: keep the existing native interface

The paper connects an ongoing duplex speech stream to an asynchronous backend:
the speech model emits a retrieval token, backend processing uses transcribed
user context, and returned text enters the model through chunked prefill. This
matches the intended split between conversational voice and external knowledge
or reasoning. The authors also identify weaknesses: all training audio was
synthesized, reasoning depends on backend ASR/LLM, and the system is not a
generalized agent. Reported Full-Duplex-Bench v3 results include substantial
interruption and filler problems. The interface therefore supplies a mechanism,
not evidence that our silence, cancellation or delegation policy works.
[Paper §§3–5](https://arxiv.org/html/2609.33443v1)

**Available:** official inference, preparation/training entrypoints, benchmark
harness, tool bank and released DuetaSpan-7B weights. The training format binds
stereo audio, word timing, retrieval markers, reference text and a lead/body
boundary; the delay sampler is constrained by that boundary. The public example
and pipeline are useful, but do not establish release of the complete original
training corpus. Code is MIT; the weight card declares NVIDIA Open Model License
and PersonaPlex ancestry. Code and weight licenses are distinct.
[Repository](https://github.com/mindlogic-ai/ContextSpanning),
[training format](https://github.com/mindlogic-ai/ContextSpanning/blob/main/docs/TRAINING.md),
[weight card](https://huggingface.co/mindlogicinc/context-spanning-7b)

**Disposition:** continue native fine-tuning. Evaluate the trigger decision,
backend work, result timing and spoken answer separately. A good external answer
does not prove correct native delivery. A retrieval token is also not a calibrated
decision about whether a task requires deeper reasoning.

### DuplexCascade: useful interaction curriculum, different architecture

DuplexCascade trains a Qwen2-7B text model to make micro-turn control decisions
between streaming ASR and TTS. Its synthetic interaction construction includes
pauses, user thinking, interruptions and backchannels; it adapts 50K UltraChat
dialogues. This offers useful examples of explicit *continue listening* targets
and of separating a short backchannel from a genuine takeover. It is a cascaded
system, rather than our native speech model. Its paper's reported adaptation
uses eight H100s for five hours, so the method is not evidence for a tiny repair
budget.
[Paper §§3–4](https://arxiv.org/html/2603.09180v1)

**Available:** MIT inference code with a browser/server bridge and separate
Kyutai streaming ASR/TTS dependencies. The checked root contains `model.py` and
`server.py`; the README documents inference rather than a complete trainer or
released 50K derived corpus. The linked model card declares MIT but requires
access acceptance and currently displays an EU download restriction; availability
must not be inferred from a public GitHub repository.
[Inference repository](https://github.com/sbintuitions/DuplexCascade),
[model card](https://huggingface.co/sbintuitions/DuplexCascade)

**Disposition:** borrow interaction categories and compare outcomes later; do
not replace the native model with the Qwen cascade. Convert any useful curriculum
idea into newly authored, audited native histories with actual audio timing.

### FLM-Audio: evidence for text planning and interaction coverage

FLM-Audio uses contiguous text monologues and waiting tokens instead of
word-by-word speech alignment, with a staged scheme in which text leads or trails
audio. The paper describes roughly one million hours of speech post-training and
200K synthetic instruction dialogues. Its final full-duplex stage introduces
interruptions at arbitrary times; meaningful interruption stops both monologue
and speech, with a delay intended to avoid reacting to short backchannels. This
supports testing semantic interruption intent and preserving answer quality.
Changing our text/audio layout to this scheme would require new alignment and
training, not just different loss weights on the current caches.
[Paper §§2–4](https://arxiv.org/html/2509.02521v3)

**Available:** model weights/custom inference code and an English/Chinese voice
server. The examined README documents deployment, not reproduction of the entire
training corpus and four-stage training procedure. Its code notice says
Apache-2.0, with MIT Moshi portions, while also retaining default-voice rights and
stating research-only use/commercial contact. Treat this as an unresolved
deployment-permission question, rather than assume that the code badge clears
model/voice rights.
[Official repository](https://github.com/cofe-ai/flm-audio),
[model and usage notice](https://huggingface.co/CofeAI/FLM-Audio)

**Disposition:** use as motivation for a later controlled planning/alignment
experiment if present supervision still fails. Do not change the native clock,
codec or approved histories during the current control repair.

### BayLing-Duplex: learned timing and a valuable evaluation counterexample

BayLing uses a GLM-4-Voice backbone, block-interleaved user audio, assistant text
and assistant audio, plus dialogue-state tokens. It reports SFT on 400K synthetic
samples followed by timing-oriented DPO. Its ablation is especially relevant:
uniform token weighting produces near-permanent silence, yet can still score
100% interruption success. Silence success must therefore be paired with valid
speech onset, correct answers and successful resume. The paper does not establish
cue-conditioned silence or asynchronous reasoning delegation. Its default
10-token audio block imposes a 0.8-second decision granularity; that is not a
latency improvement over our 80ms native frame clock. Synthetic near-field data,
and untested backchannels/multi-party behavior, constrain the reported results.
[Paper §§2–4 and Limitations](https://arxiv.org/html/2606.14528v1)

**Available:** inference repository, model card and checkpoint download route;
the model card identifies a 9B backbone. The checked public documentation does
not supply a complete 400K training-data release or executable SFT/DPO recipe.
The repository carries a GLM-4-Voice model license with commercial registration
and attribution conditions, plus separate upstream component notices.
[Repository](https://github.com/BayLing-Models/BayLing-Duplex),
[model card](https://github.com/BayLing-Models/BayLing-Duplex/blob/main/MODEL_CARD.md),
[license](https://github.com/BayLing-Models/BayLing-Duplex/blob/main/LICENSE),
[component notice](https://github.com/BayLing-Models/BayLing-Duplex/blob/main/NOTICE.md)

**Disposition:** keep quiet, release, resume and answer correctness as separate
required outcomes. Consider preference training on timing only after reliable
SFT behavior and an audited preference corpus exist. Do not copy BayLing's
silence weights into Moshi: token distributions and layouts differ.

### Moshi: foundation and training tooling already relevant to our lineage

Moshi models user and assistant speech as parallel streams with assistant text
and a hierarchical temporal/depth transformer. This is the foundation for native
simultaneous listening and speaking, rather than evidence of generalized
reasoning or reliable task delegation.
[Moshi paper](https://arxiv.org/html/2410.00037v2)

**Available:** PyTorch, MLX and Rust inference stacks, Mimi codec, weights, web
client and a separate LoRA fine-tuning repository. Original Moshi code licenses
are MIT for Python/client and Apache for Rust; original Moshi weights are CC-BY
4.0. Those licenses do not automatically replace the licenses on our downstream
PersonaPlex/Context Spanning weights.
[Moshi repository](https://github.com/kyutai-labs/moshi)

The official fine-tuning tooling expects assistant audio on the **left** and user
audio on the **right**; Context Spanning's documented raw format uses the opposite
channel ownership. This is a concrete ingestion hazard. Its LoRA tutorial is a
reference implementation, not an unchanged substitute for our span-aware
full154/49-history trainer.
[Fine-tuning preparation](https://github.com/kyutai-labs/moshi-finetune#-prepare-dataset)

**Data:** DailyTalkContiguous is a public stereo conversation dataset with word
timestamps and CC-BY-SA-4.0 labeling. It may help a separately reviewed natural
conversation/retention experiment; it does not itself provide cue-wait or
delegation labels. It has not been downloaded, approved or added to TRAIN here.
[Dataset card](https://huggingface.co/datasets/kyutai/DailyTalkContiguous)

### Full-duplex survey: turn the aspiration into measurable coverage

The supplied survey is arXiv2606.19453. Its authors' repository now uses the title
*Speaking While Listening: A Survey and Empirical Audit of Full-Duplex Spoken
Dialogue Systems*, and explains the redirect from `DuplexLM/DuplexSurvey` to
`MM-Speech/DuplexSurvey`. It supplies an architecture hierarchy, a temporal
relation × user intent × response ontology, and Idle/Listen/Speak/Wait/Dual
states. These are useful audit categories; a state diagram does not train or
validate a model. The repository collects models/datasets/benchmarks and visual
summaries, rather than supplying a trained replacement model. Each linked
resource retains its own license and split.
[Paper](https://arxiv.org/html/2606.19453v1),
[authors' current repository](https://github.com/MM-Speech/DuplexSurvey)

**Disposition:** annotate our existing scenarios against these categories. In
particular, distinguish user hesitation from completed turns, backchannels from
stop requests, intentional wait from a stall, and listening during external work
from talking continuously over the user.

## Ranked changes to our plan

These are implementation recommendations inferred from the sources and our
recorded failures, not claims that the papers demonstrated our target behavior.

1. **Finish and test the focused native control repair.** Require complete quiet
   intervals, correct cue release, old-cue rejection, semantic interruption and
   resume, alongside ordinary spoken answers. Keep the independent quality,
   semantic, conversion and numerical chain. Existing control40 alone cannot
   establish broad conversational quality or learned delegation.
2. **Make delegation a measured policy.** Preserve the original delegation
   histories, then plan matched, newly authored cases: answer an easy question
   directly; delegate a hard or tool-dependent one; ask for missing information;
   keep listening during delay; accept a correction/cancel; reject a stale result;
   resume and speak the correct returned answer. Score unnecessary delegation as
   well as failure to delegate. A fixed hand-written router would be a separate
   system component, not proof that the speech model learned that decision.
3. **Separate oracle-result and real-backend evaluations.** First test native
   routing and delivery with a known correct delayed result. Then test actual
   ASR/backend/tool execution with the same task family. Record trigger timing,
   request identity, completion, cancellation, result ownership, spoken accuracy,
   premature answers, stalls and latency. An API component smoke test alone
   cannot qualify the complete speech loop.
4. **Broaden voice quality only with a declared corpus and budget.** After the
   bounded control pilot, add realistic pauses, benign backchannels, prosody,
   acoustic conditions and connected multi-turn conversations in a versioned
   curriculum. Compare baseline/candidate using unseen speakers and wording.
   Preserve acoustic quality and ordinary-answer retention while improving timing.
5. **Defer expensive architecture/RL work.** FLM-style stream rearrangement,
   BayLing-style tokenization and timing DPO are separate experiments. Kyutai's
   interactivity RL and MoshiRAG were already reviewed in the
   [existing note](KYUTAI-INTERACTIVITY-MOSHIRAG-REVIEW-2026-10-20.md); that note's
   evaluation-versus-training and oracle-versus-live-backend distinctions remain
   applicable. None supplies an automatic fix for our failed native behavior.

No external checkpoint, dataset, benchmark audio or code was imported by this
review. Public evaluation samples remain evaluation material. The supplied
YouTube link was not treated as primary technical evidence; this map follows the
authors' paper/code releases instead. No new training, GPU submission or spending
was performed for this research task.
