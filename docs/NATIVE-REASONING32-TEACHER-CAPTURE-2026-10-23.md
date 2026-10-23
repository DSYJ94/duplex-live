# Reasoning32 actual teacher capture — 23 October2026

The new opt-in collector is `evals/native_reasoning_teacher.py`, using
`native_reasoning_stream.py`, the actual Responses transport and the asynchronous
lifecycle ledger. Nineteen offline lifecycle/transport/stream/artifact tests pass.
Source-v3 all40 objects/56 clips/32 TRAIN scenes independently qualify at the
existing word/waveform bars. This does not approve an assistant teacher or model.

The collector revalidates that exact whole source and quality report, the retained
native tokenizer and the complete engine audit's unchanged232-frame persona
prefix. It attempts one immutable whole32 collection, at most two live sessions
concurrently and at most six native-selected backend requests per session
(192 aggregate). It stops new sessions after a transport/readback failure;
existing sessions retain their raw evidence. No automatic retries or resume into
the same output directory. A partial collection cannot become an approval.

The voice teacher is `gpt-live-1`, client delegation, voice marin; the backend is
the existing `gpt-6-luna`. Every case gets the same general instructions and
backend interface. The application does not inspect words to classify difficulty
or force requests. Case families, expected answers and optimizer oracles never
enter model payloads. Only actual arrived USER transcript fragments and current
task state enter a backend request. Cancellation backend content is status only;
the voice teacher must itself answer any accompanying basic arithmetic question.

Timing is declared before capture, without observing whether an answer is right:

- First user clip begins at0.56seconds; all original PCM16 samples are sent once
  in continuous actual-paced80ms packets, with zeros between speech clips.
- Follow-up begins at the next packet boundary after both first-source completion
  plus0.56seconds and the first recorded actual HTTP dispatch plus0.56seconds.
  If the voice teacher never requests work, no dispatch or follow-up is invented.
- First result's minimum transport delay is the larger of the source plan's
  declared minimum and0.56seconds plus follow-up duration plus a fixed4second
  response opportunity. Later requests have zero added transport delay. Actual
  provider completion and deliberate transport hold have separate clocks.
- Close no earlier than5seconds after the last source and each current result's
  eligibility boundary, with all backend tasks completed. Retired work cannot
  impose its old delivery delay; an executing HTTP call must still finish and be
  retained. A bounded88second input ceiling fails incomplete work. Close does not
  depend on answer correctness, inferred intent or selecting a passing response.
- Preserve actual server closure, then measure complete causal output playout,
  allowing at most20seconds of backlog drain. No tail, packet or sample is cropped.

All raw input/output PCM, availability packets, events, transcripts, request JSON,
response bytes, request IDs, lifecycle revisions and hashes stay in the new output
tree. Acknowledgments must match actual commentary event IDs; acknowledgment is
not proof of speech or physical hearing. Keys remain local and unrecorded.
Independent readback reconstructs sent source samples and original output, binds
every backend request to its native event and arrived transcript, and checks the
actual response body/identity. Fixtures cannot obtain actual-transport approval.

Frame accounting includes232 prefix frames, the COMPLETE measured episode rounded
up at12.5Hz and every actually sent context span, including a span later cancelled.
Each span counts its two delimiters plus tokenizer output. More than1100 frames
fails the diagnostic; no cropping or changed context bound. This count remains
separate from actual native conversion, assembly, numerical and gradient approval.
An overlong or semantically wrong teacher cannot qualify because transport passed.

Still required: all32 actual teachers, independent output-word/audio and semantic
approval, exact actual-time conversion and native full-sequence approval, all
retained objectives, a distinct bounded pilot, independent Adam/RNG/native restore,
complete behavior/speech review and eligible sealed fresh qualification. The
current C9 control40 GPU source and one-eight-update pilot remain unchanged.

Protocol references were refreshed from the official OpenAI documentation:
[client delegation](https://developers.openai.com/api/docs/guides/live-delegation)
defines native request IDs, separate transcript context, commentary and its
acknowledgments; [session finalization](https://developers.openai.com/api/docs/guides/live-conversations)
requires retaining the connection through `session.closed` and treating client
backend work separately; [request debugging](https://developers.openai.com/api/reference/overview)
documents provider and client request identifiers. These API rules are transport
requirements, not evidence that our native model has learned delegation.


Actual run update,23 October: the v1 collection stopped after six sessions and
five successful backend HTTP calls. The cancellation teacher delivered a stale
result without a receipt; replacement control succeeded but the close timer
ended the session before context acknowledgment. All88 raw files are preserved
and independently backed up. No teacher corpus qualifies. See
[NATIVE-REASONING32-TEACHER-FAILURE-2026-10-23.json](NATIVE-REASONING32-TEACHER-FAILURE-2026-10-23.json).
The separately versioned stream_v2 waits a full fixed response window after both
actual eligible completion and context acknowledgment. It is tested but not yet
wired into a recovery collector. Original v1 modules and receipts stay unchanged.
