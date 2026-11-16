# Expanded native training: matched comparison, 16 November 2026

The adapter shows some improvement on familiar basic questions and strict speech/text agreement, but does not establish reliable quantitative reasoning, native delegation, silence or pause behaviour. Do not promote it as an accepted live model.

Training completed all 196 new updates on job8478956, four complete epochs, with the 105,070,592-coefficient broad rank64 adapter on the frozen PersonaPlex foundation. All four fresh native Adam/RNG/OPAL/fingerprint restores and the independent six-file final checkpoint backup passed. No saved updates were replayed. OPAL remained on the fixed KEEP recipe for this comparison; the staged adaptive continuation was not artificially applied after training had already finished.

Evaluation8511770/source2780df561a8dc9525b030d2bfdbd967cfc6de945 generated all192 outputs: raw base and trained adapter, each on the same32 regression inputs and64 quantitative DEV questions. Same RPBen reference, persona, sampler and native inference path; no teacher answer/audio or forced worker result entered generation. All four speech audits completed. Job8511770 is now provider-COMPLETED, exit0,8621seconds on4GPUs (9.578889GPUh), reservation0. Independent raw-result backup PASS:132,866files/4,566,065,549bytes, every-file SHA compared. The earlier pre-model31second evaluation failure remains counted separately; no training or completed inference was replayed.

| Measure | Base | Adapter | Interpretation |
| --- | ---: | ---: | --- |
| Core answer on12 basic regressions | 8/12 | 9/12 | Small mixed gain; familiar/partly retained cases, not fresh generalization |
| Correct final numeric answer on64 DEV | 1/64 | 3/64 | Two adapter answers have false explanations; a further answer mentions the right number ambiguously and is not counted |
| Correct answer without false/contradictory explanation | 1/64 | 1/64 | No reliable quantitative reasoning gain established |
| Native handoff events across96 inputs | 0 | 0 | Saying “background assistant” is not delegation |
| Silence before cue | 0/4 | 0/4 | Unsolved |
| Silence during requested pause | 0/4 | 0/4 | Unsolved |
| Strict generated-text/ASR agreement | 11/96 | 18/96 | Some numerical gain; strong scoring caveat below |
| Raw peak threshold pass | 94/96 | 93/96 | No gain on this check |

The semantic review is a local assistant review of all64 paired generated-text and ASR answers against the original references, not an external judge call. Requiring no false explanation does not require revealing reasoning steps: a correct concise answer passes. In the adapter's weight-gain example it says20% of120 is48 and a quarter is12, then gives the right150 by an inconsistent calculation. In the potato example it gives the right3hours but incorrectly says3potatoes take1hour. The worksheet preserves every case and verdict in `NATIVE-EXPANDED1072-FULL192-SEMANTIC-REVIEW-2026-11-16.json`.

Basic gains include five-minus-two, minutes-per-hour and the purpose of window glass. Regressions include answering “win” to the opposite of “loose”; its ruler answer no longer clearly identifies what is measured. Do not reduce this to a single improving average.

## Speech scoring limits

The strict agreement test is not a naturalness or accent score. ASR adds a terminal “thank you” absent from generated text in65base/62adapter outputs. Removing only that phrase hypothetically changes agreement passes to45/96 and49/96. Those are sensitivity figures, not replacement official scores: the phrase could be a recognition hallucination on the long silent tail or genuine speech/text divergence. Listening or a controlled ASR check is needed to distinguish them. Preserve both original transcripts and raw audio; do not silently remove words to improve scores. Naturalness and British accent remain uncertified.

## Delegation and coverage limits

There were no native RET handoff events and no actual worker events. No real backend was connected, so operational cancellation, asynchronous completion, grounding and live microphone latency were not tested. Some adapter responses promise or invent background results without a handoff. Do not count those as successful delegation.

The64 DEV questions are quantitative word problems, not broad technical/domain reasoning. They were used in training evaluation/model selection and are not a fresh unseen final test. The12 basic regressions include inspected/retained examples. The turtle question's “twice more” wording is ambiguous; both answers fail either plausible interpretation. The population question incorrectly calls New England a state; its numeric hypothetical is still defined. These reference flags are preserved, not removed retrospectively.

## Next substantial step

1. Verify parity between the learned native handoff/action representation and the inference path. Zero events could reflect a representation, supervision or runtime problem; do not assume more parameters will repair it.
2. Compare matched audio questions with correct text conditioning on the same foundation to separate perception/number capture from reasoning limits. Include simple basics and progressively harder quantitative questions, with new held-out examples.
3. Use those results to choose the next data/recipe/foundation experiment. Broaden basic/domain reasoning while retaining speech/control coverage; never train on these exact DEV questions. Use failure families in the MiRL-style plan, rather than memorising evaluation items. Test capacity or fuller tuning only when the bottleneck warrants it.
4. Keep the staged adaptive OPAL controller for a genuinely new safe continuation with actual restored states and independent boundary evidence. Keep RL in the wider plan, tied to verified correctness and actual handoff/control rewards; promises of delegation and ASR similarity are insufficient reward signals.

We can discuss these results now. The combined recorded A/B previews are immediately usable. A live interactive diagnostic is a separate serving test; this report does not claim that the new adapter is deployed or accepted. No further training is needed merely to listen to the saved outputs.

## Preservation

Checkpoint196 SHA: `96cff747f42fff1c36572800debcee2d435f20efaebf2c8ccfe9720e9bec3c87`.
Independent checkpoint receipt SHA: `d9791ff61ee209516c5223bdd05cc3287146fbc2e4317e8069485325db3e6420`.
Full original evaluation: `/projects/u6ada/duplex-live/native-expanded1072-evaluation-repair-v2`.
Independent result destination: `/home/u6ada/dsyj94.u6ada/duplex-expanded1072-evaluation-backups-v1/job-8511770`.
Only small JSON reports and listening previews were copied to the Mac; full checkpoints/raw archives remain on cluster filesystems.

Final archive inventory SHA: `5c817a61bec7eb88cc2da4c22b1253757b60506241f8bafebb4257cfecaa19c5`.
