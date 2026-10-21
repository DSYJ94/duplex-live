# Independent local ASR of the preserved native cue diagnostics

The complete audit finished on21 October2026. Whisper turbo independently heard all three synthetic inputs exactly as authored (normalized WER0.0 each). It detected words in the nominal quiet interval of ALL9 native output recordings. This corroborates the existing failed silence-control diagnosis; it does not establish native understanding or qualify a model.

| Input | Word error rate | Original peak |
| --- | ---: | ---: |
| input-ready | 0.00 | 0.740570 |
| input-geronimo | 0.00 | 0.928864 |
| input-cue | 0.00 | 0.910553 |

| Prompt variant | Case | ASR words overlapping nominal quiet | Original peak |
| --- | --- | ---: | ---: |
| original | output-00-ready-no-cue | 20 | 1.198475 |
| original | output-01-geronimo-no-cue | 12 | 0.494229 |
| original | output-02-geronimo-with-cue | 25 | 0.560643 |
| scoped_project_clause | output-03-ready-no-cue | 15 | 0.597006 |
| scoped_project_clause | output-04-geronimo-no-cue | 16 | 0.692735 |
| scoped_project_clause | output-05-geronimo-with-cue | 23 | 0.645247 |
| generic_quiet_rule | output-06-ready-no-cue | 17 | 0.698705 |
| generic_quiet_rule | output-07-geronimo-no-cue | 20 | 0.929121 |
| generic_quiet_rule | output-08-geronimo-with-cue | 18 | 0.635857 |

One output peaks at1.198475, exceeding the unchanged0.995 ceiling. No gain reduction, clipping concealment, resampling of originals, word replacement, subset acceptance or new inference was performed. All original input/output hashes were checked before and after ASR.

Input fragments and the full paced input waveforms were also verified as the exact original fragment/silence assemblies. The three inputs plus nine outputs were all retained. ASR was unprompted, with fixed official turbo weight SHA aff26ae408abcba5fbf8813c21e62b0941638c5f6eebfb145be0c9839262a19a, beam5/best5/temperature0, VAD disabled and CPU float32/4threads. Native audio timestamps are concatenated output-frame time, not wall-clock playout or latency.

Output windows may truncate speech, and a recognizer may introduce errors or repetition. Their disagreement scores are diagnostics, not full-answer WER qualification. No fullnative10, fresh40, numerical approval, new teacher collection or training occurred. The original nine failures remain frozen. The ordinary-reasoning and asynchronous-delegation goals remain unqualified.

The isolated Mac environment uses Python3.12.13, Torch2.9.0, NumPy2.2.6, Whisper20250625, whisper-timestamped1.15.9 and SciPy1.18.0. SciPy1.15.3 actually failed to import its PROPACK library on this Mac; the replacement imports and actual12-file ASR completed. The maintainer describes the PROPACK C port in [SciPy issue25635](https://github.com/scipy/scipy/issues/25635). No native training/runtime package was changed.

Source:1356b2c396569597b6716a230d849eaf7ea06768. Actual process78533 / exec session96569 completed with exit0. No API key, new cloud GPU or owner recording was used. The public1,617,941,637-byte ASR checkpoint is cached locally with its official hash; it was not copied to Runpod.

All19 ASR JSON/text metadata files (213,099B) were copied to existing volume49yytzyoge under migration/local-cue-asr-20261021-v1/ through the approved S3 backup route. Independent GET/read-back hashes matched every source file. No new GPU was used for backup. Original synthetic WAVs remain in their previously verified backup. Full native archive preservation remains passed; whole-project/Mac source backup remains partial.

- [Complete independent ASR evidence](NATIVE-CUE-INDEPENDENT-ASR-2026-10-21.json)
- [Summary](NATIVE-CUE-INDEPENDENT-ASR-SUMMARY-2026-10-21.json)
- [Local runtime and weight inventory](NATIVE-LOCAL-ASR-RUNTIME-2026-10-21.json)
- [Independent Runpod persistence verification](NATIVE-CUE-ASR-PERSISTENCE-2026-10-21.json)

Next empirical action requires the still-blank local API credential: generate the complete new40-scene TRAIN source, independently audit all168 clips with this recognizer, then collect the actual40 raw teachers. Whole-teacher ASR/semantic/native conversion and numerical approvals remain to be completed on real artifacts. Preserve all28 actual histories,49 approved teachers, KL1/hidden10 and154 original objectives. Complete retained baseline/candidate native10+speech/ordinary4 before choosing a bounded repair parent, then require all40 sealed paired fresh outputs before adoption. No unchanged extension or frontend keyword mute is a repair.
