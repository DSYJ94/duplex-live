# Local native continuation — 19 October 2026

Local is the active executor; cloud must not submit until explicit owner takeover.
The 21 October allocation deadline remains. Native/whole-system verdict FAIL.

## Access and preservation

Official local Clifton sign-in renewed the local certificate to 22:51:23 Berlin
19 October. Actual bounded SSH succeeded at08:51:48UTC; account squeue empty.
7531496 COMPLETED; 7531499 FAILED(1:0); 7531501/02/03 CANCELLED. No duplicate jobs.
Source baseline4ad4367, ownership handoff156b9cf pushed to main.

Completed the stopped four-directory source backup and compared all scoped
remote/local files by size and SHA256: 1,450 files / 661,640,011 bytes, all match.
Scope also includes native7510331 checkpoint/optimizer/RNG state, rejected
7530408 checkpoint/state, 28 old actual-history components, scope7508779,
streamed native approval7508795 and numerical7509062 plus launch/audit receipts.
This is a scoped verified backup, not a claim that every cluster asset is exported.
Ignored Mac evidence: runs/local-continuation-20261019/{remote-critical-inventory.json,
backup-verification.json,source-diagnosis.json,failed-source-waveforms.png}.
Portable artifacts: runs/cloud-export-20261016/tree/. Credentials, owner sessions
and quarantined core are excluded. Existing full model/runtime export remains
separate; approved teacher source/cache dependency coverage is being refreshed.

## Reproduced source failures and bounded repair

Command: PYTHONDONTWRITEBYTECODE=1 .venv-api/bin/python -m
 evals.native_v2_source_repair_v2 --source
 runs/cloud-export-20261016/tree/runs/native-v2-source-repair-7531496
Expected exit1 reproduces the actual failed sources, with exact audited hashes.

TRAIN015 index3 Go. is3.05seconds and contains two distinct acoustic bursts.
Independent audit measures Go. Go. at0.10–0.42 and1.48–1.80seconds; raw RMS in
those intervals is0.123345 and0.146301. This is actual repeated acoustic content,
not just an unsupported ASR repetition hypothesis.

TRAIN017 index0 What is the parcel label? is7.4seconds with50 samples >=0.999
at3.541833–3.741125seconds. The entire raw first utterance equals both the user
stem and mixed waveform at the declared800ms offset. Clipping originates in
raw generated audio; it is not introduced by mixing/overlap. Repaired index2 is
clean. Additional acoustic bursts after the independently aligned question remain
visible; rerender the diagnosed raw source rather than lowering gain/cropping.

Repair v2 renders only TRAIN015/index3 and TRAIN017/index0, exact sealed words
and same voices. General instruction asks for one reading and silence after the
last word. Each gets one successful generation, never a local ASR reroll loop.
Original failed WAV/receipt are copied under repair-originals/; originals remain
in prior directory. Every other utterance and held-out audio/receipt stays hashed
unchanged. Rebuild only the two TRAIN scenes and their manifest durations.
No thresholds change. All24 TRAIN scenes must pass independent full re-audit.

22 focused tests pass, including original/held-out preservation, fail-closed
held-out corruption and destination overwrite rejection; Slurm shell syntax and
diff whitespace checks pass. No model qualification follows from these tests.
Continue teacher/independent quality/semantic/conversion/native numerical chain
only after the complete source gate. Retain old histories, four-reference KL and
hidden retention and25 approved native teacher histories. Do not extend rejected
7530408 unchanged. Independent state restore and full native behavioral evaluation
remain mandatory before extension or adoption.
