# Cloud continuation — 18 October 2026

Plain words: This is the entry point for continuing from a cloud chat after the
owner shuts down the Mac. At 18 October 23:29 Berlin / 21:29 UTC, the owner
explicitly stopped local work for cloud takeover. This local goal is PAUSED,
not achieved; the cloud chat must establish its own working continuation. The native model
and whole system remain unqualified. Start here, then read the newest entries in
HANDOFF.md and docs/QUALIFICATION-2026-10-18.md. Older handoff paragraphs describe
historical states, not current instructions to resubmit old jobs.

Plain words: The intended product is one native full-duplex voice model that can
answer ordinary questions sensibly, listen while speaking, deliberately stay
quiet, obey stop/pause/resume cues and delegate harder work asynchronously. The
native starting point is Context Spanning 7B, in the PersonaPlex/Moshi family.
This is fine-tuning pretrained weights, not scratch pretraining. Qwen3.5-9B is a
separate experimental controller/fallback; do not restart that campaign or call
its hybrid demo the native final product. The owner accepted the Context Spanning
voice and the repaired hybrid demo's smooth playback, not overall model behavior.

## Access and shutdown boundary

Plain words: A new cloud workspace does NOT inherit the Mac's SSH keys, Clifton
certificate, SSH aliases, local processes, ignored runs directory or active goal.
It can obtain Git-tracked code and this handoff from DSYJ94/duplex-live, main.
Cloud operation still needs a supported outbound SSH route and separately
configured Isambard access. Establish those before claiming autonomous cluster
continuation. Do not copy private keys into Git or chat. Use approved environment
credential facilities, and a fresh Clifton authentication if required. Renewals
may still require the owner to complete institutional sign-in from another device.

Plain words: On the Mac the working command is ssh u6si.aip2.isambard. The alias
and ProxyJump configuration are machine-specific; do not invent a public hostname
from the alias. Official institutional agent guidance is at
https://docs.isambard.ac.uk/user-documentation/guides/using_ai_agents/ . The existing
certificate was renewed until 19 October 09:39 Berlin / 07:39 UTC, but does not
transfer automatically. An HTTPS domain allowlist alone does not prove SSH works.

Plain words: Submitted Slurm jobs survive local disconnection/shutdown. This
local chat and localhost:8766/8767 services do not keep running on a powered-off
Mac. The last native data pipeline is already terminal and failed its gate; do
not tell the owner training is running. No cloud GPU validation or extra cloud
compute purchase is requested. The previous $25 validation allowance remains
unused and cloud GPU validation remains paused.

## Authoritative current native work

Plain words: Source 1b3ae989e95f70a053bee777509910d68aad483e is committed/pushed and
has an immutable Isambard worktree. Source 7771e82188fbce799e18be4010aff8248e6304de
introduced the sealed new corpus. It has 40 independently authored scenes,
24 TRAIN / 8 validation / 8 test, 188 utterances and 6,973 characters. It includes
varied ordinary questions, four quiet-cue scenes across folds, one TRAIN
pause/continue scene, clarification and cancellation followed by a new question.
It is not captured model output, owner audio or evaluation-derived labels.

Plain words: Original generation 7531436 COMPLETED. Independent audio audit
7531437 FAILED: only 19/24 TRAIN scenes passed; validation also lacked a verified
behavior. Failed TRAIN clips were train-002 index4, train-003 index0, train-006
index3, train-015 index3, train-017 index2. Five exact-word TRAIN-only re-renders
ran in 7531496, COMPLETED. Held-out audio hashes were required unchanged. Re-audit
7531499 FAILED because the whole 24-scene TRAIN fold still did not pass.
Dependent teacher 7531501, teacher ASR 7531502 and semantic/conversion 7531503
were CANCELLED. No new teacher corpus, codec cache, optimizer updates or model
qualification came from this pipeline. Inspect the remaining TRAIN failures
before proposing another bounded repair; never silently take a passing subset.
The re-audit admits 22/24 TRAIN scenes. Remaining failures: native2-train-015
clip3 (Go. transcribed Go. Go.) and native2-train-017 (mixed waveform clipping;
all its word checks now pass). Inspect the actual clipped source utterance before
repairing it; do not merely lower the saved gain to conceal clipping.

Plain words: Cluster root is /projects/u6si/duplex-live, repo under repo/, runs
under runs/, immutable sources under worktrees/recovery-SHA7. Inspect:

- runs/native-curriculum-v2-7531436/{specifications.json,filter_report.jsonl,quality.jsonl}
- runs/native-curriculum-v2-data-7531437/ingestion.json
- runs/native-v2-source-repair-7531496/{generation.json,filter_report.jsonl,quality.jsonl}
- runs/native-curriculum-v2-data-7531499/{ingestion.json,train.jsonl}
- runs/native-curriculum-v2-launch.json
- runs/native-v2-teacher-pipeline-launch.json
- repo/logs/dl-native-source-v2-audit-7531499.out

Plain words: Local equivalents of launch receipts are under ignored
runs/recovery-continuation-20261016/. They are not supplied by a fresh Git clone.
The raw cluster directories above are the independent source of truth.

## What the completed native pilot established

Plain words: Actual assistant-only joint text/audio pilot 7530408 completed eight
updates after independent target/numerical audit 7530407. Exact independent
adapter/optimizer/RNG/logit restore passed; peak GPU allocation was 39.42GB.
It supervises assistant text and eight assistant audio heads, with ZERO user
stream prediction loss. That ownership mask is NOT waveform echo cancellation.

Plain words: REJECT the pilot for adoption. Ordinary acceptable answers fell to
1/4; private-result, clarification and cancellation behavior did not qualify.
It omitted the earlier actual-history and KL/hidden retention objectives, so it
is NOT an isolated acoustic-loss ablation. Do not extend this unchanged recipe
or attribute all regressions to audio loss. Parent native 7510331 retains the
four ordinary answers but still fails private/control tests.

Plain words: Context-format and voice controls are complete. The same-speaker
fridge excerpt improves some private values but does not solve clarification or
cancellation. The user's questions also mention lightning; voice-content copying
is not proved. Close the voice sweep. Released-voice clip 7531044 has INVALID ASR
word timestamps and must not count as a spoken pass. All twelve new voice-control
WAV hashes and strict final native loads were verified locally. No new voice,
checkpoint or owner demo was adopted. The latest Qwen1599 run 7530021 is also
REJECTED; independent role holdout 7530022 stays 20/24 before and after.

## Next training process

Plain words: Repair only the remaining failed TRAIN source clips in another
versioned directory, preserving exact words, failed originals, API receipts and
held-out audio. Keep existing speech, timing and data thresholds. The explicit
new native-v2 source schema approves only a COMPLETE 24-scene TRAIN fold; failed
held-out clips stay excluded and do not make a full qualification set. Old corpus
rules and default behavior remain unchanged.

Plain words: The prepared teacher chain is: recovery_teacher --scope native-v2
--protocol native-dialogue-v2; independent recovery_teacher_quality; separate
native_v2_semantic_review using actual gpt-6-astra requests; then
recovery_teacher_convert --semantic-approval. All 24 teachers must pass measured
speech, actual record/cancel receipts, quiet intervals, post-cancellation new
answers and independent semantic review. Conversion cannot accept a passing
subset. Twenty focused/regression CPU tests passed. The chain is implemented,
but has NOT executed successfully. Native codec/grid/input/numerical approval is
still required after conversion; do not confuse semantic review with final
training eligibility or product qualification.

Plain words: Reuse the existing native pipeline and retain the 28 old actual
student-history components, four-reference KL/hidden retention and 25 approved
native teacher histories. Add approved new native teacher conversations through
a new complete scope, streamed-user input approval and numerical preflight.
Do not weaken the old 25-scene pilot guards. Existing generic loaders support
expanded inventories; old Slurm wrappers hard-code cache paths and need explicit
immutable new wrappers. The next controlled native fit should preserve retention
rather than repeat the standalone joint pilot's confounding omission. Require an
independent full training-state restore and free-running ordinary/private/
clarification/cancellation/quiet/pause-resume checks before any extension/adoption.

## Operational rules and artifacts

Plain words: Code reaches Isambard only through Git commits/push/fetch and clean
immutable worktrees. No source scp/rsync. Administrative SSH Python is only for
Git/Slurm/status/artifact metadata. Artifact rsync must use -ar and
--exclude=session-token. Never export owner conversations, session credentials or
runs/private-core-quarantine-20261018/core-344dded. No coauthor attribution.
No captured evaluation or held-out model outputs may become TRAIN targets.

Plain words: Full models/data are NOT in GitHub. Cluster weights, pinned SIF,
environments and checkpoints remain under /projects/u6si/duplex-live. A substantial
verified local export is under runs/cloud-export-20261016/tree/runs on the Mac;
it does not appear automatically in cloud. It includes native joint audit/train,
first task evaluation, matched context arms and completed voice controls. New
source-generation/repair runs may still need copying. The cloud successor should
preserve them while access exists. Local source bundle/evidence manifests need
refresh after this handoff; old bundles are valid historical backups, not latest.

Plain words: Login Python is 3.6 (use universal_newlines=True, no shlex.join).
Cluster runtime is pinned pytorch-new.sif with Python3.12 and existing environments.
Local .venv-api has NumPy/audio tools but no torch/pytest. Preserve unrelated
roadmap.html, evals/capability-retention-probe.slurm,
evals/capability_retention_probe.py and the pre-existing tracked pycache change.
Transient SSH banner timeouts are observation failures, not permission expiry or
job failure; re-read the SAME job before submitting anything. Read goal state in
the new chat; do not claim the original goal migrated automatically.


## Final local stop — 18 October 23:29 Berlin

Plain words: The owner explicitly stopped this chat for cloud takeover. No new
repair, teacher or training job was submitted after the failed 7531499 chain.
Current committed implementation remains 1b3ae98; the later commits are handoff
documentation. The last brief continuation only fetched source artifacts and
started read-only waveform inspection. It did not establish a new diagnosis or
fix. Do not interpret the hypothesis commentary as a result. Complete diagnosis
of the short Go clip and the precise clipped utterance before any new repair.

Plain words: Focused copies of the failed TRAIN015/017 utterances and the source
quality report were successfully fetched to the Mac's ignored local export.
The broader four-directory artifact transfer was also started; verify its
completion/file inventory before calling that new backup complete. These local
files are not available through GitHub. Remote originals and failed audit
reports remain the authoritative artifacts for the cloud agent.

Plain words: Preserve unrelated local-only roadmap.html,
evals/capability-retention-probe.slurm, evals/capability_retention_probe.py and the
tracked timeline pycache modification. They were not authored in this handoff
and are intentionally not included in its GitHub commits. There are no
uncommitted implementation edits from the stopped continuation.
