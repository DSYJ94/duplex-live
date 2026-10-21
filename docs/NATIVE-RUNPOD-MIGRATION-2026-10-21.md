# Current execution — 21 October12:00UTC

The initial approved trial has ended; no trial GPU remains. Read
[NATIVE-RUNPOD-EXECUTION-2026-10-21.json](NATIVE-RUNPOD-EXECUTION-2026-10-21.json)
and the newest HANDOFF.md for exact current resources, restoration and backup status.
The full archive is verified on Mac but not yet complete on Runpod. The initial
36-file model/state export and priority proof were verified on Runpod. Independent
adapter/Adam/allRNG restoration passed; full native numerical/behavioral approval
and a usable voice audition remain incomplete. No new training occurred.

Replacement100GB volume49yytzyoge in S3-supported EUR-IS-1 has actual authenticated
storage access. The owner approved temporary overlap with oldvolume cyjzip53ap,
which must retain its unique receipts until recovered. $7/month each while both
exist, $7/month ongoing after retirement. Paid compute remains within the original
$25 budget; no replacement GPU has been started. The following earlier assessment
is historical; its no-resource/no-approval/pending-copy statements describe06:43UTC.

---

# AIRR grace-period backup and Runpod migration

LOCAL remains the sole executor. CLOUD is read-only until the owner explicitly
hands execution back. No new Slurm jobs or paid cloud resources were created.
The finished allocation does not require repeating the completed training run.

## Access and preservation

The owner's AIRR award screenshot says the award ended 21 October 2026, data is
accessible through **19 November**, and access ends **20 November**, followed by
automatic deletion. If more copying time is needed, contact the allocator before
10 November with evidence that copying has started; requests after 15 November
are rejected. These dates come from the screenshot, not an inferred SSH expiry.

Actual alias SSH succeeded at 21Oct06:34:26UTC; `squeue --all -u "$USER"` was empty.
The current LOCAL certificate is valid 21Oct06:27:59–18:27:59UTC. No authentication
material was copied. Project data remains readable. Read-only `du` reported runs
159G, models 243G, vendor 1.6G and the ARM runtime SIF 8.0G. Those totals include
material outside the native migration scope and do not imply it was all copied.

The expanded explicit inventory is:

`runs/local-migration-20261021/native-migration-v3.json`

It covers **28,470 regular files / 67,764,285,650 bytes**, plus eight preserved ARM
runtime symlinks. SHA256:
`4eac15590dc9def6dfb5e34b2cea92f30d9a067fabe0940e2876b6acdcbe6dce`.
It combines the 25,275-file preservation-ledger union, the original export's
historical checksums, the pinned Context Spanning/PersonaPlex/codec/tokenizer
assets, original synthetic histories, vendor sources and three vendor Git bundles.
All sizes, hashes and symlink targets passed a fresh local read-back verification.
The local artifact root is `runs/cloud-export-20261016/tree`.

The remote root census additionally found 126 earlier native training-proof roots
outside that selection. All **839 files / 1,494,007,624 bytes** were inventoried on
Isambard and verified locally; 242 files required transfer, while existing local
bytes satisfied the rest. Every remote `context-training-proof-*` directory from
the census is represented in the expanded inventory. This is metadata/integrity
coverage, not approval of the historical models. Evidence lives in:

- `runs/local-migration-20261021/additional-native-proofs-inventory.json`
- `runs/local-migration-20261021/additional-native-proofs-verification.json`
- `runs/local-migration-20261021/native-migration-local-verification-v3-final.json`
- `runs/local-migration-20261021/access-and-source-receipt.json`

The original stopped whole-Mac source backup remains **PARTIAL**. This expanded
native archive is not a whole-project backup: unrelated model caches, earlier
research families and non-native environments remain outside it. No owner
conversations, credentials, session tokens or quarantined core dump are included.
OneDrive placement does not independently prove an uploaded second copy. A second
verified storage destination and the actual Runpod transfer remain pending.

Full main history through `aaeebca2a9db770727634244b2aa618f2ce0ffa9` was bundled
and verified at `runs/local-migration-20261021/source-main-aaeebca.bundle`:
3,257,740 bytes, SHA256
`3a93e131cf42ea6d7fdc1f4c2a6628daca688d1fb647560b937aa2c5e6ee347d`.
Fetch newer migration source/docs from GitHub. Recreate immutable checkouts using
Git, not copied worktree `.git` pointer files.

Independent ASR weights are separately preserved under
`runs/local-continuation-20261019/asr-weights`; their new receipt is
`runs/local-migration-20261021/local-asr-weights-fresh-verification.json`.
They add 4,705,313,252 bytes if transferred. They are outside the 67.76GB total.

## Runpod viability and concrete first validation shape

Runpod is technically viable for this native workload, subject to destination GPU
validation. Use a Secure Cloud **Pod** with a persistent **network volume**. Keep
the local verified archive. The live catalog read on 21Oct around06:35–06:40UTC,
for POD / SECURE / minimum CUDA12.8, returned:

| GPU | VRAM | USD per GPU-hour | Stock for one GPU |
| --- | ---: | ---: | --- |
| H100 SXM | 80 GB | $3.99 | HIGH overall; LOW in individual listed centers |
| A100 SXM | 80 GB | $1.79 | MEDIUM overall |
| L40S | 48 GB | $1.09 | MEDIUM overall |

The separate **two-H100** read also returned HIGH overall and LOW in EUR-IS-3.
That center also appeared in the Europe STANDARD-network-volume catalog. Stock
is an observation, not a reservation. Recheck count, center, driver floor and price
immediately before creating resources. Raw reads are retained locally in
`runs/local-migration-20261021/runpod-live-catalog.json`.

Recommend H100 first because its Hopper GPU architecture is closest to Isambard's
GH200. The latest single-GPU training window peaked at22,071,380,480 allocated
bytes; that measured window is not a capacity guarantee for all evaluation cases.
The existing complete native evaluation requires up to two GPUs and requested160G
host RAM. Confirm the actual host RAM allocation before deployment. Cheaper cards
are options after measuring correctness and throughput on the new runtime.

Concrete initial proposal: **one two-H100 Pod for at most two hours**
($7.98/hour; **$15.96 GPU compute**), **100GB STANDARD network volume**
(**$7/month** while retained), and sufficient ephemeral container disk for the
pinned runtime (50GB is $5/month at the published container-disk rate, prorated
while billed). Setup, transfer and validation all consume Pod time. A **$25 initial
validation budget** is a proposed ceiling, not a spend authorization or a platform
enforced cap. Establish and verify deadline/stop monitoring before launch; do not
rely on removed `--stop-after`/`--terminate-after` flags. Reassess volume size if the
archive is expanded further. No paid deployment or transfer destination exists yet.

Prices agree with [Runpod pricing](https://www.runpod.io/pricing).
[Network volumes](https://docs.runpod.io/storage/network-volumes) persist beyond
Pod termination, bill independently, and constrain placement to their data center.

## Port and approval order

1. Preserve the candidate, then restore the explicit files to a volume. Deploy
   source with Git fetch/checkout or verified Git bundles. Mount the artifact tree
   at its provenance path `/projects/u6si/duplex-live` inside the container.
2. Build native x86 wheels with the pinned NGC25.10 Torch stack using
   `portable/context/Dockerfile`. The recipe already had an amd64 build/import and
   historical CPU-state proof; it does **not** establish destination GPU restore
   for the current candidate. Never execute the copied ARM venv/SIF as an x86
   runtime. Verify current image digest, architecture and library versions.
3. On immutable payload source
   `741872fe2f9e49f35d4deba9aba96caff26c51d4`, independently restore candidate
   **7590052**, its saved optimizer/RNG, complete original28, acoustic49 and
   recorded-native49 references. Preserve BF16, original TF32, MATH attention,
   native prefix/span/call convention and every existing loss/approval. Cross-host
   bitwise GPU identity is unproven; discrepancies must be diagnosed with the
   existing strict gates rather than silently changing tolerance or recipe.
4. Port the full native10 execution to actual Pod/process/device identities in a
   NEW versioned execution receipt. Do not fabricate Slurm job/step IDs. Explicitly
   record evaluation source (controller/backend fixes are in `d08ef7a...`), runtime,
   model hashes, all ten attempts, paired baseline and independent speech results.
   The old portable `train.sh` lacks the current stream49/full154 orchestration;
   do not launch it unchanged as a continuation of this candidate.
5. Require ordinary4/4 and the complete native10/audio review before any extension.
   Only the approved chain can unlock all40 baseline/candidate fresh comparisons
   on the sealed20. Native behavior remains UNTESTED and candidate not adopted.

TRAIN7590052 already completed eight updates and all49 supplemental teacher
exposures, with all original154 components at each update. Adam520→528 at2e-7.
Independent Isambard fresh-process original28/acoustic49/native49 fingerprints,
optimizer and all saved RNG states matched exactly. Preserve:

- Checkpoint: `7bbb2e9f1822cb43db50f6ead7c14d510e7c280d520cc7e1436dced6edbcfeb5`
- Training state: `6a9fc6cb0949d1562f654af74cb6e86a827a0214a486f869e7bd92b1ae0b63b4`
- Reload report: `88056cfa1129a795f52d974d1bbb25fa4984d670829c44b30dcb0a9fe84e9f0b`

Readiness7589157 and training7590052 completed; watcher7589159 failed because its
native10 submission was rejected by AssocGrpCPUMinutesLimit. No native10 job was
accepted. The canonical overnight receipt is TERMINAL/FROZEN. Never restart that
watcher, clear its attempts, repeat the completed training or extend rejected
7530408 unchanged. There is no Qwen fallback in this migration.

## Transfer and verification commands

The manifest and its `native-migration-v3.files-from.txt` enumerate only approved
artifact paths relative to the local export tree. After an authorized destination
exists, use the existing public SSH registration process; never transfer the local
private key or Clifton certificate. The data-copy shape is:

```sh
rsync -ar --exclude=session-token --exclude=owner-conversations \
  --exclude=keys --exclude=credentials --exclude='core-*' \
  --files-from=runs/local-migration-20261021/native-migration-v3.files-from.txt \
  -e 'ssh -p DESTINATION_PORT' \
  runs/cloud-export-20261016/tree/ \
  root@DESTINATION_HOST:/workspace/duplex-live/
```

Copy the manifest itself separately as an explicit artifact. Transfer the two ASR
weight files only if needed, with their saved hashes. Do not rsync a whole home
directory or whole remote project. Existing worktree file snapshots in the archive
are historical evidence; execute recreated Git checkouts instead.

From a committed source checkout on the destination, verify without `--build`:

```sh
python -m evals.native_migration_inventory \
  --manifest /workspace/migration/native-migration-v3.json \
  --tree /workspace/duplex-live \
  --report /workspace/migration/destination-verification-v1.json
```

The verifier rejects missing/changed regular bytes, excluded paths and parent
symlink escapes, and checks the eight archived symlink targets without executing
them. It never submits jobs, transfers data or grants model approval. Historical
manifests/receipts remain immutable. New inventory/report filenames are required.
