# Native training storage: keep the working tree small

Observed 9 November 2026, approximately 15:09 UTC. LOCAL remains sole executor; Isambard job **8234444** is submitted and pending resources. Storage work must not cancel, restart or duplicate that job.

## What occupies the Mac

Metadata-only `du` reports 355,126,316 KiB for the repository (363.65 decimal GB), including 354,677,880 KiB under ignored `runs` (363.19 GB). `.git` is only 28,100 KiB. These are filesystem allocation measurements, not a promise of exclusive reclaimable APFS bytes: hard links, clones and File Provider accounting matter.

| Artifact group | Approximate allocated size | Purpose / disposition |
| --- | ---: | --- |
| `runs/cloud-export-20261016` | 108.69 GB | Historical cluster export; includes about 64.54 GB of public model files and 34.82 GB of historical runs. Cold archive; preserve inventories and provenance. |
| `runs/local-migration-20261021` | 68.13 GB | Mostly a 67.82 GB, 36-part native transfer archive. Overlaps the exported material; not all of this is unique useful working data. Cold archive, not a local training dependency. |
| `runs/checkpoints-retained-20261108` | 35.17 GB | Five selected useful historical Venus states, already reduced from the larger obsolete checkpoint collection. Preserve; online-only is preferred. |
| `runs/archives` | 14.40 GB | Verified compressed Documents recovery archive. Preserve online-only; do not recreate the removed Documents copy. |
| Other generated runs, speech, caches and current backups | About 137 GB | Classify by dependency and verified backup before deletion; current matched-training states remain protected. |

The first four groups alone account for approximately **226 GB**. This is a source checkout containing an artifact warehouse; Git itself is small. Some redundancy is deliberate preservation during migrations, but keeping every archive and extracted copy downloaded is unnecessary.

Do not add the apparent sizes of hard-linked paths. A prior metadata scan found 232 large paths under `local-migration-20261023`, but only 11 unique inodes occupying about 6.43 GB, rather than the 125.93 GB sum of path sizes. Removing a hard link need not free the underlying bytes.

The actual `/Users/daryl/Documents` directory was separately measured at about 0.67 GB; macOS's Documents storage category is not the size of that one directory.

## Why OneDrive still uses local storage

OneDrive synchronizes files and supports online-only placeholders. Files created or downloaded here remain locally available unless evicted; moving them beneath OneDrive does not automatically evict them. Opening or hashing an online-only weight downloads its contents again. Use metadata inventories instead of repeated whole-archive hash scans after eviction.

Microsoft's **Free Up Space** retains the cloud copy and removes local contents when the provider can complete eviction. Deleting a synchronized file also deletes its cloud copy; deletion is not a substitute for eviction. See [Microsoft's Mac Files On-Demand guidance](https://support.microsoft.com/en-us/onedrive/save-disk-space-with-onedrive-files-on-demand-for-mac) and [supported Mac pin-state controls](https://learn.microsoft.com/en-gb/sharepoint/files-on-demand-mac).

### Actual eviction result, not a space-saving claim

Finder's official **Free Up Space** action was requested for `venus-baseline-526.pt`. Apple File Provider reported uploaded=true, uploading=false, downloaded=true, keep-downloaded=false, sync-paused=false and no conflict. At 15:09 UTC the file still allocated **8,380,948,480 bytes** and was not dataless. Finder displayed Syncing. Therefore **eviction is not confirmed and no Mac space reclaimed by this action is claimed**. Uploaded status on this file does not establish that every archive is uploaded.

The documented OneDrive `/getpin` query printed `Unpinned`, then its one-off helper exited SIGABRT with a mutex error. This does not establish that the main sync client crashed. No sync-client termination, database editing, unsupported eviction command, folder-wide unpin, or local artifact deletion was performed. Avoid repeated failing helper/UI attempts while training waits or runs.

## Retention policy for this project

1. **Keep local:** source, recipes, manifests, hashes, receipts, evaluation summaries, and small listening samples. These explain how to reconstruct and assess a run.
2. **Active compute:** models, corpus caches and optimizer state stay on Isambard during training. Preserve the exact initial states and successful epoch/window checkpoints; never replay saved updates or remove the only restart state. The existing observer backs up newly committed immutable matched-training states, without downloading the four existing initialization backups again.
3. **Cold cloud archive:** historical exports, transfer archives, recovery archive and selected old checkpoints should be online-only after upload is confirmed. Keep their exact inventory/hash records local. This can remove their local payload without losing Finder access. It does not promise all 226 GB is immediately reclaimable.
4. **No indefinite warm history:** after the matched experiment, retain each arm's initialization, latest resumable state and selected final/best state plus full run records. Intermediate rolling states may be removed only after a newer complete state and its off-machine backup are independently verified. Failed outputs and provenance remain preserved; no broad delete of historical training audio or unique receipts.
5. **Future transfers:** fetch useful new checkpoint/results files, not complete remote working directories or repeated carrier archives. Do not create additional Documents copies. Keep existing absolute references valid; do not move the active repository during a live run.

OneDrive online-only is sufficient for Finder-visible cold storage without buying another service. A separate private object-storage artifact bucket can later provide explicit upload/download, retention and lifecycle control, but has not been provisioned or charged here. Isambard is active working storage, not the sole long-term backup after an allocation ends. GitHub remains the source/recipe record, not a destination for large weight files.

## Completed cleanup distinguished from pending Mac eviction

The cluster deployment initially hit a **project file-count quota**, not GPU memory or training failure. Forty-nine obsolete, clean Git worktrees were removed only after verifying no unrelated/untracked non-cache data and that every source commit is retained under Git anchor `d9e6ac679cce5deb091183c89562c40ed163f73b` and archive ref `refs/archive/reproducible-native-source-20261109`. This removed 112,463 reconstructable files (about 1.31 GB estimated allocation) on Isambard. Checkpoints, corpora and receipts were untouched. Cleanup terminal was actual exit 0; job submission then succeeded. This is separate from the unconfirmed OneDrive eviction on the Mac.

Ignored evidence lives under `runs/isambard-native-20261109/cluster-worktree-quota-v1`, `training-only-approved-launch-v1`, and `storage-check-20261109.json`. Original checkpoint and recovery archive hashes remain in `NATIVE-STORAGE-CLEANUP-2026-11-08.json`; no bulk rehash or new copy was needed for this audit.

## Follow-up execution after explicit owner online-only instruction

Finder's actual Free Up Space action was applied to all seven retained-checkpoint folder items, all four recovery archive items, all 36 migration archive parts/index, and all 21 export top-level items including its tree. No active source/current matched-training state was selected. These are provider eviction requests, not completed-space-saving receipts. Three subsequent File Provider probes still showed uploaded/unpinned/downloaded and Syncing; large checkpoint/archive allocations remain. The supported OneDrive CLI also aborted after a clean app shutdown; the official client was then successfully relaunched. No archive, checkpoint or cloud copy was deleted. Detailed ignored receipts: `runs/isambard-native-20261109/onedrive-online-only-v1`.

To apply online-only across OneDrive: in Finder open the OneDrive root, select its contents, then Control-click and choose **Free Up Space**. Do not choose Always Keep on This Device. Online-only files remain visible and download when opened; it is not a permanent prohibition on local contents. For strictly no local file downloads, use OneDrive in a browser instead of opening files through Finder. Keep working code available locally and let cold artifacts remain online-only; do not pin the entire training repo, which would also download every large run artifact.
