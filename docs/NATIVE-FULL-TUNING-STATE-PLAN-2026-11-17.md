# Native tuning: capacity, precision and checkpoint storage

17 November. Read-only measurements and implementation preparation; no native weights loaded, new GPU allocation, learning, deletion or storage migration for this investigation. Collection8553179 and queued corpus review8555523 remain unchanged.

## What the live measurements establish

The pinned PersonaPlex safetensors header contains475 tensors with8,371,408,896 elements. The `transformer.*` and `text_linear.*` prefixes account for6,708,002,816 elements. These are exact published-tensor counts, **not yet actual loaded-model parameter/ownership proofs**. Selective tuning could therefore expose billions of existing coefficients without adding random layers; the105M adapter is a separate additional parameterization.

The live user HOME quota reports88,139,904KiB used against105,468,748KiB hard limit:16.53GiB headroom at observation. Filesystem-wide free space is not the user's quota. The standard FP32 model plus two FP32 Adam moments forecasts are:

| Candidate state | Forecast |
|---|---:|
| Full published model at FP32 |31.19GiB|
| Full model plus full Adam moments |93.56GiB|
| Full model plus selective-prefix Adam moments |81.17GiB|

These forecasts exclude OPAL registry duplication, rank-specific state, diagnostics, serialization overhead and concurrent archives. They are not measured checkpoint sizes. The current HOME destination cannot admit such new dense checkpoints at its observed headroom. This is a **future dense-training storage requirement**, not evidence that current collection/review has failed. Neither running nor queued work is changed.

Isambard documents HOME as100GiB, project storage as200TiB and scratch as5TiB for Phase2. It provides working storage rather than automatic user-data backups. Separate directories on the same filesystem do not establish independent recovery. Verify actual project/scratch quotas and filesystem identity before choosing destinations. [Official storage documentation](https://docs.isambard.ac.uk/user-documentation/information/system-storage/).

## The implementation consequence

`training/personaplex/mixed_native_setup.py` prepares the three methods from the **same fresh released PersonaPlex foundation**: broad105M LoRA, selective temporal-trunk/text-head tuning and full native-LM tuning. Preserve196 as a fixed reference; never replay its saved updates or casually merge it into just one arm.

The setup requires a separate mixed-phase lease and actual four-GPU process group. It captures/repeats the original native references, configures ownership, then checks the wrapped native BF16 function before any learning. It deliberately does not compare an unwrapped FP32-master forward against the original BF16 compute and call that an initialization defect. Exact post-wrap equivalence and later native gradients/full Adam/RNG/OPAL/cursor restoration remain mandatory and unproved.

Before dense learning, select and verify a destination with enough capacity for complete states and independent recovery. Preserve existing checkpoints. Do not replace Adam with an unverified low-bit optimizer, drop OPAL state, substitute adapter-only recovery, delete old lineages or stream large files onto the Mac as an implicit workaround. Lossless deduplication or remote cloud archival may be evaluated, but recoverability and authorization must be concrete before use.

Proceed meanwhile with whole-corpus review, semantic/family qualification, the existing native loss/schedule integration and full behavioural evaluation. Do not turn this storage discovery into a kernel or filesystem tuning sweep. More trainable coefficients remain an experiment; only reasoning, actual backend delegation, speech and interaction outcomes can select a model.
