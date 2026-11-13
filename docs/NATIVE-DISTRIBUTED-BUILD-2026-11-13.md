# Distributed native training build: first executable slice

13 November2026. Actual code/test producer7aea19aa73fdbf463f908ad7fcf2b3b54ea91b31. No new native model update, GPU job, external teacher call or repeated training. Existing196 states and full192 comparison/backup remain unchanged.

## Implemented

- `distributed_native.py`: full native-LM FP32 master ownership with BF16 FSDP computation; broad-LoRA DDP control; optimizer construction after wrapping; sharding-aware clipping; full model/Adam collection and loading for the full arm; exact adapter-only CPU state for the control, avoiding duplicate frozen7B backups.
- Native custom text/acoustic forwards pass through the distributed module's actual forward hooks. The old Objective can use a native view rather than calling an unwrapped sharded submodule. Original depth-context, streaming-state and activation-checkpoint interfaces remain available. No loss coefficients or old training files were modified.
- `distributed_stream_plan.py`: aligns unequal recorded-call counts across ranks, preserves every actual call boundary, and declares padding with ZERO new targets. All ranks enter the same forward/backward collective slots. This is an execution plan, not an integrated native streaming runner yet.
- `native_curriculum_intake.py`: versioned source intake preserving exact words/folds/provenance, rejecting duplicates/subsets, flagging16 shared arithmetic templates betweenTRAIN/DEV, and leaving routing labels/native-audio approval unset. It reports coverage gaps explicitly.

## Actual verification

10 tests passed in the real Isambard ARM/container environment. A real two-process Gloo CPU proof accumulated text/audio gradients and matched a single-process global-mean reference through two toy updates. Final rank parameters were bit-identical. Adapter save/reload and complete membership rejection passed. These were tiny CPU fixture updates, not updates to any native voice checkpoint. First two-rank fixture failed because its plain temporal Linear was not a realistic nested adapter target; the fixture was corrected and the real proof passed. Full CUDA/FSDP/native numerical admission is still outstanding.

PyTorch's documented custom-submodule forward and sharded clipping constraints motivated the wrapper: [FSDP reference](https://docs.pytorch.org/docs/2.15/fsdp.html). Source API and installed runtime are not assumed identical; actual container CPU checks establish only the exercised paths. No claim that these tests qualify the full native model.

## Source data recovered and preserved

The historical local1008TRAIN/64DEV curriculum file was missing. Re-downloaded ONLY pinned public GSM8K TRAIN and its MIT licence, rebuilt the existing pool and curriculum with the preserved code, and recovered exact historical hashes:

- Public pool7473rows:201840725b7dca1d76348d669cbaca1fe34083e41dd5aa4edf23d76284894b8e.
- Curriculum:882ddfa593a7caa5abca9140cd263cde767198d40f439d945fe75cf65b81ffa2.

Both were transferred and independently re-read bySHA on Isambard under `/projects/u6ada/duplex-live/expanded-native-source-intake-20261113-v1`. Licence/raw source/manifests remain alongside them. No officialTEST download, owner data, paid model call, new source audio or replacement of an old artifact. This is maths text preparation, not new qualified native dialogue. It does not fill the zero-count concepts/technical-dialogue/routing/control/worker/British-audio gaps.

## Next executable work

1. Integrate collective stream slots into the retained Objective loop, including zero-loss backward participation without creating prefix/span targets; verify weighting against the existing single-rank schedule.
2. Complete all-rank RNG/OPAL/recipe/data-bound distributed checkpoint state and fresh-process native restore. Current module state helpers cover model/Adam, not the whole approved native training-state contract.
3. Build the expanded dialogue/teacher plan with direct/clarify/delegate contrasts and actual asynchronous task lifecycles; preserve all legacy history/KL/hidden/fit15/folds. Qualify native source/teacher/timing/audio before new targets enter training.
4. Admit one representative four-GPU native full-LM execution/state/throughput path, then run substantial matched adapter/full tuning with complete epochs and evaluation. Growth and CPT remain planned parallel branches. Do not silently extend the old112hour/max12 experiment or mistake CPU fixture evidence for GPU training admission. Prepare one coherent new phase resource/data/recipe package, not repeated per-command permissions.

Remora monitoring owner reported8385685 completed400updates/1440seconds/oneGPU, plus earlier29GPU-seconds; no future reservation. Current direct sharedqueue check during this build was empty. Its public synthetic task and accounting remain separate from duplex. No Remora adapter superiority/adoption is inferred.
