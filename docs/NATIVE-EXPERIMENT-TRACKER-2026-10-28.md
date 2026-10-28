# Native model experiment tracker

The machine-readable [run index](NATIVE-EXPERIMENT-INDEX-2026-10-28.json) covers
22 locally recorded Venus controller runs and their actual stage receipts as of
its timestamp. It is not a live provider monitor or an exhaustive index of earlier
Moshi Slurm jobs. Earlier Moshi outcomes are linked through the retained reviews.
Refresh it after a run ends with `python -m training.venus.run_ledger`.

| Model state | Actual training | Data and recipe | Outcome |
| --- | --- | --- | --- |
| PersonaPlex/Moshi saved528 | Earlier native training, including the final eight updates520→528 | 1,894,400 adapter parameters in that experiment; original data/objectives and full archive retained | Rejected; paired ordinary answers2/4→1/4. Not continued by Venus training. |
| Venus190 | Five complete152-scene TRAIN epochs;190 updates | Final-layer q/v LoRA851,968 parameters; rank/alpha64; LR1e-4; AdamW; control/text CE and ordinary KL/hidden retention | Restore passed; native behavior rejected. |
| Venus244 | One complete216-scene TRAIN epoch;54 additional updates | Same adapter architecture; corrected word-paced targets, LR5e-5; full52DEV; two logged agent hold decisions | DEV loss.96549→.84252; silence loss slightly worse; complete native comparison pending. |

Exact recipes, source/data/checkpoint hashes, actual terminals and backup receipts
are stored in the index. The190 and244 development losses use different corpora
from the earlier190 run: compare244 only with its matched parent190 baseline on
the corrected52DEV, not with the old36DEV figure. This is not an isolated
learning-rate ablation: data targets and exposure also changed.

Each run retains its event journal, provider interval/guards, exact source SHA,
admitted recipe, data/target approvals and hashes, per-update training metrics,
full DEV records, optimizer/RNG checkpoints, independent restore proof, adaptive
decisions/application receipts, actual native WAVs/traces, independent speech
audit, and verified backup inventory where produced. Missing stages remain
missing; failed runs and proof-only branches never become successful candidates.

The26October move to Venus was a provisional foundation experiment, motivated by
its native duplex/delegation interface and initial ordinary generated answers.
Those early answers were not a controlled spoken-accuracy comparison proving
Venus superior. PersonaPlex artifacts remain intact; neither model is declared
production-ready. See [the original product reset](PROJECT-RESET-2026-10-25.md)
and [the complete Venus comparison](NATIVE-VENUS-COMPLETE10-REVIEW-2026-10-26.json).

Owner voice ratings are preserved separately in the [feedback summary](NATIVE-OWNER-VOICE-FEEDBACK-2026-10-27.json).
They guide George-first voice trials; they were not a reward in the completed
control training, and raw private feedback/live conversations are not exported
into this index.
