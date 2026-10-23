# Actual state-decision race and whole32 recovery v3

V2 repaired the replacement recorder: the actual old revision was retired,
the corrected12-token/41-point result returned, and the full spoken response
was retained within959 prefix/live/span frames. It remains unapproved until
whole32 independent speech and semantic checks pass.

Cancellation exposed a different application race. A real newer native request
started at49.620s; the original result was published at54.996s; the real backend
returned `cancel_reasoning` at59.747s. The correct cancel action was then rejected
because its target had already been delivered. All four new backend calls were
HTTP200. This is a delivery/state ordering failure, not missing authentication
or a demonstrated explanation for the older7590052 model's reasoning regression.
V1 and V2 raw recordings and code remain immutable.

The versioned lifecycle_v2 and stream_v3 add a causal delivery barrier:

- A newer native request that is reserved, or is still deciding an action with
  an older target offered in its tools, holds that older result's delivery.
- The model still chooses the operation. No word/intent classifier, case family,
  oracle, forced RET or fabricated cancellation is used.
- A completed unrelated answer releases the original. A committed cancel/replace
  retires it. Failed or uncertain decisions fail closed. The application never
  claims remote computation was aborted.
- The barrier is separate from actual HTTP execution and the predeclared minimum
  transport delay; the ledger records its wait/release. Publication rechecks it
  under the same lock as the bounded, non-retried wire write. Input listening and
  independent backend calls continue outside that lock.
- Provider completion and matching context acknowledgment still each receive
  the fixed response window. The complete measured1100-frame limit stays fixed.

The counterexample tests show the prior recorder sending stale content and
rejecting the late cancel, while the new recorder holds the original and sends
only the committed cancellation. Independent tasks can still complete; uncertain
control failures cannot release the old fact. These are fixture proofs, not
native behavioral qualification.

The new recovery_v3 requires the exact181-file119,510,244-byte parent-v2 archive,
independently backed up on the retained volume. It revalidates the nested original
v1 parent, the corrected v2 replacement and the actual cancellation race before
any new call. The five records000,001,003,004,005 are copied unchanged with their
own actual v1/v2 recorder and explicit parent-receipt hashes.

Only002 is re-recorded first. Then013 is tested alone against the tight full
context budget. Only after both pass do the remaining25 unattempted scenes run,
at most two concurrently. Maximum27 new sessions,35 aggregate attempts including
the prior8, and at most9+27×6=171 backend attempts within192. No retries or resume
into an existing output directory. Every failure stops new submissions and
preserves active attempts. All32 still require independent words/audio, spoken
semantics, actual-time/native conversion, numerical/gradient/retention approval
and a distinct bounded training/restore/behavior chain. A reused subset cannot
qualify. Source WAVs/words/voices/folds and all old native objectives are unchanged.

This application-state fix is currently scoped to the new teacher pipeline.
The existing immutable C9 GPU job and user-facing native voice runtime are
unchanged. Runtime integration and native learned behavior still need proof.
