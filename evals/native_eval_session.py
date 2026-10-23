"""Reuse one strictly loaded native model, with fresh per-conversation state.

This changes loading lifetime only. GPU cold/reused output parity is required by
the portable collector before its complete development results can be reviewed.
No conversations, worlds, worker outputs or sampling state are cached here.
"""
from pathlib import Path


def identity(path):
    p = Path(path).resolve(strict=True)
    s = p.stat()
    return str(p), s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


class NativeEvalSession:
    def __init__(self,scope='development'):
        if scope not in ('development','fresh-full20'):raise ValueError('Explicit native evaluation session scope required')
        self.scope=scope
        self.assets = {}
        self.engine = None
        self.binding = None
        self.cases = 0
        self.failed = False

    def verify_asset(self, path, expected):
        from evals.moshi_model_download import sha256
        key = str(Path(path).absolute())
        now = identity(path)
        if key in self.assets:
            if self.assets[key] != (now, expected):
                raise RuntimeError('Previously hashed native asset identity changed')
        else:
            if sha256(path) != expected or identity(path) != now:
                raise RuntimeError('Native asset hash or identity changed during verification')
            self.assets[key] = (now, expected)

    def _proof_files(self, proof):
        return tuple(identity(p) for p in sorted(Path(proof).rglob('*')) if p.is_file())

    def _parameters(self):
        rows = []
        for label in ('lm', 'mimi', 'user_mimi'):
            module = getattr(self.engine, label, None)
            if module is not None:
                rows.extend((label, name, id(p), p.data_ptr(), p._version,
                             str(p.device), str(p.dtype), tuple(p.shape), p.requires_grad)
                            for name, p in module.named_parameters())
        return tuple(rows)

    def load(self, args, factory):
        if self.failed:
            raise RuntimeError('Failed native session cannot be reused')
        # Only the unchanged production-style development profile is supported.
        forbidden = ('adapter_disabled', 'adapter_zero', 'warm_audio', 'pending_status',
                     'native_sampling_diagnostic', 'native_seen_retention_diagnostic',
                     'native_persona_diagnostic', 'native_ancestor_diagnostic',
                     'native_precision_diagnostic', 'native_precision_no_tf32')
        fresh=self.scope=='fresh-full20'
        if fresh and (getattr(args,'source_split',None)!='fresh-evaluation' or getattr(args,'train_history',False)
                or not getattr(args,'native_fresh_behavior_review',None) or not getattr(args,'native_fresh_candidate_proof',None)
                or not 0<=args.lookup_delivery_delay_s<=30):
            raise ValueError('Fresh reuse requires independently guarded evaluation input/review and declared bounded transport delay')
        if (any(getattr(args, n, False) for n in forbidden) or args.adapter_proof is None
                or args.layout_audit or args.ret_logit_bias or (args.lookup_delivery_delay_s and not fresh)
                or args.native_text_temperature != .7 or args.persona_profile != 'recorded-records'
                or args.backend_reference_profile != 'answer-only' or args.event_log_mode != 'immediate'
                or args.voice != 'accepted-synthetic'):
            raise ValueError('Persistent evaluation requires the unchanged native development profile')
        binding = tuple(str(getattr(args, n)) for n in
                        ('assets', 'base', 'adapter_proof', 'voice_reference', 'voice_provenance')) + (
                            getattr(args,'native_runtime_profile','container-default'),
                            getattr(args,'native_asr_placement','separate-gpu'))
        if fresh:
            binding+=tuple(identity(getattr(args,n)) for n in
                           ('native_fresh_behavior_review','native_fresh_quality'))
            binding+=(str(Path(args.native_fresh_candidate_proof).resolve()),
                      getattr(args,'native_fresh_source_profile','sealed-large-v3'))
        from evals.context_execution import snapshot
        if self.engine is None:
            self.engine, self.records, self.adapter = factory()
            self.binding = binding
            self.proof_files = self._proof_files(args.adapter_proof)
            self.parameters = self._parameters()
            self.execution = snapshot(self.engine.lm)
            self.raw_inject = self.engine.inject_context_span
        else:
            if (binding != self.binding or self._proof_files(args.adapter_proof) != self.proof_files
                    or self._parameters() != self.parameters or snapshot(self.engine.lm) != self.execution):
                raise RuntimeError('Native model, proof or execution policy changed between conversations')
        self.engine.inject_context_span = self.raw_inject
        if hasattr(self.engine, '_dl_student_history'):
            raise RuntimeError('Previous native conversation recorder was not removed')
        self.cases += 1
        return self.engine, self.records, self.adapter

    def finish_case(self, succeeded):
        if self.engine is None:
            self.failed = True
            return
        history = getattr(self.engine, '_dl_student_history', None)
        if history is not None:
            # finish() already restores these on success. Also unwind an error
            # without ever trying another conversation in a contaminated session.
            self.engine.lm.forward_codes = history.forward
            history.state.graphed_main = history.graph
            history.prefix_trace.close()
            del self.engine._dl_student_history
        self.engine.inject_context_span = self.raw_inject
        if not succeeded:
            self.failed = True
        if self._parameters() != self.parameters:
            self.failed = True
            raise RuntimeError('Native inference changed parameters')

    def receipt(self):
        return dict(schema='dl-native-persistent-evaluation-v1', conversation_index=self.cases,
                    evaluation_scope=self.scope,
                    strictly_loaded_model_reused=self.cases > 1,
                    seed_and_all_streams_reset_by_native_probe=True,
                    immutable_asset_hashes_verified_before_first_use=True,
                    immutable_asset_identity_checked_each_use=True,
                    proof_and_parameter_identity_checked_each_conversation=True,
                    conversation_or_backend_result_cached=False,
                    parity_and_complete_behavior_review_still_required=True)
