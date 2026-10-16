"""Record actual native TRAIN conditioning, including graph and span inputs.

This is raw evidence, never approved supervision. A future target assembler
must mask every recorded past label and independently approve new continuations.
"""
import json
from evals.moshi_model_download import sha256


class Recorder:
    def __init__(self, engine, folder, source):
        if source['split'] != 'train':
            raise RuntimeError('Student-history collection is TRAIN-only')
        self.engine, self.folder, self.source = engine, folder, source
        self.chunks, self.calls, self.frames, self.states = [], [], 0, []
        self.mode = 'prefix'
        self.active = False
        self.forward = engine.lm.forward_codes
        self.state = engine.lm_gen._streaming_state
        self.graph = self.state.graphed_main
        engine.lm.forward_codes = self._wrap(self.forward, 'direct')
        self.state.graphed_main = self._wrap(self.graph, 'graph')

    def _wrap(self, original, path):
        def recorded(value):
            import torch
            if self.active or value.ndim != 3 or value.shape[:2] != (1, 17) or value.dtype != torch.int64:
                raise RuntimeError('Unexpected native history forward or duplicate graph capture')
            if (value < -1).any():
                raise RuntimeError('Ungenerated audio cannot enter recorded native history')
            self.active = True
            try:
                saved = value.detach().cpu().clone()
                result = original(value)
                self.chunks.append(saved)
                self.calls.append({'path': path, 'mode': self.mode, 'start': self.frames,
                    'frames': value.shape[2]})
                self.frames += value.shape[2]
                return result
            finally:
                self.active = False
        return recorded

    def checkpoint(self):
        """After an actual complete step, retain small generator state, not KV."""
        import torch
        state = self.state
        position = state.offset % state.cache.shape[2]
        if (state.offset != self.frames + 1 or state.provided[:, 1:9, position].any()
                or int(state.provided[:, 9:17, position].sum()) != 7):
            raise RuntimeError('Unexpected actual pending-token ownership or generator clock')
        pending = self.engine._pending_exit_cb0
        self.states.append({'virtual_frames': self.frames, 'offset': state.offset,
            'cache': state.cache.detach().cpu().clone(),
            'provided': state.provided.detach().cpu().clone(),
            'cpu_rng': torch.get_rng_state().clone(), 'cuda_rng': torch.cuda.get_rng_state().clone(),
            'pending_exit_cb0': None if pending is None else pending.detach().cpu().clone()})

    def finish(self):
        import torch
        self.engine.lm.forward_codes = self.forward
        self.state.graphed_main = self.graph
        if not self.chunks or not any(c['path'] == 'graph' for c in self.calls):
            raise RuntimeError('No actual native graph inputs were captured')
        inputs = torch.cat(self.chunks, dim=2)
        if inputs.shape != (1, 17, self.frames):
            raise RuntimeError('Native student history lost frames')
        path = self.folder / 'student-history.pt'
        if len(self.states) != sum(c['mode'] == 'live' for c in self.calls):
            raise RuntimeError('Actual generator snapshots do not cover every real frame')
        torch.save({'delayed_inputs': inputs, 'states': self.states}, path)
        result = {'schema': 'dl-context-student-history-v2', 'source_split': 'train',
            'source': self.source, 'file': path.name, 'sha256': sha256(path),
            'shape': list(inputs.shape), 'calls': self.calls,
            'conditioning_only': True, 'past_labels_must_all_be_masked': True,
            'generated_reply_tokens_forced': False, 'persona_prefix_forced': True,
            'training_approved': False,
            'snapshot_after_every_real_frame': True, 'transformer_KV_saved': False,
            'graph_input_recorded_outside_graph': True}
        (self.folder / 'student-history.json').write_text(json.dumps(result, indent=2) + '\n')
        return {'file': 'student-history.json', 'sha256': sha256(self.folder / 'student-history.json'),
            'history_sha256': result['sha256'], 'frames': self.frames, 'training_approved': False}
