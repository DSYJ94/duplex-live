"""Diagnostic spoken-text conditioning of actual, completed worker results.

This is an explicit hybrid interface, not learned ARC result integration. A
manually paced native text stream lets the frozen speech decoder articulate a
real result. These inference slots are never measured training word labels.
"""
from collections import deque


class WorkerTextQueue:
    def __init__(self, encode_word, frames_per_word=4):
        if not 3 <= frames_per_word <= 8:
            raise ValueError("Unbounded inference text pace")
        self.encode_word = encode_word
        self.frames_per_word = frames_per_word
        self.queue = deque()

    def append(self, task_id, text, receipt):
        words = text.split()
        if not words or len(words) > 150:
            raise ValueError("Missing or excessive completed worker text")
        tokens = []
        for word in words:
            encoded = list(self.encode_word(word))
            if not encoded or any(t < 5 for t in encoded):
                raise ValueError("Worker word has undefined native text tokens")
            tokens.extend(encoded)
            tokens.extend([3]*max(0, self.frames_per_word-len(encoded)-1))
            tokens.append(0)
        tokens.extend([3]*12)
        self.queue.extend((task_id, token, receipt) for token in tokens)

    def pop(self, deliverable):
        while self.queue:
            task_id, token, receipt = self.queue.popleft()
            if receipt is None or deliverable(receipt):
                return task_id, token
        return None
