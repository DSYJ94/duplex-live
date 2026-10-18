"""Bounded, ordered actual capture frames for a single live session."""
import queue
import struct
import threading
import time


class LiveFrames:
    def __init__(self,max_seconds=300):
        if not 1<=max_seconds<=300:raise ValueError('Live session duration must be bounded')
        self.max_seconds=max_seconds;self.queue=queue.Queue(maxsize=16);self.closed=threading.Event();self.sequence=0
        self.received=[];self.receipts=[];self.lock=threading.Lock();self.drain_frames=0
    def feed(self,packet):
        import numpy as np
        if self.closed.is_set():raise ValueError('Capture has already ended')
        if len(packet)!=4+1920*4:raise ValueError('Capture must contain one sequence and exactly1920 float32 samples')
        sequence=struct.unpack_from('<I',packet)[0];pcm=np.frombuffer(packet,dtype='<f4',offset=4).copy()
        if sequence!=self.sequence or not np.isfinite(pcm).all() or np.max(np.abs(pcm))>8:raise ValueError('Invalid capture sequence or PCM')
        if (sequence+1)*.08>self.max_seconds:raise ValueError('Capture duration limit reached')
        received=time.monotonic()
        try:self.queue.put_nowait((pcm,received))
        except queue.Full as error:raise RuntimeError('Capture backlog exceeded16 actual frames; no audio dropped silently') from error
        with self.lock:
            self.sequence+=1;self.received.append(pcm);self.receipts.append(dict(sequence=sequence,received_monotonic=received,samples=1920))
    def take(self):
        try:return self.queue.get_nowait()
        except queue.Empty:return None
    def finish(self):self.closed.set()
    @property
    def finished(self):return self.closed.is_set() and self.queue.empty()
    def pcm(self):
        import numpy as np
        with self.lock:return np.concatenate(self.received) if self.received else np.zeros(0,dtype=np.float32)
