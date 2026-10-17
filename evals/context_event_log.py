"""Measure immediate log I/O or defer it until all user frames were consumed."""
import io
import time


class EventLog:
    def __init__(self,path,mode='immediate'):
        if mode not in ('immediate','deferred'):raise ValueError('Unknown event logging mode')
        self.path=path;self.mode=mode;self.calls=0;self.total=0.;self.maximum=0.;self.final_write_s=0.

    def __enter__(self):
        self.stream=io.StringIO() if self.mode=='deferred' else self.path.open('w')
        return self

    def measured(self,method,*args):
        start=time.monotonic()
        try:return method(*args)
        finally:
            elapsed=time.monotonic()-start;self.calls+=1;self.total+=elapsed;self.maximum=max(self.maximum,elapsed)

    def write(self,value):return self.measured(self.stream.write,value)

    def flush(self):return self.measured(self.stream.flush)

    def __exit__(self,*args):
        if self.mode=='deferred':
            start=time.monotonic()
            try:self.path.write_text(self.stream.getvalue())
            finally:self.final_write_s=time.monotonic()-start;self.stream.close()
        else:self.stream.close()

    def summary(self):
        return dict(mode=self.mode,in_stream_calls=self.calls,in_stream_write_flush_s=self.total,
            maximum_in_stream_write_flush_s=self.maximum,post_input_file_write_s=self.final_write_s,
            filesystem_writes_during_input=self.mode=='immediate')
