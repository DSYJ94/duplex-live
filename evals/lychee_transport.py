"""Bounded local event transport. Congestion fails explicitly, never drops input."""
import json
import socket
import time


class EventSender:
    def __init__(self, path):
        self.socket=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
        self.socket.settimeout(.05)
        self.socket.connect(str(path))
        self.sent=0

    def send(self, value):
        data=(json.dumps(value,separators=(',',':'))+'\n').encode()
        if len(data)>65536:raise ValueError('Oversized native transport event')
        self.socket.sendall(data)
        self.sent+=1

    def start(self, epoch):
        self.send({'type':'clock_start','monotonic_epoch':epoch})

    def event(self, record):
        self.send({'type':'native_event','record':record})

    def finish(self, epoch):
        self.send({'type':'input_complete','available_s':time.monotonic()-epoch})

    def close(self):
        self.socket.close()
