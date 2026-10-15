import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path
from evals.lychee_transport import EventSender


class TransportTests(unittest.TestCase):
    def test_actual_socket_retains_clock_and_every_native_event(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.sock';server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM)
            server.bind(str(path));server.listen(1);received=[]
            def read():
                connection,_=server.accept()
                with connection,connection.makefile('rb') as stream:
                    for line in stream:received.append(json.loads(line))
            thread=threading.Thread(target=read);thread.start()
            sender=EventSender(path);epoch=time.monotonic();sender.start(epoch)
            records=[{'available_s':i*.04,'event':{'type':'speaking_token','stoken':151696+i}} for i in range(40)]
            for record in records:sender.event(record)
            sender.finish(epoch);sender.close();thread.join(timeout=2);server.close()
            self.assertFalse(thread.is_alive())
            self.assertEqual(received[0],{'type':'clock_start','monotonic_epoch':epoch})
            self.assertEqual([r['record'] for r in received[1:-1]],records)
            self.assertEqual(received[-1]['type'],'input_complete')
            self.assertEqual(sender.sent,42)

    def test_oversized_event_fails_before_sending(self):
        sender=object.__new__(EventSender)
        with self.assertRaisesRegex(ValueError,'Oversized'):
            sender.send({'event':'x'*70000})


if __name__=='__main__':unittest.main()
