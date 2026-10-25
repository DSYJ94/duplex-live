"""Wait for an independently running guest guard without mistaking startup for failure."""
import json
import time
from datetime import datetime


def await_guest_guard(read, unit, status, *, attempts=20, pause=time.sleep, now=time.time):
    """Require fresh armed status and exact live systemd PID; never relax permission."""
    for _ in range(attempts):
        props = dict(line.split('=', 1) for line in read(
            'sudo systemctl show '+unit+' -p MainPID -p ActiveState -p InvocationID'
        ).splitlines())
        if props.get('ActiveState') in ('failed', 'inactive', 'deactivating'):
            raise RuntimeError('Guest guard service is not active: '+repr(props))
        payload = read('if test -f '+status+'; then cat '+status+'; else echo pending; fi').strip()
        if payload != 'pending':
            guest = json.loads(payload)
            if guest.get('phase') != 'armed':
                raise RuntimeError('Guest guard denied permission: '+repr(guest))
            if (str(guest['pid']) == props.get('MainPID')
                    and props.get('ActiveState') == 'active'
                    and props.get('InvocationID')
                    and 0 <= now()-datetime.fromisoformat(guest['at_utc']).timestamp() < 15):
                return guest, props
        pause(1)
    raise TimeoutError('No fresh same-process guest guard; model launch refused')
