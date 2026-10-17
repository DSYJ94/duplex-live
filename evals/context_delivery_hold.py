"""Explicit inference-only transport-delay diagnostic; never fabricate a result."""
import math


def hold_until(result, dispatched_s, minimum_delay_s, now_s):
    if (not all(math.isfinite(x) for x in (dispatched_s,minimum_delay_s,now_s))
            or not 0<=minimum_delay_s<=30 or dispatched_s<0 or now_s<dispatched_s):
        raise ValueError('Invalid bounded delivery-delay clock')
    lookup=(result.get('route') in ('answer','delegate') and any(
        r.get('tool')=='lookup_record' and r.get('executed') is True for r in result.get('receipts',[])))
    release=dispatched_s+minimum_delay_s
    return release if lookup and now_s<release else None
