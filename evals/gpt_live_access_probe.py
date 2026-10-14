"""Verify actual GPT-Live session access without sending audio or exposing keys."""
import argparse
import asyncio
import hashlib
import json
import os
from pathlib import Path


async def probe(out):
    from websockets.asyncio.client import connect
    key = os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("OPENAI_API_KEY is not configured")
    headers = {"Authorization": "Bearer " + key,
               "OpenAI-Safety-Identifier": hashlib.sha256(b"duplex-live-evaluation").hexdigest()}
    report = {"model": "gpt-live-1", "audio_sent": False, "access_verified": False,
              "endpoint": "wss://api.openai.com/v1/live/sessions"}
    try:
        async with connect(report["endpoint"], additional_headers=headers,
                           open_timeout=20, max_size=2**22) as ws:
            await ws.send(json.dumps({"type": "session.start", "event_id": "access_start",
                "session": {"model": "gpt-live-1", "delegation": {"type": "client"},
                            "audio": {"format": {"type": "audio/pcm", "rate": 24000},
                                      "output": {"voice": "marin"}}}}))
            events = []
            async with asyncio.timeout(30):
                async for message in ws:
                    event = json.loads(message)
                    events.append(event)
                    if event.get("type") == "session.started":
                        report["access_verified"] = True
                        await ws.send(json.dumps({"type": "session.close"}))
                    if event.get("type") in ("session.closed", "error"):
                        break
            report["events"] = events
    except Exception as error:
        # Handshake messages can contain request details. Record status/type only.
        report["error_type"] = type(error).__name__
        response = getattr(error, "response", None)
        report["http_status"] = getattr(response, "status_code", None)
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "events"}), flush=True)
    if not report["access_verified"]:
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=False)
    asyncio.run(probe(args.out))


if __name__ == "__main__":
    main()
