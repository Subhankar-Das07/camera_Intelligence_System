"""Official WhatsApp Cloud API adapter. No-ops when credentials are missing."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, Dict, List, Optional


def _token() -> str:
    return os.environ.get("WHATSAPP_TOKEN", "").strip()


def _phone_id() -> str:
    return os.environ.get("WHATSAPP_PHONE_NUMBER_ID", "").strip()


def configured() -> bool:
    return bool(_token() and _phone_id())


def send_text(to_e164: str, body: str) -> Dict[str, Any]:
    """Send a plain text message via WhatsApp Cloud API. Returns status dict."""
    to_e164 = (to_e164 or "").replace("+", "").replace(" ", "")
    if not configured():
        return {"ok": False, "skipped": True, "reason": "WhatsApp env not configured"}
    if not to_e164:
        return {"ok": False, "skipped": True, "reason": "No recipient number"}

    url = f"https://graph.facebook.com/v20.0/{_phone_id()}/messages"
    payload = {
        "messaging_product": "whatsapp",
        "to": to_e164,
        "type": "text",
        "text": {"body": body[:4000]},
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        method="POST",
        headers={
            "Authorization": f"Bearer {_token()}",
            "Content-Type": "application/json",
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
            return {"ok": True, "skipped": False, "response": raw[:500]}
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        return {"ok": False, "skipped": False, "reason": err}
    except Exception as e:
        return {"ok": False, "skipped": False, "reason": str(e)}


def notify_numbers(numbers: Optional[List[str]], body: str) -> List[Dict[str, Any]]:
    results = []
    for n in numbers or []:
        n = str(n).strip()
        if n:
            results.append({"to": n, **send_text(n, body)})
    if not results:
        results.append(send_text(os.environ.get("WHATSAPP_DEFAULT_TO", ""), body))
    return results
