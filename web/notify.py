"""Notification channels (stdlib only). Each returns "ok" or an error text, never raises."""
import json
import os
import urllib.request


def _post(url, payload, headers=None, timeout=10):
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), method="POST",
                                 headers={"Content-Type": "application/json", **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status


def _ha():
    """Inside the add-on the Supervisor proxy needs no token from you; outside, use HA_URL + HA_TOKEN."""
    sup = os.environ.get("SUPERVISOR_TOKEN")
    if sup and not os.environ.get("HA_URL"):
        return os.environ.get("SUPERVISOR_URL", "http://supervisor/core").rstrip("/") + "/api", sup
    return os.environ.get("HA_URL", "").rstrip("/") + "/api", os.environ.get("HA_TOKEN", "")


def ha(title, body, url=None, tag=None, ev=None):
    """Push to the selected phones (Companion app) and fire a `stock_signal` event for your own automations."""
    base, tok = _ha()
    svcs = [s.strip() for s in os.environ.get("HA_NOTIFY", "").split(",") if s.strip()]
    if not tok or base == "/api":
        return "ยังไม่ได้ตั้งค่าการเชื่อมต่อ HA"
    if not svcs:
        return "ยังไม่ได้เลือกมือถือในตัวเลือก ha_notify"
    h = {"Authorization": f"Bearer {tok}"}
    data = {"channel": "Signals", "importance": "high", "ttl": 0, "priority": "high", "tag": tag or "signal"}
    if url:
        data.update({"clickAction": url, "url": url})
    out = []
    for s in svcs:
        try:
            _post(f"{base}/services/notify/{s}", {"title": title, "message": body, "data": data}, h)
            out.append("ok")
        except Exception as e:
            out.append(f"{s}: {e}")
    try:
        if ev:
            _post(f"{base}/events/stock_signal", ev, h)
    except Exception as e:
        out.append(f"event: {e}")
    return "ok" if all(x == "ok" for x in out) else "; ".join(x for x in out if x != "ok")


def persist(title, body, url=None, tag=None, ev=None):
    """A notification card inside Home Assistant itself (bell icon)."""
    base, tok = _ha()
    try:
        _post(f"{base}/services/persistent_notification/create",
              {"title": title, "message": body, "notification_id": tag or "signal"}, {"Authorization": f"Bearer {tok}"})
        return "ok"
    except Exception as e:
        return f"HA: {e}"


def ntfy(title, body, url=None, tag=None, ev=None):
    base, topic = os.environ.get("NTFY_URL", "https://ntfy.sh").rstrip("/"), os.environ.get("NTFY_TOPIC", "")
    if not topic:
        return "ยังไม่ได้ตั้ง ntfy_topic"
    p = {"topic": topic, "title": title, "message": body, "priority": 4}
    if url:
        p["click"] = url
    try:
        _post(base, p)
        return "ok"
    except Exception as e:
        return f"ntfy: {e}"


def telegram(title, body, url=None, tag=None, ev=None):
    tok, chat = os.environ.get("TG_TOKEN", ""), os.environ.get("TG_CHAT", "")
    if not (tok and chat):
        return "ยังไม่ได้ตั้ง tg_token/tg_chat"
    try:
        _post(f"https://api.telegram.org/bot{tok}/sendMessage", {"chat_id": chat, "text": f"{title}\n{body}"})
        return "ok"
    except Exception as e:
        return f"Telegram: {e}"


CHANNELS = {"ha": ha, "persist": persist, "ntfy": ntfy, "tg": telegram}


def send(enabled, title, body, url=None, tag=None, ev=None):
    return {k: CHANNELS[k](title, body, url, tag, ev) for k, on in enabled.items() if on and k in CHANNELS}
