"""
Signal Hub (Home Assistant add-on). This process uses only the standard library, so it idles at a few tens of MB.
All pandas/yfinance work runs in worker.py, a short-lived child that Linux is told to kill first when RAM runs out.
"""
import base64
import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

HERE = Path(__file__).parent
try:
    OPT = json.loads((Path(os.environ.get("DATA_DIR", "/data")) / "options.json").read_text())  # add-on options from the HA UI
except Exception:
    OPT = {}


def opt(k, d=None):
    v = OPT.get(k)
    return d if v in (None, "") else v


for env, key in (("NTFY_URL", "ntfy_url"), ("NTFY_TOPIC", "ntfy_topic"), ("TG_TOKEN", "tg_token"),
                 ("TG_CHAT", "tg_chat"), ("COST_PCT", "cost_pct")):
    if opt(key) is not None and not os.environ.get(env):
        os.environ[env] = str(opt(key))
os.environ.setdefault("HA_NOTIFY", ",".join(opt("ha_notify", [])))
import notify  # noqa: E402

DATA = Path(os.environ.get("DATA_DIR", "/data"))
DATA.mkdir(parents=True, exist_ok=True)
WEB = HERE / "web"
TZ = ZoneInfo(os.environ.get("TZ") or "Asia/Bangkok")
CAT = json.loads((HERE / "catalog.json").read_text(encoding="utf-8"))
STRATS = CAT["strategies"]
INGRESS_IP = "172.30.32.2"  # the Supervisor's ingress gateway: HA has already logged the user in
DEFAULTS = {"time": opt("scan_time", "05:15"), "events": {"buy": True, "sell": True, "sl": True, "tp": False},
            "channels": {"ha": True, "persist": False, "ntfy": False, "tg": False}, "capital": 10000}
EV_TH = {"buy": "🟢 สัญญาณซื้อ", "sell": "🔴 สัญญาณขาย", "sl": "🛑 โดน Stop-loss", "tp": "🎯 ถึงเป้ากำไร"}
LOCK, SCAN_LOCK = threading.Lock(), threading.Lock()
JOBS = {}
STATUS = {"scanning": False, "last_scan": None, "errors": {}}


def mem_free_mb():
    try:
        for ln in open("/proc/meminfo"):
            if ln.startswith("MemAvailable"):
                return int(ln.split()[1]) // 1024
    except Exception:
        pass
    return 10 ** 6


def _key():
    p = DATA / "secret.key"
    if not p.exists():
        p.write_text(secrets.token_hex(32))
    return p.read_text().strip().encode()


SECRET = _key()


# ------------------------------------------------------------------ storage (persists in /data, included in HA backups)
def db():
    c = sqlite3.connect(DATA / "hub.db", check_same_thread=False)
    c.row_factory = sqlite3.Row
    c.executescript("""CREATE TABLE IF NOT EXISTS watch(ticker TEXT PRIMARY KEY, strategy TEXT, params TEXT, alert INTEGER DEFAULT 1);
        CREATE TABLE IF NOT EXISTS sig(ticker TEXT, d TEXT, ev TEXT, price REAL, msg TEXT, ts TEXT, PRIMARY KEY(ticker,d,ev));
        CREATE TABLE IF NOT EXISTS snap(ticker TEXT PRIMARY KEY, js TEXT);
        CREATE TABLE IF NOT EXISTS kv(k TEXT PRIMARY KEY, v TEXT);""")
    return c


def kv_get(k, default=None):
    with LOCK, db() as c:
        r = c.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
    return json.loads(r["v"]) if r else default


def kv_set(k, v):
    with LOCK, db() as c:
        c.execute("INSERT OR REPLACE INTO kv VALUES(?,?)", (k, json.dumps(v, ensure_ascii=False)))


def settings():
    s = kv_get("settings", {})
    return {**DEFAULTS, **s, "events": {**DEFAULTS["events"], **s.get("events", {})},
            "channels": {**DEFAULTS["channels"], **s.get("channels", {})}}


def add_watch(t, name, params=None, alert=1):
    with LOCK, db() as c:
        c.execute("INSERT OR REPLACE INTO watch VALUES(?,?,?,?)", (t, name, json.dumps(params or STRATS[name]["params"]), alert))


def run_worker(cmd, payload, timeout=600):
    return subprocess.run([sys.executable, str(HERE / "worker.py"), cmd], input=json.dumps(payload), capture_output=True,
                          text=True, timeout=timeout, cwd=HERE)


# ------------------------------------------------------------------ scanning + alerts
def handle(snap, events, alert, st):
    with LOCK, db() as c:
        c.execute("INSERT OR REPLACE INTO snap VALUES(?,?)", (snap["ticker"], json.dumps(snap, ensure_ascii=False)))
    for ev, detail in events:
        with LOCK, db() as c:
            new = c.execute("INSERT OR IGNORE INTO sig VALUES(?,?,?,?,?,?)", (snap["ticker"], snap["date"], ev, snap["price"], detail,
                            datetime.now(TZ).isoformat(timespec="seconds"))).rowcount
        if new and alert and st["events"].get(ev):
            lv = snap.get("levels")
            body = f"{detail}\nกลยุทธ์: {STRATS.get(snap['strategy'], {}).get('label', snap['strategy'])} · ข้อมูลวันที่ {snap['date']}"
            if ev == "buy" and lv:
                body += f"\nSL {lv['sl']:,.2f} · TP1 {lv['tp'][0]:,.2f} · TP2 {lv['tp'][1]:,.2f} · TP3 {lv['tp'][2]:,.2f}"
            notify.send(st["channels"], f"{EV_TH[ev]} {snap['ticker']} @ {snap['price']:,.2f}", body, os.environ.get("PUBLIC_URL") or None,
                        f"{snap['ticker']}-{ev}", {"ticker": snap["ticker"], "event": ev, "price": snap["price"], "action": snap["action"]})


def run_scan(only=None):
    """Returns True when finished (even with per-ticker errors), False when postponed for lack of RAM."""
    if not SCAN_LOCK.acquire(blocking=bool(only)):
        return True
    STATUS["scanning"] = True
    try:
        need, free = int(opt("min_free_mb", 250)), mem_free_mb()
        if free < need:
            STATUS["errors"] = {"_": f"แรมว่างเหลือ {free}MB (ต่ำกว่า {need}MB) เลื่อนการสแกนไปลองใหม่"}
            return False
        st = settings()
        with LOCK, db() as c:
            rows = [dict(r) for r in c.execute("SELECT * FROM watch" + (" WHERE ticker=?" if only else ""), (only,) if only else ())]
        errs = {}
        for i in range(0, len(rows), 5):  # small batches bound memory and survive a kill
            chunk = rows[i:i + 5]
            p = run_worker("scan", {"items": [{"ticker": r["ticker"], "strategy": r["strategy"], "params": json.loads(r["params"])} for r in chunk]})
            done = set()
            for line in p.stdout.splitlines():
                try:
                    o = json.loads(line)
                except ValueError:
                    continue
                t = o["snap"]["ticker"] if o.get("ok") else o.get("ticker")
                done.add(t)
                if o.get("ok"):
                    handle(o["snap"], o["events"], next(r["alert"] for r in chunk if r["ticker"] == t), st)
                else:
                    errs[t] = o["error"]
            for r in chunk:
                if r["ticker"] not in done:
                    errs[r["ticker"]] = "ถูกปิดกลางทาง (แรมไม่พอ?)" if p.returncode else "ไม่มีผลลัพธ์"
        STATUS["errors"] = errs
        STATUS["last_scan"] = datetime.now(TZ).isoformat(timespec="minutes")
        return True
    except Exception as e:
        STATUS["errors"] = {"_": str(e)[:200]}
        return True
    finally:
        STATUS["scanning"] = False
        SCAN_LOCK.release()


def run_backtest(t):
    """Runs in a background thread: ~1-3 minutes on a Pi, result cached for 24 h."""
    JOBS[t] = {"status": "running"}
    try:
        with SCAN_LOCK:  # one heavy job at a time
            if mem_free_mb() < int(opt("min_free_mb", 250)):
                raise RuntimeError("แรมว่างไม่พอ ลองใหม่ภายหลัง")
            p = run_worker("backtest", {"ticker": t}, 1500)
        try:
            out = json.loads(p.stdout)
        except ValueError:
            raise RuntimeError((p.stderr.strip().splitlines() or ["ทดสอบไม่สำเร็จ (อาจถูกปิดเพราะแรมไม่พอ)"])[-1][:200])
        (DATA / f"bt_{t}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        JOBS[t] = {"status": "done"}
    except Exception as e:
        JOBS[t] = {"status": "error", "error": str(e)[:200]}


def scheduler():
    retry_at = 0
    while True:
        try:
            now, st = datetime.now(TZ), settings()
            today = now.strftime("%Y-%m-%d")
            # US sessions Mon-Fri finish early Tue-Sat Thai time
            if 1 <= now.weekday() <= 5 and now.strftime("%H:%M") >= st["time"] and kv_get("last_run") != today and time.time() >= retry_at:
                if run_scan():
                    kv_set("last_run", today)
                else:
                    retry_at = time.time() + 1800  # not enough RAM: try again in 30 minutes
        except Exception:
            pass
        time.sleep(30)


# ------------------------------------------------------------------ auth (ingress = already logged into HA; direct port = password)
def make_token(days=30):
    exp = str(int(time.time()) + days * 86400)
    return base64.urlsafe_b64encode(f"{exp}.{hmac.new(SECRET, exp.encode(), hashlib.sha256).hexdigest()}".encode()).decode()


def check_token(tok):
    try:
        exp, sig = base64.urlsafe_b64decode(tok.encode()).decode().split(".")
        return int(exp) > time.time() and hmac.compare_digest(sig, hmac.new(SECRET, exp.encode(), hashlib.sha256).hexdigest())
    except Exception:
        return False


FAILS = {"n": 0, "t": 0.0}


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send_json(self, obj, code=200):
        b = json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def body(self):
        n = int(self.headers.get("Content-Length") or 0)
        try:
            return json.loads(self.rfile.read(min(n, 100000)) or b"{}")
        except ValueError:
            return {}

    def authed(self):
        return self.client_address[0] == INGRESS_IP or check_token(self.headers.get("Authorization", "").replace("Bearer ", ""))

    def static(self, path):
        p = (WEB / ("index.html" if path in ("", "/") else path.lstrip("/"))).resolve()
        if WEB.resolve() not in p.parents or not p.is_file():
            p = WEB / "index.html"
        b = p.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", (mimetypes.guess_type(p.name)[0] or "application/octet-stream") + "; charset=utf-8")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def route(self, method):
        path = urlparse(self.path).path
        if not path.startswith("/api/"):
            return self.static(path) if method == "GET" else self.send_json({"error": "not found"}, 404)
        if path == "/api/login" and method == "POST":
            pw = opt("password", "")
            if not pw:
                return self.send_json({"error": "ยังไม่ได้ตั้งรหัสผ่านสำหรับเข้าผ่านพอร์ตตรง ใช้ผ่านแผงใน HA แทน"}, 403)
            if FAILS["n"] >= 5 and time.time() - FAILS["t"] < 900:
                return self.send_json({"error": "ลองผิดหลายครั้ง รอ 15 นาที"}, 429)
            if hmac.compare_digest(self.body().get("password", "").encode(), str(pw).encode()):
                FAILS["n"] = 0
                return self.send_json({"token": make_token()})
            FAILS.update(n=FAILS["n"] + 1, t=time.time())
            time.sleep(1)
            return self.send_json({"error": "รหัสผ่านไม่ถูกต้อง"}, 401)
        if not self.authed():
            return self.send_json({"error": "ต้องเข้าสู่ระบบ"}, 401)
        b = self.body() if method in ("POST", "PUT") else {}
        if path == "/api/catalog":
            return self.send_json(CAT)
        if path == "/api/state" and method == "GET":
            with LOCK, db() as c:
                items = [json.loads(r["js"]) for r in c.execute("SELECT js FROM snap")]
                wl = {r["ticker"]: dict(r) for r in c.execute("SELECT * FROM watch")}
            have = {i["ticker"] for i in items}
            items += [{"ticker": t, "strategy": w["strategy"], "action": "WAIT", "msg": "ยังไม่เคยสแกน"} for t, w in wl.items() if t not in have]
            items = [{**i, "alert": bool(wl[i["ticker"]]["alert"])} for i in items if i["ticker"] in wl]
            return self.send_json({"items": items, "settings": settings(), "status": {**STATUS, "free_mb": mem_free_mb()}})
        if path == "/api/signals" and method == "GET":
            with LOCK, db() as c:
                return self.send_json([dict(r) for r in c.execute("SELECT * FROM sig ORDER BY ts DESC LIMIT 100")])
        if path == "/api/watchlist" and method == "POST":
            t = re.sub(r"[^A-Z0-9.\-]", "", str(b.get("ticker", "")).upper())[:12]
            name = b.get("strategy") or opt("default_strategy", "gtr_default")
            if not t or name not in STRATS:
                return self.send_json({"error": "ชื่อหุ้นหรือกลยุทธ์ไม่ถูกต้อง"}, 400)
            add_watch(t, name, b.get("params"))
            threading.Thread(target=run_scan, kwargs={"only": t}, daemon=True).start()
            return self.send_json({"ok": True})
        m = re.match(r"/api/watchlist/([A-Za-z0-9.\-]+)$", path)
        if m:
            t = m.group(1).upper()
            if method == "DELETE":
                with LOCK, db() as c:
                    c.execute("DELETE FROM watch WHERE ticker=?", (t,))
                    c.execute("DELETE FROM snap WHERE ticker=?", (t,))
                return self.send_json({"ok": True})
            if method == "PUT":
                with LOCK, db() as c:
                    r = c.execute("SELECT * FROM watch WHERE ticker=?", (t,)).fetchone()
                if not r or b.get("strategy", r["strategy"]) not in STRATS:
                    return self.send_json({"error": "ไม่พบหุ้นหรือกลยุทธ์"}, 400)
                add_watch(t, b.get("strategy", r["strategy"]), b.get("params", json.loads(r["params"])), int(bool(b.get("alert", r["alert"]))))
                if "strategy" in b or "params" in b:
                    threading.Thread(target=run_scan, kwargs={"only": t}, daemon=True).start()
                return self.send_json({"ok": True})
        m = re.match(r"/api/chart/([A-Za-z0-9.\-]+)$", path)
        if m and method == "GET":
            t = m.group(1).upper()
            cache = DATA / f"chart_{t}.json"
            if cache.exists() and time.time() - cache.stat().st_mtime < 900:
                return self.send_json(json.loads(cache.read_text(encoding="utf-8")))
            with LOCK, db() as c:
                r = c.execute("SELECT * FROM watch WHERE ticker=?", (t,)).fetchone()
            name, params = (r["strategy"], json.loads(r["params"])) if r else ("gtr_default", {})
            with SCAN_LOCK:  # one heavy job at a time
                if mem_free_mb() < int(opt("min_free_mb", 250)):
                    return self.send_json({"error": "แรมว่างไม่พอ ลองใหม่ภายหลัง"}, 503)
                p = run_worker("chart", {"ticker": t, "strategy": name, "params": params}, 300)
            try:
                out = json.loads(p.stdout)
            except ValueError:
                return self.send_json({"error": (p.stderr.strip().splitlines() or ["คำนวณไม่สำเร็จ"])[-1][:200]}, 400)
            cache.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
            return self.send_json(out)
        m = re.match(r"/api/backtest/([A-Za-z0-9.\-]+)$", path)
        if m and method in ("GET", "POST"):
            t = m.group(1).upper()
            f = DATA / f"bt_{t}.json"
            if method == "POST":  # (re)run
                if JOBS.get(t, {}).get("status") != "running":
                    f.unlink(missing_ok=True)
                    threading.Thread(target=run_backtest, args=(t,), daemon=True).start()
                return self.send_json({"status": "running"})
            if JOBS.get(t, {}).get("status") == "running":
                return self.send_json({"status": "running"})
            if f.exists() and time.time() - f.stat().st_mtime < 86400:
                return self.send_json({"status": "done", "result": json.loads(f.read_text(encoding="utf-8"))})
            return self.send_json(JOBS.get(t) if JOBS.get(t, {}).get("status") == "error" else {"status": "none"})
        if path == "/api/settings" and method == "PUT":
            cur = settings()
            if re.fullmatch(r"\d\d:\d\d", str(b.get("time", ""))):
                cur["time"] = b["time"]
            for g in ("events", "channels"):
                if isinstance(b.get(g), dict):
                    cur[g].update({k: bool(v) for k, v in b[g].items() if k in cur[g]})
            if isinstance(b.get("capital"), (int, float)) and b["capital"] >= 0:
                cur["capital"] = b["capital"]
            kv_set("settings", cur)
            return self.send_json(cur)
        if path == "/api/scan" and method == "POST":
            threading.Thread(target=run_scan, daemon=True).start()
            return self.send_json({"started": True})
        if path == "/api/test-notify" and method == "POST":
            ch = b.get("channel")
            en = {ch: True} if ch in notify.CHANNELS else settings()["channels"]
            return self.send_json(notify.send(en, "🔔 ทดสอบแจ้งเตือน", "ถ้าเห็นข้อความนี้ แปลว่าช่องทางนี้ใช้งานได้", os.environ.get("PUBLIC_URL") or None, "test"))
        return self.send_json({"error": "not found"}, 404)

    def do_GET(self): self.route("GET")
    def do_POST(self): self.route("POST")
    def do_PUT(self): self.route("PUT")
    def do_DELETE(self): self.route("DELETE")


def main():
    strat = opt("default_strategy", "gtr_default")
    for t in opt("watchlist", []):  # seed from add-on options; edits made in the app are kept
        t = re.sub(r"[^A-Z0-9.\-]", "", str(t).upper())[:12]
        with LOCK, db() as c:
            have = c.execute("SELECT 1 FROM watch WHERE ticker=?", (t,)).fetchone()
        if t and not have and strat in STRATS and not kv_get("seeded_" + t):
            add_watch(t, strat)
            kv_set("seeded_" + t, True)
    threading.Thread(target=scheduler, daemon=True).start()
    if opt("notify_on_start", False):  # quick way to check that alerts reach your phone
        threading.Thread(target=lambda: print("ทดสอบแจ้งเตือน:", notify.send(settings()["channels"], "✅ Signal Hub เริ่มทำงานแล้ว",
                         "ถ้าเห็นข้อความนี้ ช่องทางแจ้งเตือนใช้งานได้", os.environ.get("PUBLIC_URL") or None, "start"), flush=True), daemon=True).start()
    if opt("scan_on_start", False):
        threading.Thread(target=run_scan, daemon=True).start()
    port = int(os.environ.get("PORT", "8099"))
    print(f"Signal Hub พร้อมที่พอร์ต {port}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", port), H).serve_forever()


if __name__ == "__main__":
    main()
