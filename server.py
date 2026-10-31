"""
Signal Hub (Home Assistant add-on). This process uses only the standard library, so it idles at a few tens of MB.
All pandas/yfinance work runs in worker.py, a short-lived child that Linux is told to kill first when RAM runs out.
"""
import base64
import csv
import gzip
import hashlib
import hmac
import html as _html
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
import urllib.request
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
            "channels": {"ha": True, "persist": False, "ntfy": False, "tg": False}, "capital": 10000, "auto_refresh": True}
EV_TH = {"buy": "🟢 สัญญาณซื้อ", "sell": "🔴 สัญญาณขาย", "sl": "🛑 โดน Stop-loss", "tp": "🎯 ถึงเป้ากำไร"}
LOCK, SCAN_LOCK = threading.Lock(), threading.Lock()
JOBS = {}
JOBS2 = {}
HEAVY = {"p": None}
RES = DATA / "results"
RES.mkdir(parents=True, exist_ok=True)
SJ = {"status": "none"}
FALLBACK = "AAPL MSFT NVDA AMZN GOOGL META AVGO TSLA BRK-B JPM LLY V UNH XOM MA COST HD PG JNJ ABBV NFLX BAC WMT CRM ORCL CVX KO MRK AMD CSCO PEP ADBE TMO LIN MCD ACN ABT DIS WFC GE INTU QCOM TXN AMAT".split()
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


def run_worker(cmd, payload, timeout=600, heavy=False):
    """Heavy jobs register their child so an urgent alert scan can stop them (alerts always come first)."""
    p = subprocess.Popen([sys.executable, str(HERE / "worker.py"), cmd], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, text=True, cwd=HERE)
    if heavy:
        HEAVY["p"] = p
    try:
        out, err = p.communicate(json.dumps(payload), timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        out, err = p.communicate()
    finally:
        if heavy:
            HEAVY["p"] = None
    return subprocess.CompletedProcess(p.args, p.returncode, out, err)


def kill_heavy():
    p = HEAVY["p"]
    if p and p.poll() is None:
        p.terminate()


def expected_bar():
    """Date of the newest US session that should already be final (after 16:20 New York time)."""
    et = datetime.now(ZoneInfo("America/New_York"))
    d = et.date()
    if et.hour * 60 + et.minute < 16 * 60 + 20:
        d = d.fromordinal(d.toordinal() - 1)
    while d.weekday() >= 5:
        d = d.fromordinal(d.toordinal() - 1)
    return d.isoformat()


def run_job(kind, key, cmd, payload, timeout):
    """Generic heavy job: result is written to /data/results and only READ by the app. Re-queued if an alert scan pre-empts it."""
    jk = f"{kind}:{key}"
    JOBS2[jk] = {"status": "running"}
    try:
        for _ in range(6):
            with SCAN_LOCK:
                if mem_free_mb() < int(opt("min_free_mb", 250)):
                    raise RuntimeError("แรมว่างไม่พอ ลองใหม่ภายหลัง")
                p = run_worker(cmd, payload, timeout, heavy=True)
            if p.returncode == -15:  # stopped for an urgent alert scan
                JOBS2[jk] = {"status": "queued"}
                time.sleep(120)
                continue
            break
        try:
            out = json.loads(p.stdout)
        except ValueError:
            raise RuntimeError((p.stderr.strip().splitlines() or ["ไม่สำเร็จ (อาจถูกปิดเพราะแรมไม่พอ)"])[-1][:200])
        (RES / f"{kind}_{key}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        JOBS2.pop(jk, None)
    except Exception as e:
        JOBS2[jk] = {"status": "error", "error": str(e)[:200]}


def build_job(kind, key, b):
    num = lambda v, lo, hi, d: max(lo, min(hi, int(float(v)))) if isinstance(v, (int, float, str)) and str(v).replace(".", "", 1).isdigit() else d
    if kind == "walkforward":
        if b.get("strategy") not in STRATS:
            raise ValueError("กลยุทธ์ไม่ถูกต้อง")
        return {"ticker": key, "strategy": b["strategy"], "ty": num(b.get("ty"), 2, 15, 5), "tm": num(b.get("tm"), 3, 24, 12)}, 1800
    if kind == "custom":
        sels = []
        for s in (b.get("sels") or [])[:6]:
            sp = CAT["custom"].get(s.get("key"))
            if not sp:
                raise ValueError("อินดิเคเตอร์ไม่ถูกต้อง")
            sels.append({"key": s["key"], "params": {p[0]: (float(s["params"][p[0]]) if isinstance(p[4], float) else int(float(s["params"][p[0]]))) if p[0] in s.get("params", {}) else p[4] for p in sp["params"]}})
        if not sels:
            raise ValueError("เลือกอย่างน้อย 1 อินดิเคเตอร์")
        return {"ticker": key, "sels": sels, "mode": b.get("mode") if b.get("mode") in ("all", "any", "vote") else "all", "hold": num(b.get("hold"), 0, 250, 0)}, 900
    if kind == "ai_train":
        import random
        pool = sorted(i["t"] for i in universe())
        rnd = random.Random(42)
        return {"tickers": sorted(rnd.sample(pool, min(num(b.get("n"), 20, 60, 30), len(pool)))), "h": b.get("h") if b.get("h") in (5, 10, 20) else 10}, 3600
    with LOCK, db() as c:
        return {"tickers": [r["ticker"] for r in c.execute("SELECT ticker FROM watch")]}, 900


def maintenance(wd):
    """After the daily alert scan: refresh stored studies (AI scores daily; backtest + walk-forward on Saturday mornings)."""
    if not settings().get("auto_refresh", True):
        return
    with LOCK, db() as c:
        rows = [dict(r) for r in c.execute("SELECT ticker, strategy FROM watch")]
    if (Path(os.environ.get("BACKTEST_MODEL_DIR", "/data/models")) / "ai_model.json").exists():
        run_job("ai_score", "all", "ai_score", {"tickers": [r["ticker"] for r in rows]}, 900)
    if wd == 5:
        for r in rows:
            run_backtest(r["ticker"])
            run_job("walkforward", r["ticker"], "walkforward", {"ticker": r["ticker"], "strategy": r["strategy"], "ty": 5, "tm": 12}, 1800)


# ------------------------------------------------------------------ scanning + alerts
def handle(snap, events, alert, st, expected="", final=True):
    snap["stale"] = bool(expected) and snap["date"] < expected
    with LOCK, db() as c:
        c.execute("INSERT OR REPLACE INTO snap VALUES(?,?)", (snap["ticker"], json.dumps(snap, ensure_ascii=False)))
    for ev, detail in events:
        if snap["stale"] and not final:  # the newest session is not in the data yet: wait for a retry instead of alerting on old bars
            continue
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


def run_scan(only=None, urgent=False, final=True):
    """True = finished, False = postponed for lack of RAM, "stale" = newest session not in the data yet (retry soon)."""
    if urgent:
        kill_heavy()  # alerts pre-empt any heavy study that is running
        SCAN_LOCK.acquire()
    elif not SCAN_LOCK.acquire(blocking=bool(only)):
        return True
    STATUS["scanning"] = True
    stale = []
    exp = expected_bar()
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
                    handle(o["snap"], o["events"], next(r["alert"] for r in chunk if r["ticker"] == t), st, exp, final)
                    if o["snap"]["date"] < exp:
                        stale.append(t)
                else:
                    errs[t] = o["error"]
            for r in chunk:
                if r["ticker"] not in done:
                    errs[r["ticker"]] = "ถูกปิดกลางทาง (แรมไม่พอ?)" if p.returncode else "ไม่มีผลลัพธ์"
        STATUS["errors"] = errs
        STATUS["last_scan"] = datetime.now(TZ).isoformat(timespec="minutes")
        return "stale" if stale and not final else True
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
            p = run_worker("backtest", {"ticker": t}, 1500, heavy=True)
        try:
            out = json.loads(p.stdout)
        except ValueError:
            raise RuntimeError((p.stderr.strip().splitlines() or ["ทดสอบไม่สำเร็จ (อาจถูกปิดเพราะแรมไม่พอ)"])[-1][:200])
        (DATA / f"bt_{t}.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        JOBS[t] = {"status": "done"}
    except Exception as e:
        JOBS[t] = {"status": "error", "error": str(e)[:200]}


def universe():
    """S&P 500 from Wikipedia (parsed with the standard library, cached 7 days); small built-in list if offline."""
    f = DATA / "sp500.json"
    if f.exists() and time.time() - f.stat().st_mtime < 7 * 86400:
        return json.loads(f.read_text(encoding="utf-8"))
    try:
        req = urllib.request.Request("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", headers={"User-Agent": "Mozilla/5.0 (signal-hub)"})
        page = urllib.request.urlopen(req, timeout=20).read().decode("utf-8", "ignore")
        tbl = page.split('id="constituents"', 1)[1].split("</table>", 1)[0]
        clean = lambda x: _html.unescape(re.sub(r"<.*?>", "", x, flags=re.S)).strip()
        out = []
        for r in re.findall(r"<tr>(.*?)</tr>", tbl, re.S):
            td = re.findall(r"<td.*?>(.*?)</td>", r, re.S)
            if len(td) >= 3:
                out.append({"t": clean(td[0]).replace(".", "-"), "n": clean(td[1]), "s": clean(td[2])})
        if len(out) < 400:
            raise ValueError("unexpected list")
        f.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return out
    except Exception:
        return [{"t": t, "n": "", "s": ""} for t in FALLBACK]


def run_scanu(cfg):
    """Market scan in a background thread: 20 tickers per short-lived worker, politely waiting for free RAM."""
    tks = cfg["tickers"]
    SJ.clear()
    SJ.update(status="running", done=0, total=len(tks), rows=[], cfg=cfg, error=None, started=datetime.now(TZ).isoformat(timespec="minutes"))
    meta = {u["t"]: u for u in universe()} if cfg["universe"] == "sp500" else {}
    try:
        for i in range(0, len(tks), 20):
            for _ in range(40):  # wait up to ~20 min for RAM instead of failing
                if mem_free_mb() >= int(opt("min_free_mb", 250)):
                    break
                time.sleep(30)
            with SCAN_LOCK:
                p = run_worker("scanu", {"tickers": tks[i:i + 20], "names": cfg["names"], "specs": cfg["specs"]}, 900)
            for line in p.stdout.splitlines():
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                m = meta.get(r["ticker"], {})
                SJ["rows"].append({**r, "name": m.get("n", ""), "sector": m.get("s", "")})
            SJ["done"] = min(len(tks), i + 20)
            time.sleep(1)
        SJ["status"] = "done"
        (DATA / "scanu.json").write_text(json.dumps({k: SJ[k] for k in ("rows", "cfg", "started")}, ensure_ascii=False), encoding="utf-8")
    except Exception as e:
        SJ.update(status="error", error=str(e)[:200])


def scheduler():
    retry_at = 0
    while True:
        try:
            now, st = datetime.now(TZ), settings()
            today = now.strftime("%Y-%m-%d")
            # US sessions Mon-Fri finish early Tue-Sat Thai time
            if 1 <= now.weekday() <= 5 and now.strftime("%H:%M") >= st["time"] and kv_get("last_run") != today and time.time() >= retry_at:
                n = (kv_get("tries") or {}).get(today, 0)
                r = run_scan(urgent=True, final=n >= 5)  # up to 6 attempts, 10 minutes apart, while data is still arriving
                if r is True:
                    kv_set("last_run", today)
                    threading.Thread(target=maintenance, args=(now.weekday(),), daemon=True).start()
                elif r == "stale":
                    kv_set("tries", {today: n + 1})
                    retry_at = time.time() + 600
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
        if path == "/api/scanu" and method == "GET":
            if SJ["status"] == "none" and (DATA / "scanu.json").exists():
                SJ.update(status="done", done=0, total=0, **json.loads((DATA / "scanu.json").read_text(encoding="utf-8")))
            return self.send_json({k: SJ.get(k) for k in ("status", "done", "total", "error", "started", "cfg")} | ({"rows": SJ.get("rows", [])} if SJ["status"] == "done" else {}))
        if path == "/api/scanu" and method == "POST":
            if SJ.get("status") == "running":
                return self.send_json({"error": "กำลังสแกนอยู่"}, 409)
            base = CAT["scan_specs"]
            names = [n for n in b.get("names", []) if n in base]
            if not names:
                return self.send_json({"error": "เลือกสัญญาณอย่างน้อย 1 ตัว"}, 400)
            specs = {}
            for n in names:  # keep each parameter's original type (int stays int)
                sp = (b.get("specs") or {}).get(n, {})
                specs[n] = {}
                for k, d in base[n].items():
                    try:
                        specs[n][k] = float(sp.get(k, d)) if isinstance(d, float) else int(round(float(sp.get(k, d))))
                    except (TypeError, ValueError):
                        specs[n][k] = d
            if b.get("universe") == "custom":
                tks = [re.sub(r"[^A-Z0-9.\-]", "", str(t).upper())[:12] for t in b.get("tickers", [])][:150]
            else:
                tks = [u["t"] for u in universe()]
            tks = list(dict.fromkeys(t for t in tks if t))
            if not tks:
                return self.send_json({"error": "ไม่มีรายชื่อหุ้น"}, 400)
            cfg = {"universe": "custom" if b.get("universe") == "custom" else "sp500", "tickers": tks, "names": names, "specs": specs}
            threading.Thread(target=run_scanu, args=(cfg,), daemon=True).start()
            return self.send_json({"status": "running", "total": len(tks)})
        m = re.match(r"/api/job/(walkforward|custom|ai_train|ai_score)/([A-Za-z0-9._\-]+)$", path)
        if m and method in ("GET", "POST"):
            kind, key = m.groups()
            key = key.upper() if kind in ("walkforward", "custom") else key
            jk, f = f"{kind}:{key}", RES / f"{kind}_{key}.json"
            if method == "POST":
                if JOBS2.get(jk, {}).get("status") not in ("running", "queued"):
                    try:
                        pl, to = build_job(kind, key, b)
                    except (ValueError, KeyError, TypeError) as e:
                        return self.send_json({"error": str(e) or "ข้อมูลไม่ถูกต้อง"}, 400)
                    threading.Thread(target=run_job, args=(kind, key, kind, pl, to), daemon=True).start()
                    JOBS2[jk] = {"status": "running"}
                return self.send_json({"status": "running"})
            if JOBS2.get(jk):
                return self.send_json(JOBS2[jk])
            if f.exists():
                return self.send_json({"status": "done", "age_h": round((time.time() - f.stat().st_mtime) / 3600, 1), "result": json.loads(f.read_text(encoding="utf-8"))})
            return self.send_json({"status": "none"})
        m = re.match(r"/api/bars/([A-Za-z0-9.\-]+)$", path)
        if m and method == "GET":  # read straight from the price cache, standard library only (no pandas, no RAM spike)
            f = DATA / "prices" / f"{m.group(1).upper()}.csv.gz"
            if not f.exists():
                return self.send_json({"error": "ยังไม่มีข้อมูลราคาในเครื่อง (จะมีหลังสแกนหรือทดสอบครั้งแรก)"}, 404)
            with gzip.open(f, "rt", encoding="utf-8") as fh:
                rows = list(csv.reader(fh))[1:][-260:]
            return self.send_json({"bars": [{"d": r[0][:10], "o": float(r[1]), "h": float(r[2]), "l": float(r[3]), "c": float(r[4])} for r in rows]})
        if path == "/api/settings" and method == "PUT":
            cur = settings()
            if re.fullmatch(r"\d\d:\d\d", str(b.get("time", ""))):
                cur["time"] = b["time"]
            for g in ("events", "channels"):
                if isinstance(b.get(g), dict):
                    cur[g].update({k: bool(v) for k, v in b[g].items() if k in cur[g]})
            if isinstance(b.get("capital"), (int, float)) and b["capital"] >= 0:
                cur["capital"] = b["capital"]
            if "auto_refresh" in b:
                cur["auto_refresh"] = bool(b["auto_refresh"])
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
