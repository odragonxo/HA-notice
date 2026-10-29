"""Heavy analysis in a short-lived child process:  python worker.py scan|chart  (JSON on stdin, JSON lines on stdout).
It asks Linux to kill THIS process first if memory runs out, so Home Assistant itself is not the victim."""
import gc
import json
import math
import os
import sys
from datetime import datetime

try:
    with open("/proc/self/oom_score_adj", "w") as f:
        f.write("1000")
except Exception:
    pass
import yfinance as yf  # noqa: E402

import backtest as bt  # noqa: E402

COST = float(os.environ.get("COST_PCT", "0.20")) / 100


def load(t):
    d = bt._clean(yf.download(t, period="5y", auto_adjust=True, progress=False, threads=False))
    if len(d) < 300:
        raise ValueError("ข้อมูลไม่พอ (ต้องมีประวัติอย่างน้อยราว 1 ปี)")
    return d


def analyse(t, name, params):
    df = load(t)
    _, _, trades, sig, state = bt.simulate(df, name, params, COST)
    act, msg = bt.current_action_for({"signal": sig, "state": state})
    lv = None
    if state and state["status"] == "long":
        lv = {"entry": state["entry_price"], "sl": state["sl"], "tp": list(state["tps"]), "hit": state["tp_hit"]}
    elif state and state["preview"] and state["pending"]:
        p = state["preview"]
        lv = {"entry": p["entry"], "sl": p["sl"], "tp": [p["tp1"], p["tp2"], p["tp3"]], "hit": 0, "est": True}
    events = []
    if act in ("BUY", "SELL"):
        events.append((act.lower(), msg))
    if len(trades) and trades.iloc[-1]["status"] == "ปิดแล้ว" and trades.iloc[-1]["exit_date"] == df.index[-1]:
        rs = str(trades.iloc[-1]["reason"] or "")
        px = trades.iloc[-1]["exit_price"]
        if rs.startswith("SL"):
            events.append(("sl", f"ออกที่ {px:,.2f} ({rs})"))
        elif rs.startswith("TP"):
            events.append(("tp", f"{rs} ที่ {px:,.2f}"))
    last, prev = df.iloc[-1], df.iloc[-2]
    snap = {"ticker": t, "strategy": name, "params": params, "action": act, "msg": msg,
            "date": df.index[-1].strftime("%Y-%m-%d"), "price": float(last["Close"]),
            "chg": float(last["Close"] / prev["Close"] - 1), "levels": lv,
            "spark": [round(float(x), 2) for x in df["Close"].iloc[-40:]],
            "updated": datetime.now().isoformat(timespec="minutes")}
    return snap, events, df, trades


def main():
    cmd, req = sys.argv[1], json.load(sys.stdin)
    if cmd == "scan":
        for it in req["items"]:
            try:
                snap, ev, _, _ = analyse(it["ticker"], it["strategy"], it["params"])
                print(json.dumps({"ok": True, "snap": snap, "events": ev}, ensure_ascii=False), flush=True)
            except Exception as e:
                print(json.dumps({"ok": False, "ticker": it["ticker"], "error": str(e)[:200]}, ensure_ascii=False), flush=True)
            gc.collect()
    elif cmd == "backtest":
        t = req["ticker"]
        df = bt._clean(yf.download(t, period="10y", auto_adjust=True, progress=False, threads=False))
        if len(df) < 800:
            raise ValueError("ข้อมูลไม่พอสำหรับทดสอบ (ต้องมีประวัติราว 3 ปีขึ้นไป)")
        split = df.index[int(len(df) * 0.7)]  # first 70% = choose parameters, last 30% = judge
        fits, bh, bhd = bt.evaluate(df, split, COST)
        keys = ("total_return", "cagr", "max_drawdown", "sharpe", "trades", "win_rate", "avg_days", "time_in_market", "expectancy")
        clean = lambda m: {k: (float(m[k]) if k in m and math.isfinite(float(m[k])) else None) for k in keys}

        def eq(d, n=60):
            e = (1 + d[d.index >= split]).cumprod()
            return [round(float(e.iloc[int(i * (len(e) - 1) / (n - 1))]), 4) for i in range(n)]
        rows = [{"name": n, "params": f["params"], "train_sharpe": f["train"].get("sharpe"), "test": clean(f["test"]), "eq": eq(f["daily"])}
                for n, f in fits.items()]
        print(json.dumps({"ticker": t, "start": df.index[0].strftime("%Y-%m-%d"), "split": split.strftime("%Y-%m-%d"),
                          "end": df.index[-1].strftime("%Y-%m-%d"), "cost": COST * 100, "bh": {**clean(bh), "eq": eq(bhd)}, "rows": rows},
                         ensure_ascii=False))
    elif cmd == "chart":
        snap, _, df, tr = analyse(req["ticker"], req["strategy"], req["params"])
        d = df.iloc[-260:]
        print(json.dumps({"ticker": req["ticker"], "strategy": req["strategy"], "snap": snap,
                          "bars": [{"d": i.strftime("%Y-%m-%d"), "o": r.Open, "h": r.High, "l": r.Low, "c": r.Close} for i, r in d.iterrows()],
                          "trades": [{"entry_date": x.entry_date.strftime("%Y-%m-%d"), "entry_price": x.entry_price,
                                      "exit_date": x.exit_date.strftime("%Y-%m-%d"), "exit_price": x.exit_price,
                                      "ret": x["return"], "status": x.status} for _, x in tr.iterrows()]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
