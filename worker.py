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
import pandas as pd  # noqa: E402
import yfinance as yf  # noqa: E402

import backtest as bt  # noqa: E402

COST = float(os.environ.get("COST_PCT", "0.20")) / 100


PDIR = os.path.join(os.environ.get("DATA_DIR", "/data"), "prices")


def get_prices(t, years=10):
    """Price history cached on disk (gz CSV). Normally only the last few days are downloaded, which keeps RAM, time and
    network use low and makes alerts faster. If the overlap disagrees with the cache (split/dividend adjustment) or the
    cache is too short, the whole history is re-downloaded so the numbers stay accurate."""
    os.makedirs(PDIR, exist_ok=True)
    f, old = os.path.join(PDIR, f"{t}.csv.gz"), None
    try:
        old = pd.read_csv(f, index_col=0, parse_dates=True)
    except Exception:
        pass
    if old is not None and len(old) > 300 and (old.index[-1] - old.index[0]).days >= years * 365 * 0.9:
        new = yf.download(t, start=(old.index[-1] - pd.Timedelta(days=7)).strftime("%Y-%m-%d"), auto_adjust=True, progress=False, threads=False)
        new = bt._clean(new) if len(new) else new
        if not len(new):
            return old
        both = old.index.intersection(new.index)
        if len(both) and ((new.loc[both, "Close"] / old.loc[both, "Close"] - 1).abs().max() <= 0.005):
            d = pd.concat([old[~old.index.isin(new.index)], new]).sort_index()
            d.to_csv(f, compression="gzip")
            return d
    d = bt._clean(yf.download(t, period=f"{years}y", auto_adjust=True, progress=False, threads=False))
    if len(d) >= 300:
        d.to_csv(f, compression="gzip")
    return d


def load(t):
    d = get_prices(t, 10).iloc[-1500:]
    if len(d) < 300:
        raise ValueError("ข้อมูลไม่พอ (ต้องมีประวัติอย่างน้อยราว 1 ปี)")
    return d


KEYS = ("total_return", "cagr", "max_drawdown", "sharpe", "trades", "win_rate", "avg_days", "time_in_market", "expectancy")
cm = lambda m: {k: (float(m[k]) if k in m and math.isfinite(float(m[k])) else None) for k in KEYS}
ds = lambda x: x.strftime("%Y-%m-%d")
fin = lambda v: float(v) if v is not None and math.isfinite(float(v)) else None


def eqs(d, split=None, n=60):
    e = (1 + (d if split is None else d[d.index >= split])).cumprod()
    return [round(float(e.iloc[int(i * (len(e) - 1) / (n - 1))]), 4) for i in range(n)]


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
    elif cmd == "scanu":  # one batch (~20 tickers) of the market scan
        import pandas as pd
        tks = req["tickers"]
        raw = yf.download(tks, period="2y", auto_adjust=True, group_by="ticker", threads=False, progress=False)
        multi = isinstance(raw.columns, pd.MultiIndex)
        for t in tks:
            try:
                d = bt._clean((raw[t] if multi else raw).dropna(how="all").copy())
                r = bt.scan_ticker(d, req["names"], specs=req["specs"])
                if r:
                    r["ticker"], r["date"] = t, str(r["date"])
                    print(json.dumps({k: (None if isinstance(v, float) and not math.isfinite(v) else v) for k, v in r.items()}, ensure_ascii=False), flush=True)
            except Exception:
                pass
        gc.collect()
    elif cmd == "backtest":
        t = req["ticker"]
        df = get_prices(t, 10)
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
    elif cmd == "walkforward":
        t, name = req["ticker"], req["strategy"]
        df = get_prices(t, 10)
        w = bt.walk_forward(df, name, COST, int(req.get("ty", 5)), int(req.get("tm", 12)))
        act, msg = bt.current_action_for({"signal": w["latest_signal"], "state": w["latest_state"]})
        print(json.dumps({"ticker": t, "strategy": name, "ty": req.get("ty", 5), "tm": req.get("tm", 12), "asof": ds(df.index[-1]),
                          "m": cm(w["metrics"]), "bh": cm(w["bh_metrics"]), "eq": eqs(w["daily"]), "eqb": eqs(w["bh_daily"]),
                          "folds": [{"a": ds(f["test_start"]), "b": ds(f["test_end"]), "p": f["params"], "ts": fin(f["train_sharpe"]),
                                     "r": fin(f["test_return"]), "bh": fin(f["bh_return"])} for f in w["folds"]],
                          "latest": {"params": w["latest_params"], "action": act, "msg": msg}}, ensure_ascii=False))
    elif cmd == "custom":
        import custom_signals as cs
        t, sels, mode, hold = req["ticker"], req["sels"], req.get("mode", "all"), int(req.get("hold", 0))
        df = get_prices(t, 10)
        errs = cs.validate(sels, df)
        if errs:
            raise ValueError("; ".join(errs))
        sig = cs.build_signal(df, sels, mode, hold)
        split = df.index[int(len(df) * 0.7)]
        f = bt.fit_from_signal(df, sig, split, COST)
        act, msg = bt.current_action(sig)
        print(json.dumps({"ticker": t, "split": ds(split), "asof": ds(df.index[-1]), "desc": cs.describe(sels, mode, hold), "action": act,
                          "msg": msg, "train": cm(f["train"]), "test": cm(f["test"]), "bh": cm(bt.bh_test_metrics(df, split)),
                          "eq": eqs(f["daily"], split), "eqb": eqs(df["Close"].pct_change().fillna(0), split),
                          "trades": [{"a": ds(x.entry_date), "ap": x.entry_price, "b": ds(x.exit_date), "bp": x.exit_price, "r": x["return"],
                                      "s": x.status} for _, x in f["trades"].tail(200).iterrows()]}, ensure_ascii=False))
    elif cmd == "ai_train":
        import ml_model as ml
        frames = {}
        for tk in req["tickers"]:  # one at a time from the disk cache
            try:
                d = get_prices(tk, 15)
                if len(d) > 600 and {"High", "Low", "Open"} <= set(d.columns):
                    frames[tk] = d
            except Exception:
                pass
            gc.collect()
        if len(frames) < 15:
            raise ValueError(f"ดึงราคาสำเร็จแค่ {len(frames)} ตัว (ต้องอย่างน้อย 15)")
        m = ml.train_pipeline(frames, horizon=int(req.get("h", 10)))
        ml.save(m)
        wf, ho = m["metrics"]["walk_forward"], m["metrics"]["holdout"]
        print(json.dumps({"verdict": m["verdict"], "universe": len(m["universe"]), "trained_at": m["trained_at"], "horizon": m["horizon"],
                          "range": m["data_range"], "base_rate": m["base_rate"], "wf_auc": fin(wf["auc"]), "ho_auc": fin(ho["auc"])}))
    elif cmd == "ai_score":
        import ml_model as ml
        m = ml.load()
        if not m:
            raise ValueError("ยังไม่มีโมเดล AI ให้ฝึกก่อน")
        out = {}
        for tk in req["tickers"]:
            try:
                d = get_prices(tk, 10)
                sc = ml.score(d, m, "full")
                p = float(sc.iloc[-1])
                if p != p:
                    continue
                ct, (bd, sg) = ml.contributions(d, m, "full"), ml.band(p, m["base_rate"])
                out[tk] = {"p": p, "band": bd, "sgn": sg, "pct": float((sc.dropna() <= p).mean()), "asof": ds(d.index[-1]),
                           "top": [{"l": r.label, "c": round(float(r.contrib), 3)} for r in ct.head(6).itertuples()] if ct is not None else []}
            except Exception:
                pass
            gc.collect()
        print(json.dumps({"base": m["base_rate"], "verdict": m["verdict"], "scores": out}, ensure_ascii=False))
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
