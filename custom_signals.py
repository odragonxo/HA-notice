"""
Pick-your-own technical indicators -> buy/sell signal (an extra system next to the built-in strategies).

Every indicator becomes a 0/1 "long state". Several indicators are combined with
  all  = every selected indicator must agree (strict, fewer trades)
  any  = at least one agrees (loose, more trades)
  vote = strict majority agrees
A BUY is the combined state turning 0 -> 1, a SELL is 1 -> 0 (executed at the next open, like everything else).
"""
import numpy as np
import pandas as pd

import backtest as bt

# params: (key, label, min, max, default, step)  - float defaults give float inputs
SPEC = {
    "sma_cross": {"label": "SMA ตัดกัน", "needs_hl": False,
                  "desc": "ซื้อเมื่อ SMA เร็วอยู่เหนือ SMA ช้า ขายเมื่อกลับกัน",
                  "params": [("fast", "SMA เร็ว (วัน)", 2, 200, 20, 1), ("slow", "SMA ช้า (วัน)", 5, 400, 50, 1)]},
    "ema_cross": {"label": "EMA ตัดกัน", "needs_hl": False,
                  "desc": "ซื้อเมื่อ EMA เร็วอยู่เหนือ EMA ช้า ขายเมื่อกลับกัน (ไวกว่า SMA)",
                  "params": [("fast", "EMA เร็ว (วัน)", 2, 200, 9, 1), ("slow", "EMA ช้า (วัน)", 5, 400, 21, 1)]},
    "price_sma": {"label": "ราคาเหนือเส้นค่าเฉลี่ย", "needs_hl": False,
                  "desc": "ถือเมื่อราคาปิดอยู่เหนือ SMA ขายเมื่อหลุดเส้น",
                  "params": [("n", "SMA (วัน)", 5, 400, 200, 1)]},
    "macd": {"label": "MACD", "needs_hl": False,
             "desc": "ซื้อเมื่อเส้น MACD อยู่เหนือเส้นสัญญาณ ขายเมื่อกลับกัน",
             "params": [("fast", "เร็ว", 2, 100, 12, 1), ("slow", "ช้า", 5, 200, 26, 1), ("sig", "สัญญาณ", 2, 50, 9, 1)]},
    "supertrend": {"label": "Supertrend (ATR)", "needs_hl": True,
                   "desc": "ซื้อเมื่อราคาปิดเหนือเส้น Supertrend ขายเมื่อหลุดเส้น",
                   "params": [("n", "ATR (วัน)", 2, 50, 10, 1), ("mult", "ตัวคูณ", 0.5, 10.0, 3.0, 0.5)]},
    "rsi": {"label": "RSI กลับตัว", "needs_hl": False,
            "desc": "ซื้อเมื่อ RSI ต่ำกว่าระดับล่าง (ขายมากเกิน) ขายเมื่อ RSI สูงกว่าระดับบน",
            "params": [("n", "RSI (วัน)", 2, 50, 14, 1), ("low", "ซื้อเมื่อต่ำกว่า", 1, 49, 30, 1),
                       ("high", "ขายเมื่อสูงกว่า", 51, 99, 70, 1)]},
    "stoch": {"label": "Stochastic กลับตัว", "needs_hl": True,
              "desc": "ซื้อเมื่อ %K ต่ำกว่าระดับล่าง ขายเมื่อสูงกว่าระดับบน",
              "params": [("k", "%K (วัน)", 3, 50, 14, 1), ("d", "ทำเรียบ (วัน)", 1, 10, 3, 1),
                         ("low", "ซื้อเมื่อต่ำกว่า", 1, 49, 20, 1), ("high", "ขายเมื่อสูงกว่า", 51, 99, 80, 1)]},
    "bollinger": {"label": "Bollinger Bands กลับตัว", "needs_hl": False,
                  "desc": "ซื้อเมื่อราคาปิดต่ำกว่าแถบล่าง ขายเมื่อกลับถึงเส้นกลาง",
                  "params": [("n", "ช่วง (วัน)", 5, 100, 20, 1), ("k", "ส่วนเบี่ยงเบน (เท่า)", 0.5, 4.0, 2.0, 0.1)]},
    "donchian": {"label": "Breakout (Donchian)", "needs_hl": False,
                 "desc": "ซื้อเมื่อปิดทะลุสูงสุด n วัน ขายเมื่อหลุดต่ำสุด m วัน",
                 "params": [("n", "ทะลุสูงสุด (วัน)", 3, 250, 20, 1), ("m", "หลุดต่ำสุด (วัน)", 2, 120, 10, 1)]},
}
MODES = {"all": "ทุกตัวต้องเห็นด้วย (เข้มงวด เทรดน้อย)", "any": "ตัวใดตัวหนึ่ง (หลวม เทรดมาก)",
         "vote": "เสียงข้างมาก"}


def defaults(key):
    return {p[0]: p[4] for p in SPEC[key]["params"]}


def validate(sels, df=None):
    errs = []
    for s in sels:
        k, p = s["key"], s["params"]
        if k in ("sma_cross", "ema_cross", "macd") and p["fast"] >= p["slow"]:
            errs.append(f"{SPEC[k]['label']}: ค่าเร็วต้องน้อยกว่าค่าช้า")
        if k in ("rsi", "stoch") and p["low"] >= p["high"]:
            errs.append(f"{SPEC[k]['label']}: ระดับซื้อต้องต่ำกว่าระดับขาย")
        if df is not None and SPEC[k]["needs_hl"] and not {"High", "Low"} <= set(df.columns):
            errs.append(f"{SPEC[k]['label']}: ต้องใช้ข้อมูล High/Low")
    return errs


def state(df, key, **p):
    """0/1 long state of one indicator."""
    c = df["Close"]
    if key == "sma_cross":
        return bt.sig_sma_cross(df, p["fast"], p["slow"])
    if key == "ema_cross":
        out = (c.ewm(span=p["fast"], adjust=False).mean() > c.ewm(span=p["slow"], adjust=False).mean()).astype(int)
        out.iloc[: p["slow"]] = 0  # warm-up
        return out
    if key == "price_sma":
        return bt.sig_sma_filter(df, p["n"])
    if key == "macd":
        return bt.sig_macd(df, p["fast"], p["slow"], p["sig"])
    if key == "supertrend":
        return bt.sig_supertrend(df, p["n"], p["mult"])
    if key == "rsi":
        return bt.sig_rsi_revert(df, p["n"], p["low"], p["high"])
    if key == "stoch":
        hh, ll = df["High"].rolling(p["k"]).max(), df["Low"].rolling(p["k"]).min()
        kk = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(p["d"]).mean()
        return bt._stateful(c.index, kk < p["low"], kk > p["high"])
    if key == "bollinger":
        return bt.sig_bollinger(df, p["n"], p["k"])
    if key == "donchian":
        return bt.sig_breakout(df, p["n"], p["m"])
    raise KeyError(key)


def build_signal(df, sels, mode="all", max_hold=0):
    """sels: [{"key":..., "params": {...}}]. Returns the combined 0/1 signal (decided at each close)."""
    st_ = [state(df, s["key"], **s["params"]).astype(int) for s in sels]
    tot, n = sum(st_), len(st_)
    comb = {"all": tot == n, "any": tot >= 1, "vote": tot * 2 > n}[mode].astype(int)
    if max_hold and max_hold > 0:
        buy = (comb == 1) & (comb.shift(1).fillna(0) == 0)
        sell = (comb == 0) & (comb.shift(1).fillna(0) == 1)
        comb = bt._timed(df.index, buy, sell, int(max_hold))
    return comb


def describe(sels, mode, max_hold):
    parts = [f"{SPEC[s['key']]['label']} {s['params']}" for s in sels]
    joiner = {"all": " และ ", "any": " หรือ ", "vote": " / โหวตเสียงข้างมาก: "}[mode]
    txt = joiner.join(parts) if len(parts) > 1 else parts[0]
    return txt + (f" · ถือไม่เกิน {int(max_hold)} วันทำการ" if max_hold else "")


def overlays(df, key, p):
    """Lines for the price panel: (label, series, colour key, dash)."""
    c = df["Close"]
    if key == "sma_cross":
        return [(f"SMA{p['fast']}", c.rolling(p["fast"]).mean(), "blue", None),
                (f"SMA{p['slow']}", c.rolling(p["slow"]).mean(), "orange", None)]
    if key == "ema_cross":
        return [(f"EMA{p['fast']}", c.ewm(span=p["fast"], adjust=False).mean(), "blue", None),
                (f"EMA{p['slow']}", c.ewm(span=p["slow"], adjust=False).mean(), "orange", None)]
    if key == "price_sma":
        return [(f"SMA{p['n']}", c.rolling(p["n"]).mean(), "orange", None)]
    if key == "supertrend":
        td, ln = bt.supertrend(df, p["n"], p["mult"])
        return [("Supertrend ขาขึ้น", ln.where(td == 1), "green", None), ("Supertrend ขาลง", ln.where(td == -1), "red", None)]
    if key == "bollinger":
        mid, sd = c.rolling(p["n"]).mean(), c.rolling(p["n"]).std()
        return [("แถบบน", mid + p["k"] * sd, "grey", "dot"), ("เส้นกลาง", mid, "orange", None),
                ("แถบล่าง", mid - p["k"] * sd, "green", "dot")]
    if key == "donchian":
        return [(f"สูงสุด {p['n']} วัน", c.rolling(p["n"]).max().shift(1), "green", "dot"),
                (f"ต่ำสุด {p['m']} วัน", c.rolling(p["m"]).min().shift(1), "red", "dot")]
    return []


def panel_data(df, key, p):
    """Oscillator panel: dict(title, lines=[(label, series, colour key)], hist=Series|None, hlines=[(y, colour key)], yrange)."""
    c = df["Close"]
    if key == "rsi":
        return {"title": f"RSI({p['n']})", "lines": [(f"RSI({p['n']})", bt.rsi(c, p["n"]), "purple")], "hist": None,
                "hlines": [(p["low"], "green"), (p["high"], "red")], "yrange": (0, 100)}
    if key == "stoch":
        hh, ll = df["High"].rolling(p["k"]).max(), df["Low"].rolling(p["k"]).min()
        kk = (100 * (c - ll) / (hh - ll).replace(0, np.nan)).rolling(p["d"]).mean()
        return {"title": "Stochastic %K", "lines": [("%K", kk, "purple")], "hist": None,
                "hlines": [(p["low"], "green"), (p["high"], "red")], "yrange": (0, 100)}
    if key == "macd":
        m, s_ = bt.macd_calc(c, p["fast"], p["slow"], p["sig"])
        return {"title": "MACD", "lines": [("MACD", m, "blue"), ("Signal", s_, "orange")], "hist": m - s_,
                "hlines": [], "yrange": None}
    return None
