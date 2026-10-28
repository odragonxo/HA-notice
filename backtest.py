"""
Backtest engine: trend / breakout / RSI strategies vs buy-and-hold.

CLI:
    pip install yfinance pandas numpy matplotlib
    python backtest.py --ticker NVDA --start 2010-01-01 --split 2020-01-01
    python backtest.py --csv data.csv     # CSV: Date, Close (optional Open, High, Low, Volume)

Rules that keep the test honest:
  * Parameters are chosen on the TRAIN period only, then judged on TEST.
  * Signal is computed from the CLOSE of day t; the trade is executed at the
    OPEN of day t+1 (or the next close if the data has no Open column).
  * Fee + slippage are charged on every buy and every sell.
  * Long-only, all-in / all-out, no leverage, no shorting. Taxes not modelled.
"""
import argparse
import itertools

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_TRAIN_TRADES = 5


# ---------- data ----------
def _clean(df):
    if isinstance(df.columns, pd.MultiIndex):  # newer yfinance
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.title)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].apply(pd.to_numeric, errors="coerce").dropna(subset=["Close"])
    if "Open" in df:
        df = df.dropna(subset=["Open"])
    return df.sort_index()


def load_prices(ticker=None, start=None, csv=None):
    if csv:
        df = pd.read_csv(csv, parse_dates=["Date"], index_col="Date")
    else:
        import yfinance as yf

        df = yf.download(ticker, start=str(start), auto_adjust=True, progress=False)
    return _clean(df)


# ---------- indicators / signals (1 = long, 0 = flat), computed from the CLOSE ----------
def rsi(close, n):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def atr(df, n=14):
    if not {"High", "Low"} <= set(df.columns):
        return None
    pc = df["Close"].shift(1)
    tr = pd.concat([df["High"] - df["Low"], (df["High"] - pc).abs(), (df["Low"] - pc).abs()],
                   axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def macd_calc(close, fast, slow, sig):
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    return m, m.ewm(span=sig, adjust=False).mean()


def supertrend(df, n, mult):
    """Classic Supertrend. Returns (trend series of +1/-1, trailing line)."""
    h, l, c = df["High"].values, df["Low"].values, df["Close"].values
    a = atr(df, n).values
    hl2 = (h + l) / 2
    ub, lb = hl2 + mult * a, hl2 - mult * a
    k = len(c)
    fub, flb, tr = np.full(k, np.nan), np.full(k, np.nan), np.ones(k)
    start = n  # skip ATR warm-up
    if k <= start:
        return pd.Series(1, index=df.index), pd.Series(np.nan, index=df.index)
    fub[start], flb[start] = ub[start], lb[start]
    for i in range(start + 1, k):
        fub[i] = ub[i] if (ub[i] < fub[i - 1] or c[i - 1] > fub[i - 1]) else fub[i - 1]
        flb[i] = lb[i] if (lb[i] > flb[i - 1] or c[i - 1] < flb[i - 1]) else flb[i - 1]
        if tr[i - 1] == 1 and c[i] < flb[i]:
            tr[i] = -1
        elif tr[i - 1] == -1 and c[i] > fub[i]:
            tr[i] = 1
        else:
            tr[i] = tr[i - 1]
    tr[:start] = 0
    line = np.where(tr == 1, flb, fub)
    return pd.Series(tr, index=df.index), pd.Series(line, index=df.index)


def _stateful(index, buy, sell):
    pos = pd.Series(np.nan, index=index)
    pos[buy] = 1
    pos[sell] = 0
    return pos.ffill().fillna(0).astype(int)


def _timed(index, buy, sell, max_hold):
    """Like _stateful, plus a TIME STOP: force an exit after max_hold bars.
    Signal at close i -> position from the next open. Held ~= max_hold trading days."""
    b, x = buy.fillna(False).values, sell.fillna(False).values
    pos, cur, held = np.zeros(len(index), dtype=int), 0, 0
    for i in range(len(index)):
        if cur == 0:
            if b[i]:
                cur, held = 1, 0
        else:
            held += 1
            if x[i] or held >= max_hold:
                cur = 0
        pos[i] = cur
    return pd.Series(pos, index=index)


def ibs(df):
    """Internal Bar Strength: where the close sits inside the day's range (0 = low, 1 = high)."""
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    return (df["Close"] - df["Low"]) / rng


def sig_rsi2_trend(df, low, trend_n, max_hold):
    """Short-term pullback in an uptrend: RSI(2) very low while above the long SMA; exit above SMA5."""
    c = df["Close"]
    return _timed(c.index, (rsi(c, 2) < low) & (c > c.rolling(trend_n).mean()),
                  c > c.rolling(5).mean(), max_hold)


def sig_ibs(df, low, max_hold, trend):
    """Buy a weak close (IBS low), sell when price closes above the prior day's high."""
    c = df["Close"]
    ok = (c > c.rolling(trend).mean()) if trend else pd.Series(True, index=c.index)
    return _timed(c.index, (ibs(df) < low) & ok, c > df["High"].shift(1), max_hold)


def sig_pullback_low(df, n, max_hold):
    """Buy an n-day closing low while above SMA200; exit when price recovers above SMA5."""
    c = df["Close"]
    return _timed(c.index, (c <= c.rolling(n).min()) & (c > c.rolling(200).mean()),
                  c > c.rolling(5).mean(), max_hold)


def sig_short_breakout(df, n, exit_n, max_hold):
    """Short-term breakout: new n-day closing high, exit on exit_n-day low or time stop."""
    c = df["Close"]
    return _timed(c.index, c > c.rolling(n).max().shift(1), c < c.rolling(exit_n).min().shift(1), max_hold)


def sig_boll_pull(df, k, max_hold):
    """Bollinger(20) lower-band dip inside an uptrend (above SMA200); exit at the middle band."""
    c = df["Close"]
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    return _timed(c.index, (c < mid - k * sd) & (c > c.rolling(200).mean()), c > mid, max_hold)


def sig_sma_cross(df, fast, slow):
    c = df["Close"]
    return (c.rolling(fast).mean() > c.rolling(slow).mean()).astype(int)


def sig_sma_filter(df, n):
    """Long while price is above its n-day average (Faber-style trend filter)."""
    c = df["Close"]
    return (c > c.rolling(n).mean()).astype(int)


def sig_tsmom(df, n):
    """Time-series momentum: long while the n-day return is positive."""
    c = df["Close"]
    return (c / c.shift(n) - 1 > 0).astype(int)


def sig_breakout(df, n, exit_n):
    """Donchian-style: enter on a new n-day closing high, exit on a new exit_n-day low."""
    c = df["Close"]
    return _stateful(c.index, c > c.rolling(n).max().shift(1), c < c.rolling(exit_n).min().shift(1))


def sig_macd(df, fast, slow, sig):
    m, s_ = macd_calc(df["Close"], fast, slow, sig)
    out = (m > s_).astype(int)
    out.iloc[: slow * 2] = 0  # warm-up
    return out


def sig_supertrend(df, n, mult):
    tr, _ = supertrend(df, n, mult)
    return (tr == 1).astype(int)


def sig_bollinger(df, n, k):
    """Mean reversion: buy below the lower band, sell when price returns to the middle band."""
    c = df["Close"]
    mid, sd = c.rolling(n).mean(), c.rolling(n).std()
    return _stateful(c.index, c < mid - k * sd, c > mid)


def sig_rsi_revert(df, n, low, exit_level):
    r = rsi(df["Close"], n)
    return _stateful(df.index, r < low, r > exit_level)


def vote_components(df):
    """Fixed, NOT optimised rules (so there is nothing to overfit)."""
    c = df["Close"]
    v = {
        "ราคา > SMA200": (c > c.rolling(200).mean()).astype(int),
        "ผลตอบแทน 12 เดือน > 0": (c / c.shift(252) - 1 > 0).astype(int),
        "MACD(12,26,9) > signal": sig_macd(df, 12, 26, 9),
        "Breakout 100/50": sig_breakout(df, 100, 50),
    }
    if {"High", "Low"} <= set(df.columns):
        v["Supertrend(10,3)"] = sig_supertrend(df, 10, 3.0)
    return v


def sig_trend_vote(df):
    v = vote_components(df)
    total = sum(v.values())
    return (total * 2 > len(v)).astype(int)  # strict majority


# ======================================================================================
# GTR engine: Python port of the SIGNAL logic of "Gold The Richer (GTR) v2.0.0" (CrystalBoy).
# Only the parts that decide trades are ported (Supertrend flip / EMA-cross early entry, ATR
# volatility filter, cooldown, ATR-based SL & TP1-3, momentum-fade exit). The display-only layers
# (MACD, EWO, VWAP, Fibonacci, volume...) are in the helpers below.
# Differences from the Pine script, on purpose (so the test is realistic):
#   * long-only (a SELL signal closes a long instead of opening a short)
#   * orders fill at the NEXT bar's open (the Pine script assumes the signal bar's close)
#   * SL/TP are resolved inside the bar with the worst case first (SL before TP, gaps fill at the open)
#   * costs are charged on every fill; TP1/TP2/TP3 can scale out in thirds (scale_out=1)
#   * optional "wait for retrace" limit entry (the author's own advice), see `retrace`
# ======================================================================================
GTR_DEFAULTS = dict(factor=2.0, early=1, scale=1.0, retrace=0.0, wait=3, sl=1.5, tp1=1.2, tp2=2.4, tp3=3.6,
                    scale_out=1, breakeven=0, htf=0, vol_filter=1, vol_mult=0.8, cooldown=3,
                    atr_len=10, risk_atr_len=14, ema_fast=9, ema_slow=21, rsi_len=14, rsi_ob=70)


def trend_state(df):
    """GTR dashboard trend: +1 bull / -1 bear / 0 neutral (close vs EMA50, EMA21 vs EMA50, Supertrend(10, 3))."""
    c = df["Close"]
    e1, e2 = c.ewm(span=21, adjust=False).mean(), c.ewm(span=50, adjust=False).mean()
    d, _ = supertrend(df, 10, 3.0)
    out = pd.Series(0, index=df.index)
    out[(c > e2) & (e1 > e2) & (d == 1)] = 1
    out[(c < e2) & (e1 < e2) & (d == -1)] = -1
    return out


def _resample_ohlc(df, rule):
    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last"}
    if "Volume" in df.columns:
        agg["Volume"] = "sum"
    return df.resample(rule).agg(agg).dropna(subset=["Close"])


def htf_trend_daily(df):
    """Weekly trend mapped to daily rows using COMPLETED weeks only (no look-ahead)."""
    t = trend_state(_resample_ohlc(df, "W-FRI"))
    return t.reindex(df.index, method="ffill").fillna(0).astype(int)


def gtr_engine(df, cost=0.0, **kw):
    """Bar-by-bar simulation with a real cash/units ledger. Returns dict(daily, pos, trades, target, state, equity)."""
    P = {**GTR_DEFAULTS, **kw}
    if not {"High", "Low"} <= set(df.columns):
        raise ValueError("GTR ต้องใช้ข้อมูล Open/High/Low/Close")
    idx, n = df.index, len(df)
    c = df["Close"].values.astype(float)
    o = (df["Open"] if "Open" in df.columns else df["Close"]).values.astype(float)
    h, l = df["High"].values.astype(float), df["Low"].values.astype(float)
    tr_dir = supertrend(df, P["atr_len"], P["factor"])[0].values
    ef = df["Close"].ewm(span=P["ema_fast"], adjust=False).mean().values
    es = df["Close"].ewm(span=P["ema_slow"], adjust=False).mean().values
    r = rsi(df["Close"], P["rsi_len"]).values
    a_s = atr(df, P["risk_atr_len"])
    a, a_avg = a_s.values, a_s.rolling(50).mean().values
    htfv = htf_trend_daily(df).values if P["htf"] else np.zeros(n, dtype=int)
    scale, so = P["scale"], bool(P["scale_out"])

    cash, units, units0 = 1.0, 0.0, 0.0
    eq, inv, target = np.empty(n), np.zeros(n), np.zeros(n, dtype=int)
    pend_entry, pend_exit, last_sig = None, None, -10 ** 9
    sl = e_px = np.nan
    tps, tp_hit, tr = [np.nan] * 3, 0, None
    trades = []

    def sell(q, px, i, reason):
        nonlocal cash, units, tr
        cash += q * px * (1 - cost)
        units -= q
        tr["exits"].append((q, px))
        tr["last_i"], tr["reason"] = i, reason
        if units <= 1e-12:
            units = 0.0
            q_tot = sum(x[0] for x in tr["exits"])
            trades.append((idx[tr["entry_i"]], tr["entry_px"], idx[i], sum(x[0] * x[1] for x in tr["exits"]) / q_tot,
                           cash / tr["cash0"] - 1, "ปิดแล้ว", reason))
            tr = None

    for i in range(n):
        by_limit = False
        # 1) orders decided at the previous close are executed now
        if pend_exit is not None and units > 0:
            sell(units, o[i], i, pend_exit)
            pend_exit = None
        elif pend_entry is not None and units == 0:
            lim, fp = pend_entry["limit"], None
            if lim is None or o[i] <= lim:
                fp = o[i]
            elif l[i] <= lim:
                fp, by_limit = lim, True
            if fp is not None:
                atr_sig = pend_entry["atr"]
                tr = {"entry_i": i, "entry_px": fp, "cash0": cash, "exits": []}
                units = units0 = cash / (fp * (1 + cost))
                cash, e_px, tp_hit = 0.0, fp, 0
                sl = fp - scale * P["sl"] * atr_sig
                tps = [fp + scale * P[k] * atr_sig for k in ("tp1", "tp2", "tp3")]
                pend_entry = None
            elif i >= pend_entry["expires"]:
                pend_entry = None
        # 2) intrabar stop / targets (worst case first)
        if units > 0:
            new_sl = None
            if o[i] <= sl:
                sell(units, o[i], i, "SL" if tp_hit == 0 else f"SL หลัง TP{tp_hit}")
            elif l[i] <= sl:
                sell(units, sl, i, "SL" if tp_hit == 0 else f"SL หลัง TP{tp_hit}")
            elif not by_limit:
                for k in (1, 2, 3):
                    if units > 0 and tp_hit < k and h[i] >= tps[k - 1]:
                        px = max(o[i], tps[k - 1])
                        tp_hit = k
                        if k == 3:
                            sell(units, px, i, "TP3")
                        elif so:
                            sell(min(units0 / 3, units), px, i, f"TP{k}")
                        if k == 1 and P["breakeven"]:
                            new_sl = e_px * (1 + 2 * cost)
            if units > 0 and new_sl is not None:
                sl = max(sl, new_sl)  # break-even applies from the next bar
        eq[i] = cash + units * c[i]
        inv[i] = 1.0 if units > 0 else 0.0
        # 3) decisions at this bar's close -> orders for the next open
        ok = i >= 1 and not np.isnan(a[i]) and tr_dir[i] != 0 and tr_dir[i - 1] != 0
        if ok:
            can = i - last_sig >= P["cooldown"]
            trend_buy, trend_sell = tr_dir[i] == 1 and tr_dir[i - 1] == -1, tr_dir[i] == -1 and tr_dir[i - 1] == 1
            early_buy = P["early"] and ef[i] > es[i] and ef[i - 1] <= es[i - 1] and r[i] > 50 and c[i] > ef[i]
            early_sell = P["early"] and ef[i] < es[i] and ef[i - 1] >= es[i - 1] and r[i] < 50 and c[i] < ef[i]
            vol_ok = (not P["vol_filter"]) or (not np.isnan(a_avg[i]) and a[i] > a_avg[i] * P["vol_mult"])
            buy = can and (trend_buy or early_buy) and units == 0 and pend_entry is None and vol_ok \
                and (not P["htf"] or htfv[i] >= 0)
            sell_sig = can and (trend_sell or early_sell) and vol_ok and (not P["htf"] or htfv[i] <= 0)
            if buy and sell_sig:  # conflicting: follow the trend engine
                buy, sell_sig = tr_dir[i] == 1, tr_dir[i] != 1
            if units > 0 and pend_exit is None:
                if P["rsi_ob"] and r[i - 1] >= P["rsi_ob"] and r[i] < P["rsi_ob"]:
                    pend_exit = "Momentum fade"
                elif sell_sig:
                    pend_exit = "สัญญาณขาย"
            if sell_sig:
                pend_entry = None
            if buy:
                pend_entry = {"limit": c[i] - P["retrace"] * a[i] if P["retrace"] > 0 else None, "atr": a[i],
                              "expires": i + (P["wait"] if P["retrace"] > 0 else 1)}
            if buy or sell_sig:
                last_sig = i
        target[i] = 1 if ((units > 0 and pend_exit is None) or pend_entry is not None) else 0

    last = n - 1
    if tr is not None:  # still open
        trades.append((idx[tr["entry_i"]], tr["entry_px"], idx[last], c[last], eq[last] / tr["cash0"] - 1,
                       "ถืออยู่", "ถืออยู่"))
    tdf = pd.DataFrame(trades, columns=["entry_date", "entry_price", "exit_date", "exit_price", "return",
                                        "status", "reason"])
    tdf["entry_date"], tdf["exit_date"] = pd.to_datetime(tdf["entry_date"]), pd.to_datetime(tdf["exit_date"])
    for k in ["entry_price", "exit_price", "return"]:
        tdf[k] = tdf[k].astype(float)
    tdf["days"] = (tdf["exit_date"] - tdf["entry_date"]).dt.days

    atr_last = a[last]
    pend = None
    if pend_exit is not None:
        pend = "exit"
    elif pend_entry is not None:
        pend = "entry" if pend_entry["limit"] is None else "entry_limit"
    preview = None
    if not np.isnan(atr_last):
        preview = {"entry": c[last], "sl": c[last] - scale * P["sl"] * atr_last,
                   "tp1": c[last] + scale * P["tp1"] * atr_last, "tp2": c[last] + scale * P["tp2"] * atr_last,
                   "tp3": c[last] + scale * P["tp3"] * atr_last}
    state = {"status": "long" if units > 0 else "flat", "pending": pend, "reason": pend_exit,
             "limit": pend_entry["limit"] if pend_entry else None,
             "wait_left": (pend_entry["expires"] - last) if pend_entry else None,
             "entry_price": e_px if units > 0 else None, "sl": sl if units > 0 else None,
             "tps": tps if units > 0 else None, "tp_hit": tp_hit if units > 0 else 0,
             "atr": float(atr_last), "last_close": float(c[last]), "preview": preview,
             "scale_out": so, "scale": scale, "sl_mult": P["sl"]}
    equity = pd.Series(eq, index=idx)
    return {"daily": equity.pct_change().fillna(0.0), "pos": pd.Series(inv, index=idx), "trades": tdf,
            "target": pd.Series(target, index=idx), "state": state, "equity": equity}


def sig_gtr(df, **p):
    """Target-position series (1 = long after this close, or an entry order pending for the next open)."""
    return gtr_engine(df, cost=0.0, **p)["target"]


ENGINE_STRATS = {"gtr": gtr_engine, "gtr_default": gtr_engine}


# ---------- display layers from the GTR script (daily-bar versions) ----------
def ewo(df, fast=5, slow=35):
    """Elliott Wave Oscillator = SMA(fast) - SMA(slow) of the median price."""
    src = (df["High"] + df["Low"]) / 2
    return src.rolling(fast).mean() - src.rolling(slow).mean()


def buy_sell_volume(df, n=10):
    """ESTIMATE only: each bar's volume is split by where the close sits in the bar's range.
    Returns (buy share % over the last n bars, per-bar buy volume, per-bar sell volume)."""
    v = df["Volume"].astype(float)
    rng = (df["High"] - df["Low"]).replace(0, np.nan)
    buy = (v * (df["Close"] - df["Low"]) / rng).fillna(v * 0.5)
    return buy.rolling(n).sum() / v.rolling(n).sum() * 100, buy, v - buy


def major_swings(df, k=14, smooth=3, ob=80, os_=20):
    """Credit: swing logic follows the "Major Swing" idea used in GTR v2.0.0 (CrystalBoy), which credits
    SYNTRADE_FB (Mozilla Public License 2.0). Keep this credit if you redistribute.
    Swing High = highest high while Stochastic %K is overbought, confirmed when %K later reaches
    oversold (mirror for lows). Each swing is only KNOWN on `confirm_date`, long after its own bar."""
    hh, ll = df["High"].rolling(k).max(), df["Low"].rolling(k).min()
    kk = (100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)).rolling(smooth).mean().values
    h, l, idx = df["High"].values, df["Low"].values, df.index
    out, hi_on, lo_on = [], False, False
    hi_px = hi_i = lo_px = lo_i = None
    for i in range(len(df)):
        if np.isnan(kk[i]):
            continue
        if kk[i] > ob:
            if not hi_on:
                hi_on, hi_px, hi_i = True, h[i], i
            elif h[i] > hi_px:
                hi_px, hi_i = h[i], i
        elif kk[i] < os_ and hi_on:
            out.append(("High", idx[hi_i], float(hi_px), idx[i]))
            hi_on = False
        if kk[i] < os_:
            if not lo_on:
                lo_on, lo_px, lo_i = True, l[i], i
            elif l[i] < lo_px:
                lo_px, lo_i = l[i], i
        elif kk[i] > ob and lo_on:
            out.append(("Low", idx[lo_i], float(lo_px), idx[i]))
            lo_on = False
    return pd.DataFrame(out, columns=["kind", "bar_date", "price", "confirm_date"])


def fib_levels(swings):
    """Retracement between the latest swing High and latest swing Low: 0 = the newer swing, 1 = the older."""
    if swings.empty or swings["kind"].nunique() < 2:
        return None
    hi = swings[swings["kind"] == "High"].iloc[-1]
    lo = swings[swings["kind"] == "Low"].iloc[-1]
    new, old = (hi, lo) if hi["bar_date"] > lo["bar_date"] else (lo, hi)
    lv = {x: float(new["price"] + (old["price"] - new["price"]) * x) for x in (0.0, 0.236, 0.382, 0.5, 0.618, 0.786, 1.0)}
    return {"levels": lv, "new": new, "old": old}


def dashboard(df):
    """Rows for the summary table: (indicator, value text, state +1/-1/0)."""
    c = df["Close"]
    rows = []

    def tr_row(label, d):
        if len(d) < 60:
            return (label, "ข้อมูลไม่พอ", 0)
        s = int(trend_state(d).iloc[-1])
        return (label, {1: "Bullish", -1: "Bearish", 0: "Neutral"}[s], s)

    rows.append(tr_row("เทรนด์ รายวัน", df))
    for lbl, rule_try in (("เทรนด์ รายสัปดาห์ (ปิดแล้ว)", ["W-FRI"]), ("เทรนด์ รายเดือน (ปิดแล้ว)", ["ME", "M"])):
        w = None
        for rule in rule_try:
            try:
                w = _resample_ohlc(df, rule)
                break
            except ValueError:
                continue
        if w is not None and len(w) > 1:
            end_label = w.index[-1]
            complete = df.index[-1] >= end_label  # last bucket already finished?
            rows.append(tr_row(lbl, w if complete else w.iloc[:-1]))
    r14 = float(rsi(c, 14).iloc[-1])
    rows.append(("RSI(14)", f"{r14:.0f} " + ("(Overbought)" if r14 >= 70 else "(Oversold)" if r14 <= 30 else ""),
                 -1 if r14 >= 70 else 1 if r14 <= 30 else 0))
    m, s_ = macd_calc(c, 13, 35, 6)
    rows.append(("MACD (13,35,6)", f"{m.iloc[-1]:.2f} vs {s_.iloc[-1]:.2f}", 1 if m.iloc[-1] > s_.iloc[-1] else -1))
    e = ewo(df).iloc[-1]
    rows.append(("Elliott Wave Osc.", f"{e:+.2f}", 1 if e > 0 else -1 if e < 0 else 0))
    ema9 = c.ewm(span=9, adjust=False).mean().iloc[-1]
    rows.append(("ราคา vs EMA9", f"{(c.iloc[-1] / ema9 - 1) * 100:+.1f}%", 1 if c.iloc[-1] > ema9 else -1))
    if "Volume" in df.columns and df["Volume"].iloc[-10:].sum() > 0:
        share = float(buy_sell_volume(df, 10)[0].iloc[-1])
        rows.append(("Volume ซื้อ/ขาย 10 แท่ง (ประมาณ)", f"ซื้อ {share:.0f}% / ขาย {100 - share:.0f}%",
                     1 if share >= 55 else -1 if share <= 45 else 0))
    return pd.DataFrame(rows, columns=["ตัวชี้วัด", "ค่า", "state"])


# name -> (function, param grid). Grids are kept small on purpose (less overfitting).
STRATEGIES = {
    "sma_cross": (sig_sma_cross, {"fast": [10, 20, 50], "slow": [50, 100, 200]}),
    "sma_filter": (sig_sma_filter, {"n": [50, 100, 150, 200]}),
    "tsmom": (sig_tsmom, {"n": [63, 126, 252]}),
    "breakout": (sig_breakout, {"n": [20, 50, 100], "exit_n": [10, 20, 50]}),
    "macd": (sig_macd, {"fast": [8, 12, 19], "slow": [21, 26, 39], "sig": [5, 9]}),
    "supertrend": (sig_supertrend, {"n": [7, 10, 14], "mult": [2.0, 3.0, 4.0]}),
    "bollinger": (sig_bollinger, {"n": [20, 50], "k": [2.0, 2.5]}),
    "rsi_revert": (sig_rsi_revert, {"n": [2, 14], "low": [10, 30], "exit_level": [50, 70]}),
    "trend_vote": (lambda df: sig_trend_vote(df), {}),
    # --- short-term / swing (hold a few days; time stop built in) ---
    "rsi2_trend": (sig_rsi2_trend, {"low": [5, 10, 20], "trend_n": [100, 200], "max_hold": [5, 10]}),
    "ibs": (sig_ibs, {"low": [0.1, 0.2, 0.3], "max_hold": [3, 5, 10], "trend": [0, 200]}),
    "pullback_low": (sig_pullback_low, {"n": [3, 5, 7], "max_hold": [5, 10]}),
    "short_breakout": (sig_short_breakout, {"n": [5, 10, 20], "exit_n": [3, 5], "max_hold": [10, 20]}),
    "boll_pull": (sig_boll_pull, {"k": [1.5, 2.0], "max_hold": [5, 10]}),
    # --- GTR engine (SL/TP bracket trades; see gtr_engine) ---
    "gtr_default": (lambda df: sig_gtr(df), {}),
    "gtr": (sig_gtr, {"factor": [2.0, 3.0, 3.5], "early": [0, 1], "scale": [1.0, 2.0], "retrace": [0.0, 0.5]}),
}
NEEDS_HL = {"supertrend", "ibs", "gtr", "gtr_default"}

# group: "trend" = long holds, "swing" = short holds
STRATEGY_GROUP = {n: "trend" for n in ["sma_cross", "sma_filter", "tsmom", "breakout", "macd",
                                       "supertrend", "trend_vote"]}
STRATEGY_GROUP.update({n: "swing" for n in ["bollinger", "rsi_revert", "rsi2_trend", "ibs",
                                            "pullback_low", "short_breakout", "boll_pull",
                                            "gtr_default", "gtr"]})
GROUP_LABELS = {"all": "ทั้งหมด", "trend": "เทรนด์ (ถือยาว)", "swing": "เทรดสั้น (swing ถือไม่กี่วัน)"}

STRATEGY_LABELS = {
    "sma_cross": "ตามเทรนด์: SMA cross",
    "sma_filter": "ตามเทรนด์: ราคาเหนือเส้นค่าเฉลี่ย",
    "tsmom": "โมเมนตัมตามเวลา (Time-series momentum)",
    "breakout": "เบรกเอาต์ (Donchian)",
    "macd": "MACD",
    "supertrend": "Supertrend (ATR)",
    "bollinger": "กลับตัว: Bollinger Bands",
    "rsi_revert": "กลับตัว: RSI",
    "trend_vote": "โหวตหลายเทรนด์ (พารามิเตอร์คงที่)",
    "rsi2_trend": "สั้น: RSI(2) ย่อตัวในขาขึ้น",
    "ibs": "สั้น: IBS ปิดอ่อนแรง",
    "pullback_low": "สั้น: ซื้อย่อที่ต่ำสุด n วัน",
    "short_breakout": "สั้น: เบรกเอาต์ระยะสั้น",
    "boll_pull": "สั้น: Bollinger ย่อตัวในขาขึ้น",
    "gtr_default": "GTR ค่าตั้งต้น (Supertrend+EMA, SL/TP ตาม ATR)",
    "gtr": "GTR ปรับพารามิเตอร์ (ฝึกจากช่วง Train)",
}

# (what it is, evidence / caveat)
STRATEGY_INFO = {
    "sma_cross": ("ซื้อเมื่อค่าเฉลี่ยเร็วตัดขึ้นเหนือค่าเฉลี่ยช้า ขายเมื่อกลับกัน",
                  "ตามเทรนด์แบบคลาสสิก ได้ผลเมื่อมีเทรนด์ยาว แต่โดนหลอกบ่อยในตลาดไซด์เวย์"),
    "sma_filter": ("ถือเมื่อราคาปิดอยู่เหนือค่าเฉลี่ย n วัน (เช่น 200 วัน) ไม่งั้นอยู่นอกตลาด",
                   "แนวคิดจากงานของ Faber (2007) เรื่อง tactical asset allocation เทรดน้อย ค่าใช้จ่ายต่ำ "
                   "ช่วยลด drawdown ในตลาดหมี แต่มักตามหลังตลาดในขาขึ้นแรง"),
    "tsmom": ("ถือเมื่อผลตอบแทน n วันที่ผ่านมาเป็นบวก",
              "อิงงาน Time series momentum (Moskowitz, Ooi, Pedersen 2012) ที่พบในหลายสินทรัพย์ "
              "แต่ผลในอดีตไม่การันตีว่ายังได้ผลต่อ"),
    "breakout": ("ซื้อเมื่อปิดทะลุสูงสุด n วัน ขายเมื่อหลุดต่ำสุด m วัน",
                 "แนวคิด Donchian channel / Turtle Traders ชนะน้อยครั้งแต่ได้กำไรต่อครั้งใหญ่ "
                 "ต้องทนแพ้ติดกันได้"),
    "macd": ("ซื้อเมื่อเส้น MACD ตัดขึ้นเหนือเส้นสัญญาณ",
             "ใช้แพร่หลายมาก แต่หลักฐานเชิงสถิติว่าชนะตลาดอย่างเสถียรค่อนข้างอ่อน "
             "โดยพื้นฐานเป็นตัวตามเทรนด์ที่ตอบสนองเร็ว"),
    "supertrend": ("เส้นตามหลังราคาโดยใช้ ATR ซื้อเมื่อราคาปิดเหนือเส้น ขายเมื่อหลุดเส้น",
                   "เป็น trailing stop ที่ปรับตามความผันผวน ช่วยให้ถือเทรนด์ได้นาน แต่ถูกเขย่าออกในตลาดแกว่ง "
                   "ต้องใช้ข้อมูล High/Low"),
    "bollinger": ("ซื้อเมื่อราคาปิดต่ำกว่าแถบล่าง ขายเมื่อกลับมาถึงเส้นกลาง",
                  "เป็นกลยุทธ์กลับตัว เหมาะตลาดแกว่งกรอบ แต่กับหุ้นเทรนด์แรงอย่างหุ้นเทค "
                  "อาจซื้อตอนราคายังร่วงต่อ ควรเทียบผลกับ buy & hold ให้ดี"),
    "rsi_revert": ("ซื้อเมื่อ RSI ต่ำมาก ขายเมื่อ RSI กลับมาสูง",
                   "แนวคิดกลับตัวระยะสั้นแบบ RSI(2) ของ Connors เทรดถี่ ต้นทุนมีผลมาก "
                   "และเสี่ยงโดนล้างเมื่อราคาตกต่อเนื่อง"),
    "trend_vote": ("รวมกฎเทรนด์ 4-5 ข้อ (SMA200, โมเมนตัม 12 เดือน, MACD, Breakout, Supertrend) "
                   "แล้วถือเมื่อเสียงข้างมากเป็นซื้อ",
                   "ไม่มีพารามิเตอร์ให้ปรับ จึงเสี่ยง overfit น้อยที่สุดในกลุ่ม และมักนิ่งกว่าตัวเดี่ยว "
                   "เป็นจุดเริ่มต้นที่สมเหตุสมผลสำหรับตั้งเป็นเกณฑ์อ้างอิง"),
    "rsi2_trend": ("ซื้อเมื่อ RSI(2) ต่ำมากขณะราคายังอยู่เหนือ SMA ยาว ขายเมื่อปิดเหนือ SMA5 หรือถือครบกำหนด",
                   "แนวคิดจากหนังสือ Short Term Trading Strategies That Work (Connors & Alvarez 2008) "
                   "หลักฐานส่วนใหญ่มาจากดัชนีและหุ้นใหญ่ในอดีต ข้อได้เปรียบอาจลดลงแล้ว "
                   "และต้นทุนต่อเทรดมีผลมาก"),
    "ibs": ("ซื้อเมื่อราคาปิดใกล้ราคาต่ำสุดของวัน (IBS ต่ำ) ขายเมื่อปิดเหนือราคาสูงสุดของวันก่อน",
            "เป็นกลยุทธ์กลับตัวที่นักเทรดเชิงปริมาณใช้ ส่วนใหญ่ทดสอบกับดัชนีและ ETF "
            "กับหุ้นเดี่ยวที่เทรนด์แรงหลักฐานน้อยกว่า ต้องมีข้อมูล High/Low"),
    "pullback_low": ("ซื้อเมื่อราคาปิดต่ำสุดในรอบ n วันขณะที่ยังอยู่เหนือ SMA200 ขายเมื่อฟื้นเหนือ SMA5 หรือถือครบกำหนด",
                     "ซื้อย่อในขาขึ้นแบบ swing ได้ผลดีเมื่อเทรนด์ใหญ่ยังอยู่ "
                     "แต่ถ้าเทรนด์เปลี่ยนจะซื้อตอนราคายังร่วง"),
    "short_breakout": ("ซื้อเมื่อปิดทะลุสูงสุด n วัน (สั้น) ขายเมื่อหลุดต่ำสุด m วันหรือถือครบกำหนด",
                       "เป็นการตามโมเมนตัมระยะสั้น มีสัญญาณหลอกเยอะและเทรดถี่ "
                       "ค่าธรรมเนียมและสลิปเพจกินกำไรง่าย"),
    "boll_pull": ("ซื้อเมื่อราคาปิดหลุดแถบล่าง Bollinger ขณะอยู่เหนือ SMA200 ขายเมื่อกลับถึงเส้นกลางหรือถือครบกำหนด",
                  "เวอร์ชันมีฟิลเตอร์เทรนด์และจำกัดเวลาถือ ของกลยุทธ์ Bollinger ตัวเดิม "
                  "ลดความเสี่ยงซื้อตอนเทรนด์ลง แต่ยังเป็นการสวนราคาระยะสั้น"),
    "gtr_default": ("แปลงจากอินดิเคเตอร์ GTR v2.0.0 (CrystalBoy) ซื้อเมื่อ Supertrend(10, 2.0) พลิกขึ้น หรือ EMA9 ตัดขึ้นเหนือ EMA21 "
                    "ขณะ RSI>50 โดยต้องมีความผันผวนพอ ตั้ง SL=1.5 ATR, ทยอยขายที่ TP1/2/3 = 1.2/2.4/3.6 ATR "
                    "ออกเมื่อ RSI หลุด 70, เกิดสัญญาณขาย หรือโดน SL",
                    "ใช้ค่าตั้งต้นของผู้เขียน ไม่มีการปรับ จึงเสี่ยง overfit น้อย จุดต่างจากต้นฉบับ: ซื้อที่ราคาเปิดวันถัดไป (ไม่ใช่ราคาปิด) "
                    "เทรดฝั่งซื้ออย่างเดียว ตรวจ SL ก่อน TP เมื่อชนทั้งคู่ในแท่งเดียว ต้นฉบับออกแบบมาสำหรับทองคำระยะสั้น "
                    "กับหุ้นรายวันอาจต่างไป และการทยอยขาย 3 ส่วนเป็นข้อสมมติของผม ไม่ได้อยู่ในต้นฉบับ"),
    "gtr": ("เอนจินเดียวกับ GTR ค่าตั้งต้น แต่เลือก ตัวคูณ Supertrend, เปิด/ปิด Early Entry, สเกล SL/TP และการรอราคาย้อน (retrace) "
            "จากช่วง Train",
            "ตัวเลือก retrace = ตั้ง Limit รอราคาย้อน 0.5 ATR ภายใน 3 วัน ตามคำแนะนำของผู้เขียนที่ว่า 'รอราคากลับมา retrace' "
            "ซึ่งทดสอบได้ว่าจริงไหม ปรับพารามิเตอร์มากขึ้น ความเสี่ยง overfit จึงสูงกว่าค่าตั้งต้น ดูผล Test และ walk-forward ประกอบ"),
}


def available_strategies(df):
    has_hl = {"High", "Low"} <= set(df.columns)
    return [n for n in STRATEGIES if has_hl or n not in NEEDS_HL]


def param_grid(space):
    keys = list(space)
    for vals in itertools.product(*space.values()):
        p = dict(zip(keys, vals))
        if "fast" in p and p["fast"] >= p["slow"]:
            continue
        if "n" in p and "exit_n" in p and p["exit_n"] >= p["n"]:
            continue
        yield p


# ---------- engine ----------
def _exec_price(df):
    return df["Open"] if "Open" in df.columns else df["Close"]


def run(df, signal, cost):
    """Daily strategy returns. Signal at close t -> trade at open t+1."""
    close, ex = df["Close"], _exec_price(df)
    s1 = signal.shift(1).fillna(0)  # position held during the day
    s2 = signal.shift(2).fillna(0)  # position held overnight into the day
    overnight = (ex / close.shift(1) - 1).fillna(0)
    intraday = (close / ex - 1).fillna(0)
    daily = s2 * overnight + s1 * intraday - (s1 - s2).abs() * cost
    return daily, s1


def trade_log(df, signal, cost):
    """One row per trade, net of costs. The last trade may still be open."""
    close, ex = df["Close"], _exec_price(df)
    s1 = signal.shift(1).fillna(0).astype(int)
    chg = s1.diff().fillna(s1)
    rows, entry = [], None
    for d, c in chg.items():
        if c == 1:
            entry = (d, float(ex[d]))
        elif c == -1 and entry is not None:
            xp = float(ex[d])
            rows.append((entry[0], entry[1], d, xp, xp / entry[1] - 1 - 2 * cost, "ปิดแล้ว"))
            entry = None
    if entry is not None:
        xp = float(close.iloc[-1])
        rows.append((entry[0], entry[1], close.index[-1], xp, xp / entry[1] - 1 - cost, "ถืออยู่"))
    out = pd.DataFrame(rows, columns=["entry_date", "entry_price", "exit_date", "exit_price",
                                      "return", "status"])
    # make dtypes explicit so an EMPTY table (no trades) still works with .dt
    out["entry_date"] = pd.to_datetime(out["entry_date"])
    out["exit_date"] = pd.to_datetime(out["exit_date"])
    for c in ["entry_price", "exit_price", "return"]:
        out[c] = out[c].astype(float)
    out["days"] = (out["exit_date"] - out["entry_date"]).dt.days
    return out


def metrics(daily, pos=None, trades=None):
    daily = daily.dropna()
    if len(daily) < 2:
        return {}
    eq = (1 + daily).cumprod()
    years = len(daily) / TRADING_DAYS
    m = {
        "total_return": eq.iloc[-1] - 1,
        "cagr": eq.iloc[-1] ** (1 / years) - 1,
        "max_drawdown": (eq / eq.cummax() - 1).min(),
        "sharpe": daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() > 0 else 0.0,
    }
    if pos is not None:
        m["time_in_market"] = float(pos.mean())
    if trades is not None:
        t = trades["return"].values if len(trades) else np.array([0.0])
        wins, losses = t[t > 0], t[t <= 0]
        m["trades"] = len(trades)
        m["win_rate"] = float((t > 0).mean()) if len(trades) else 0.0
        m["expectancy"] = float(t.mean()) if len(trades) else 0.0
        m["avg_win"] = float(wins.mean()) if len(wins) else 0.0
        m["avg_loss"] = float(losses.mean()) if len(losses) and len(trades) else 0.0
        m["profit_factor"] = float(wins.sum() / abs(losses.sum())) if losses.sum() != 0 else np.inf
        m["avg_days"] = float(trades["days"].mean()) if len(trades) else 0.0  # calendar days
    return m


def simulate(df, name, params, cost):
    """One strategy run -> (daily returns, position flag, trades, signal/target series, state or None)."""
    if name in ENGINE_STRATS:
        r = ENGINE_STRATS[name](df, cost=cost, **params)
        return r["daily"], r["pos"], r["trades"], r["target"], r["state"]
    sig = STRATEGIES[name][0](df, **params)
    daily, pos = run(df, sig, cost)
    return daily, pos, trade_log(df, sig, cost), sig, None


def pick_params(train, name, cost):
    """Grid-search the strategy's parameters on a TRAIN slice (best Sharpe). Returns (params, sharpe)."""
    _, space = STRATEGIES[name]
    best, best_s = None, -np.inf
    for p in param_grid(space):
        d, _, tl, _, _ = simulate(train, name, p, cost)
        s = metrics(d).get("sharpe", -np.inf)
        if len(tl) < MIN_TRAIN_TRADES:
            s -= 10  # too few trades to trust
        if s > best_s:
            best, best_s = p, s
    return best, best_s


def fit_strategy(df, split, name, cost):
    """Choose params on TRAIN (best Sharpe), then report TRAIN and TEST separately."""
    split = pd.Timestamp(split)
    train = df[df.index < split]
    best, _ = pick_params(train, name, cost)

    daily, pos, trades, signal, state = simulate(df, name, best, cost)  # indicators warmed up on full history
    tr_mask, te_mask = daily.index < split, daily.index >= split
    m_tr = metrics(daily[tr_mask], pos[tr_mask], trades[trades["entry_date"] < split])
    m_te = metrics(daily[te_mask], pos[te_mask], trades[trades["entry_date"] >= split])
    return {"name": name, "params": best, "signal": signal, "daily": daily, "pos": pos,
            "trades": trades, "train": m_tr, "test": m_te, "state": state}


def params_valid(p):
    """Human-readable problems with a hand-edited parameter set (empty list = fine)."""
    errs = []
    if "fast" in p and "slow" in p and p["fast"] >= p["slow"]:
        errs.append("ค่าเร็วต้องน้อยกว่าค่าช้า")
    if "exit_n" in p and "n" in p and p["exit_n"] >= p["n"]:
        errs.append("ช่วงออก (exit_n) ต้องสั้นกว่าช่วงเข้า (n)")
    if "low" in p and "exit_level" in p and p["low"] >= p["exit_level"]:
        errs.append("ระดับซื้อ (low) ต้องต่ำกว่าระดับขาย (exit_level)")
    for k in ("n", "exit_n", "fast", "slow", "sig", "trend_n", "max_hold", "mult", "factor", "k"):
        if k in p and p[k] <= 0:
            errs.append(f"{k} ต้องมากกว่า 0")
    return errs


def fit_fixed(df, split, name, params, cost):
    """Like fit_strategy but with parameters chosen by the user (no search). Works for engine strategies too."""
    split = pd.Timestamp(split)
    daily, pos, trades, signal, state = simulate(df, name, params, cost)
    tr_mask, te_mask = daily.index < split, daily.index >= split
    return {"name": name, "params": params, "signal": signal, "daily": daily, "pos": pos, "trades": trades,
            "train": metrics(daily[tr_mask], pos[tr_mask], trades[trades["entry_date"] < split]),
            "test": metrics(daily[te_mask], pos[te_mask], trades[trades["entry_date"] >= split]), "state": state}


def bh_test_metrics(df, split):
    bh = df["Close"].pct_change().fillna(0)
    bh = bh[bh.index >= pd.Timestamp(split)]
    return metrics(bh, pd.Series(1.0, index=bh.index), None)


def fit_from_signal(df, signal, split, cost, name="custom", params=None):
    """Same result shape as fit_strategy, but for a ready-made 0/1 signal (no parameter search)."""
    split = pd.Timestamp(split)
    daily, pos = run(df, signal, cost)
    trades = trade_log(df, signal, cost)
    tr_mask, te_mask = daily.index < split, daily.index >= split
    return {"name": name, "params": params or {}, "signal": signal, "daily": daily, "pos": pos, "trades": trades,
            "train": metrics(daily[tr_mask], pos[tr_mask], trades[trades["entry_date"] < split]),
            "test": metrics(daily[te_mask], pos[te_mask], trades[trades["entry_date"] >= split]), "state": None}


def trade_pnl(trades, capital, alloc=1.0, compound=True):
    """Money P/L per trade. alloc = share of the balance put into each trade; compound=False always sizes
    from the starting capital. `return` is already net of fees/slippage. An open trade shows unrealised P/L."""
    t = trades.copy().reset_index(drop=True)
    bal, rows = float(capital), []
    for r in t["return"].values:
        invest = min((bal if compound else capital) * alloc, bal)
        pnl = invest * r
        bal += pnl
        rows.append((invest, pnl, bal))
    t["invested"] = [x[0] for x in rows]
    t["pnl"] = [x[1] for x in rows]
    t["balance"] = [x[2] for x in rows]
    return t


def capital_summary(trades, close, start, capital, alloc=1.0, compound=True):
    """What would `capital` have become from `start` following the trades, versus just holding."""
    start = pd.Timestamp(start)
    p = trade_pnl(trades[trades["entry_date"] >= start], capital, alloc, compound)
    final = float(p["balance"].iloc[-1]) if len(p) else float(capital)
    c = close[close.index >= start]
    bh = float(capital * c.iloc[-1] / c.iloc[0]) if len(c) > 1 else float(capital)
    return {"final": final, "pnl": final - capital, "bh_final": bh, "bh_pnl": bh - capital, "n": len(p),
            "wins": int((p["pnl"] > 0).sum()) if len(p) else 0,
            "best": float(p["pnl"].max()) if len(p) else 0.0, "worst": float(p["pnl"].min()) if len(p) else 0.0}


def evaluate(df, split, cost):
    """Returns ({strategy: fit}, buy_hold_test_metrics, buy_hold_daily)."""
    split = pd.Timestamp(split)
    fits = {n: fit_strategy(df, split, n, cost) for n in available_strategies(df)}
    bh = df["Close"].pct_change().fillna(0)
    bh_te = bh[bh.index >= split]
    m_bh = metrics(bh_te, pd.Series(1.0, index=bh_te.index), None)
    return fits, m_bh, bh


def cost_stress(df, fit, split, cost, mults=(1, 2, 3)):
    """Re-run the chosen strategy's test period with higher per-side costs."""
    split = pd.Timestamp(split)
    rows = []
    for k in mults:
        if fit["name"] in STRATEGIES:
            d, pos, tl, _, _ = simulate(df, fit["name"], fit["params"], cost * k)
        else:  # user-built signal (custom system)
            d, pos = run(df, fit["signal"], cost * k)
            tl = trade_log(df, fit["signal"], cost * k)
        te = d.index >= split
        m = metrics(d[te], pos[te], tl[tl["entry_date"] >= split])
        rows.append({"mult": k, "cost_pct": cost * k * 100, "cagr": m["cagr"], "sharpe": m["sharpe"],
                     "expectancy": m["expectancy"], "total_return": m["total_return"]})
    return pd.DataFrame(rows)


def current_action(signal):
    """What to do at the NEXT open, given the signal at the latest close."""
    now = int(signal.iloc[-1])
    prev = int(signal.iloc[-2]) if len(signal) > 1 else 0
    if now == 1 and prev == 0:
        return "BUY", "สัญญาณซื้อใหม่ → เข้าซื้อตอนเปิดตลาดวันถัดไป"
    if now == 0 and prev == 1:
        return "SELL", "สัญญาณขาย → ขายตอนเปิดตลาดวันถัดไป"
    if now == 1:
        return "HOLD", "ถืออยู่ต่อ (ยังไม่มีสัญญาณขาย)"
    return "WAIT", "อยู่นอกตลาด รอสัญญาณซื้อ"


def current_action_for(fit):
    """Like current_action, but understands bracket-order engines (needs fit['signal'] and optional fit['state'])."""
    st_ = fit.get("state")
    if not st_:
        return current_action(fit["signal"])
    p = st_["pending"]
    if p == "entry":
        return "BUY", "สัญญาณซื้อใหม่ → เข้าซื้อตอนเปิดตลาดวันถัดไป (SL/TP คำนวณจากราคาที่ได้จริง)"
    if p == "entry_limit":
        return "BUY", (f"สัญญาณซื้อใหม่ → ตั้งคำสั่ง Limit ที่ {st_['limit']:,.2f} รอราคาย้อน "
                       f"ใช้ได้อีก {st_['wait_left']} วันทำการ ถ้าไม่ถึงให้ยกเลิก")
    if p == "exit":
        return "SELL", f"ขายตอนเปิดตลาดวันถัดไป ({st_['reason']})"
    if st_["status"] == "long":
        return "HOLD", "ถืออยู่ ตั้ง SL/TP ตามระดับด้านล่าง"
    return "WAIT", "อยู่นอกตลาด รอสัญญาณซื้อ"


def trigger_levels(name, params, df):
    """Human-readable distance to the next signal change, based on the latest close."""
    close = df["Close"]
    last = float(close.iloc[-1])
    out = []
    if name == "sma_cross":
        f, s_ = close.rolling(params["fast"]).mean().iloc[-1], close.rolling(params["slow"]).mean().iloc[-1]
        out += [(f"SMA{params['fast']} (เร็ว)", f"{f:,.2f}"), (f"SMA{params['slow']} (ช้า)", f"{s_:,.2f}"),
                ("ช่องว่างเร็ว-ช้า", f"{(f / s_ - 1) * 100:+.2f}%  (ซื้อเมื่อ > 0)")]
    elif name == "sma_filter":
        m = close.rolling(params["n"]).mean().iloc[-1]
        out += [(f"SMA{params['n']}", f"{m:,.2f}"),
                ("ราคาห่างจากเส้น", f"{(last / m - 1) * 100:+.2f}%  (ถือเมื่อ > 0)")]
    elif name == "tsmom":
        n = params["n"]
        ref = float(close.iloc[-n])
        out += [(f"ผลตอบแทน {n} วัน", f"{(last / close.iloc[-n - 1] - 1) * 100:+.1f}%  (ถือเมื่อ > 0)"),
                ("ราคาปิดพรุ่งนี้ต้องสูงกว่า (เพื่อคงสถานะบวก)", f"{ref:,.2f}")]
    elif name == "breakout":
        up, dn = close.iloc[-params["n"]:].max(), close.iloc[-params["exit_n"]:].min()
        out += [(f"ปิดเหนือสูงสุด {params['n']} วัน = ซื้อ", f"{up:,.2f}  ({(up / last - 1) * 100:+.1f}% จากราคาปิด)"),
                (f"ปิดต่ำกว่าต่ำสุด {params['exit_n']} วัน = ขาย", f"{dn:,.2f}  ({(dn / last - 1) * 100:+.1f}%)")]
    elif name == "macd":
        m, s_ = macd_calc(close, params["fast"], params["slow"], params["sig"])
        out += [("MACD", f"{m.iloc[-1]:.3f}"), ("เส้นสัญญาณ", f"{s_.iloc[-1]:.3f}"),
                ("Histogram", f"{m.iloc[-1] - s_.iloc[-1]:+.3f}  (ซื้อเมื่อ > 0)")]
    elif name == "supertrend":
        tr, line = supertrend(df, params["n"], params["mult"])
        ln = float(line.iloc[-1])
        out += [("เทรนด์ปัจจุบัน", "ขาขึ้น" if tr.iloc[-1] == 1 else "ขาลง"),
                ("เส้น Supertrend", f"{ln:,.2f}  ({(ln / last - 1) * 100:+.1f}% จากราคาปิด)"),
                ("กติกา", "ปิดหลุดเส้นขณะขาขึ้น = ขาย · ปิดเหนือเส้นขณะขาลง = ซื้อ")]
    elif name == "bollinger":
        mid, sd = close.rolling(params["n"]).mean().iloc[-1], close.rolling(params["n"]).std().iloc[-1]
        lo = mid - params["k"] * sd
        out += [("แถบล่าง (ซื้อเมื่อปิดต่ำกว่า)", f"{lo:,.2f}  ({(lo / last - 1) * 100:+.1f}%)"),
                ("เส้นกลาง (ขายเมื่อปิดสูงกว่า)", f"{mid:,.2f}  ({(mid / last - 1) * 100:+.1f}%)"),
                ("หมายเหตุ", "ระดับแถบเป็นค่าโดยประมาณ จะขยับตามราคาวันถัดไป")]
    elif name == "rsi_revert":
        r = rsi(close, params["n"]).iloc[-1]
        out += [(f"RSI({params['n']}) ล่าสุด", f"{r:.1f}"), ("ซื้อเมื่อ RSI ต่ำกว่า", f"{params['low']}"),
                ("ขายเมื่อ RSI สูงกว่า", f"{params['exit_level']}")]
    elif name in ("rsi2_trend", "pullback_low", "boll_pull", "ibs", "short_breakout"):
        sma5 = close.rolling(5).mean().iloc[-1]
        tn = params.get("trend_n", 200 if name != "ibs" else params.get("trend", 0))
        if name == "rsi2_trend":
            out.append(("RSI(2) ล่าสุด", f"{rsi(close, 2).iloc[-1]:.1f}  (ซื้อเมื่อ < {params['low']})"))
        elif name == "pullback_low":
            lvl = close.iloc[-(params["n"] - 1):].min()
            out.append((f"ซื้อเมื่อปิดไม่สูงกว่า (ต่ำสุด {params['n']} วัน)", f"{lvl:,.2f}  ({(lvl / last - 1) * 100:+.1f}%)"))
        elif name == "boll_pull":
            mid, sd = close.rolling(20).mean().iloc[-1], close.rolling(20).std().iloc[-1]
            lo = mid - params["k"] * sd
            out += [("แถบล่าง (ซื้อเมื่อปิดต่ำกว่า)", f"{lo:,.2f}  ({(lo / last - 1) * 100:+.1f}%)"),
                    ("เส้นกลาง (ขายเมื่อปิดสูงกว่า)", f"{mid:,.2f}  ({(mid / last - 1) * 100:+.1f}%)")]
        elif name == "ibs":
            cur = ibs(df).iloc[-1]
            out += [("IBS ล่าสุด", f"{cur:.2f}  (ซื้อเมื่อ < {params['low']})"),
                    ("ขายเมื่อปิดเหนือ (สูงสุดของวันล่าสุด)", f"{df['High'].iloc[-1]:,.2f}")]
        elif name == "short_breakout":
            up, dn = close.iloc[-params["n"]:].max(), close.iloc[-params["exit_n"]:].min()
            out += [(f"ปิดเหนือสูงสุด {params['n']} วัน = ซื้อ", f"{up:,.2f}  ({(up / last - 1) * 100:+.1f}%)"),
                    (f"ปิดต่ำกว่าต่ำสุด {params['exit_n']} วัน = ขาย", f"{dn:,.2f}  ({(dn / last - 1) * 100:+.1f}%)")]
        if name in ("rsi2_trend", "pullback_low", "boll_pull") or (name == "ibs" and tn):
            n_ = 200 if name in ("pullback_low", "boll_pull") else tn
            ma = close.rolling(n_).mean().iloc[-1]
            out.append((f"ฟิลเตอร์เทรนด์: ราคา > SMA{n_}", f"{'✅ ผ่าน' if last > ma else '❌ ไม่ผ่าน'} (SMA{n_} = {ma:,.2f})"))
        if name in ("rsi2_trend", "pullback_low"):
            out.append(("ขายเมื่อปิดเหนือ SMA5", f"{sma5:,.2f}  ({(sma5 / last - 1) * 100:+.1f}%)"))
        out.append(("ถือสูงสุด (time stop)", f"{params['max_hold']} วันทำการ แล้วออกอัตโนมัติ"))
    elif name in ("gtr", "gtr_default"):
        P = {**GTR_DEFAULTS, **params}
        stt = gtr_engine(df, **params)["state"]
        tdir, tline = supertrend(df, P["atr_len"], P["factor"])
        ln = float(tline.iloc[-1])
        out.append((f"Supertrend({P['atr_len']}, {P['factor']})",
                    f"{'ขาขึ้น' if tdir.iloc[-1] == 1 else 'ขาลง'} · เส้น {ln:,.2f} ({(ln / last - 1) * 100:+.1f}%)"))
        ef_, es_ = close.ewm(span=P["ema_fast"], adjust=False).mean().iloc[-1], close.ewm(span=P["ema_slow"], adjust=False).mean().iloc[-1]
        out.append((f"EMA{P['ema_fast']} / EMA{P['ema_slow']}", f"{ef_:,.2f} / {es_:,.2f} ({'ขาขึ้น' if ef_ > es_ else 'ขาลง'})"))
        out.append(("RSI(14)", f"{rsi(close, 14).iloc[-1]:.0f}  (Early Entry ต้อง > 50, ออกเมื่อหลุด {P['rsi_ob']})"))
        a14 = atr(df, 14)
        okv = a14.iloc[-1] > a14.rolling(50).mean().iloc[-1] * P["vol_mult"]
        out.append(("ตัวกรองความผันผวน (ATR)", "✅ ผ่าน" if okv else "❌ ไม่ผ่าน สัญญาณใหม่จะถูกข้าม"))
        if stt["status"] == "long":
            e_, sl_ = stt["entry_price"], stt["sl"]
            R_ = e_ - sl_
            out.append(("ราคาเข้า", f"{e_:,.2f}"))
            out.append(("SL", f"{sl_:,.2f} ({(sl_ / e_ - 1) * 100:+.1f}%)"))
            for k_, tp_ in enumerate(stt["tps"], 1):
                out.append((f"TP{k_}", f"{tp_:,.2f} ({(tp_ / e_ - 1) * 100:+.1f}% = {(tp_ - e_) / R_:.1f}R)"
                            + (" ✓ ถึงแล้ว" if stt["tp_hit"] >= k_ else "")))
        elif stt["preview"]:
            pv = stt["preview"]
            R_ = pv["entry"] - pv["sl"]
            out.append(("ถ้าเข้าตอนนี้ (คำนวณจากราคาปิดล่าสุด)", f"SL {pv['sl']:,.2f} ({(pv['sl'] / pv['entry'] - 1) * 100:+.1f}%)"))
            for k_ in (1, 2, 3):
                out.append((f"  TP{k_}", f"{pv[f'tp{k_}']:,.2f} ({(pv[f'tp{k_}'] - pv['entry']) / R_:.1f}R)"))
    elif name == "trend_vote":
        v = vote_components(df)
        for k, ser in v.items():
            out.append((k, "✅ ซื้อ" if ser.iloc[-1] == 1 else "❌ ไม่ซื้อ"))
        out.append(("รวม", f"{sum(int(x.iloc[-1]) for x in v.values())}/{len(v)} (ถือเมื่อเกินครึ่ง)"))
    return out


def overlay_lines(name, params, df):
    """Lines to draw on the price chart: list of (label, series, color, dash)."""
    c = df["Close"]
    if name == "sma_cross":
        return [(f"SMA{params['fast']}", c.rolling(params["fast"]).mean(), "blue", None),
                (f"SMA{params['slow']}", c.rolling(params["slow"]).mean(), "orange", None)]
    if name == "sma_filter":
        return [(f"SMA{params['n']}", c.rolling(params["n"]).mean(), "orange", None)]
    if name == "breakout":
        return [(f"สูงสุด {params['n']} วัน (ซื้อ)", c.rolling(params["n"]).max().shift(1), "green", "dot"),
                (f"ต่ำสุด {params['exit_n']} วัน (ขาย)", c.rolling(params["exit_n"]).min().shift(1), "red", "dot")]
    if name == "bollinger":
        mid, sd = c.rolling(params["n"]).mean(), c.rolling(params["n"]).std()
        return [("แถบบน", mid + params["k"] * sd, "grey", "dot"), ("เส้นกลาง (ขาย)", mid, "orange", None),
                ("แถบล่าง (ซื้อ)", mid - params["k"] * sd, "green", "dot")]
    if name == "supertrend":
        tr, line = supertrend(df, params["n"], params["mult"])
        return [("Supertrend ขาขึ้น", line.where(tr == 1), "green", None),
                ("Supertrend ขาลง", line.where(tr == -1), "red", None)]
    if name == "trend_vote":
        return [("SMA200", c.rolling(200).mean(), "orange", None)]
    if name in ("gtr", "gtr_default"):
        P = {**GTR_DEFAULTS, **params}
        tdir, tline = supertrend(df, P["atr_len"], P["factor"])
        return [("Supertrend ขาขึ้น", tline.where(tdir == 1), "green", None),
                ("Supertrend ขาลง", tline.where(tdir == -1), "red", None),
                (f"EMA{P['ema_fast']}", c.ewm(span=P["ema_fast"], adjust=False).mean(), "blue", "dot"),
                (f"EMA{P['ema_slow']}", c.ewm(span=P["ema_slow"], adjust=False).mean(), "orange", "dot")]
    if name in ("rsi2_trend", "pullback_low"):
        tn = params.get("trend_n", 200)
        return [(f"SMA{tn} (ฟิลเตอร์เทรนด์)", c.rolling(tn).mean(), "orange", None),
                ("SMA5 (จุดออก)", c.rolling(5).mean(), "blue", "dot")]
    if name == "ibs":
        return [(f"SMA{params['trend']} (ฟิลเตอร์เทรนด์)", c.rolling(params["trend"]).mean(), "orange", None)] \
            if params["trend"] else []
    if name == "boll_pull":
        mid, sd = c.rolling(20).mean(), c.rolling(20).std()
        return [("แถบล่าง (ซื้อ)", mid - params["k"] * sd, "green", "dot"), ("เส้นกลาง (ขาย)", mid, "blue", "dot"),
                ("SMA200", c.rolling(200).mean(), "orange", None)]
    if name == "short_breakout":
        return [(f"สูงสุด {params['n']} วัน (ซื้อ)", c.rolling(params["n"]).max().shift(1), "green", "dot"),
                (f"ต่ำสุด {params['exit_n']} วัน (ขาย)", c.rolling(params["exit_n"]).min().shift(1), "red", "dot")]
    return []  # tsmom, macd, rsi_revert: nothing on the price panel


def walk_forward(df, name, cost, train_years=5, test_months=12):
    """Rolling re-training. Every fold: pick params on the previous `train_years`, trade the next
    `test_months` with them (never seen before), then roll forward. All test segments are stitched
    into ONE out-of-sample track record. Also returns the params fitted on the most recent window,
    which is what you should use for today's live signal.
    Signal strategies are stitched exactly. Bracket-order engines (GTR) are re-simulated per fold with the
    fold's parameters over the whole history and only that fold's returns are kept (positions carried
    across a fold boundary are an approximation)."""
    n = len(df)
    tdays = int(train_years * TRADING_DAYS)
    sdays = max(21, int(test_months * TRADING_DAYS / 12))
    if n < tdays + sdays:
        raise ValueError("ข้อมูลไม่พอสำหรับ walk-forward ลองเลื่อนวันเริ่มต้นให้เก่าขึ้นหรือลดจำนวนปีฝึก")
    engine = name in ENGINE_STRATS
    sig = pd.Series(0, index=df.index, dtype=int)
    daily = pd.Series(0.0, index=df.index)
    pos = pd.Series(0.0, index=df.index)
    parts = []
    folds, i = [], tdays
    while i < n:
        j = min(i + sdays, n)
        best, s_tr = pick_params(df.iloc[i - tdays:i], name, cost)
        if engine:
            d_, p_, t_, sg_, _ = simulate(df.iloc[:j], name, best, cost)
            daily.iloc[i:j], pos.iloc[i:j], sig.iloc[i:j] = d_.iloc[i:j].values, p_.iloc[i:j].values, sg_.iloc[i:j].values
            parts.append(t_[(t_["entry_date"] >= df.index[i]) & (t_["entry_date"] <= df.index[j - 1])])
        else:
            sig.iloc[i:j] = STRATEGIES[name][0](df.iloc[:j], **best).iloc[i:j].values  # warmed up on history
        folds.append({"train_start": df.index[i - tdays], "test_start": df.index[i],
                      "test_end": df.index[j - 1], "params": best, "train_sharpe": s_tr})
        i = j
    if engine:
        trades = pd.concat(parts, ignore_index=True)
    else:
        daily, pos = run(df, sig, cost)
        trades = trade_log(df, sig, cost)
    for f in folds:
        seg = daily[(daily.index >= f["test_start"]) & (daily.index <= f["test_end"])]
        bseg = df["Close"].pct_change().fillna(0)[seg.index]
        f["test_return"] = float((1 + seg).prod() - 1)
        f["bh_return"] = float((1 + bseg).prod() - 1)
        f["test_sharpe"] = metrics(seg).get("sharpe", 0.0)
    oos0 = folds[0]["test_start"]
    mask = daily.index >= oos0
    bh = df["Close"].pct_change().fillna(0)[mask]
    latest, _ = pick_params(df.iloc[-tdays:], name, cost)
    _, _, _, latest_sig, latest_state = simulate(df, name, latest, cost)
    return {
        "name": name, "folds": folds, "signal": sig, "daily": daily[mask], "bh_daily": bh,
        "trades": trades[trades["entry_date"] >= oos0],
        "metrics": metrics(daily[mask], pos[mask], trades[trades["entry_date"] >= oos0]),
        "bh_metrics": metrics(bh, pd.Series(1.0, index=bh.index), None),
        "latest_params": latest, "latest_signal": latest_sig, "latest_state": latest_state,
    }


# ---------- market scanner (fixed, standard parameters: NOT optimised per stock) ----------
SCAN_SPECS = {
    "trend_vote": {},
    "sma_filter": {"n": 200},
    "tsmom": {"n": 252},
    "breakout": {"n": 50, "exit_n": 20},
    "macd": {"fast": 12, "slow": 26, "sig": 9},
    "supertrend": {"n": 10, "mult": 3.0},
    "rsi2_trend": {"low": 10, "trend_n": 200, "max_hold": 5},
    "pullback_low": {"n": 5, "max_hold": 10},
    "ibs": {"low": 0.2, "max_hold": 5, "trend": 200},
    "boll_pull": {"k": 2.0, "max_hold": 10},
    "short_breakout": {"n": 10, "exit_n": 5, "max_hold": 10},
    "gtr": {"factor": 2.0, "early": 1, "scale": 1.0, "retrace": 0.0},
}


def scan_ticker(df, names, specs=None, extra=None):
    """Today's state of each chosen strategy plus a few ranking stats.
    specs: {name: params} overriding SCAN_SPECS; extra: {name: callable(df) -> 0/1 signal} for user-built signals."""
    if len(df) < 260:
        return None
    c = df["Close"]
    has_hl = {"High", "Low"} <= set(df.columns)
    fresh, held = [], []
    for nm in names:
        try:
            if extra and nm in extra:
                sg = extra[nm](df)
            else:
                if nm in NEEDS_HL and not has_hl:
                    continue
                sg = STRATEGIES[nm][0](df, **((specs or {}).get(nm, SCAN_SPECS[nm])))
        except Exception:
            continue
        if int(sg.iloc[-1]) == 1:
            held.append(nm)
            if int(sg.iloc[-2]) == 0:
                fresh.append(nm)
    a = atr(df)
    vol = (c * df["Volume"]).iloc[-21:].mean() if "Volume" in df.columns else np.nan
    return {
        "close": float(c.iloc[-1]), "date": c.index[-1].date(),
        "ret_1m": float(c.iloc[-1] / c.iloc[-22] - 1), "ret_3m": float(c.iloc[-1] / c.iloc[-64] - 1),
        "ret_12m": float(c.iloc[-1] / c.iloc[-253] - 1),
        "from_high": float(c.iloc[-1] / c.iloc[-252:].max() - 1),
        "rsi14": float(rsi(c, 14).iloc[-1]),
        "above_sma200": bool(c.iloc[-1] > c.rolling(200).mean().iloc[-1]),
        "atr_pct": float(a.iloc[-1] / c.iloc[-1]) if a is not None else np.nan,
        "dollar_vol": float(vol),
        "n_fresh": len(fresh), "n_long": len(held), "fresh": fresh, "long": held,
    }


def rank_scan(rows):
    """rows: {ticker: scan_ticker() dict}. Rank by fresh BUY signals, then trend strength, then momentum."""
    df = pd.DataFrame.from_dict({k: v for k, v in rows.items() if v}, orient="index")
    if df.empty:
        return df
    df.index.name = "ticker"
    return df.sort_values(["n_fresh", "n_long", "ret_3m"], ascending=False).reset_index()


# ---------- CLI ----------
COLS = ["total_return", "cagr", "max_drawdown", "sharpe", "time_in_market",
        "trades", "win_rate", "expectancy", "profit_factor"]
PCT = {"total_return", "cagr", "max_drawdown", "time_in_market", "win_rate", "expectancy"}


def _fmt(m, k):
    if k not in m:
        return ""
    v = m[k]
    if k in PCT:
        return f"{v * 100:.1f}%"
    return f"{int(v)}" if k == "trades" else f"{v:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="NVDA")
    ap.add_argument("--start", default="2010-01-01")
    ap.add_argument("--split", default="2020-01-01", help="train < split <= test")
    ap.add_argument("--fee", type=float, default=0.001)
    ap.add_argument("--slippage", type=float, default=0.0005)
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    cost = args.fee + args.slippage

    df = load_prices(args.ticker, args.start, args.csv)
    fits, m_bh, _ = evaluate(df, args.split, cost)
    print(f"{args.ticker}: {df.index[0].date()} -> {df.index[-1].date()}  cost/side {cost * 100:.2f}%\n")
    print("=== OUT-OF-SAMPLE (TEST) ===")
    print(f"{'strategy':<12}" + "".join(f"{c:>15}" for c in COLS))
    for n, f in fits.items():
        print(f"{n:<12}" + "".join(f"{_fmt(f['test'], c):>15}" for c in COLS))
    print(f"{'buy&hold':<12}" + "".join(f"{_fmt(m_bh, c):>15}" for c in COLS))
    print()
    for n, f in fits.items():
        act = current_action(f["signal"])[1]
        print(f"{n:<12} {f['params']}  train_sharpe={f['train'].get('sharpe', 0):.2f} "
              f"test_sharpe={f['test'].get('sharpe', 0):.2f}  -> {act}")


if __name__ == "__main__":
    main()
