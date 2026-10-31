"""
Statistical "AI assistant": a small logistic-regression model on technical features, trained on many stocks.

Question it answers:  "after buying at the next open, is the price higher than entry + hurdle after H trading days?"
It outputs a probability. It is NOT an oracle: the app only trusts it as far as its out-of-sample numbers allow.

Honesty rules baked in
  * Features at day t use only data up to day t (checked by a truncation test).
  * Training/test are split by TIME with an embargo of ~H days, so overlapping labels can't leak.
  * Two weight sets are stored:  `cut`  (trained only up to a cutoff, judged on the data after it)
    and `full` (trained on everything, used for today's score).  Performance is ALWAYS quoted from `cut`
    and the yearly walk-forward, never from `full`.
  * Plain JSON + numpy only: no pickle (safe to load), no scikit-learn needed.
"""
import json
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import backtest as bt

MODEL_DIR = Path(os.environ.get("BACKTEST_MODEL_DIR", "models"))
MODEL_FILE = MODEL_DIR / "ai_model.json"
FEATURES = ["ret_1", "ret_5", "ret_10", "ret_21", "ret_63", "ret_126", "vol_21", "vol_ratio", "rsi_14", "rsi_2",
            "dist_sma20", "dist_sma50", "dist_sma200", "sma50_200", "macd_hist", "bb_pos", "stoch_k", "atr_pct",
            "range_pct", "from_high_252", "from_low_252", "gap", "ibs", "vol_surge"]
FEATURE_TH = {
    "ret_1": "ผลตอบแทน 1 วัน", "ret_5": "ผลตอบแทน 5 วัน", "ret_10": "ผลตอบแทน 10 วัน", "ret_21": "ผลตอบแทน 1 เดือน",
    "ret_63": "ผลตอบแทน 3 เดือน", "ret_126": "ผลตอบแทน 6 เดือน", "vol_21": "ความผันผวน 21 วัน",
    "vol_ratio": "ความผันผวนสั้น/ยาว", "rsi_14": "RSI(14)", "rsi_2": "RSI(2)", "dist_sma20": "ห่างจาก SMA20",
    "dist_sma50": "ห่างจาก SMA50", "dist_sma200": "ห่างจาก SMA200", "sma50_200": "SMA50 เทียบ SMA200",
    "macd_hist": "MACD histogram", "bb_pos": "ตำแหน่งใน Bollinger", "stoch_k": "Stochastic %K", "atr_pct": "ATR % ของราคา",
    "range_pct": "ช่วงราคาวันนี้ %", "from_high_252": "ห่างจากสูงสุด 52 สัปดาห์", "from_low_252": "ห่างจากต่ำสุด 52 สัปดาห์",
    "gap": "ช่องว่างราคาเปิด", "ibs": "ราคาปิดในช่วงราคาวัน (IBS)", "vol_surge": "ปริมาณซื้อขายเทียบปกติ"}


# ---------------------------------------------------------------- features / labels
def make_features(df):
    """One row per day, using ONLY information available at that day's close."""
    c, h, l, o = df["Close"], df["High"], df["Low"], df["Open"]
    r1 = c.pct_change()
    f = pd.DataFrame(index=df.index)
    for n in (1, 5, 10, 21, 63, 126):
        f[f"ret_{n}"] = c.pct_change(n)
    f["vol_21"] = r1.rolling(21).std()
    f["vol_ratio"] = r1.rolling(5).std() / r1.rolling(63).std().replace(0, np.nan)
    f["rsi_14"], f["rsi_2"] = bt.rsi(c, 14), bt.rsi(c, 2)
    for n in (20, 50, 200):
        f[f"dist_sma{n}"] = c / c.rolling(n).mean() - 1
    f["sma50_200"] = c.rolling(50).mean() / c.rolling(200).mean() - 1
    m, s_ = bt.macd_calc(c, 12, 26, 9)
    f["macd_hist"] = (m - s_) / c
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    f["bb_pos"] = (c - mid) / (2 * sd.replace(0, np.nan))
    hh, ll = h.rolling(14).max(), l.rolling(14).min()
    f["stoch_k"] = (c - ll) / (hh - ll).replace(0, np.nan) * 100
    f["atr_pct"] = bt.atr(df, 14) / c
    f["range_pct"] = (h - l) / c
    f["from_high_252"] = c / c.rolling(252).max() - 1
    f["from_low_252"] = c / c.rolling(252).min() - 1
    f["gap"] = o / c.shift(1) - 1
    f["ibs"] = (c - l) / (h - l).replace(0, np.nan)
    if "Volume" in df.columns:
        v = df["Volume"].astype(float)
        f["vol_surge"] = np.log(v.replace(0, np.nan) / v.rolling(20).mean().replace(0, np.nan))
    else:
        f["vol_surge"] = np.nan
    return f[FEATURES].replace([np.inf, -np.inf], np.nan)


def forward_return(df, horizon):
    """Entry at the NEXT open, exit at the close `horizon` days after the signal day."""
    return df["Close"].shift(-horizon) / df["Open"].shift(-1) - 1


def build_dataset(frames, horizon=10, hurdle=0.003):
    parts = []
    for tk, df in frames.items():
        try:
            f = make_features(df)
        except Exception:
            continue
        fwd = forward_return(df, horizon)
        d = f.copy()
        d["fwd"] = fwd
        d["y"] = np.where(fwd.notna(), (fwd > hurdle).astype(float), np.nan)
        d["ticker"] = tk
        d["date"] = df.index
        parts.append(d)
    data = pd.concat(parts, ignore_index=True)
    return data.dropna(subset=FEATURES).reset_index(drop=True)


# ---------------------------------------------------------------- logistic regression (numpy only)
def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def fit_logistic(X, y, l2=2.0, iters=400, lr=0.05, seed=0):
    """Standardise, then L2-regularised logistic regression by full-batch Adam. Returns a plain dict."""
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    Z = np.clip((X - mu) / sd, -5, 5)
    n, d = Z.shape
    w, b = np.zeros(d), float(np.log(y.mean() / (1 - y.mean())))
    m_w, v_w, m_b, v_b = np.zeros(d), np.zeros(d), 0.0, 0.0
    for t in range(1, iters + 1):
        p = _sigmoid(Z @ w + b)
        g_w, g_b = Z.T @ (p - y) / n + l2 * w / n, float((p - y).mean())
        m_w, v_w = 0.9 * m_w + 0.1 * g_w, 0.999 * v_w + 0.001 * g_w ** 2
        m_b, v_b = 0.9 * m_b + 0.1 * g_b, 0.999 * v_b + 0.001 * g_b ** 2
        w -= lr * (m_w / (1 - 0.9 ** t)) / (np.sqrt(v_w / (1 - 0.999 ** t)) + 1e-8)
        b -= lr * (m_b / (1 - 0.9 ** t)) / (np.sqrt(v_b / (1 - 0.999 ** t)) + 1e-8)
    return {"mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist(), "b": float(b)}


def predict(params, X):
    Z = np.clip((np.asarray(X, float) - np.array(params["mean"])) / np.array(params["std"]), -5, 5)
    return _sigmoid(Z @ np.array(params["w"]) + params["b"])


def auc(score, y):
    """Rank-based AUC (0.5 = coin flip)."""
    y = np.asarray(y)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    ranks = pd.Series(score).rank().values
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def _quintiles(p, fwd, y, groups=None):
    """Mean forward return and hit rate per score quintile (quintiles formed within each group, e.g. a year)."""
    df = pd.DataFrame({"p": p, "fwd": fwd, "y": y, "g": groups if groups is not None else 0})
    df["q"] = df.groupby("g")["p"].transform(lambda s: pd.qcut(s.rank(method="first"), 5, labels=False))
    out = df.groupby("q").agg(mean_fwd=("fwd", "mean"), hit=("y", "mean"), n=("y", "size")).reset_index()
    return [{"q": int(r.q) + 1, "mean_fwd": float(r.mean_fwd), "hit": float(r.hit), "n": int(r.n)} for r in out.itertuples()]


def _slice_metrics(model_params, test):
    p = predict(model_params, test[FEATURES].values)
    return p, {"auc": auc(p, test["y"].values), "n": int(len(test)), "base_rate": float(test["y"].mean()),
               "all_mean_fwd": float(test["fwd"].mean()),
               "quintiles": _quintiles(p, test["fwd"].values, test["y"].values)}


def walk_forward_eval(data, horizon, min_train_years=4, l2=2.0):
    """Expanding window, one test year at a time, embargo ~horizon days between train and test."""
    years = sorted(data["date"].dt.year.unique())
    lab = data.dropna(subset=["y"])
    by_year, P, F, Y, G = [], [], [], [], []
    for Y_ in range(years[0] + min_train_years, years[-1] + 1):
        start = pd.Timestamp(f"{Y_}-01-01")
        train = lab[lab["date"] <= start - pd.Timedelta(days=int(horizon * 1.6) + 3)]
        test = lab[lab["date"].dt.year == Y_]
        if len(train) < 5000 or len(test) < 300:
            continue
        params = fit_logistic(train[FEATURES].values, train["y"].values, l2=l2)
        p, m = _slice_metrics(params, test)
        by_year.append({"year": int(Y_), "auc": m["auc"], "n": m["n"], "base_rate": m["base_rate"],
                        "top_mean": m["quintiles"][-1]["mean_fwd"], "all_mean": m["all_mean_fwd"],
                        "bottom_mean": m["quintiles"][0]["mean_fwd"]})
        P += list(p); F += list(test["fwd"]); Y += list(test["y"]); G += [Y_] * len(test)
    if not P:
        return {"auc": float("nan"), "by_year": [], "quintiles": [], "n": 0}
    return {"auc": auc(np.array(P), np.array(Y)), "by_year": by_year, "n": len(P),
            "quintiles": _quintiles(np.array(P), np.array(F), np.array(Y), np.array(G))}


def verdict(metrics):
    """Plain-language gate. The app shows AI scores prominently only if the numbers earn it."""
    ho, wf = metrics.get("holdout", {}), metrics.get("walk_forward", {})
    a_ho, a_wf = ho.get("auc", np.nan), wf.get("auc", np.nan)
    q = ho.get("quintiles") or []
    spread = (q[-1]["mean_fwd"] - q[0]["mean_fwd"]) if len(q) == 5 else np.nan
    yrs = wf.get("by_year", [])
    pos = sum(1 for y in yrs if y["auc"] > 0.5)
    if not (a_ho == a_ho and a_wf == a_wf):
        return "none", "ยังประเมินไม่ได้ (ข้อมูลไม่พอ)"
    if a_wf < 0.51 or a_ho < 0.50:
        return "fail", (f"ไม่ผ่าน: AUC walk-forward {a_wf:.3f}, holdout {a_ho:.3f} ใกล้เคียงการเดา (0.5) "
                        "อย่านำคะแนน AI มาประกอบการตัดสินใจ")
    if a_wf < 0.53 or not spread > 0 or pos < max(1, len(yrs) // 2):
        return "weak", (f"อ่อนมาก: AUC walk-forward {a_wf:.3f}, holdout {a_ho:.3f}, ปีที่ AUC>0.5 = {pos}/{len(yrs)} "
                        "มีสัญญาณเล็กน้อยแต่ไม่นิ่ง ใช้เป็นข้อมูลประกอบเท่านั้น")
    return "ok", (f"พอใช้เป็นข้อมูลประกอบ: AUC walk-forward {a_wf:.3f}, holdout {a_ho:.3f}, ปีที่ AUC>0.5 = {pos}/{len(yrs)} "
                  "(ยังไม่ใช่ตัวการันตี และเคยดีไม่ได้แปลว่าจะดีต่อ)")


# ---------------------------------------------------------------- training pipeline
def train_pipeline(frames, horizon=10, hurdle=0.003, holdout_days=365, l2=2.0, progress=None):
    """frames: {ticker: OHLCV DataFrame}. Returns the model dict (JSON-serialisable)."""
    say = progress or (lambda *_: None)
    say(0.1, "สร้างฟีเจอร์และป้ายคำตอบ...")
    data = build_dataset(frames, horizon, hurdle)
    lab = data.dropna(subset=["y"])
    if len(lab) < 20000:
        raise ValueError(f"ข้อมูลฝึกน้อยเกินไป ({len(lab):,} แถว) ลองเพิ่มจำนวนหุ้นหรือเริ่มข้อมูลให้เก่าขึ้น")
    last = lab["date"].max()
    cutoff = last - pd.Timedelta(days=holdout_days)
    say(0.3, "ประเมินแบบ walk-forward ทีละปี...")
    wf = walk_forward_eval(data, horizon, l2=l2)
    say(0.6, "ฝึกโมเดล cutoff (ใช้วัดผลจริง)...")
    tr = lab[lab["date"] <= cutoff - pd.Timedelta(days=int(horizon * 1.6) + 3)]
    ho = lab[lab["date"] > cutoff]
    cut = fit_logistic(tr[FEATURES].values, tr["y"].values, l2=l2)
    _, ho_m = _slice_metrics(cut, ho)
    say(0.85, "ฝึกโมเดลเต็ม (ใช้ให้คะแนนวันนี้)...")
    full = fit_logistic(lab[FEATURES].values, lab["y"].values, l2=l2)
    metrics = {"walk_forward": wf, "holdout": ho_m}
    lvl, txt = verdict(metrics)
    say(1.0, "เสร็จแล้ว")
    return {"version": 1, "trained_at": datetime.now().isoformat(timespec="seconds"), "horizon": horizon,
            "hurdle": hurdle, "features": FEATURES, "universe": sorted(frames), "n_rows": int(len(lab)),
            "data_range": [str(lab["date"].min().date()), str(last.date())], "cutoff": str(cutoff.date()),
            "base_rate": float(lab["y"].mean()), "full": full, "cut": cut, "metrics": metrics,
            "verdict": {"level": lvl, "text": txt}}


def save(model, path=MODEL_FILE):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(model, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, path)


def load(path=MODEL_FILE):
    try:
        return validate_model(json.loads(Path(path).read_text(encoding="utf-8")))
    except Exception:
        return None


def validate_model(m):
    """Reject files that are not a model we wrote (the file may come from an upload)."""
    if not isinstance(m, dict) or m.get("version") != 1 or m.get("features") != FEATURES:
        raise ValueError("ไฟล์โมเดลไม่ตรงรูปแบบ (เวอร์ชันหรือรายชื่อฟีเจอร์ไม่ตรง)")
    for k in ("full", "cut"):
        if len(m[k]["w"]) != len(FEATURES) or len(m[k]["mean"]) != len(FEATURES) or len(m[k]["std"]) != len(FEATURES):
            raise ValueError("ไฟล์โมเดลไม่ครบ")
    return m


# ---------------------------------------------------------------- using a model
def score(df, model, which="full"):
    """Probability series by date (NaN where features are missing)."""
    f = make_features(df)
    ok = f.notna().all(axis=1)
    out = pd.Series(np.nan, index=df.index)
    if ok.any():
        out[ok] = predict(model[which], f[ok].values)
    return out


def contributions(df, model, which="full"):
    """Why the latest score is what it is: per-feature contribution in logit units."""
    f = make_features(df).iloc[-1]
    if f.isna().any():
        return None
    p = model[which]
    z = np.clip((f.values - np.array(p["mean"])) / np.array(p["std"]), -5, 5)
    out = pd.DataFrame({"feature": FEATURES, "value": f.values, "z": z, "contrib": z * np.array(p["w"])})
    out["label"] = out["feature"].map(FEATURE_TH)
    return out.reindex(out["contrib"].abs().sort_values(ascending=False).index).reset_index(drop=True)


def band(prob, base_rate):
    """Describe a score relative to the model's own base rate."""
    d = prob - base_rate
    if d >= 0.04:
        return "สูง", 1
    if d <= -0.04:
        return "ต่ำ", -1
    return "กลาง ๆ", 0


# ---------------------------------------------------------------- data download (used by the app / CLI)
def download_history(tickers, start="2010-01-01", chunk=25, progress=None):
    import yfinance as yf

    out = {}
    for i in range(0, len(tickers), chunk):
        part = list(tickers[i:i + chunk])
        try:
            raw = yf.download(part, start=start, auto_adjust=True, group_by="ticker", threads=True, progress=False)
        except Exception:
            continue
        multi = isinstance(raw.columns, pd.MultiIndex)
        for t in part:
            try:
                d = bt._clean((raw[t] if multi else raw).dropna(how="all").copy())
                if len(d) > 600 and {"High", "Low", "Open"} <= set(d.columns):
                    out[t] = d
            except Exception:
                pass
        if progress:
            progress(min(1.0, (i + chunk) / len(tickers)), f"ดึงราคา {min(i + chunk, len(tickers))}/{len(tickers)}")
    return out


if __name__ == "__main__":  # python ml_model.py  -> trains on a sample of the S&P 500 and saves models/ai_model.json
    import universe

    try:
        tks = universe.fetch_sp500()["ticker"].tolist()
    except Exception:
        tks = universe.FALLBACK
    rng = np.random.default_rng(42)
    tks = sorted(rng.choice(tks, size=min(60, len(tks)), replace=False).tolist())
    fr = download_history(tks, progress=lambda f, m: print(f"{f * 100:3.0f}% {m}"))
    mdl = train_pipeline(fr, progress=lambda f, m: print(f"{f * 100:3.0f}% {m}"))
    save(mdl)
    print(mdl["verdict"]["text"])
