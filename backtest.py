"""
BTCUSD 1H backtest - 3 low-frequency trend setups.
Replaces the old 5m scalp strategies (they all lost to fees).

Data: Delta Exchange public candles (no key), fallback Bybit.
Optional: set env NTFY_TOPIC to get the summary on your phone.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ---------------- CONFIG ----------------
SYMBOL = "BTCUSD"
DAYS = 180            # history length
FEE = 0.0005          # 0.05% per side
SLIP = 0.0001         # 0.01% per side
RISK_PER_TRADE = 0.01 # 1% of equity risked per trade (for equity curve)
MAX_HOLD_BARS = 48    # force exit after 48 hours
OOS_SPLIT = 0.70      # first 70% in-sample, last 30% out-of-sample


# ---------------- NTFY ----------------
def notify(title, message):
    topic = (os.getenv("NTFY_TOPIC") or "").strip()
    if not topic:
        print("WARNING: NTFY_TOPIC env var is empty - no ntfy notification sent. "
              "Pass it in the workflow yml (see env: block).")
        return
    topic = topic.replace("https://ntfy.sh/", "")
    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"),
                          headers={"Title": title}, timeout=15)
        print(f"ntfy status: {r.status_code}")
    except Exception as ex:
        print("ntfy failed:", ex)


# ---------------- DATA ----------------
def fetch_delta(base):
    end = int(time.time())
    start = end - DAYS * 86400
    step = 1500 * 3600
    rows, s = [], start
    while s < end:
        e = min(s + step, end)
        r = requests.get(
            f"{base}/v2/history/candles",
            params={"resolution": "1h", "symbol": SYMBOL, "start": s, "end": e},
            timeout=30,
        )
        r.raise_for_status()
        rows += r.json()["result"]
        s = e
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    return df[["time", "open", "high", "low", "close", "volume"]]


def fetch_bybit():
    end = int(time.time() * 1000)
    start = end - DAYS * 86400 * 1000
    rows, cur_end = [], end
    while cur_end > start:
        r = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params={"category": "linear", "symbol": "BTCUSDT", "interval": "60",
                    "end": cur_end, "limit": 1000},
            timeout=30,
        )
        r.raise_for_status()
        lst = r.json()["result"]["list"]
        if not lst:
            break
        rows += lst
        cur_end = int(lst[-1][0]) - 1
    df = pd.DataFrame(rows, columns=["time", "open", "high", "low", "close", "volume", "turnover"])
    df["time"] = pd.to_datetime(df["time"].astype("int64"), unit="ms")
    return df[["time", "open", "high", "low", "close", "volume"]]


def load_data():
    for name, fn in [
        ("Delta global", lambda: fetch_delta("https://api.delta.exchange")),
        ("Delta India", lambda: fetch_delta("https://api.india.delta.exchange")),
        ("Bybit", fetch_bybit),
    ]:
        try:
            df = fn()
            if len(df) > 500:
                print(f"Data source: {name}, {len(df)} candles")
                df = df.drop_duplicates("time").sort_values("time").set_index("time")
                return df.astype(float), name
        except Exception as ex:
            print(f"{name} failed: {ex}")
    notify("Backtest FAILED", "No data source worked (Delta/Bybit). Check the workflow log.")
    sys.exit("No data source worked")


# ---------------- INDICATORS ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


# ---------------- SETUPS ----------------
# Each returns (long_signal, short_signal, sl_atr_mult, tp_atr_mult)
# Signal on bar close -> entry at next bar open.
def setup_a_donchian(df):
    """Donchian 20 breakout, only in direction of EMA200 trend."""
    e200 = ema(df["close"], 200)
    hi = df["high"].rolling(20).max().shift()
    lo = df["low"].rolling(20).min().shift()
    long = (df["close"] > hi) & (df["close"] > e200)
    short = (df["close"] < lo) & (df["close"] < e200)
    return long, short, 1.5, 3.0


def setup_b_pullback(df):
    """Trend pullback: EMA50>EMA200, price dips to EMA21 zone, RSI turns up from <45."""
    c = df["close"]
    e21, e50, e200 = ema(c, 21), ema(c, 50), ema(c, 200)
    r = rsi(c)
    long = (e50 > e200) & (df["low"] <= e21) & (c > e21) & (r.shift() < 45) & (r > r.shift())
    short = (e50 < e200) & (df["high"] >= e21) & (c < e21) & (r.shift() > 55) & (r < r.shift())
    return long, short, 1.5, 2.5


def setup_c_squeeze(df):
    """Bollinger squeeze (width in lowest 20% of last 100 bars) then breakout with trend."""
    c = df["close"]
    ma = c.rolling(20).mean()
    sd = c.rolling(20).std()
    up, dn = ma + 2 * sd, ma - 2 * sd
    width = (up - dn) / ma
    squeezed = width.shift() <= width.rolling(100).quantile(0.2).shift()
    e100 = ema(c, 100)
    long = squeezed & (c > up) & (c > e100)
    short = squeezed & (c < dn) & (c < e100)
    return long, short, 1.5, 3.0


SETUPS = {
    "A_donchian_breakout": setup_a_donchian,
    "B_trend_pullback": setup_b_pullback,
    "C_bb_squeeze": setup_c_squeeze,
}


# ---------------- ENGINE ----------------
def run(df, long, short, sl_m, tp_m):
    a = atr(df).values
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    ls, ss = long.fillna(False).values, short.fillna(False).values
    idx = df.index
    n, i, trades = len(df), 0, []
    while i < n - 2:
        d = 1 if ls[i] else (-1 if ss[i] else 0)
        if d == 0 or np.isnan(a[i]):
            i += 1
            continue
        entry = o[i + 1]
        sl = entry - d * sl_m * a[i]
        tp = entry + d * tp_m * a[i]
        risk_pct = sl_m * a[i] / entry
        exit_px, j = None, i + 1
        while j < n:
            hit_sl = l[j] <= sl if d == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if d == 1 else l[j] <= tp
            if hit_sl:               # SL first if both hit in same candle (conservative)
                exit_px = sl
                break
            if hit_tp:
                exit_px = tp
                break
            if j - (i + 1) >= MAX_HOLD_BARS:
                exit_px = c[j]
                break
            j += 1
        if exit_px is None:
            break
        net = d * (exit_px - entry) / entry - 2 * (FEE + SLIP)
        trades.append({"time": idx[i + 1], "dir": d, "net_pct": net, "R": net / risk_pct})
        i = j
    return pd.DataFrame(trades)


def stats(t):
    if t is None or len(t) == 0:
        return dict(trades=0, win=0, pf=0, exp_R=0, ret=0, maxdd=0)
    wins, losses = t.loc[t.R > 0, "R"].sum(), -t.loc[t.R <= 0, "R"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = (1 + RISK_PER_TRADE * t.R).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(trades=len(t), win=(t.R > 0).mean() * 100, pf=pf,
                exp_R=t.R.mean(), ret=(eq.iloc[-1] - 1) * 100, maxdd=dd * 100)


def fmt(name, tag, s):
    return (f"{name:22s} {tag:4s} trades={s['trades']:3d} win={s['win']:4.0f}% "
            f"PF={s['pf']:.2f} expR={s['exp_R']:+.2f} ret={s['ret']:+.1f}% dd={s['maxdd']:.1f}%")


# ---------------- MAIN ----------------
def main():
    df, src = load_data()
    split_time = df.index[int(len(df) * OOS_SPLIT)]
    print(f"Range {df.index[0]} -> {df.index[-1]} | OOS starts {split_time}\n")

    lines, passed = [], []
    for name, fn in SETUPS.items():
        long, short, sl_m, tp_m = fn(df)
        t = run(df, long, short, sl_m, tp_m)
        s_all = stats(t)
        s_is = stats(t[t.time < split_time]) if len(t) else stats(None)
        s_oos = stats(t[t.time >= split_time]) if len(t) else stats(None)
        for tag, s in [("ALL", s_all), ("IS", s_is), ("OOS", s_oos)]:
            print(fmt(name, tag, s))
        print()
        ok = s_all["trades"] >= 25 and s_all["pf"] >= 1.2 and s_oos["pf"] >= 1.0
        if ok:
            passed.append(name)
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: n={s_all['trades']} "
                     f"PF={s_all['pf']:.2f} OOS_PF={s_oos['pf']:.2f} ret={s_all['ret']:+.1f}%")

    summary = f"BTC 1H backtest {DAYS}d ({src})\n" + "\n".join(lines) + \
              f"\nPassed {len(passed)}/{len(SETUPS)}"
    print(summary)

    notify("Backtest result", summary)


if __name__ == "__main__":
    main()
      
