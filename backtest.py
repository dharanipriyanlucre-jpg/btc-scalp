"""
1m backtest of two Freqtrade strategies on several Delta symbols (long only, one position at a time,
unleveraged). Data is downloaded ONCE per symbol and both strategies run on it.

  Combined_Indicators   : BinHV45 + ClucMay72018 dip buy. Stoploss -6.58%, trailing stop, ROI 1.5%,
                          sell signal = close above Bollinger middle band.
  StrategyScalpingFast  : oversold scalp (open below EMA5-low, ADX>30, MFI<30, StochFast<30 cross up,
                          CCI<-150). ROI 1%, stoploss -50%, sell signals are DISABLED in the original,
                          so a trade that never reaches +1% is held until it does (bag holding).

Order inside one candle (conservative): stoploss -> trailing -> ROI -> sell signal.
Entry is at the OPEN of the candle after the signal. Your Delta fee + GST + slippage on both sides.
A second "stress" result repeats the same trades with 0.05% slippage per side.
Result goes to ntfy via env NTFY_TOPIC.

NOTE: 1m data is big (about 350 requests per symbol per year), so the download takes a while.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests
from numpy.lib.stride_tricks import sliding_window_view

# ---------------- CONFIG (change numbers here) ----------------
SYMBOLS = ["BTCUSD", "ETHUSD", "XRPUSD", "XAUTUSD"]   # a symbol Delta does not list is skipped
DAYS = 365             # 1m history length (730 gives more trades, just slower)
TF_MIN = 1
FEE = 0.0002           # 0.02% per side (Delta fee)
GST = 0.18             # 18% GST on the fee -> effective 0.0236% per side
SLIP = 0.0001          # 0.01% slippage per side (set 0 if you only use limit orders)
STRESS_SLIP = 0.0005   # slippage per side for the stress test (fast dips fill badly)
OOS_SPLIT = 0.70       # first 70% in-sample, last 30% out-of-sample

# pass rules (on the pooled trades of all symbols)
MIN_TRADES = 40
MIN_PF = 1.3
MIN_OOS_PF = 1.1
MIN_STRESS_PF = 1.0    # must still not lose money with the harsher slippage
MAX_MTM_DD = -25.0     # worst drawdown including open losses, per symbol

SIDE = FEE * (1 + GST) + SLIP
SIDE_STRESS = FEE * (1 + GST) + STRESS_SLIP


# ---------------- NTFY ----------------
def notify(title, message):
    topic = (os.getenv("NTFY_TOPIC") or "").strip()
    if not topic:
        print("WARNING: NTFY_TOPIC env var is empty - no ntfy notification sent.")
        return
    topic = topic.replace("https://ntfy.sh/", "")
    try:
        r = requests.post(f"https://ntfy.sh/{topic}", data=message.encode("utf-8"),
                          headers={"Title": title}, timeout=15)
        print(f"ntfy status: {r.status_code}")
    except Exception as ex:
        print("ntfy failed:", ex)


# ---------------- DATA ----------------
def fetch_delta(base, symbol):
    end = int(time.time())
    start = end - DAYS * 86400
    step = 1500 * TF_MIN * 60
    rows, s, calls = [], start, 0
    while s < end:
        e = min(s + step, end)
        for attempt in range(5):
            r = requests.get(
                f"{base}/v2/history/candles",
                params={"resolution": f"{TF_MIN}m", "symbol": symbol, "start": s, "end": e},
                timeout=30,
            )
            if r.status_code == 429:
                time.sleep(2 + attempt * 2)
                continue
            r.raise_for_status()
            break
        rows += r.json()["result"]
        s = e
        calls += 1
        if calls % 50 == 0:
            print(f"  {symbol}: {calls} requests, {len(rows)} candles so far")
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df[["time", "open", "high", "low", "close", "volume"]]
    return df.drop_duplicates("time").sort_values("time").set_index("time").astype(float)


def clean(df):
    """Drop bad candles (zeros, NaN, broken highs/lows, crazy spikes)."""
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)]
    df = df[df["high"] >= df["low"]].copy()
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    jump = (df["close"] / df["close"].shift() - 1).abs()
    df = df[~(jump > 0.2)]
    print(f"Cleaned {n0 - len(df)} bad candles out of {n0}")
    return df


def load_data(symbol):
    for name, base in [("Delta global", "https://api.delta.exchange"),
                       ("Delta India", "https://api.india.delta.exchange")]:
        try:
            df = fetch_delta(base, symbol)
            if len(df) > 5000:
                print(f"Data source: {name}, {len(df)} {symbol} candles")
                return clean(df), name
        except Exception as ex:
            print(f"{symbol} on {name} failed: {ex}")
    return None, None


# ---------------- INDICATORS ----------------
def crossed_above(a, b):
    return ((a > b) & (a.shift(1) <= b.shift(1))).fillna(False)


def adx(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    plus = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    minus = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = 1 / n
    atr = tr.ewm(alpha=a, adjust=False).mean()
    pdi = 100 * plus.ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * minus.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def cci(h, l, c, n=14):
    tp = (h + l + c) / 3
    sma = tp.rolling(n).mean()
    x = tp.values
    w = sliding_window_view(x, n)
    mad = np.abs(w - w.mean(axis=1, keepdims=True)).mean(axis=1)
    mad = pd.Series(np.r_[np.full(n - 1, np.nan), mad], index=tp.index)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))


def mfi(h, l, c, v, n=14):
    tp = (h + l + c) / 3
    rmf = tp * v
    d = tp.diff()
    pos = rmf.where(d > 0, 0.0).rolling(n).sum()
    neg = rmf.where(d < 0, 0.0).rolling(n).sum()
    return 100 * pos / (pos + neg).replace(0, np.nan)


# ---------------- STRATEGIES (same maths as the Freqtrade files) ----------------
def strat_combined(df):
    c, h, l, v = df["close"], df["high"], df["low"], df["volume"]
    # BinHV45
    mean40 = c.rolling(40).mean()
    lower40 = (mean40 - 2 * c.rolling(40).std()).fillna(0)
    mid40 = mean40.fillna(0)
    bbdelta = (mid40 - lower40).abs()
    closedelta = (c - c.shift()).abs()
    tail = (c - l).abs()
    s1 = ((lower40.shift() > 0) & (bbdelta > c * 0.00703) & (closedelta > c * 0.01476)
          & (tail < bbdelta * 0.03632) & (c < lower40.shift()) & (c <= c.shift()))
    # ClucMay72018
    typ = (h + l + c) / 3
    bb_mid = typ.rolling(20).mean()
    bb_lower = bb_mid - 2 * typ.rolling(20).std()
    ema50 = c.ewm(span=50, adjust=False).mean()
    s2 = (c < ema50) & (c < 0.98863 * bb_lower) & (v < v.rolling(30).mean().shift(1) * 34)
    buy = (s1 | s2).fillna(False).values
    sell = (c > bb_mid).fillna(False).values
    return buy, sell, int(s1.sum() + s2.sum())


def strat_scalp(df):
    o, h, l, c, v = (df[k] for k in ["open", "high", "low", "close", "volume"])
    ema_low = l.ewm(span=5, adjust=False).mean()
    ll, hh = l.rolling(5).min(), h.rolling(5).max()
    fastk = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    fastd = fastk.rolling(3).mean()
    buy = ((o < ema_low) & (adx(h, l, c) > 30) & (mfi(h, l, c, v) < 30) & (fastk < 30) & (fastd < 30)
           & crossed_above(fastk, fastd) & (cci(h, l, c) < -150))   # original uses CCI with default 14
    buy = buy.fillna(False).values
    return buy, None, int(buy.sum())          # original has use_sell_signal = False


STRATEGIES = {
    "Combined_Indicators": dict(fn=strat_combined, stop=0.0658, roi=0.015,
                                trail_off=0.03082, trail_pos=0.0198),
    "ScalpingFast": dict(fn=strat_scalp, stop=0.5, roi=0.01, trail_off=None, trail_pos=None),
}


# ---------------- ENGINE ----------------
def simulate(df, buy, sell, p):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    idx, n = df.index, len(df)
    trades, i = [], 0
    while i < n - 2:
        if not buy[i]:
            i += 1
            continue
        k = i + 1
        entry = o[k]
        stop, roi = entry * (1 - p["stop"]), entry * (1 + p["roi"])
        trig = entry * (1 + p["trail_off"]) if p["trail_off"] else None
        maxh, minl = entry, entry
        exit_px, reason, j = None, "open_at_end", k
        while j < n:
            minl, maxh = min(minl, l[j]), max(maxh, h[j])
            if l[j] <= stop:
                exit_px, reason = stop, "stop"
                break
            if trig and maxh >= trig and l[j] <= maxh * (1 - p["trail_pos"]):
                exit_px, reason = maxh * (1 - p["trail_pos"]), "trail"
                break
            if h[j] >= roi:
                exit_px, reason = roi, "roi"
                break
            if sell is not None and sell[j] and j + 1 < n:
                exit_px, reason, j = o[j + 1], "signal", j + 1
                break
            j += 1
        if exit_px is None:
            j, exit_px = n - 1, c[n - 1]
        net = (exit_px - entry) / entry - 2 * SIDE
        trades.append((idx[k], net, net - 2 * (SIDE_STRESS - SIDE), (minl - entry) / entry,
                       j - k + 1, reason))
        i = j
    return pd.DataFrame(trades, columns=["time", "ret", "ret_stress", "mae", "bars", "reason"])


def stats(t, col="ret"):
    if t is None or len(t) == 0:
        return dict(trades=0, win=0, pf=0, avg=0, ret=0, dd=0, mtm=0, worst=0, hold=0, maxhold=0)
    r = t[col]
    wins, losses = r[r > 0].sum(), -r[r <= 0].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = (1 + r).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    before = eq.shift(1).fillna(1.0)
    mtm = ((before * (1 + t.mae)) / np.maximum(before, peak.shift(1).fillna(1.0)) - 1).min()
    return dict(trades=len(t), win=(r > 0).mean() * 100, pf=pf, avg=r.mean() * 100,
                ret=(eq.iloc[-1] - 1) * 100, dd=dd * 100, mtm=min(mtm, dd) * 100,
                worst=r.min() * 100, hold=t.bars.mean() * TF_MIN,
                maxhold=t.bars.max() * TF_MIN / 1440)


def fmt(name, tag, s):
    return (f"{name:20s} {tag:6s} n={s['trades']:4d} win={s['win']:3.0f}% PF={s['pf']:.2f} "
            f"avg={s['avg']:+.2f}% ret={s['ret']:+.1f}% MTMdd={s['mtm']:.1f}% worst={s['worst']:.1f}% "
            f"hold={s['hold']:.0f}min maxhold={s['maxhold']:.1f}d")


# ---------------- MAIN ----------------
def main():
    data, skipped = {}, []
    for symbol in SYMBOLS:
        df, _ = load_data(symbol)
        if df is None:
            skipped.append(symbol)
        else:
            data[symbol] = df
    if not data:
        notify("Backtest FAILED", "No symbol could be downloaded. Check the workflow log.")
        sys.exit("No data")

    start = min(d.index[0] for d in data.values())
    end = max(d.index[-1] for d in data.values())
    split_time = start + (end - start) * OOS_SPLIT        # one common out-of-sample date
    print(f"\nSymbols: {list(data)} | skipped: {skipped} | OOS from {split_time.date()} | "
          f"cost/side {SIDE*100:.4f}% (stress {SIDE_STRESS*100:.4f}%)")

    out = [f"1m {DAYS}d unlevered | {len(data)} symbols" + (f" (skipped {', '.join(skipped)})" if skipped else "")]
    for sname, p in STRATEGIES.items():
        print(f"\n=========== {sname} ===========")
        all_t, per, worst_mtm = [], [], 0.0
        for symbol, df in data.items():
            buy, sell, nsig = p["fn"](df)
            t = simulate(df, buy, sell, p)
            t["symbol"] = symbol
            all_t.append(t)
            s = stats(t)
            worst_mtm = min(worst_mtm, s["mtm"])
            print(f"{symbol}: signals={nsig}")
            print(fmt(symbol, "ALL", s))
            per.append(f"{symbol}: sig={nsig} n={s['trades']} PF={s['pf']:.2f} ret={s['ret']:+.1f}% "
                       f"MTMdd={s['mtm']:.0f}%")
        T = pd.concat(all_t).sort_values("time")
        p_all, p_is, p_oos = stats(T), stats(T[T.time < split_time]), stats(T[T.time >= split_time])
        p_str = stats(T, "ret_stress")
        print("\nPOOLED")
        for lab, st in [("ALL", p_all), ("IS", p_is), ("OOS", p_oos), ("STRESS", p_str)]:
            print(fmt("POOLED", lab, st))
        if len(T):
            print("Exits:", T["reason"].value_counts().to_dict())
        ok = (p_all["trades"] >= MIN_TRADES and p_all["pf"] >= MIN_PF and p_oos["pf"] >= MIN_OOS_PF
              and p_str["pf"] >= MIN_STRESS_PF and worst_mtm >= MAX_MTM_DD)
        out.append(f"-- {sname}\n" + "\n".join(per)
                   + f"\n{'PASS' if ok else 'FAIL'} POOLED: n={p_all['trades']} win={p_all['win']:.0f}% "
                     f"PF={p_all['pf']:.2f} IS={p_is['pf']:.2f} OOS={p_oos['pf']:.2f} "
                     f"(OOS n={p_oos['trades']}) stressPF={p_str['pf']:.2f} "
                     f"worstMTMdd={worst_mtm:.0f}% worst={p_all['worst']:.0f}% maxhold={p_all['maxhold']:.0f}d")
    summary = "\n".join(out)
    print("\n" + summary)
    notify("Backtest result", summary)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as ex:          # any crash also goes to ntfy
        notify("Backtest ERROR", f"{type(ex).__name__}: {ex}"[:500])
        raise
