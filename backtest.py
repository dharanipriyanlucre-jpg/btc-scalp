"""
BTCUSD 1m backtest - port of the Freqtrade strategy "Combined_Indicators"
(BinHV45 + ClucMay72018 dip buying, long only).

Same rules as the Freqtrade file:
  Buy  : BinHV45 condition OR Cluc condition on the signal candle's close, entry at the NEXT open
  Exit : stoploss -6.58%, trailing stop (1.98% trail once profit reached 3.082%), ROI 1.5%,
         or the sell signal (close > Bollinger middle band of typical price) executed at next open
  Order inside one candle (conservative): stoploss -> trailing -> ROI -> sell signal.

Single position at a time, unleveraged. Your Delta fee + GST + slippage is charged on both sides.
Result goes to ntfy via env NTFY_TOPIC.

NOTE: 1m history is big (about 350 requests per year of data), so the download takes a few minutes.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ---------------- CONFIG (change numbers here) ----------------
SYMBOLS = ["BTCUSD", "ETHUSD", "XRPUSD", "XAUTUSD"]   # a symbol Delta does not list is skipped
DAYS = 365             # 1m history length (730 works too, just slower)
TF_MIN = 1
FEE = 0.0002           # 0.02% per side (Delta fee)
GST = 0.18             # 18% GST on the fee -> effective 0.0236% per side
SLIP = 0.0001          # 0.01% slippage per side (set 0 if you only use limit orders)
OOS_SPLIT = 0.70       # first 70% in-sample, last 30% out-of-sample

# Freqtrade settings (copied from the strategy file)
ROI = 0.015
STOPLOSS = 0.0658
TRAIL_POS = 0.0198
TRAIL_OFFSET = 0.03082

# pass rules
MIN_TRADES = 40
MIN_PF = 1.3
MIN_OOS_PF = 1.1
MAX_MTM_DD = -25.0

SIDE = FEE * (1 + GST) + SLIP   # cost per side as a fraction of price


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


# ---------------- INDICATORS / SIGNALS (same maths as the Freqtrade file) ----------------
def build_signals(df):
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
    vmean = v.rolling(30).mean()
    s2 = (c < ema50) & (c < 0.98863 * bb_lower) & (v < vmean.shift(1) * 34)

    buy = (s1 | s2).fillna(False)
    tag = np.where(s1.fillna(False), "BinHV45", "Cluc")
    sell = (c > bb_mid).fillna(False)
    return buy.values, tag, sell.values, int(s1.sum()), int(s2.sum())


# ---------------- ENGINE ----------------
def simulate(df, buy, tag, sell):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    idx, n = df.index, len(df)
    trades, i = [], 0
    while i < n - 2:
        if not buy[i]:
            i += 1
            continue
        k = i + 1
        entry = o[k]
        stop, roi, trig = entry * (1 - STOPLOSS), entry * (1 + ROI), entry * (1 + TRAIL_OFFSET)
        maxh, minl = entry, entry
        exit_px, reason, j = None, "end", k
        while j < n:
            minl, maxh = min(minl, l[j]), max(maxh, h[j])
            if l[j] <= stop:
                exit_px, reason = stop, "stop"
                break
            if maxh >= trig and l[j] <= maxh * (1 - TRAIL_POS):
                exit_px, reason = maxh * (1 - TRAIL_POS), "trail"
                break
            if h[j] >= roi:
                exit_px, reason = roi, "roi"
                break
            if sell[j] and j + 1 < n:
                exit_px, reason, j = o[j + 1], "signal", j + 1   # sell signal fills at next open
                break
            j += 1
        if exit_px is None:
            j, exit_px = n - 1, c[n - 1]
        net = (exit_px - entry) / entry - 2 * SIDE
        trades.append((idx[k], net, (minl - entry) / entry, j - k + 1, reason, tag[i]))
        i = j
    return pd.DataFrame(trades, columns=["time", "ret", "mae", "bars", "reason", "tag"])


def stats(t):
    if t is None or len(t) == 0:
        return dict(trades=0, win=0, pf=0, avg=0, ret=0, dd=0, mtm=0, worst=0, hold=0)
    wins, losses = t.loc[t.ret > 0, "ret"].sum(), -t.loc[t.ret <= 0, "ret"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = (1 + t.ret).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    before = eq.shift(1).fillna(1.0)
    mtm = ((before * (1 + t.mae)) / np.maximum(before, peak.shift(1).fillna(1.0)) - 1).min()
    return dict(trades=len(t), win=(t.ret > 0).mean() * 100, pf=pf, avg=t.ret.mean() * 100,
                ret=(eq.iloc[-1] - 1) * 100, dd=dd * 100, mtm=min(mtm, dd) * 100,
                worst=t.ret.min() * 100, hold=t.bars.mean() * TF_MIN)


def fmt(name, tag, s):
    return (f"{name:12s} {tag:4s} n={s['trades']:4d} win={s['win']:3.0f}% PF={s['pf']:.2f} "
            f"avg={s['avg']:+.2f}% ret={s['ret']:+.1f}% dd={s['dd']:.1f}% MTMdd={s['mtm']:.1f}% "
            f"worst={s['worst']:.1f}% hold={s['hold']:.0f}min")


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
    split_time = start + (end - start) * OOS_SPLIT        # one common out-of-sample date for all symbols
    print(f"\nSymbols: {list(data)} | skipped: {skipped} | OOS from {split_time.date()} | "
          f"cost per side {SIDE*100:.4f}%")

    lines, all_t, worst_mtm = [], [], 0.0
    for symbol, df in data.items():
        buy, tag, sell, n1, n2 = build_signals(df)
        t = simulate(df, buy, tag, sell)
        t["symbol"] = symbol
        all_t.append(t)
        s_all = stats(t)
        s_oos = stats(t[t.time >= split_time])
        worst_mtm = min(worst_mtm, s_all["mtm"])
        c = df["close"]
        bh = (c.iloc[-1] / c.iloc[0] - 1) * 100
        print(f"\n{symbol}: {len(df)} candles | signals BinHV45={n1} Cluc={n2}")
        print(fmt(symbol, "ALL", s_all))
        print(fmt(symbol, "OOS", s_oos))
        print(f"buy&hold ret={bh:+.0f}%")
        lines.append(f"{symbol}: signals={n1 + n2} n={s_all['trades']} win={s_all['win']:.0f}% "
                     f"PF={s_all['pf']:.2f} ret={s_all['ret']:+.1f}% MTMdd={s_all['mtm']:.0f}% "
                     f"(buy&hold {bh:+.0f}%)")

    T = pd.concat(all_t).sort_values("time")
    p_all = stats(T)
    p_is, p_oos = stats(T[T.time < split_time]), stats(T[T.time >= split_time])
    print("\nPOOLED (all symbols together)")
    for lab, st in [("ALL", p_all), ("IS", p_is), ("OOS", p_oos)]:
        print(fmt("POOLED", lab, st))
    if len(T):
        print("Exits:", T["reason"].value_counts().to_dict())

    ok = (p_all["trades"] >= MIN_TRADES and p_all["pf"] >= MIN_PF and p_oos["pf"] >= MIN_OOS_PF
          and worst_mtm >= MAX_MTM_DD)
    verdict = "PASS" if ok else "FAIL"
    summary = (f"Combined_Indicators 1m {DAYS}d, unlevered, {len(data)} symbols"
               + (f" (skipped {', '.join(skipped)})" if skipped else "") + "\n"
               + "\n".join(lines)
               + f"\n{verdict} POOLED: n={p_all['trades']} win={p_all['win']:.0f}% PF={p_all['pf']:.2f} "
                 f"IS={p_is['pf']:.2f} OOS={p_oos['pf']:.2f} (OOS n={p_oos['trades']}) "
                 f"worst MTMdd={worst_mtm:.0f}% worst trade={p_all['worst']:.0f}%")
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
  
