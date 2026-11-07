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
SYMBOLS = ["BTCUSD"]   # add "ETHUSD" here to get more trades (BTC rarely moves enough for these signals)
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
            print(f"{name} failed: {ex}")
    notify("Backtest FAILED", f"No data source worked for {symbol}. Check the workflow log.")
    sys.exit("No data source worked")


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
    lines, passed = [], []
    for symbol in SYMBOLS:
        df, src = load_data(symbol)
        buy, tag, sell, n1, n2 = build_signals(df)
        split_time = df.index[int(len(df) * OOS_SPLIT)]
        print(f"\n{symbol}: {len(df)} candles | signals: BinHV45={n1} Cluc={n2} | "
              f"OOS from {split_time.date()} | cost per side {SIDE*100:.4f}%")
        t = simulate(df, buy, tag, sell)
        t_is, t_oos = t[t.time < split_time], t[t.time >= split_time]
        s_all, s_is, s_oos = stats(t), stats(t_is), stats(t_oos)
        for lab, s in [("ALL", s_all), ("IS", s_is), ("OOS", s_oos)]:
            print(fmt(symbol, lab, s))
        if len(t):
            print("Exits:", t["reason"].value_counts().to_dict(), "| entries:", t["tag"].value_counts().to_dict())
        c = df["close"]
        bench = (f"buy&hold ret={(c.iloc[-1]/c.iloc[0]-1)*100:+.0f}% "
                 f"maxdd={(c/c.cummax()-1).min()*100:.0f}%")
        print(bench)
        ok = (s_all["trades"] >= MIN_TRADES and s_all["pf"] >= MIN_PF and s_oos["pf"] >= MIN_OOS_PF
              and s_all["mtm"] >= MAX_MTM_DD)
        if ok:
            passed.append(symbol)
        lines.append(f"{'PASS' if ok else 'FAIL'} {symbol}: signals={n1 + n2} n={s_all['trades']} "
                     f"win={s_all['win']:.0f}% PF={s_all['pf']:.2f} OOS={s_oos['pf']:.2f} "
                     f"ret={s_all['ret']:+.1f}% MTMdd={s_all['mtm']:.0f}% worst={s_all['worst']:.0f}% | {bench}")

    summary = (f"Combined_Indicators (BinHV45+Cluc) 1m {DAYS}d, unlevered\n" + "\n".join(lines)
               + f"\nPassed {len(passed)}/{len(SYMBOLS)}")
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
  
