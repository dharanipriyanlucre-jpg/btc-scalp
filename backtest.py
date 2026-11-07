"""
BTCUSD backtest - NostalgiaForInfinity-style DIP BUYING, long only, on one pair.

Idea (same as NFI): in an uptrend, buy when price is oversold after a dip, sell the small bounce.
  Trend filter : 1h EMA50 > 1h EMA200 and price above 1h EMA200 (last completed 1h candle only)
  Dip trigger  : RSI14 < 30 and close below the lower Bollinger Band (20, 2) on 30m candles
  Safe dip     : skip if price already fell more than 6% in the last 6 hours (crash guard)

Three ways to manage the trade, to show what NFI's style really does:
  V1_strict_stop : 1.5% target, hard 3% stop, 24h time exit           (normal risk control)
  V2_no_stop     : 1.5% target or RSI>70, NO stop (-20% doom exit), hold up to 14 days  (bag holding)
  V3_rebuy       : like V2 but buys more at -3% and -6%, target 1.5% above average entry,
                   doom exit at -12% from the first entry                                 (averaging down)

Returns are UNLEVERAGED. V3 reserves capital for 3 units. "MTM dd" is the worst drawdown including
open losses (what you would actually feel), not just closed trades.
Result goes to ntfy via env NTFY_TOPIC.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ---------------- CONFIG (change numbers here) ----------------
SYMBOL = "BTCUSD"
DAYS = 730             # history length
TF_MIN = 30            # candle size in minutes
FEE = 0.0002           # 0.02% per side (Delta fee)
GST = 0.18             # 18% GST on the fee -> effective 0.0236% per side
SLIP = 0.0001          # 0.01% slippage per side (set 0 if you only use limit orders)
OOS_SPLIT = 0.70       # first 70% in-sample, last 30% out-of-sample
MAX_TRADES_PER_DAY = 2
WINDOW_START_UTC_MIN = 30   # 06:00 IST = 00:30 UTC

# entry (fixed)
RSI_LEN, RSI_BUY = 14, 30
BB_LEN, BB_STD = 20, 2.0
CRASH_BARS, CRASH_DROP = 12, 0.06     # skip if price fell > 6% over the last 12 candles (6h)

VARIANTS = {
    "V1_strict_stop": dict(tp=0.015, stop=0.03, rsi_exit=None, rebuy=[], max_hold=48, cap=1),
    "V2_no_stop":     dict(tp=0.015, stop=0.20, rsi_exit=70, rebuy=[], max_hold=672, cap=1),
    "V3_rebuy":       dict(tp=0.015, stop=0.12, rsi_exit=70, rebuy=[0.03, 0.06], max_hold=672, cap=3),
}

# pass rules
MIN_TRADES = 40
MIN_PF = 1.3
MIN_OOS_PF = 1.1
MAX_MTM_DD = -25.0     # worst drawdown including open losses must be better than -25%

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
def fetch_delta(base):
    end = int(time.time())
    start = end - DAYS * 86400
    step = 1500 * TF_MIN * 60
    rows, s = [], start
    while s < end:
        e = min(s + step, end)
        r = requests.get(
            f"{base}/v2/history/candles",
            params={"resolution": f"{TF_MIN}m", "symbol": SYMBOL, "start": s, "end": e},
            timeout=30,
        )
        r.raise_for_status()
        rows += r.json()["result"]
        s = e
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
    jump = (df["close"] / df["close"].shift() - 1).abs()
    df = df[~(jump > 0.2)]
    print(f"Cleaned {n0 - len(df)} bad candles out of {n0}")
    return df


def load_data():
    for name, base in [("Delta global", "https://api.delta.exchange"),
                       ("Delta India", "https://api.india.delta.exchange")]:
        try:
            df = fetch_delta(base)
            if len(df) > 2000:
                print(f"Data source: {name}, {len(df)} candles")
                return clean(df), name
        except Exception as ex:
            print(f"{name} failed: {ex}")
    notify("Backtest FAILED", "No data source worked (Delta). Check the workflow log.")
    sys.exit("No data source worked")


# ---------------- INDICATORS / SIGNAL ----------------
def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(s, n=14):
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def build_signal(df):
    c = df["close"]
    r = rsi(c, RSI_LEN)
    ma, sd = c.rolling(BB_LEN).mean(), c.rolling(BB_LEN).std()
    lower = ma - BB_STD * sd
    # 1h trend filter using the last COMPLETED 1h candle only (no look-ahead)
    h1 = c.resample("1h").last().dropna()
    e50, e200 = ema(h1, 50), ema(h1, 200)
    trend = ((e50 > e200) & (h1 > e200)).astype(float).shift(1)
    trend30 = trend.reindex(df.index, method="ffill") == 1
    safe = (c / c.shift(CRASH_BARS) - 1) > -CRASH_DROP
    sig = trend30 & (r < RSI_BUY) & (c < lower) & safe
    return sig.fillna(False), r


# ---------------- ENGINE ----------------
def simulate(df, sig, r, v):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    rv, sv = r.values, sig.values
    idx, n = df.index, len(df)
    day_key = (idx - pd.Timedelta(minutes=WINDOW_START_UTC_MIN)).floor("D")
    trades, per_day, i = [], {}, 0
    while i < n - 2:
        if not sv[i]:
            i += 1
            continue
        k = i + 1
        day = day_key[k]
        if per_day.get(day, 0) >= MAX_TRADES_PER_DAY:
            i += 1
            continue
        first = o[k]
        fills, nxt = [first], 0
        last = min(k + v["max_hold"], n - 1)
        exit_px, reason, j = None, "time", k
        stop_px = first * (1 - v["stop"])
        while j <= last:
            if l[j] <= stop_px:                                   # 1) stop / doom exit first (conservative)
                exit_px, reason = stop_px, "stop"
                break
            while nxt < len(v["rebuy"]) and l[j] <= first * (1 - v["rebuy"][nxt]):
                fills.append(first * (1 - v["rebuy"][nxt]))       # 2) rebuy fills
                nxt += 1
            avg = float(np.mean(fills))
            tp_px = avg * (1 + v["tp"])
            if h[j] >= tp_px:                                     # 3) take profit
                exit_px, reason = tp_px, "tp"
                break
            if v["rsi_exit"] and rv[j] > v["rsi_exit"] and c[j] > avg * (1 + 2 * SIDE):
                exit_px, reason = c[j], "rsi"                     # 4) overbought exit at close
                break
            j += 1
        if exit_px is None:
            j, exit_px = last, c[last]
        units = len(fills)
        pnl = sum((exit_px - f) / f for f in fills) - 2 * SIDE * units
        mae = sum((l[k:j + 1].min() - f) / f for f in fills) / v["cap"]   # worst open loss on capital
        trades.append((idx[k], pnl / v["cap"], mae, j - k + 1, reason, units))
        per_day[day] = per_day.get(day, 0) + 1
        i = j
    return pd.DataFrame(trades, columns=["time", "ret", "mae", "bars", "reason", "units"])


def stats(t):
    if t is None or len(t) == 0:
        return dict(trades=0, win=0, pf=0, avg=0, ret=0, dd=0, mtm=0, worst=0, hold=0, stops=0)
    wins, losses = t.loc[t.ret > 0, "ret"].sum(), -t.loc[t.ret <= 0, "ret"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = (1 + t.ret).cumprod()
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    before = eq.shift(1).fillna(1.0)
    mtm = ((before * (1 + t.mae)) / np.maximum(before, peak.shift(1).fillna(1.0)) - 1).min()
    return dict(trades=len(t), win=(t.ret > 0).mean() * 100, pf=pf, avg=t.ret.mean() * 100,
                ret=(eq.iloc[-1] - 1) * 100, dd=dd * 100, mtm=min(mtm, dd) * 100,
                worst=t.ret.min() * 100, hold=t.bars.mean() * TF_MIN / 60,
                stops=(t.reason == "stop").mean() * 100)


def fmt(name, tag, s):
    return (f"{name:15s} {tag:4s} n={s['trades']:3d} win={s['win']:3.0f}% PF={s['pf']:.2f} "
            f"avg={s['avg']:+.2f}% ret={s['ret']:+.1f}% dd={s['dd']:.1f}% MTMdd={s['mtm']:.1f}% "
            f"worst={s['worst']:.1f}% hold={s['hold']:.0f}h stops={s['stops']:.0f}%")


# ---------------- MAIN ----------------
def main():
    df, src = load_data()
    sig, r = build_signal(df)
    split_time = df.index[int(len(df) * OOS_SPLIT)]
    print(f"{len(df)} candles | {int(sig.sum())} dip signals | OOS from {split_time.date()} | "
          f"cost per side {SIDE*100:.4f}%\n")

    c = df["close"]
    bh_ret = (c.iloc[-1] / c.iloc[0] - 1) * 100
    bh_dd = (c / c.cummax() - 1).min() * 100
    bench = f"Benchmark buy & hold BTC: ret={bh_ret:+.0f}% maxdd={bh_dd:.0f}%"
    print(bench + "\n")

    lines, passed = [], []
    for name, v in VARIANTS.items():
        t = simulate(df, sig, r, v)
        t_is, t_oos = t[t.time < split_time], t[t.time >= split_time]
        s_all, s_is, s_oos = stats(t), stats(t_is), stats(t_oos)
        for tag, s in [("ALL", s_all), ("IS", s_is), ("OOS", s_oos)]:
            print(fmt(name, tag, s))
        print()
        ok = (s_all["trades"] >= MIN_TRADES and s_all["pf"] >= MIN_PF and s_oos["pf"] >= MIN_OOS_PF
              and s_all["mtm"] >= MAX_MTM_DD)
        if ok:
            passed.append(name)
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: n={s_all['trades']} win={s_all['win']:.0f}% "
                     f"PF={s_all['pf']:.2f} OOS={s_oos['pf']:.2f} ret={s_all['ret']:+.1f}% "
                     f"MTMdd={s_all['mtm']:.0f}% worst={s_all['worst']:.0f}%")

    summary = (f"BTC NFI-style dip buy {DAYS}d {TF_MIN}m ({src}), unlevered\n" + "\n".join(lines)
               + f"\n{bench}\nPassed {len(passed)}/{len(VARIANTS)}")
    print(summary)
    notify("Backtest result", summary)


if __name__ == "__main__":
    try:
        main()
    except SystemExit:
        raise
    except Exception as ex:          # any crash also goes to ntfy
        notify("Backtest ERROR", f"{type(ex).__name__}: {ex}"[:500])
        raise
