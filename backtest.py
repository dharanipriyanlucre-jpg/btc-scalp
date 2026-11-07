"""
BTCUSD backtest - 3 NEW setups built on different concepts (no trend/EMA logic):
  S1  Liquidity sweep reversal : wick takes the previous day's high/low, closes back inside -> fade it
  S2  London opening-range breakout : breakout of the 12:30-13:30 IST range
  S3  VWAP mean reversion : stretched from the daily-anchored VWAP + rejection candle -> back to VWAP

Window = 06:00 IST -> next day 06:00 IST (00:30 UTC -> 00:30 UTC). Max 2 trades a day.
Setups only trade when their condition appears, so some days have no trade.
Parameters are FIXED (no grid search), so only 3 tests are run - less room for luck.

Data: Delta Exchange public candles (no key), fallback Bybit.
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
RISK_PER_TRADE = 0.01  # 1% of equity risked per trade (equity curve only)
MAX_TRADES_PER_DAY = 2
OOS_SPLIT = 0.70       # first 70% in-sample, last 30% out-of-sample
WINDOW_START_UTC_MIN = 30   # 06:00 IST = 00:30 UTC

# setup parameters (x = multiple of the average daily range, "datr")
RR = 1.5               # reward : risk for S1 and S2
SWEEP_BUFFER = 0.05    # S1: stop sits this far beyond the sweep wick
RISK_MIN, RISK_MAX = 0.10, 0.60   # S1/S2: skip trades whose stop is outside this range
ORB_START, ORB_END = 420, 480     # S2: 07:00-08:00 UTC = 12:30-13:30 IST (London open), minutes of day
ORB_WIDTH_MIN, ORB_WIDTH_MAX = 0.08, 0.60
VW_DEV = 0.35          # S3: price must be this far from VWAP
VW_SL = 0.25           # S3: stop distance
VW_MIN_BAR = 6         # S3: wait 3 hours after window start so VWAP is stable

# pass rules
MIN_TRADES = 120
MIN_PF = 1.2
MIN_OOS_PF = 1.05

BARS_PER_DAY = 24 * 60 // TF_MIN
COST = 2 * (FEE * (1 + GST) + SLIP)   # round-trip cost as a fraction of price


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
    return df[["time", "open", "high", "low", "close", "volume"]]


def fetch_bybit():
    end = int(time.time() * 1000)
    start = end - DAYS * 86400 * 1000
    rows, cur_end = [], end
    while cur_end > start:
        r = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params={"category": "linear", "symbol": "BTCUSDT", "interval": str(TF_MIN),
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


def load_data():
    for name, fn in [
        ("Delta global", lambda: fetch_delta("https://api.delta.exchange")),
        ("Delta India", lambda: fetch_delta("https://api.india.delta.exchange")),
        ("Bybit", fetch_bybit),
    ]:
        try:
            df = fn()
            if len(df) > 2000:
                print(f"Data source: {name}, {len(df)} candles")
                df = df.drop_duplicates("time").sort_values("time").set_index("time")
                return clean(df.astype(float)), name
        except Exception as ex:
            print(f"{name} failed: {ex}")
    notify("Backtest FAILED", "No data source worked (Delta/Bybit). Check the workflow log.")
    sys.exit("No data source worked")


# ---------------- PREP ----------------
def prepare(df):
    df = df.copy()
    df["win"] = (df.index - pd.Timedelta(minutes=WINDOW_START_UTC_MIN)).floor("D")
    g = df.groupby("win")
    w = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "n": g.size(),
    })
    w = w[w["n"] >= int(BARS_PER_DAY * 0.9)]          # full windows only
    w["datr"] = (w["high"] - w["low"]).rolling(14).mean().shift(1)   # avg daily range known at start
    w["pdh"] = w["high"].shift(1)                                    # previous window high
    w["pdl"] = w["low"].shift(1)                                     # previous window low

    wins = []
    for win, wi in w.iterrows():
        if np.isnan(wi["datr"]) or np.isnan(wi["pdh"]):
            continue
        seg = df[df["win"] == win]
        h, l, c, v = (seg[x].values for x in ["high", "low", "close", "volume"])
        if v.sum() <= 0:
            v = np.ones_like(v)
        typ = (h + l + c) / 3
        vwap = np.cumsum(typ * v) / np.cumsum(v)           # VWAP anchored at window start
        wins.append(dict(
            win=win, idx=seg.index, datr=wi["datr"], pdh=wi["pdh"], pdl=wi["pdl"],
            o=seg["open"].values, h=h, l=l, c=c, vwap=vwap,
            mod=(seg.index.hour * 60 + seg.index.minute).values,
        ))
    return wins, w


# ---------------- SETUPS ----------------
# signal(k, W, n_tr) -> None or (direction, stop_price, target_price)
# Entry is at the OPEN of candle k; only candles before k may be used.
def sig_sweep(k, W, n_tr):
    """S1 liquidity sweep: wick beyond previous day high/low, close back inside -> fade."""
    if k < 1:
        return None
    o, h, l, c, datr = W["o"], W["h"], W["l"], W["c"], W["datr"]
    entry = o[k]
    if h[k - 1] > W["pdh"] and c[k - 1] < W["pdh"]:
        sl = h[k - 1] + SWEEP_BUFFER * datr
        risk = sl - entry
        if RISK_MIN * datr <= risk <= RISK_MAX * datr:
            return -1, sl, entry - RR * risk
    if l[k - 1] < W["pdl"] and c[k - 1] > W["pdl"]:
        sl = l[k - 1] - SWEEP_BUFFER * datr
        risk = entry - sl
        if RISK_MIN * datr <= risk <= RISK_MAX * datr:
            return 1, sl, entry + RR * risk
    return None


def sig_orb(k, W, n_tr):
    """S2 London opening-range breakout (one trade per day)."""
    mod = W["mod"]
    if n_tr >= 1 or k < 1 or mod[k - 1] < ORB_END:
        return None
    sel = (mod[:k] >= ORB_START) & (mod[:k] < ORB_END)
    if sel.sum() < 2:
        return None
    rh, rl = W["h"][:k][sel].max(), W["l"][:k][sel].min()
    datr, entry, c1 = W["datr"], W["o"][k], W["c"][k - 1]
    if not (ORB_WIDTH_MIN * datr <= rh - rl <= ORB_WIDTH_MAX * datr):
        return None
    if c1 > rh:
        risk = entry - rl
        if risk > 0 and risk <= 0.8 * datr:
            return 1, rl, entry + RR * risk
    if c1 < rl:
        risk = rh - entry
        if risk > 0 and risk <= 0.8 * datr:
            return -1, rh, entry - RR * risk
    return None


def sig_vwap(k, W, n_tr):
    """S3 VWAP mean reversion: stretched from VWAP + rejection candle -> target VWAP."""
    if k < VW_MIN_BAR:
        return None
    datr, entry = W["datr"], W["o"][k]
    vw, c1, o1 = W["vwap"][k - 1], W["c"][k - 1], W["o"][k - 1]
    dev = (c1 - vw) / datr
    risk = VW_SL * datr
    if dev >= VW_DEV and c1 < o1 and entry - vw >= risk:
        return -1, entry + risk, vw
    if dev <= -VW_DEV and c1 > o1 and vw - entry >= risk:
        return 1, entry - risk, vw
    return None


SETUPS = {
    "S1_liquidity_sweep": sig_sweep,
    "S2_london_orb": sig_orb,
    "S3_vwap_reversion": sig_vwap,
}


# ---------------- ENGINE ----------------
def run(wins, signal_fn):
    trades = []
    for W in wins:
        o, h, l, c, idx = W["o"], W["h"], W["l"], W["c"], W["idx"]
        m, k, n_tr = len(o), 0, 0
        while k < m and n_tr < MAX_TRADES_PER_DAY:
            sig = signal_fn(k, W, n_tr)
            if sig is None:
                k += 1
                continue
            d, sl, tp = sig
            entry = o[k]
            valid = np.isfinite(entry) and entry > 0 and (
                (d == 1 and sl < entry < tp) or (d == -1 and tp < entry < sl))
            if not valid:
                k += 1
                continue
            risk_pct = abs(entry - sl) / entry
            exit_px, j = c[m - 1], m - 1          # default: close at window end
            for jj in range(k, m):
                hit_sl = l[jj] <= sl if d == 1 else h[jj] >= sl
                hit_tp = h[jj] >= tp if d == 1 else l[jj] <= tp
                if hit_sl:                         # SL first if both in one candle (conservative)
                    exit_px, j = sl, jj
                    break
                if hit_tp:
                    exit_px, j = tp, jj
                    break
            net = d * (exit_px - entry) / entry - COST
            R = net / risk_pct
            if np.isfinite(R):
                trades.append((idx[k], d, net, R))
            n_tr += 1
            k = j + 1
    return pd.DataFrame(trades, columns=["time", "dir", "net_pct", "R"])


def stats(t, n_windows):
    if t is None or len(t) == 0 or n_windows == 0:
        return dict(trades=0, win=0, pf=0, exp_R=0, ret=0, maxdd=0, cov=0, tpd=0)
    wins, losses = t.loc[t.R > 0, "R"].sum(), -t.loc[t.R <= 0, "R"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = (1 + RISK_PER_TRADE * t.R).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    days_with = t["time"].dt.floor("D").nunique()
    return dict(trades=len(t), win=(t.R > 0).mean() * 100, pf=pf, exp_R=t.R.mean(),
                ret=(eq.iloc[-1] - 1) * 100, maxdd=dd * 100,
                cov=days_with / n_windows, tpd=len(t) / n_windows)


def fmt(name, tag, s):
    return (f"{name:19s} {tag:4s} n={s['trades']:4d} win={s['win']:3.0f}% PF={s['pf']:.2f} "
            f"expR={s['exp_R']:+.2f} ret={s['ret']:+.1f}% dd={s['maxdd']:.1f}% "
            f"trades/day={s['tpd']:.2f} days_traded={s['cov']*100:.0f}%")


# ---------------- MAIN ----------------
def main():
    raw, src = load_data()
    wins, w = prepare(raw)
    split_time = wins[int(len(wins) * OOS_SPLIT)]["win"]
    nw_is = sum(1 for x in wins if x["win"] < split_time)
    nw_oos = len(wins) - nw_is
    print(f"{len(wins)} daily windows | in-sample {nw_is} days | out-of-sample {nw_oos} days "
          f"(from {split_time.date()})")
    print(f"Round-trip cost {COST*100:.3f}% (fee {FEE*100:.2f}% + {GST*100:.0f}% GST, "
          f"slip {SLIP*100:.2f}% per side)\n")

    lines, passed = [], []
    for name, fn in SETUPS.items():
        t = run(wins, fn)
        t_is, t_oos = t[t.time < split_time], t[t.time >= split_time]
        s_all, s_is, s_oos = stats(t, len(wins)), stats(t_is, nw_is), stats(t_oos, nw_oos)
        for tag, s in [("ALL", s_all), ("IS", s_is), ("OOS", s_oos)]:
            print(fmt(name, tag, s))
        print()
        ok = (s_all["trades"] >= MIN_TRADES and s_all["pf"] >= MIN_PF and s_oos["pf"] >= MIN_OOS_PF)
        if ok:
            passed.append(name)
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: n={s_all['trades']} win={s_all['win']:.0f}% "
                     f"PF={s_all['pf']:.2f} IS={s_is['pf']:.2f} OOS={s_oos['pf']:.2f} "
                     f"days={s_all['cov']*100:.0f}% ret={s_all['ret']:+.1f}%")

    summary = (f"BTC new-concept backtest {DAYS}d {TF_MIN}m ({src})\n" + "\n".join(lines)
               + f"\nPassed {len(passed)}/{len(SETUPS)}")
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
 
