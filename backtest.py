"""
BTCUSD daily-window backtest - TUNING Setup C (trend pullback).
Window = 06:00 IST -> next day 06:00 IST (00:30 UTC -> 00:30 UTC).
1 to 2 trades per day, every trade closed by the window end.

How the tuning stays honest:
  * every combo is ranked using only the first 70% of the data (in-sample)
  * the best ones are then shown on the last 30% (out-of-sample)
  * a setup only PASSES if the in-sample winner also works out-of-sample

Data: Delta Exchange public candles (no key), fallback Bybit.
Result goes to ntfy via env NTFY_TOPIC.
"""
import itertools
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

# parameter grid (stop/target are x average daily range)
SL_GRID = [0.3, 0.4, 0.5, 0.7, 1.0]
TP_GRID = [0.2, 0.3, 0.4, 0.6, 0.8]
SPAN_GRID = [24, 48, 96]       # pullback EMA length in candles (24 = 12h, 48 = 1 day, 96 = 2 days)
FORCE_GRID = [12, 24, 36]      # candle index to force an entry if no trade yet (12 = 12:00 IST)

# pass rules for the in-sample winner
MIN_IS_TRADES = 300
MIN_IS_PF = 1.15
MIN_OOS_PF = 1.05
MIN_COVERAGE = 0.90    # share of days with at least 1 trade

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
    for sp in SPAN_GRID:
        df[f"ema{sp}"] = df["close"].ewm(span=sp, adjust=False).mean()
    g = df.groupby("win")
    w = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "n": g.size(),
    })
    w = w[w["n"] >= int(BARS_PER_DAY * 0.9)]          # full windows only
    w["datr"] = (w["high"] - w["low"]).rolling(14).mean().shift(1)   # avg daily range known at start
    ema10 = w["close"].ewm(span=10, adjust=False).mean()
    w["trend"] = np.where(w["close"].shift(1) > ema10.shift(1), 1, -1)  # trend known at start

    # pre-extract numpy arrays per window (fast grid search)
    wins = []
    for win, wi in w.iterrows():
        if np.isnan(wi["datr"]):
            continue
        seg = df[df["win"] == win]
        wins.append(dict(
            win=win, idx=seg.index, datr=wi["datr"], trend=int(wi["trend"]),
            o=seg["open"].values, h=seg["high"].values, l=seg["low"].values,
            c=seg["close"].values, e={sp: seg[f"ema{sp}"].values for sp in SPAN_GRID},
        ))
    return wins, w


# ---------------- ENGINE (setup C: trend pullback) ----------------
def entry_dir(k, h, l, e, trend, force_bar, n_tr):
    """Long/short on a dip to the EMA in trend direction; forced entry if nothing by force_bar."""
    if k >= 1:
        if trend == 1 and l[k - 1] <= e[k - 1]:
            return 1
        if trend == -1 and h[k - 1] >= e[k - 1]:
            return -1
    if k == force_bar and n_tr == 0:
        return trend
    return 0


def run(wins, sl_m, tp_m, span, force_bar):
    trades = []
    for W in wins:
        o, h, l, c, e = W["o"], W["h"], W["l"], W["c"], W["e"][span]
        datr, trend, idx = W["datr"], W["trend"], W["idx"]
        m, k, n_tr = len(o), 0, 0
        while k < m and n_tr < MAX_TRADES_PER_DAY:
            d = entry_dir(k, h, l, e, trend, force_bar, n_tr)
            if d == 0:
                k += 1
                continue
            entry = o[k]
            if not np.isfinite(entry) or entry <= 0 or not datr > 0:
                k += 1
                continue
            sl = entry - d * sl_m * datr
            tp = entry + d * tp_m * datr
            risk_pct = sl_m * datr / entry
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


# ---------------- MAIN ----------------
def main():
    raw, src = load_data()
    wins, w = prepare(raw)
    split_time = wins[int(len(wins) * OOS_SPLIT)]["win"]
    nw_is = sum(1 for x in wins if x["win"] < split_time)
    nw_oos = len(wins) - nw_is
    print(f"{len(wins)} daily windows | in-sample {nw_is} days | out-of-sample {nw_oos} days "
          f"(from {split_time.date()})")
    print(f"Round-trip cost {COST*100:.3f}% (fee {FEE*100:.2f}% + {GST*100:.0f}% GST, slip {SLIP*100:.2f}% per side)")

    combos = list(itertools.product(SL_GRID, TP_GRID, SPAN_GRID, FORCE_GRID))
    print(f"Testing {len(combos)} parameter combos...\n")

    res = []
    for sl_m, tp_m, span, force in combos:
        t = run(wins, sl_m, tp_m, span, force)
        t_is, t_oos = t[t.time < split_time], t[t.time >= split_time]
        res.append(dict(p=(sl_m, tp_m, span, force), s_all=stats(t, len(wins)),
                        s_is=stats(t_is, nw_is), s_oos=stats(t_oos, nw_oos)))

    # how much of the grid is profitable at all (luck check)
    pf_is = np.array([r["s_is"]["pf"] for r in res])
    pf_oos = np.array([r["s_oos"]["pf"] for r in res])
    share_is = (pf_is > 1).mean() * 100
    share_oos = (pf_oos > 1).mean() * 100
    share_both = ((pf_is > 1) & (pf_oos > 1)).mean() * 100
    print(f"Combos with PF>1: in-sample {share_is:.0f}% | out-of-sample {share_oos:.0f}% | both {share_both:.0f}%")
    print(f"Median PF: in-sample {np.median(pf_is):.2f} | out-of-sample {np.median(pf_oos):.2f}\n")

    # rank by IN-SAMPLE profit factor only
    cands = [r for r in res if r["s_is"]["trades"] >= MIN_IS_TRADES and r["s_all"]["cov"] >= MIN_COVERAGE]
    cands.sort(key=lambda r: r["s_is"]["pf"], reverse=True)
    top = cands[:8]

    print("Top 8 by in-sample PF (SL, TP, EMA, force-bar):")
    for r in top:
        sl_m, tp_m, span, force = r["p"]
        a, i, o = r["s_all"], r["s_is"], r["s_oos"]
        print(f"SL{sl_m} TP{tp_m} EMA{span} F{force} | IS PF {i['pf']:.2f} | OOS PF {o['pf']:.2f} | "
              f"win {a['win']:.0f}% | n {a['trades']} | tr/day {a['tpd']:.2f} | days {a['cov']*100:.0f}% | "
              f"ret {a['ret']:+.1f}% | dd {a['maxdd']:.1f}%")

    if not top:
        summary = (f"BTC daily-window tuning (setup C) {DAYS}d ({src})\n"
                   f"No combo had enough trades/coverage.\nPF>1 in-sample {share_is:.0f}% | OOS {share_oos:.0f}%")
        print("\n" + summary)
        notify("Backtest result", summary)
        return

    best = top[0]
    sl_m, tp_m, span, force = best["p"]
    i, o, a = best["s_is"], best["s_oos"], best["s_all"]
    ok = (i["trades"] >= MIN_IS_TRADES and i["pf"] >= MIN_IS_PF and o["pf"] >= MIN_OOS_PF
          and a["cov"] >= MIN_COVERAGE)
    verdict = "PASS" if ok else "FAIL"

    summary = (
        f"BTC daily-window tuning (setup C) {DAYS}d {TF_MIN}m ({src})\n"
        f"Tested {len(combos)} combos | PF>1 in-sample {share_is:.0f}%, OOS {share_oos:.0f}%, both {share_both:.0f}%\n"
        f"Best in-sample: SL{sl_m} TP{tp_m} EMA{span} F{force}\n"
        f"{verdict}: IS PF {i['pf']:.2f} | OOS PF {o['pf']:.2f} | win {a['win']:.0f}% | "
        f"n={a['trades']} | days {a['cov']*100:.0f}% | ret {a['ret']:+.1f}% | dd {a['maxdd']:.1f}%"
    )
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
    
