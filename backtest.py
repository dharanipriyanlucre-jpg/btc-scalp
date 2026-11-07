"""
BTCUSD daily-window backtest.
Window = 06:00 IST -> next day 06:00 IST (00:30 UTC -> 00:30 UTC).
Goal: at least 1 trade per day, max 2, all closed by the window end.

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
FEE = 0.0005           # 0.05% per side
SLIP = 0.0001          # 0.01% per side
SL_MULT = 0.5          # stop   = 0.5 x average daily range
TP_MULT = 0.3          # target = 0.3 x average daily range (smaller TP = higher win rate)
RISK_PER_TRADE = 0.01  # 1% of equity risked per trade (equity curve only)
MAX_TRADES_PER_DAY = 2
FORCE_BAR = 24         # setup C: if no trade yet, enter 12h after window start (18:00 IST)
OOS_SPLIT = 0.70       # first 70% in-sample, last 30% out-of-sample
WINDOW_START_UTC_MIN = 30   # 06:00 IST = 00:30 UTC

# pass rules
MIN_TRADES = 150
MIN_WIN = 60.0
MIN_PF = 1.2
MIN_OOS_PF = 1.0
MIN_COVERAGE = 0.90    # share of days with at least 1 trade

BARS_PER_DAY = 24 * 60 // TF_MIN


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
    df["ema_d"] = df["close"].ewm(span=BARS_PER_DAY, adjust=False).mean()
    g = df.groupby("win")
    w = pd.DataFrame({
        "open": g["open"].first(), "high": g["high"].max(),
        "low": g["low"].min(), "close": g["close"].last(), "n": g.size(),
    })
    w = w[w["n"] >= int(BARS_PER_DAY * 0.9)]          # full windows only
    w["datr"] = (w["high"] - w["low"]).rolling(14).mean().shift(1)   # avg daily range, known at start
    w["prev_ret"] = (w["close"] / w["open"] - 1).shift(1)            # previous window return
    ema10 = w["close"].ewm(span=10, adjust=False).mean()
    w["trend"] = np.where(w["close"].shift(1) > ema10.shift(1), 1, -1)  # trend known at start
    return df, w


# ---------------- SETUPS ----------------
# entry_fn(k, o, h, l, c, e, wi, n_tr) -> direction (1 long, -1 short, 0 none)
# k = candle index inside the window, entry is at the OPEN of candle k,
# only data before candle k may be used.
def setup_a_trend_open(k, o, h, l, c, e, wi, n_tr):
    """Every day at 06:00 IST trade in the direction of the daily trend."""
    return int(wi["trend"]) if k == 0 else 0


def setup_b_fade_prev_day(k, o, h, l, c, e, wi, n_tr):
    """Every day at 06:00 IST fade the previous window's move."""
    if k == 0 and wi["prev_ret"] != 0 and not np.isnan(wi["prev_ret"]):
        return int(-np.sign(wi["prev_ret"]))
    return 0


def setup_c_trend_pullback(k, o, h, l, c, e, wi, n_tr):
    """Trade trend on a dip to the daily EMA; if no trade by 18:00 IST, enter anyway."""
    t = int(wi["trend"])
    if k >= 1:
        if t == 1 and l[k - 1] <= e[k - 1]:
            return 1
        if t == -1 and h[k - 1] >= e[k - 1]:
            return -1
    if k == FORCE_BAR and n_tr == 0:
        return t
    return 0


SETUPS = {
    "A_trend_at_6am": setup_a_trend_open,
    "B_fade_prev_day": setup_b_fade_prev_day,
    "C_trend_pullback": setup_c_trend_pullback,
}


# ---------------- ENGINE ----------------
def run(df, w, entry_fn):
    groups = {k: v for k, v in df.groupby("win")}
    trades, n_windows = [], 0
    cost = 2 * (FEE + SLIP)
    for win, wi in w.iterrows():
        if np.isnan(wi["datr"]) or win not in groups:
            continue
        n_windows += 1
        seg = groups[win]
        o, h, l, c, e = (seg[x].values for x in ["open", "high", "low", "close", "ema_d"])
        idx = seg.index
        m, k, n_tr = len(seg), 0, 0
        while k < m and n_tr < MAX_TRADES_PER_DAY:
            d = entry_fn(k, o, h, l, c, e, wi, n_tr)
            if d == 0:
                k += 1
                continue
            entry = o[k]
            if not np.isfinite(entry) or entry <= 0 or not wi["datr"] > 0:
                k += 1
                continue
            sl = entry - d * SL_MULT * wi["datr"]
            tp = entry + d * TP_MULT * wi["datr"]
            risk_pct = SL_MULT * wi["datr"] / entry
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
            net = d * (exit_px - entry) / entry - cost
            if not np.isfinite(net / risk_pct):
                k = j + 1
                continue
            trades.append({"time": idx[k], "dir": d, "net_pct": net, "R": net / risk_pct})
            n_tr += 1
            k = j + 1
    return pd.DataFrame(trades), n_windows


def stats(t, n_windows=None):
    if t is not None and len(t):
        t = t[np.isfinite(t["R"])]
    if t is None or len(t) == 0:
        return dict(trades=0, win=0, pf=0, exp_R=0, ret=0, maxdd=0, cov=0, tpd=0)
    wins, losses = t.loc[t.R > 0, "R"].sum(), -t.loc[t.R <= 0, "R"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = (1 + RISK_PER_TRADE * t.R).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    days_with = t["time"].dt.floor("D").nunique() if n_windows else 0
    return dict(trades=len(t), win=(t.R > 0).mean() * 100, pf=pf, exp_R=t.R.mean(),
                ret=(eq.iloc[-1] - 1) * 100, maxdd=dd * 100,
                cov=(days_with / n_windows) if n_windows else 0,
                tpd=(len(t) / n_windows) if n_windows else 0)


def fmt(name, tag, s):
    return (f"{name:18s} {tag:4s} n={s['trades']:4d} win={s['win']:3.0f}% PF={s['pf']:.2f} "
            f"expR={s['exp_R']:+.2f} ret={s['ret']:+.1f}% dd={s['maxdd']:.1f}% "
            f"trades/day={s['tpd']:.2f} days_traded={s['cov']*100:.0f}%")


# ---------------- MAIN ----------------
def main():
    raw, src = load_data()
    df, w = prepare(raw)
    split_time = w.index[int(len(w) * OOS_SPLIT)]
    print(f"Range {w.index[0]} -> {w.index[-1]} | {len(w)} daily windows | OOS from {split_time}")
    print(f"SL {SL_MULT}x / TP {TP_MULT}x daily range | fee {FEE*100:.3f}% slip {SLIP*100:.3f}% per side\n")

    lines, passed = [], []
    for name, fn in SETUPS.items():
        t, nw = run(df, w, fn)
        s_all = stats(t, nw)
        if len(t):
            t_is, t_oos = t[t.time < split_time], t[t.time >= split_time]
            nw_is = int((w.index[w["datr"].notna()] < split_time).sum())
            nw_oos = nw - nw_is
        else:
            t_is = t_oos = t
            nw_is = nw_oos = 0
        s_is, s_oos = stats(t_is, nw_is), stats(t_oos, nw_oos)
        for tag, s in [("ALL", s_all), ("IS", s_is), ("OOS", s_oos)]:
            print(fmt(name, tag, s))
        print()
        ok = (s_all["trades"] >= MIN_TRADES and s_all["win"] >= MIN_WIN and s_all["pf"] >= MIN_PF
              and s_oos["pf"] >= MIN_OOS_PF and s_all["cov"] >= MIN_COVERAGE)
        if ok:
            passed.append(name)
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: n={s_all['trades']} win={s_all['win']:.0f}% "
                     f"PF={s_all['pf']:.2f} OOS_PF={s_oos['pf']:.2f} "
                     f"days={s_all['cov']*100:.0f}% ret={s_all['ret']:+.1f}%")

    summary = (f"BTC daily-window backtest {DAYS}d {TF_MIN}m ({src})\n" + "\n".join(lines)
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
    
