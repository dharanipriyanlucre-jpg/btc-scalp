"""
BTCUSD backtest - 3 setups on concepts not tried yet. Uses only free Delta public candles (no key).
  X1  ETH->BTC lead-lag     : when ETH has run far ahead of / behind BTC over 3h, BTC catches up
  X2  Time-of-day pattern   : 30-min slots with a statistically strong drift, picked on the
                              FIRST 70% of data only, then tested on the last 30% (the real test)
  X3  Volume climax reversal: huge volume + huge range candle closing at its low/high -> fade it

Max 2 trades per 06:00-to-06:00 IST window. Every trade has a stop and a fixed time exit.
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
OTHER = "ETHUSD"
DAYS = 730             # history length
TF_MIN = 30            # candle size in minutes
FEE = 0.0002           # 0.02% per side (Delta fee)
GST = 0.18             # 18% GST on the fee -> effective 0.0236% per side
SLIP = 0.0001          # 0.01% slippage per side (set 0 if you only use limit orders)
RISK_PER_TRADE = 0.01  # 1% of equity risked per trade (equity curve only)
MAX_TRADES_PER_DAY = 2
OOS_SPLIT = 0.70       # first 70% in-sample, last 30% out-of-sample
WINDOW_START_UTC_MIN = 30   # 06:00 IST = 00:30 UTC
SL_ATR = 2.0           # stop = 2 x ATR(14) of the 30m candle

# X1 lead-lag
X1_LOOKBACK = 6        # compare 6-candle (3h) returns
X1_Z = 2.0             # spread must be 2 std from normal
X1_Z_WINDOW = 336      # 1 week of 30m candles
X1_HOLD = 6            # exit after 3h
# X2 time-of-day
X2_HOLD = 4            # exit after 2h
X2_T = 3.0             # in-sample t-stat needed for a slot (strict, 48 slots are tested)
X2_MAX_SLOTS = 4
# X3 volume climax
X3_VOL_MULT = 3.0
X3_RANGE_MULT = 1.8
X3_AVG = 96            # 2 days of 30m candles
X3_CLV = 0.25          # close in the bottom/top 25% of the candle
X3_HOLD = 8            # exit after 4h

# pass rules
MIN_TRADES = 100
MIN_PF = 1.2
MIN_OOS_PF = 1.05
X2_MIN_OOS_TRADES = 40
X2_MIN_OOS_PF = 1.2    # X2 is picked on in-sample, so only the out-of-sample result counts

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
def fetch_delta(base, symbol):
    end = int(time.time())
    start = end - DAYS * 86400
    step = 1500 * TF_MIN * 60
    rows, s = [], start
    while s < end:
        e = min(s + step, end)
        r = requests.get(
            f"{base}/v2/history/candles",
            params={"resolution": f"{TF_MIN}m", "symbol": symbol, "start": s, "end": e},
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
    df["volume"] = df["volume"].fillna(0).clip(lower=0)
    jump = (df["close"] / df["close"].shift() - 1).abs()
    df = df[~(jump > 0.2)]
    print(f"Cleaned {n0 - len(df)} bad candles out of {n0}")
    return df


def load_data():
    for name, base in [("Delta global", "https://api.delta.exchange"),
                       ("Delta India", "https://api.india.delta.exchange")]:
        try:
            btc = fetch_delta(base, SYMBOL)
            if len(btc) < 2000:
                continue
            print(f"Data source: {name}, {len(btc)} {SYMBOL} candles")
            btc = clean(btc)
            eth = None
            try:
                eth = clean(fetch_delta(base, OTHER))
            except Exception as ex:
                print(f"{OTHER} failed: {ex}")
            return btc, eth, name
        except Exception as ex:
            print(f"{name} failed: {ex}")
    notify("Backtest FAILED", "No data source worked (Delta). Check the workflow log.")
    sys.exit("No data source worked")


# ---------------- FEATURES / SETUPS ----------------
def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


# each setup returns (long_signal, short_signal, hold_bars) evaluated at the CLOSE of a candle;
# the trade enters at the OPEN of the next candle.
def setup_x1_leadlag(btc, eth, a, split_time):
    e = eth["close"].reindex(btc.index)
    spread = (e / e.shift(X1_LOOKBACK) - 1) - (btc["close"] / btc["close"].shift(X1_LOOKBACK) - 1)
    z = spread / spread.rolling(X1_Z_WINDOW).std().replace(0, np.nan)
    return z >= X1_Z, z <= -X1_Z, X1_HOLD       # ETH ahead -> BTC catches up (long); ETH behind -> short


def setup_x2_timeofday(btc, eth, a, split_time):
    slot = pd.Series(btc.index.hour * 2 + btc.index.minute // 30, index=btc.index)
    fwd = btc["close"].shift(-(X2_HOLD - 1)) / btc["open"] - 1      # open of bar -> close of bar+hold-1
    train_end = split_time - pd.Timedelta(minutes=TF_MIN * X2_HOLD)  # no peeking past the split
    train = fwd[btc.index < train_end].dropna()
    g = train.groupby(slot.loc[train.index])
    mean, std, n = g.mean(), g.std(), g.count()
    t = (mean / (std / np.sqrt(n))).dropna()
    chosen = t[t.abs() >= X2_T].abs().sort_values(ascending=False).head(X2_MAX_SLOTS).index
    direction = {int(s): int(np.sign(mean[s])) for s in chosen}
    print(f"X2 slots picked on in-sample data (UTC slot = hour*2 + half hour): "
          f"{ {k: ('long' if v > 0 else 'short') for k, v in direction.items()} }")
    nxt = (slot + 1) % BARS_PER_DAY                  # slot of the candle we would enter on
    long = nxt.map(lambda s: direction.get(int(s), 0) == 1)
    short = nxt.map(lambda s: direction.get(int(s), 0) == -1)
    return long, short, X2_HOLD


def setup_x3_climax(btc, eth, a, split_time):
    rng = btc["high"] - btc["low"]
    big_vol = btc["volume"] > X3_VOL_MULT * btc["volume"].rolling(X3_AVG).mean().shift()
    big_rng = rng > X3_RANGE_MULT * rng.rolling(X3_AVG).mean().shift()
    clv = (btc["close"] - btc["low"]) / rng.replace(0, np.nan)
    climax = big_vol & big_rng
    return climax & (clv <= X3_CLV), climax & (clv >= 1 - X3_CLV), X3_HOLD   # capitulation -> buy, blow-off -> sell


SETUPS = {
    "X1_eth_btc_leadlag": (setup_x1_leadlag, True, False),
    "X2_time_of_day": (setup_x2_timeofday, False, True),
    "X3_volume_climax": (setup_x3_climax, False, False),
}   # name: (function, needs_eth, selected_on_in_sample)


# ---------------- ENGINE ----------------
def simulate(df, long, short, a, hold):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    ls, ss, av = long.fillna(False).values, short.fillna(False).values, a.values
    idx, n = df.index, len(df)
    day_key = (idx - pd.Timedelta(minutes=WINDOW_START_UTC_MIN)).floor("D")
    trades, per_day, i = [], {}, 0
    while i < n - 2:
        d = 1 if ls[i] else (-1 if ss[i] else 0)
        if d == 0 or not np.isfinite(av[i]) or av[i] <= 0:
            i += 1
            continue
        day = day_key[i + 1]
        if per_day.get(day, 0) >= MAX_TRADES_PER_DAY:
            i += 1
            continue
        entry = o[i + 1]
        sl = entry - d * SL_ATR * av[i]
        risk_pct = SL_ATR * av[i] / entry
        last = min(i + hold, n - 1)
        exit_px, j = c[last], last                  # default: close after the hold time
        for jj in range(i + 1, last + 1):
            if (l[jj] <= sl) if d == 1 else (h[jj] >= sl):
                exit_px, j = sl, jj
                break
        net = d * (exit_px - entry) / entry - COST
        R = net / risk_pct
        if np.isfinite(R):
            trades.append((idx[i + 1], d, net, R))
            per_day[day] = per_day.get(day, 0) + 1
        i = j
    return pd.DataFrame(trades, columns=["time", "dir", "net_pct", "R"])


def stats(t, n_days):
    if t is None or len(t) == 0 or n_days == 0:
        return dict(trades=0, win=0, pf=0, exp_R=0, ret=0, maxdd=0, cov=0)
    wins, losses = t.loc[t.R > 0, "R"].sum(), -t.loc[t.R <= 0, "R"].sum()
    pf = wins / losses if losses > 0 else float("inf")
    eq = (1 + RISK_PER_TRADE * t.R).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return dict(trades=len(t), win=(t.R > 0).mean() * 100, pf=pf, exp_R=t.R.mean(),
                ret=(eq.iloc[-1] - 1) * 100, maxdd=dd * 100,
                cov=t["time"].dt.floor("D").nunique() / n_days)


def fmt(name, tag, s):
    return (f"{name:19s} {tag:4s} n={s['trades']:4d} win={s['win']:3.0f}% PF={s['pf']:.2f} "
            f"expR={s['exp_R']:+.2f} ret={s['ret']:+.1f}% dd={s['maxdd']:.1f}% days_traded={s['cov']*100:.0f}%")


# ---------------- MAIN ----------------
def main():
    btc, eth, src = load_data()
    a = atr(btc)
    n_days = btc.index.floor("D").nunique()
    split_time = btc.index[int(len(btc) * OOS_SPLIT)]
    d_is = btc.index[btc.index < split_time].floor("D").nunique()
    print(f"{len(btc)} candles, {n_days} days | out-of-sample from {split_time.date()} | "
          f"round-trip cost {COST*100:.3f}%\n")

    lines, passed = [], []
    for name, (fn, needs_eth, sel_is) in SETUPS.items():
        if needs_eth and eth is None:
            print(f"{name}: skipped ({OTHER} data missing)\n")
            lines.append(f"SKIP {name}: {OTHER} data not available")
            continue
        long, short, hold = fn(btc, eth, a, split_time)
        t = simulate(btc, long, short, a, hold)
        t_is, t_oos = t[t.time < split_time], t[t.time >= split_time]
        s_all, s_is, s_oos = stats(t, n_days), stats(t_is, d_is), stats(t_oos, n_days - d_is)
        for tag, s in [("ALL", s_all), ("IS", s_is), ("OOS", s_oos)]:
            print(fmt(name, tag, s))
        print()
        if sel_is:   # picked on in-sample data, so only the out-of-sample result counts
            ok = s_oos["trades"] >= X2_MIN_OOS_TRADES and s_oos["pf"] >= X2_MIN_OOS_PF
        else:
            ok = s_all["trades"] >= MIN_TRADES and s_all["pf"] >= MIN_PF and s_oos["pf"] >= MIN_OOS_PF
        if ok:
            passed.append(name)
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: n={s_all['trades']} win={s_all['win']:.0f}% "
                     f"PF={s_all['pf']:.2f} IS={s_is['pf']:.2f} OOS={s_oos['pf']:.2f} "
                     f"(OOS n={s_oos['trades']}) ret={s_all['ret']:+.1f}%")

    summary = (f"BTC new-concept backtest 2 {DAYS}d {TF_MIN}m ({src})\n" + "\n".join(lines)
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
 
