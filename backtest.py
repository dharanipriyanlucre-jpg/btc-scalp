#!/usr/bin/env python3
"""
backtest.py - compare scalping strategies on BTCUSD (Delta Exchange candles).

- Signals are read at candle CLOSE, entry is at the NEXT candle OPEN.
- SL = sl_mult x ATR(14), TP = RR x SL. If SL and TP are both hit in one
  candle, SL is assumed first (conservative).
- Fees + slippage are charged on both sides.
- Every strategy is swept over several SL/RR settings, then split into
  in-sample (first 70%) and out-of-sample (last 30%) to spot curve fitting.

Add your own strategy: write a function f(df) -> array of 1 / -1 / 0
and register it in STRATEGIES.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

BASE = os.getenv("DELTA_BASE", "https://api.delta.exchange")
SYMBOL = os.getenv("SYMBOL", "BTCUSD")
TF = os.getenv("TIMEFRAME", "5m")
DAYS = int(os.getenv("DAYS", "30"))
FEE = float(os.getenv("FEE_PCT", "0.05")) / 100   # per side (taker)
SLIP = float(os.getenv("SLIP_PCT", "0.01")) / 100  # per side
MAX_HOLD = int(os.getenv("MAX_HOLD_BARS", "48"))
MIN_TRADES = int(os.getenv("MIN_TRADES", "30"))
SL_MULTS = [1.0, 1.5, 2.0]
RRS = [1.5, 2.0, 3.0]
TF_SEC = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800, "1h": 3600}


# ----------------------------------------------------------------- data
def fetch(symbol, tf, days):
    step = TF_SEC[tf]
    end = int(time.time())
    start = end - days * 86400
    chunk = 1900 * step  # API allows up to 2000 candles per call
    rows, s = [], start
    while s < end:
        e = min(s + chunk, end)
        r = requests.get(
            f"{BASE}/v2/history/candles",
            params={"resolution": tf, "symbol": symbol, "start": s, "end": e},
            timeout=30,
        )
        r.raise_for_status()
        rows += r.json().get("result", [])
        s = e
        time.sleep(0.3)
    if not rows:
        sys.exit("No candles returned - check SYMBOL / DELTA_BASE.")
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("dt")[["open", "high", "low", "close", "volume"]].astype(float)
    print(f"Loaded {len(df)} candles {df.index[0]} -> {df.index[-1]}")
    return df


# ----------------------------------------------------------- indicators
def add_indicators(df):
    o, h, l, c, v = df.open, df.high, df.low, df.close, df.volume
    for n in (9, 21, 50, 200):
        df[f"ema{n}"] = c.ewm(span=n, adjust=False).mean()
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    df["atr"] = tr.ewm(alpha=1 / 14, adjust=False).mean()
    d = c.diff()
    up, dn = d.clip(lower=0), -d.clip(upper=0)
    rs = up.ewm(alpha=1 / 14, adjust=False).mean() / dn.ewm(alpha=1 / 14, adjust=False).mean()
    df["rsi"] = 100 - 100 / (1 + rs)
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    df["bb_up"], df["bb_lo"] = ma + 2 * sd, ma - 2 * sd
    tp = (h + l + c) / 3
    day = df.index.date
    df["vwap"] = (tp * v).groupby(day).cumsum() / v.groupby(day).cumsum()  # resets daily (UTC)
    df["vol_ma"] = v.rolling(20).mean()
    df["hh20"] = h.rolling(20).max().shift(1)
    df["ll20"] = l.rolling(20).min().shift(1)
    return df


# ----------------------------------------------------------- strategies
def _sig(long, short):
    return np.where(long, 1, np.where(short, -1, 0))


def s_ema_cross(d):
    up = (d.ema9 > d.ema21) & (d.ema9.shift() <= d.ema21.shift())
    dn = (d.ema9 < d.ema21) & (d.ema9.shift() >= d.ema21.shift())
    return _sig(up & (d.close > d.ema200), dn & (d.close < d.ema200))


def s_vwap_reclaim(d):
    long = (d.low <= d.vwap) & (d.close > d.vwap) & (d.close > d.open) & (d.close > d.ema200)
    short = (d.high >= d.vwap) & (d.close < d.vwap) & (d.close < d.open) & (d.close < d.ema200)
    return _sig(long, short)


def s_bb_rsi_reversion(d):
    long = (d.low < d.bb_lo) & (d.close > d.bb_lo) & (d.rsi < 35)
    short = (d.high > d.bb_up) & (d.close < d.bb_up) & (d.rsi > 65)
    return _sig(long, short)


def s_donchian_breakout(d):
    vol = d.volume > 1.5 * d.vol_ma
    long = (d.close > d.hh20) & vol & (d.close > d.ema200)
    short = (d.close < d.ll20) & vol & (d.close < d.ema200)
    return _sig(long, short)


def s_liquidity_sweep(d):
    short = (d.high > d.hh20) & (d.close < d.hh20) & (d.close < d.open)
    long = (d.low < d.ll20) & (d.close > d.ll20) & (d.close > d.open)
    return _sig(long, short)


def s_volume_spike(d):
    body = (d.close - d.open).abs()
    big = (d.volume > 2.5 * d.vol_ma) & (body > 0.8 * d.atr)
    long = big & (d.close > d.open) & (d.close > d.ema50)
    short = big & (d.close < d.open) & (d.close < d.ema50)
    return _sig(long, short)


STRATEGIES = {
    "ema_cross_trend": s_ema_cross,
    "vwap_reclaim": s_vwap_reclaim,
    "bb_rsi_reversion": s_bb_rsi_reversion,
    "donchian_breakout": s_donchian_breakout,
    "liquidity_sweep": s_liquidity_sweep,
    "volume_spike_follow": s_volume_spike,
}


# --------------------------------------------------------------- engine
def simulate(df, sig, sl_mult, rr):
    o, h, l, c, a = (df[k].values for k in ("open", "high", "low", "close", "atr"))
    idx = df.index
    n = len(df)
    out, i = [], 0
    while i < n - 2:
        s = sig[i]
        if s == 0 or np.isnan(a[i]) or a[i] <= 0:
            i += 1
            continue
        entry = o[i + 1]
        risk = sl_mult * a[i]
        sl = entry - s * risk
        tp = entry + s * risk * rr
        last = min(i + 1 + MAX_HOLD, n - 1)
        exit_px, exit_j = c[last], last
        for j in range(i + 1, last + 1):
            hit_sl = l[j] <= sl if s == 1 else h[j] >= sl
            hit_tp = h[j] >= tp if s == 1 else l[j] <= tp
            if hit_sl:
                exit_px, exit_j = sl, j
                break
            if hit_tp:
                exit_px, exit_j = tp, j
                break
        gross = s * (exit_px - entry) / entry
        net = gross - 2 * (FEE + SLIP)
        out.append((idx[i + 1], s, net / (risk / entry), exit_j - i))
        i = exit_j + 1  # one trade at a time
    return pd.DataFrame(out, columns=["entry_time", "side", "R", "bars"])


def stats(t):
    if t.empty:
        return dict(trades=0, win=0.0, pf=0.0, exp=0.0, total_r=0.0, max_dd=0.0)
    w, lo = t.R[t.R > 0].sum(), -t.R[t.R < 0].sum()
    eq = t.R.cumsum()
    return dict(
        trades=len(t),
        win=round(100 * (t.R > 0).mean(), 1),
        pf=round(w / lo, 2) if lo > 0 else 99.0,
        exp=round(t.R.mean(), 3),
        total_r=round(t.R.sum(), 1),
        max_dd=round((eq.cummax() - eq).max(), 1),
    )


# ----------------------------------------------------------------- main
def main():
    df = add_indicators(fetch(SYMBOL, TF, DAYS))
    cut = df.index[int(len(df) * 0.7)]
    rows, trades_store = [], {}
    for name, fn in STRATEGIES.items():
        sig = fn(df)
        for slm in SL_MULTS:
            for rr in RRS:
                t = simulate(df, sig, slm, rr)
                full, ins, oos = stats(t), stats(t[t.entry_time < cut]), stats(t[t.entry_time >= cut])
                rows.append({
                    "strategy": name, "sl_atr": slm, "rr": rr, **full,
                    "is_pf": ins["pf"], "oos_pf": oos["pf"], "oos_trades": oos["trades"],
                })
                trades_store[(name, slm, rr)] = t
    res = pd.DataFrame(rows)
    res["verdict"] = np.where(
        (res.trades >= MIN_TRADES) & (res.pf >= 1.2) & (res.is_pf >= 1.2) & (res.oos_pf >= 1.2),
        "KEEP", "drop")

    os.makedirs("results", exist_ok=True)
    res.to_csv("results/all_results.csv", index=False)

    best = (res.sort_values(["verdict", "total_r"], ascending=[True, False])
               .groupby("strategy", as_index=False).head(1)
               .sort_values("total_r", ascending=False))
    for _, r in best.iterrows():
        trades_store[(r.strategy, r.sl_atr, r.rr)].to_csv(f"results/trades_{r.strategy}.csv", index=False)

    cols = ["strategy", "sl_atr", "rr", "trades", "win", "pf", "exp", "total_r",
            "max_dd", "is_pf", "oos_pf", "verdict"]
    md = [f"# Backtest {SYMBOL} {TF} - last {DAYS} days",
          f"Fee {FEE*100:.3f}% + slip {SLIP*100:.3f}% per side. Results in R (1R = stop distance).",
          f"Out-of-sample = last 30% of data. KEEP needs PF >= 1.2 overall, in-sample and out-of-sample, "
          f"with at least {MIN_TRADES} trades.", "",
          "| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in best.iterrows():
        md.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    text = "\n".join(md)
    print(text)
    with open("results/summary.md", "w") as f:
        f.write(text)
    if os.getenv("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(text)

    topic = os.getenv("NTFY_TOPIC")
    if topic:
        n_keep = int((best.verdict == "KEEP").sum())
        head = (f"{SYMBOL} {TF} | last {DAYS} days | {len(df)} candles\n"
                f"Fee {FEE*100:.3f}% + slip {SLIP*100:.3f}% per side | results in R\n"
                f"KEEP = PF>=1.2 overall, in-sample & out-of-sample, >={MIN_TRADES} trades\n"
                f"Passed: {n_keep} of {len(best)}\n")
        blocks = []
        for _, r in best.iterrows():
            tag = "KEEP" if r.verdict == "KEEP" else "drop"
            blocks.append(
                f"[{tag}] {r.strategy}\n"
                f"  SL {r.sl_atr} x ATR | RR 1:{r.rr}\n"
                f"  Trades {r.trades} | Win {r.win}%\n"
                f"  PF {r.pf} | Exp {r.exp}R | Total {r.total_r}R | MaxDD {r.max_dd}R\n"
                f"  In-sample PF {r.is_pf} | Out-of-sample PF {r.oos_pf} ({r.oos_trades} trades)")
        # ntfy messages are capped at ~4 KB, so split into several if needed
        parts, cur = [], head
        for b in blocks:
            if len((cur + "\n" + b).encode()) > 3500:
                parts.append(cur)
                cur = b
            else:
                cur = cur + "\n" + b
        parts.append(cur)
        for k, p in enumerate(parts, 1):
            suffix = f" ({k}/{len(parts)})" if len(parts) > 1 else ""
            try:
                requests.post(
                    f"https://ntfy.sh/{topic}", data=p.encode("utf-8"),
                    headers={"Title": f"Backtest {SYMBOL} {TF} {DAYS}d{suffix}",
                             "Tags": "chart_with_upwards_trend",
                             "Priority": "default"},
                    timeout=15)
            except Exception as e:  # never fail the run because of the notification
                print("ntfy failed:", e)


if __name__ == "__main__":
    main()
