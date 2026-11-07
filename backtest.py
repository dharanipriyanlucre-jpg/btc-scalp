"""
BTC backtest - 3 setups from DERIVATIVES POSITIONING data (not price patterns):
  D1  Funding extreme (contrarian)  : crowded longs/shorts -> trade against the crowd
  D2  Liquidation flush (reversal)  : big forced-liquidation spike on a candle -> fade the flush
  D3  OI squeeze fade               : big price move while open interest FALLS (positions closing,
                                      no new money) -> fade the move

Data: Coinalyze history (price, open interest, funding, liquidations) from ONE exchange perp.
Needs env COINALYZE_KEY.  Coinalyze only keeps a limited amount of intraday history,
so the script uses 4h candles and prints how many days it really got.
Result goes to ntfy via env NTFY_TOPIC.
"""
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

# ---------------- CONFIG (change numbers here) ----------------
DAYS = 730             # history requested (Coinalyze returns what it keeps, usually less)
INTERVAL = "4hour"     # Coinalyze interval: 1hour, 2hour, 4hour, 6hour, 12hour, daily
BAR_HOURS = 4
FEE = 0.0002           # 0.02% per side (Delta fee)
GST = 0.18             # 18% GST on the fee -> effective 0.0236% per side
SLIP = 0.0001          # 0.01% slippage per side (set 0 if you only use limit orders)
RISK_PER_TRADE = 0.01  # 1% of equity risked per trade (equity curve only)
MAX_TRADES_PER_DAY = 2
OOS_SPLIT = 0.70       # first 70% in-sample, last 30% out-of-sample

# trade management (x ATR of the 4h candle)
SL_ATR = 1.5
TP_ATR = 2.0
MAX_HOLD = 6           # close after 6 candles (24h) if neither SL nor TP hit

# setup parameters (fixed, no grid search)
FUND_LOOKBACK = 60     # D1: funding z-score window (candles)
FUND_Z = 2.0           # D1: |z| needed
LIQ_LOOKBACK = 30      # D2: average liquidation window
LIQ_MULT = 3.0         # D2: spike must be this many x the average
OI_LOOKBACK = 60       # D3: OI change z-score window
OI_Z = -1.0            # D3: OI change z-score must be below this (positions closing)
MOVE_ATR = 2.0         # D3: price move over last 3 candles, in ATR

# pass rules (small sample, so be strict)
MIN_TRADES = 60
MIN_PF = 1.3
MIN_OOS_PF = 1.1

COST = 2 * (FEE * (1 + GST) + SLIP)   # round-trip cost as a fraction of price
API = "https://api.coinalyze.net/v1"
KEY = (os.getenv("COINALYZE_KEY") or "").strip()


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


# ---------------- DATA (Coinalyze) ----------------
def api_get(path, params=None, tries=6):
    p = dict(params or {})
    p["api_key"] = KEY
    for _ in range(tries):
        r = requests.get(f"{API}/{path}", params=p, headers={"api_key": KEY}, timeout=60)
        if r.status_code == 429:
            try:
                wait = float(r.headers.get("Retry-After", "10"))
            except ValueError:
                wait = 10
            time.sleep(wait + 1)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"{path}: still rate limited after retries")


def pick_symbol():
    markets = api_get("future-markets")
    btc = [m for m in markets if m.get("base_asset") == "BTC" and m.get("is_perpetual")
           and m.get("quote_asset") in ("USDT", "USD")]
    pref = ([m for m in btc if m.get("exchange") == "A" and m.get("quote_asset") == "USDT"]
            or [m for m in btc if m.get("quote_asset") == "USDT"] or btc)
    if not pref:
        raise RuntimeError("No BTC perpetual market found on Coinalyze")
    print(f"Using symbol {pref[0]['symbol']}")
    return pref[0]["symbol"]


def history(path, symbol, extra=None):
    now = int(time.time())
    params = {"symbols": symbol, "interval": INTERVAL, "from": now - DAYS * 86400, "to": now}
    params.update(extra or {})
    data = api_get(path, params)
    time.sleep(2)                                   # stay under the rate limit
    hist = data[0]["history"] if data else []
    if not hist:
        raise RuntimeError(f"{path}: empty history")
    df = pd.DataFrame(hist)
    df.index = pd.to_datetime(df["t"], unit="s")
    return df.drop(columns=["t"]).sort_index()


def load_data():
    if not KEY:
        notify("Backtest FAILED", "COINALYZE_KEY is empty. Add it as a repo secret and in the yml env.")
        sys.exit("COINALYZE_KEY missing")
    symbol = pick_symbol()
    px = history("ohlcv-history", symbol)[["o", "h", "l", "c", "v"]]
    px.columns = ["open", "high", "low", "close", "volume"]
    df = px.copy()
    missing = []
    for col, path, extra, pick, names in [
        ("oi", "open-interest-history", {"convert_to_usd": "true"}, ["c"], ["oi"]),
        ("fund", "funding-rate-history", {}, ["c"], ["fund"]),
        ("liq", "liquidation-history", {"convert_to_usd": "true"}, ["l", "s"], ["liq_long", "liq_short"]),
    ]:
        try:
            x = history(path, symbol, extra)[pick]
            x.columns = names
            df = df.join(x, how="left")
        except Exception as ex:
            print(f"{path} failed: {ex}")
            missing.append(path)
    return clean(df), symbol, missing


def clean(df):
    n0 = len(df)
    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["open", "high", "low", "close"])
    df = df[(df[["open", "high", "low", "close"]] > 0).all(axis=1)].copy()
    df["high"] = df[["high", "open", "close"]].max(axis=1)
    df["low"] = df[["low", "open", "close"]].min(axis=1)
    for col in ["oi", "fund"]:
        if col in df:
            df[col] = df[col].ffill()
    for col in ["liq_long", "liq_short"]:
        if col in df:
            df[col] = df[col].fillna(0).clip(lower=0)
    jump = (df["close"] / df["close"].shift() - 1).abs()
    df = df[~(jump > 0.2)]
    print(f"Cleaned {n0 - len(df)} bad candles out of {n0}")
    return df


# ---------------- FEATURES / SETUPS ----------------
def atr(df, n=14):
    pc = df["close"].shift()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def zscore(s, n):
    return (s - s.rolling(n).mean()) / s.rolling(n).std().replace(0, np.nan)


# each setup returns (long_signal, short_signal) evaluated at the CLOSE of a candle;
# the trade enters at the OPEN of the next candle.
def setup_d1_funding(df, a):
    z = zscore(df["fund"], FUND_LOOKBACK)
    return z <= -FUND_Z, z >= FUND_Z          # very negative funding -> long, very positive -> short


def setup_d2_liquidation(df, a):
    spike_long = df["liq_long"] > LIQ_MULT * df["liq_long"].rolling(LIQ_LOOKBACK).mean().shift()
    spike_short = df["liq_short"] > LIQ_MULT * df["liq_short"].rolling(LIQ_LOOKBACK).mean().shift()
    red, green = df["close"] < df["open"], df["close"] > df["open"]
    return spike_long & red, spike_short & green   # longs flushed on a red candle -> buy, etc.


def setup_d3_oi_squeeze(df, a):
    oi_z = zscore(df["oi"].pct_change(), OI_LOOKBACK)
    move = (df["close"] - df["close"].shift(3)) / a
    closing = oi_z <= OI_Z
    return closing & (move <= -MOVE_ATR), closing & (move >= MOVE_ATR)   # fade moves made on falling OI


SETUPS = {
    "D1_funding_extreme": (setup_d1_funding, ["fund"]),
    "D2_liquidation_flush": (setup_d2_liquidation, ["liq_long", "liq_short"]),
    "D3_oi_squeeze_fade": (setup_d3_oi_squeeze, ["oi"]),
}


# ---------------- ENGINE ----------------
def simulate(df, long, short, a):
    o, h, l, c = (df[k].values for k in ["open", "high", "low", "close"])
    ls, ss, av = long.fillna(False).values, short.fillna(False).values, a.values
    idx, n = df.index, len(df)
    trades, per_day, i = [], {}, 0
    while i < n - 2:
        d = 1 if ls[i] else (-1 if ss[i] else 0)
        if d == 0 or not np.isfinite(av[i]) or av[i] <= 0:
            i += 1
            continue
        day = idx[i + 1].floor("D")
        if per_day.get(day, 0) >= MAX_TRADES_PER_DAY:
            i += 1
            continue
        entry = o[i + 1]
        sl, tp = entry - d * SL_ATR * av[i], entry + d * TP_ATR * av[i]
        risk_pct = SL_ATR * av[i] / entry
        last = min(i + MAX_HOLD, n - 1)
        exit_px, j = c[last], last                  # default: close after max hold
        for jj in range(i + 1, last + 1):
            hit_sl = l[jj] <= sl if d == 1 else h[jj] >= sl
            hit_tp = h[jj] >= tp if d == 1 else l[jj] <= tp
            if hit_sl:                              # SL first if both in one candle (conservative)
                exit_px, j = sl, jj
                break
            if hit_tp:
                exit_px, j = tp, jj
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
    return (f"{name:21s} {tag:4s} n={s['trades']:4d} win={s['win']:3.0f}% PF={s['pf']:.2f} "
            f"expR={s['exp_R']:+.2f} ret={s['ret']:+.1f}% dd={s['maxdd']:.1f}% days_traded={s['cov']*100:.0f}%")


# ---------------- MAIN ----------------
def main():
    df, symbol, missing = load_data()
    a = atr(df)
    n_days = df.index.floor("D").nunique()
    split_time = df.index[int(len(df) * OOS_SPLIT)]
    d_is = df.index[df.index < split_time].floor("D").nunique()
    print(f"{symbol}: {len(df)} candles ({INTERVAL}), {n_days} days, {df.index[0]} -> {df.index[-1]}")
    print(f"Out-of-sample from {split_time}. Round-trip cost {COST*100:.3f}%\n")

    lines, passed = [], []
    for name, (fn, needs) in SETUPS.items():
        if any(col not in df for col in needs):
            print(f"{name}: skipped (data missing: {missing})\n")
            lines.append(f"SKIP {name}: data not available")
            continue
        long, short = fn(df, a)
        t = simulate(df, long, short, a)
        t_is, t_oos = t[t.time < split_time], t[t.time >= split_time]
        s_all, s_is, s_oos = stats(t, n_days), stats(t_is, d_is), stats(t_oos, n_days - d_is)
        for tag, s in [("ALL", s_all), ("IS", s_is), ("OOS", s_oos)]:
            print(fmt(name, tag, s))
        print()
        ok = s_all["trades"] >= MIN_TRADES and s_all["pf"] >= MIN_PF and s_oos["pf"] >= MIN_OOS_PF
        if ok:
            passed.append(name)
        lines.append(f"{'PASS' if ok else 'FAIL'} {name}: n={s_all['trades']} win={s_all['win']:.0f}% "
                     f"PF={s_all['pf']:.2f} IS={s_is['pf']:.2f} OOS={s_oos['pf']:.2f} ret={s_all['ret']:+.1f}%")

    summary = (f"BTC derivatives backtest {INTERVAL} | {n_days} days of data\n" + "\n".join(lines)
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
      
