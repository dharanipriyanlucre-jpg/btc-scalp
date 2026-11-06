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
OKX_BAR = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m", "1h": "1H"}


def _fetch_delta(base, symbol, tf, days):
    step = TF_SEC[tf]
    end = int(time.time())
    s = end - days * 86400
    chunk = 1900 * step  # API allows up to 2000 candles per call
    rows, shown = [], False
    while s < end:
        e = min(s + chunk, end)
        r = requests.get(
            f"{base}/v2/history/candles",
            params={"resolution": tf, "symbol": symbol, "start": s, "end": e},
            timeout=30,
        )
        if r.status_code != 200:
            print(f"Delta {base} HTTP {r.status_code}: {r.text[:200]}")
            return []
        res = r.json().get("result") or []
        if not res and not shown:
            print(f"Delta {base} empty response: {r.text[:200]}")
            shown = True
        rows += res
        s = e
        time.sleep(0.3)
    return rows


def _fetch_okx(tf, days):
    inst = os.getenv("OKX_INST", "BTC-USDT-SWAP")
    cutoff = (time.time() - days * 86400) * 1000
    rows, after = [], None
    while True:
        p = {"instId": inst, "bar": OKX_BAR[tf], "limit": "100"}
        if after:
            p["after"] = after
        r = requests.get("https://www.okx.com/api/v5/market/history-candles", params=p, timeout=30)
        if r.status_code != 200:
            print(f"OKX HTTP {r.status_code}: {r.text[:200]}")
            break
        data = r.json().get("data") or []
        if not data:
            break
        rows += [dict(time=int(x[0]) // 1000, open=x[1], high=x[2], low=x[3],
                      close=x[4], volume=x[5]) for x in data]
        after = data[-1][0]
        if int(after) < cutoff:
            break
        time.sleep(0.15)
    return rows


def fetch(symbol, tf, days):
    rows = []
    for base in dict.fromkeys([BASE, "https://api.india.delta.exchange", "https://api.delta.exchange"]):
        try:
            rows = _fetch_delta(base, symbol, tf, days)
        except Exception as e:
            print(f"Delta {base} failed: {e}")
            rows = []
        if rows:
            print(f"Using Delta data from {base}")
            break
    if not rows:
        print("Delta returned nothing - falling back to OKX candles")
        rows = _fetch_okx(tf, days)
    if not rows:
        sys.exit("No candles from Delta or OKX.")
    df = pd.DataFrame(rows).drop_duplicates("time").sort_values("time")
    df["dt"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.set_index("dt")[["open", "high", "low", "close", "volume"]].astype(float)
    df = df[df.index >= pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)]
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
