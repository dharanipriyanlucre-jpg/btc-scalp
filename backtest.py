"""
backtest.py - replays the analyze.py rules on recent Delta BTCUSD data.
Prints trades, win rate, profit factor for TP1 (2R) and TP2 (3R).
Conservative: if SL and TP are hit in the same candle, it counts as a loss.
Env: DELTA_BASE (default https://api.india.delta.exchange), DAYS (default 14)
"""
import os, time, requests
from bisect import bisect_right

BASE = os.getenv("DELTA_BASE", "https://api.india.delta.exchange")
DAYS = int(os.getenv("DAYS", "14"))
SYMBOL = "BTCUSD"


def fetch_5m(days):
    need, end, rows = days * 288, int(time.time()), {}
    while len(rows) < need:
        start = end - 300 * 1500
        r = requests.get(f"{BASE}/v2/history/candles", params={
            "resolution": "5m", "symbol": SYMBOL, "start": start, "end": end},
            timeout=30)
        r.raise_for_status()
        got = r.json()["result"]
        if not got:
            break
        for c in got:
            rows[c["time"]] = c
        end = start
    return [rows[t] for t in sorted(rows)][-need:]


def resample(c5, secs):
    out, cur = [], None
    for c in c5:
        b = c["time"] - c["time"] % secs
        if cur is None or cur["time"] != b:
            cur = {"time": b, "high": c["high"], "low": c["low"],
                   "close": c["close"]}
            out.append(cur)
        else:
            cur["high"] = max(cur["high"], c["high"])
            cur["low"] = min(cur["low"], c["low"])
            cur["close"] = c["close"]
    return out


def ema(vals, p):
    k, e = 2 / (p + 1), vals[0]
    for v in vals[1:]:
        e = v * k + e * (1 - k)
    return e


def rsi(vals, p=14):
    g = l = 0
    for i in range(-p, 0):
        d = vals[i] - vals[i - 1]
        g += max(d, 0)
        l += max(-d, 0)
    return 100 if l == 0 else 100 - 100 / (1 + g / l)


def trend(cl):
    e20, e50 = ema(cl, 20), ema(cl, 50)
    if e20 > e50 and cl[-1] > e50:
        return "UP", e20
    if e20 < e50 and cl[-1] < e50:
        return "DOWN", e20
    return "FLAT", e20


def simulate(c5, i, side, entry, sl, risk, rr):
    """Return +rr, -1, or None (no fill / unresolved). Fill window 6 candles."""
    d = 1 if side == "LONG" else -1
    tp = entry + d * risk * rr
    fill = None
    for j in range(i + 1, min(i + 7, len(c5))):
        c = c5[j]
        if (d == 1 and c["low"] <= entry) or (d == -1 and c["high"] >= entry):
            fill = j
            break
    if fill is None:
        return None, i + 1
    for j in range(fill + 1, min(fill + 49, len(c5))):
        c = c5[j]
        hit_sl = c["low"] <= sl if d == 1 else c["high"] >= sl
        hit_tp = c["high"] >= tp if d == 1 else c["low"] <= tp
        if hit_sl:
            return -1.0, j
        if hit_tp:
            return float(rr), j
    return None, fill + 49


def main():
    c5 = fetch_5m(DAYS)
    c15, c1h = resample(c5, 900), resample(c5, 3600)
    s15 = [c["time"] for c in c15]
    s1h = [c["time"] for c in c1h]
    res = {2: [], 3: []}
    i = 400
    while i < len(c5) - 60:
        t = c5[i]["time"]
        n15 = bisect_right(s15, t + 300 - 900)
        n1h = bisect_right(s1h, t + 300 - 3600)
        if n15 < 60 or n1h < 60:
            i += 1
            continue
        cl5 = [c["close"] for c in c5[i - 79:i + 1]]
        t5, e20 = trend(cl5)
        t15, _ = trend([c["close"] for c in c15[:n15]][-80:])
        t1h, _ = trend([c["close"] for c in c1h[:n1h]][-80:])
        atr = sum(c["high"] - c["low"] for c in c5[i - 13:i + 1]) / 14
        last, r5 = cl5[-1], rsi(cl5)
        risk = max(atr * 1.2, 40)
        side = None
        if t1h == t15 == t5 == "UP" and r5 < 68:
            side, entry = "LONG", max(e20, last - atr * 0.5)
            sl = entry - risk
        elif t1h == t15 == t5 == "DOWN" and r5 > 32:
            side, entry = "SHORT", min(e20, last + atr * 0.5)
            sl = entry + risk
        if not side:
            i += 1
            continue
        nxt = i + 1
        for rr in (2, 3):
            out, end = simulate(c5, i, side, entry, sl, risk, rr)
            if out is not None:
                res[rr].append(out)
            nxt = max(nxt, end) if out is not None else nxt
        i = max(i + 1, nxt if nxt > i + 1 else i + 1)

    print(f"Data: {len(c5)} 5m candles (~{DAYS} days)")
    for rr, r in res.items():
        w = sum(1 for x in r if x > 0)
        gp = sum(x for x in r if x > 0)
        gl = -sum(x for x in r if x < 0)
        n = len(r)
        print(f"\nTarget 1:{rr}")
        print(f"  trades {n}, wins {w}, win rate {100*w/n:.1f}%" if n else
              "  no trades")
        if n:
            print(f"  profit factor {gp/gl:.2f}" if gl else "  no losses")
            print(f"  avg result per trade {sum(r)/n:+.2f}R "
                  f"(break-even win rate {100/(rr+1):.0f}%)")


if __name__ == "__main__":
    main()
  
