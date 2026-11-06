"""
backtest_multi.py - compares 6 BTCUSD 5m strategies on Delta data (free).
Reports win rate, profit factor, avg R per trade, INCLUDING fees, and splits
the data into train (first 2/3) and test (last 1/3) so a strategy must work
on both before you trust it.

Env: DELTA_BASE (default https://api.india.delta.exchange)
     DAYS (default 30)
     FEE_PCT round-trip fee in % of position value (default 0.06)
Conservative rules: SL and TP in the same candle counts as a loss.
Open trades time out after 4 hours and close at the candle close.
"""
import os, time, requests
from bisect import bisect_right

BASE = os.getenv("DELTA_BASE", "https://api.india.delta.exchange")
DAYS = int(os.getenv("DAYS", "30"))
FEE = float(os.getenv("FEE_PCT", "0.06"))
SYMBOL = "BTCUSD"
RRS = (1.5, 2.0, 3.0)


# ---------- data ----------
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


# ---------- indicators ----------
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
        return "UP", e20, e50
    if e20 < e50 and cl[-1] < e50:
        return "DOWN", e20, e50
    return "FLAT", e20, e50


def build_features(c5):
    c15, c1h = resample(c5, 900), resample(c5, 3600)
    s15 = [c["time"] for c in c15]
    s1h = [c["time"] for c in c1h]
    closes = [c["close"] for c in c5]
    vols = [c.get("volume", 0) or 0 for c in c5]
    F = {}
    for i in range(400, len(c5) - 60):
        t = c5[i]["time"]
        n15 = bisect_right(s15, t + 300 - 900)
        n1h = bisect_right(s1h, t + 300 - 3600)
        if n15 < 60 or n1h < 60:
            continue
        cl5 = closes[i - 79:i + 1]
        t5, e20, e50 = trend(cl5)
        t15 = trend([c["close"] for c in c15[max(0, n15 - 80):n15]])[0]
        t1h = trend([c["close"] for c in c1h[max(0, n1h - 80):n1h]])[0]
        atr = sum(c["high"] - c["low"] for c in c5[i - 13:i + 1]) / 14
        w = cl5[-20:]
        sma = sum(w) / 20
        sd = (sum((x - sma) ** 2 for x in w) / 20) ** 0.5
        va = sum(vols[i - 20:i]) / 20
        F[i] = dict(
            t5=t5, t15=t15, t1h=t1h, e20=e20, e50=e50, atr=atr,
            gap=abs(e20 - e50) / atr if atr else 0, rsi=rsi(cl5),
            last=cl5[-1], high=c5[i]["high"], low=c5[i]["low"],
            hi20=max(c["high"] for c in c5[i - 20:i]),
            lo20=min(c["low"] for c in c5[i - 20:i]),
            bb_lo=sma - 2 * sd, bb_up=sma + 2 * sd,
            vr=vols[i] / va if va > 0 else 0)
    return F


# ---------- strategies: return (side, entry_type, entry, sl) or None ----------
def pullback(f, mult, min_gap=0.0):
    risk = max(f["atr"] * mult, 40)
    if f["gap"] < min_gap:
        return None
    if f["t1h"] == f["t15"] == f["t5"] == "UP" and f["rsi"] < 68:
        e = max(f["e20"], f["last"] - f["atr"] * 0.5)
        return "LONG", "limit", e, e - risk
    if f["t1h"] == f["t15"] == f["t5"] == "DOWN" and f["rsi"] > 32:
        e = min(f["e20"], f["last"] + f["atr"] * 0.5)
        return "SHORT", "limit", e, e + risk
    return None


def s_baseline(f):
    return pullback(f, 1.2)


def s_wide_stop(f):
    return pullback(f, 2.0)


def s_strong_trend(f):
    return pullback(f, 2.0, min_gap=0.6)


def s_breakout(f):
    if f["vr"] > 1.5:
        if f["last"] > f["hi20"] and f["t1h"] == f["t15"] == "UP":
            return "LONG", "market", f["last"], f["last"] - 1.5 * f["atr"]
        if f["last"] < f["lo20"] and f["t1h"] == f["t15"] == "DOWN":
            return "SHORT", "market", f["last"], f["last"] + 1.5 * f["atr"]
    return None


def s_mean_rev(f):
    if f["rsi"] < 25 and f["last"] < f["bb_lo"] and f["t1h"] != "DOWN":
        return "LONG", "market", f["last"], f["last"] - 1.5 * f["atr"]
    if f["rsi"] > 75 and f["last"] > f["bb_up"] and f["t1h"] != "UP":
        return "SHORT", "market", f["last"], f["last"] + 1.5 * f["atr"]
    return None


def s_sweep(f):
    a = f["atr"]
    if f["vr"] > 1.3:
        if f["low"] < f["lo20"] and f["last"] > f["lo20"] and f["t1h"] != "DOWN":
            return "LONG", "market", f["last"], min(f["low"] - 0.2 * a, f["last"] - 0.5 * a)
        if f["high"] > f["hi20"] and f["last"] < f["hi20"] and f["t1h"] != "UP":
            return "SHORT", "market", f["last"], max(f["high"] + 0.2 * a, f["last"] + 0.5 * a)
    return None


STRATS = [
    ("baseline", s_baseline), ("wide-stop", s_wide_stop),
    ("strong-trend", s_strong_trend), ("vol-breakout", s_breakout),
    ("mean-revert", s_mean_rev), ("sweep-reversal", s_sweep),
]


# ---------- simulation ----------
def simulate(c5, i, sig, rr):
    side, etype, entry, sl = sig
    d = 1 if side == "LONG" else -1
    if etype == "market":
        fill = i + 1
        entry = c5[fill]["open"]
        start = fill
    else:
        fill = None
        for j in range(i + 1, min(i + 7, len(c5))):
            c = c5[j]
            if (d == 1 and c["low"] <= entry) or (d == -1 and c["high"] >= entry):
                fill = j
                break
        if fill is None:
            return None
        start = fill + 1
    risk = (entry - sl) * d
    if risk <= 0:
        return None
    tp = entry + d * risk * rr
    cost = FEE / 100 * entry / risk
    last_j = min(start + 48, len(c5)) - 1
    for j in range(start, last_j + 1):
        c = c5[j]
        hit_sl = c["low"] <= sl if d == 1 else c["high"] >= sl
        hit_tp = c["high"] >= tp if d == 1 else c["low"] <= tp
        if hit_sl:
            return -1.0 - cost, False, j
        if hit_tp:
            return rr - cost, True, j
    r = (c5[last_j]["close"] - entry) * d / risk - cost
    return r, False, last_j


def run(c5, F, fn, rr):
    trades, i, idx = [], 400, sorted(F)
    pos = 0
    while pos < len(idx):
        i = idx[pos]
        sig = fn(F[i])
        res = simulate(c5, i, sig, rr) if sig else None
        if res is None:
            pos += 1
            continue
        trades.append((i, res[0], res[1]))
        nxt = res[2] + 1
        while pos < len(idx) and idx[pos] < nxt:
            pos += 1
    return trades


def stats(tr):
    n = len(tr)
    if n == 0:
        return dict(n=0, win=0.0, pf=0.0, avg=0.0)
    wins = sum(1 for _, _, h in tr if h)
    gp = sum(r for _, r, _ in tr if r > 0)
    gl = -sum(r for _, r, _ in tr if r < 0)
    return dict(n=n, win=100 * wins / n, pf=(gp / gl if gl else 99.0),
                avg=sum(r for _, r, _ in tr) / n)


def main():
    c5 = fetch_5m(DAYS)
    F = build_features(c5)
    idx = sorted(F)
    split = idx[len(idx) * 2 // 3]
    print(f"Data: {len(c5)} 5m candles (~{DAYS} days), fee {FEE}% round trip")
    print("avgR = average result per trade in R after fees. Need > 0.\n")
    rows = []
    for name, fn in STRATS:
        for rr in RRS:
            tr = run(c5, F, fn, rr)
            a = stats(tr)
            t1 = stats([x for x in tr if x[0] < split])
            t2 = stats([x for x in tr if x[0] >= split])
            rows.append((name, rr, a, t1, t2))
            print(f"{name} 1:{rr:g}\n"
                  f"  all n={a['n']} win={a['win']:.0f}% PF={a['pf']:.2f} "
                  f"avg={a['avg']:+.2f}R\n"
                  f"  train {t1['avg']:+.2f}R (n={t1['n']}) | "
                  f"test {t2['avg']:+.2f}R (n={t2['n']})")
    print("\n=== PASS = avgR > 0 on train AND test, 30+ trades total ===")
    ok = [r for r in rows if r[3]["avg"] > 0 and r[4]["avg"] > 0 and r[2]["n"] >= 30]
    if not ok:
        print("None passed. No strategy here is profitable after fees.")
    for name, rr, a, t1, t2 in sorted(ok, key=lambda r: -r[2]["avg"]):
        print(f"{name} 1:{rr:g}  n={a['n']} win={a['win']:.0f}% "
              f"avg={a['avg']:+.2f}R train {t1['avg']:+.2f} test {t2['avg']:+.2f}")


if __name__ == "__main__":
    main()
  
