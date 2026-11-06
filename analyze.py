"""
analyze.py - BTCUSD scalp analyzer (Delta Exchange data)
Writes the final message to signal.txt (notify.py sends it to ntfy).

Env vars (all optional):
  ANTHROPIC_API_KEY  -> if set, Claude reviews the data and writes the message
  DELTA_BASE         -> default https://api.india.delta.exchange
                        (global: https://api.delta.exchange)
  CLAUDE_MODEL       -> default claude-sonnet-5-5
Without ANTHROPIC_API_KEY the built-in rules produce the signal (free).
"""
import os, time, requests

BASE = os.getenv("DELTA_BASE", "https://api.india.delta.exchange")
SYMBOL = "BTCUSD"
MODEL = os.getenv("CLAUDE_MODEL", "claude-sonnet-5-5")
API_KEY = os.getenv("ANTHROPIC_API_KEY")


def candles(res, n=80):
    secs = {"5m": 300, "15m": 900, "1h": 3600}[res]
    end = int(time.time())
    r = requests.get(f"{BASE}/v2/history/candles", params={
        "resolution": res, "symbol": SYMBOL,
        "start": end - secs * n, "end": end}, timeout=20)
    r.raise_for_status()
    return sorted(r.json()["result"], key=lambda c: c["time"])


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


def stats(res):
    c = candles(res)
    cl = [x["close"] for x in c]
    hi = [x["high"] for x in c]
    lo = [x["low"] for x in c]
    vol = [x["volume"] for x in c]
    e20, e50 = ema(cl, 20), ema(cl, 50)
    return {
        "last": cl[-1], "ema20": round(e20, 1), "ema50": round(e50, 1),
        "trend": "UP" if e20 > e50 and cl[-1] > e50 else
                 "DOWN" if e20 < e50 and cl[-1] < e50 else "FLAT",
        "rsi": round(rsi(cl), 1),
        "atr": round(sum(h - l for h, l in zip(hi[-14:], lo[-14:])) / 14, 1),
        "high20": max(hi[-20:]), "low20": min(lo[-20:]),
        "vol_ratio": round(vol[-1] / (sum(vol[-20:]) / 20), 2),
    }


def rule_signal(s5, s15, s1h):
    last, atr = s5["last"], s5["atr"]
    risk = round(max(atr * 1.2, 40), 1)
    if s1h["trend"] == s15["trend"] == s5["trend"] == "UP" and s5["rsi"] < 68:
        side, entry = "LONG", round(max(s5["ema20"], last - atr * 0.5), 1)
        sl, d = entry - risk, 1
    elif s1h["trend"] == s15["trend"] == s5["trend"] == "DOWN" and s5["rsi"] > 32:
        side, entry = "SHORT", round(min(s5["ema20"], last + atr * 0.5), 1)
        sl, d = entry + risk, -1
    else:
        return (f"NO TRADE - trends not aligned (1h {s1h['trend']}, "
                f"15m {s15['trend']}, 5m {s5['trend']}).\n"
                f"Watch: support {s5['low20']}, resistance {s5['high20']}\n"
                f"Price {last}, RSI {s5['rsi']}")
    tp1, tp2 = entry + d * risk * 2, entry + d * risk * 3
    return (f"{side} (scalp) - 1h/15m/5m trend aligned\n"
            f"ENTRY: {entry:.1f} limit\nSL: {sl:.1f}\n"
            f"TP1: {tp1:.1f} (1:2)\nTP2: {tp2:.1f} (1:3)\n"
            f"Price {last}, RSI {s5['rsi']}, ATR {atr}")


def claude_message(data, rule_msg):
    prompt = (
        "You are a BTCUSD perpetual scalping analyst. Trader uses 5m charts, "
        "limit orders, min 1:2 reward:risk, tight realistic SL.\n"
        f"Computed data (do not recompute): {data}\n"
        f"Rule-based signal: {rule_msg}\n\n"
        "Review it. Reply in max 6 short lines: BIAS, ENTRY, SL, TP1 (RR), "
        "TP2 (RR), WAIT-FOR / invalidation. If no clean setup, say NO TRADE "
        "and give the two levels to watch. Do not invent data.")
    r = requests.post("https://api.anthropic.com/v1/messages", headers={
        "x-api-key": API_KEY, "anthropic-version": "2023-06-01",
        "content-type": "application/json"},
        json={"model": MODEL, "max_tokens": 400,
              "messages": [{"role": "user", "content": prompt}]}, timeout=60)
    r.raise_for_status()
    return r.json()["content"][0]["text"].strip()


def main():
    s5, s15, s1h = stats("5m"), stats("15m"), stats("1h")
    msg = rule_signal(s5, s15, s1h)
    if API_KEY:
        try:
            msg = claude_message({"5m": s5, "15m": s15, "1h": s1h}, msg)
        except Exception as e:
            print("Claude step failed, using rule signal:", e)
    open("signal.txt", "w", encoding="utf-8").write(msg)
    print(msg)


if __name__ == "__main__":
    main()
  
