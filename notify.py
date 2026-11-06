"""
notify.py - sends signal.txt to your ntfy app.
Env var: NTFY_TOPIC (your ntfy topic name; keep it as a GitHub secret)
"""
import os, requests

TOPIC = os.environ["NTFY_TOPIC"]
msg = open("signal.txt", encoding="utf-8").read().strip()
no_trade = msg.upper().startswith("NO TRADE") or "BIAS: NO TRADE" in msg.upper()

requests.post(
    f"https://ntfy.sh/{TOPIC}",
    data=msg.encode("utf-8"),
    headers={
        "Title": "BTCUSD scalp read",
        "Priority": "3" if no_trade else "4",
        "Tags": "zzz" if no_trade else "chart_with_upwards_trend",
    },
    timeout=20,
).raise_for_status()
print("sent to ntfy")
