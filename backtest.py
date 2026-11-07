import os
import time

import pandas as pd
import requests

BASE_URL = os.getenv("DELTA_BASE_URL", "https://api.india.delta.exchange").rstrip("/")
SYMBOL = os.getenv("SYMBOL", "BTCUSD")
DAYS = int(os.getenv("DAYS", "90"))

FEE_PER_SIDE = 0.0002 * (1 + 0.18)
SLIPPAGE_PER_SIDE = float(os.getenv("SLIPPAGE_PER_SIDE", "0.0002"))

MAX_ENTRIES_PER_DAY = 2
ATR_STOP_MULTIPLE = 1.5
REWARD_RISK = 1.5

NTFY_TOPIC = os.getenv("NTFY_TOPIC")
NTFY_SERVER = os.getenv("NTFY_SERVER", "https://ntfy.sh").rstrip("/")

session = requests.Session()
session.headers.update({
 "User-Agent": "research-backtest/1.0",
 "Accept": "application/json",
})

def get_json(path, params=None):
 response = session.get(f"{BASE_URL}{path}", params=params, timeout=30)
 response.raise_for_status()
 data = response.json()
 if not data.get("success"):
 raise RuntimeError(f"Delta API error: {data}")
 return data

def find_product(symbol):
 data = get_json("/v2/products", {"states": "live", "page_size": 100})
 for product in data.get("result", []):
 if product.get("symbol", "").upper() == symbol.upper():
 return product
 raise RuntimeError(f"{symbol} not found in live products on {BASE_URL}")

def fetch_candles(symbol, start_ts, end_ts, resolution):
 rows = []
 cursor = start_ts
 chunk_seconds = 90 * 24 * 60 * 60

 while cursor < end_ts:
 chunk_end = min(cursor + chunk_seconds, end_ts)
 payload = get_json(
 "/v2/history/candles",
 {
 "resolution": resolution,
 "symbol": symbol,
 "start": cursor,
 "end": chunk_end,
 },
 )
 rows.extend(payload.get("result", []))
 cursor = chunk_end
 time.sleep(0.2)

 df = pd.DataFrame(rows)
 required = {"time", "open", "high", "low", "close"}
 if df.empty or not required.issubset(df.columns):
 raise RuntimeError(f"No usable {resolution} candles returned")

 df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
 for col in ["open", "high", "low", "close"]:
 df[col] = pd.to_numeric(df[col], errors="coerce")

 return (
 df.dropna(subset=["open", "high", "low", "close"])
.drop_duplicates("time")
.sort_values("time")
.set_index("time")
 )

def add_indicators(df):
 df = df.copy()
 df["ema_fast"] = df["close"].ewm(span=9, adjust=False).mean()
 df["ema_slow"] = df["close"].ewm(span=21, adjust=False).mean()

 previous_close = df["close"].shift(1)
 true_range = pd.concat(
 [
 df["high"] - df["low"],
 (df["high"] - previous_close).abs(),
 (df["low"] - previous_close).abs(),
 ],
 axis=1,
 ).max(axis=1)
 df["atr"] = true_range.rolling(14).mean()
 return df

def backtest(df, trend):
 df = df.copy()
 trend = trend.reindex(df.index, method="ffill")
 df["trend_ema"] = trend["ema"]

 df["long_signal"] = (
 (df["close"] > df["trend_ema"])
 & (df["ema_fast"] > df["ema_slow"])
 & (df["close"].shift(1) <= df["ema_fast"].shift(1))
 & (df["close"] > df["ema_fast"])
 )
 df["short_signal"] = (
 (df["close"] < df["trend_ema"])
 & (df["ema_fast"] < df["ema_slow"])
 & (df["close"].shift(1) >= df["ema_fast"].shift(1))
 & (df["close"] < df["ema_fast"])
 )

 trades = []
 daily_entries = {}
 i = 1

 while i < len(df) - 1:
 signal_row = df.iloc[i - 1]
 entry_row = df.iloc[i]
 day = entry_row.name.date()
 entries_today = daily_entries.get(day, 0)

 if entries_today >= MAX_ENTRIES_PER_DAY:
 i += 1
 continue

 direction = (
 1 if signal_row["long_signal"]
 else -1 if signal_row["short_signal"]
 else 0
 )
 if direction == 0 or pd.isna(signal_row["atr"]):
 i += 1
 continue

 entry = float(entry_row["open"])
 risk = float(signal_row["atr"]) * ATR_STOP_MULTIPLE
 if risk <= 0:
 i += 1
 continue

 stop = entry - direction * risk
 target = entry + direction * risk * REWARD_RISK
 exit_price = float(df.iloc[-1]["close"])
 exit_time = df.index[-1]
 reason = "end_of_data"

 for j in range(i, len(df)):
 bar = df.iloc[j]
 hit_stop = bar["low"] <= stop if direction == 1 else bar["high"] >= stop
 hit_target = bar["high"] >= target if direction == 1 else bar["low"] <= target

 # If both levels are reached in one candle, assume the stop is hit first.
 if hit_stop:
 exit_price, exit_time, reason = stop, df.index[j], "stop"
 break
 if hit_target:
 exit_price, exit_time, reason = target, df.index[j], "target"
 break

 gross_return = direction * (exit_price - entry) / entry
 net_return = gross_return - 2 * (FEE_PER_SIDE + SLIPPAGE_PER_SIDE)

 trades.append({
 "entry_time": entry_row.name,
 "exit_time": exit_time,
 "side": "long" if direction == 1 else "short",
 "entry": entry,
 "exit": exit_price,
 "exit_reason": reason,
 "gross_return_pct": gross_return * 100,
 "net_return_pct": net_return * 100,
 })

 daily_entries[day] = entries_today + 1
 i = max(i + 1, j + 1)

 return pd.DataFrame(trades)

def send_ntfy(message, title="Backtest complete"):
 if not NTFY_TOPIC:
 print("NTFY_TOPIC not set; skipping notification.")
 return

 response = requests.post(
 f"{NTFY_SERVER}/{NTFY_TOPIC}",
 data=message.encode("utf-8"),
 headers={"Title": title, "Priority": "default"},
 timeout=15,
 )
 response.raise_for_status()

def main():
 product = find_product(SYMBOL)
 end_ts = int(time.time())
 start_ts = end_ts - DAYS * 24 * 60 * 60

 candles = add_indicators(
 fetch_candles(product["symbol"], start_ts, end_ts, "5m")
 )
 trend_candles = fetch_candles(
 product["symbol"], start_ts, end_ts, "15m"
 )
 trend = trend_candles[["close"]].copy()
 trend["ema"] = trend["close"].ewm(span=50, adjust=False).mean()

 trades = backtest(candles, trend)
 if trades.empty:
 print("No trades found for these rules and dates.")
 try:
 send_ntfy(
 f"Symbol: {product['symbol']}\n"
 "No trades found. Historical backtest only."
 )
 except requests.RequestException as exc:
 print(f"Could not send ntfy notification: {exc}")
 return

 equity = (1 + trades["net_return_pct"] / 100).cumprod()
 max_drawdown = (equity / equity.cummax() - 1).min()
 wins = (trades["net_return_pct"] > 0).mean()
 avg_trades_per_day = len(trades) / max(DAYS, 1)

 print(f"Symbol: {product['symbol']}")
 print(f"Period: {candles.index[0]} to {candles.index[-1]}")
 print("Rules: 5m EMA pullback, 15m EMA trend filter, ATR stop, 1.5R target")
 print(f"Fee per side including stated GST: {FEE_PER_SIDE:.4%}")
 print(f"Assumed slippage per side: {SLIPPAGE_PER_SIDE:.4%}")
 print(f"Trades: {len(trades)}")
 print(f"Win rate: {wins:.2%}")
 print(f"Net compounded return per 1x notional: {equity.iloc[-1] - 1:.2%}")
 print(f"Trade-equity max drawdown: {max_drawdown:.2%}")
 print(f"Average entries per calendar day: {avg_trades_per_day:.2f}")
 print("\nRecent trades:")
 print(trades.tail(10).to_string(index=False))

 trades.to_csv("backtest_trades.csv", index=False)

 message = (
 f"Symbol: {product['symbol']}\n"
 f"Trades: {len(trades)}\n"
 f"Win rate: {wins:.2%}\n"
 f"Net return: {equity.iloc[-1] - 1:.2%}\n"
 f"Max drawdown: {max_drawdown:.2%}\n"
 f"Trades/day: {avg_trades_per_day:.2f}\n"
 "Historical backtest only; not a live trading signal."
 )

 try:
 send_ntfy(message)
 print("ntfy notification sent.")
 except requests.RequestException as exc:
 print(f"Could not send ntfy notification: {exc}")

if __name__ == "__main__":
 main()
  
