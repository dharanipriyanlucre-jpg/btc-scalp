import os
import time
import logging
import requests
import pandas as pd
import pandas_ta as ta
from datetime import datetime, timedelta
from dotenv import load_dotenv
from tabulate import tabulate

load_dotenv()

# ───────── CONFIG ─────────
SYMBOL = os.getenv("DELTA_SYMBOL", "BTCUSD")
START_DATE = pd.Timestamp(os.getenv("START_DATE", "2023-06-01"))
END_DATE = pd.Timestamp(os.getenv("END_DATE", "2024-06-01"))
RISK_PER_TRADE = float(os.getenv("RISK_PER_TRADE", 0.01))
COMMISSION_BPS = float(os.getenv("COMMISSION_BPS", 2)) / 10000
INITIAL_CAPITAL = float(os.getenv("INITIAL_CAPITAL", 10000.0))
SLIPPAGE_PCT = float(os.getenv("SLIPPAGE_PCT", 0.0005))
NTFY_TOPIC_URL = os.getenv("NTFY_TOPIC_URL", "")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ───────── DELTA PUBLIC API FETCHER ─────────
def fetch_delta_candles(interval: str, start_ts: int, end_ts: int) -> pd.DataFrame:
    url = f"https://api.delta.exchange/v1/candles/{interval}"
    dfs = []
    cursor = pd.Timestamp.fromtimestamp(start_ts)
    
    while cursor < pd.Timestamp.fromtimestamp(end_ts):
        chunk_start = int(cursor.timestamp())
        chunk_end = min(int((cursor + pd.Timedelta(days=30)).timestamp()), end_ts)
        
        params = {"symbol": SYMBOL, "start": chunk_start, "end": chunk_end, "count": 1000}
        retries = 0
        
        while retries < 4:
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                raw = resp.json().get("data", [])
                if not raw:
                    break
                    
                df = pd.DataFrame(raw, columns=["time","open","high","low","close","volume"])
                df["time"] = pd.to_datetime(df["time"], unit="ms")
                for c in ["open","high","low","close","volume"]:
                    df[c] = df[c].astype(float)
                    
                mask = (df["time"] >= cursor) & (df["time"] <= pd.Timestamp.fromtimestamp(chunk_end))
                dfs.append(df[mask])
                break
            except Exception as e:
                retries += 1
                wait = 2 ** retries
                logging.warning(f"Fetch failed ({retries}/4): {e}. Retrying in {wait}s...")
                time.sleep(wait)
        
        cursor += pd.Timedelta(days=30)
        if not dfs and retries == 4:
            raise RuntimeError("API consistently returned errors. Check symbol/name or network.")
            
    if not dfs:
        raise ValueError("No candles fetched. Verify symbol format and date range.")
        
    full = pd.concat(dfs).drop_duplicates(subset=["time"]).sort_values("time").set_index("time")
    return full

# ───────── MULTI-TIMEFRAME ALIGNMENT ─────────
def build_master_frame(df_5m: pd.DataFrame, df_15m: pd.DataFrame, df_1h: pd.DataFrame) -> pd.DataFrame:
    master = df_5m[["open","high","low","close","volume"]].copy()
    
    for df_tf, tf_name in [(df_15m, "15m"), (df_1h, "1h")]:
        aligned = df_tf.resample("5min").last().reindex(master.index).ffill()
        for col in aligned.columns:
            if col != "volume":
                master[f"{col}_{tf_name}"] = aligned[col]
                
    master.dropna(inplace=True)
    return master

# ───────── INDICATOR CALCULATOR ─────────
def calc_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df.ta.ema(length=21, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.adx(length=14, append=True)
    df.ta.rsi(length=14, append=True)
    df["vol_sma_20"] = df["volume"].rolling(20).mean()
    return df

# ───────── STRATEGY SIGNAL GENERATOR ─────────
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    m5 = df.copy()
    
    m5["trend_ok"] = (
        (m5["close_1h"] > m5["ema_50_1h"]) &
        (m5["close_1h"] > m5["ema_200_1h"]) &
        (m5["adx_14_1h"] > 20)
    )
    m5["pullback_ok"] = (
        (m5["close_15m"] < m5["ema_50_15m"]) &
        (m5["close_15m"] > m5["ema_21_15m"])
    )
    m5["trigger_rsi"] = m5["rsi_14"] > 50
    m5["trigger_vol"] = m5["volume"] > m5["vol_sma_20"] * 1.2
    m5["bullish_close"] = (m5["close"] > m5["open"]) & (m5["close"] > m5["low"] * 1.005)
    
    m5["long_signal"] = (
        m5["trend_ok"] & m5["pullback_ok"] &
        m5["trigger_rsi"] & m5["trigger_vol"] & m5["bullish_close"]
    )
    return m5

# ───────── TRADE SIMULATOR ─────────
def run_backtest(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["equity"] = INITIAL_CAPITAL
    df["position_side"] = None
    df["entry_price"] = 0.0
    df["sl_price"] = 0.0
    df["tp_price"] = 0.0
    df["trade_pnl"] = 0.0
    df["cumulative_pnl"] = 0.0
    
    capital = INITIAL_CAPITAL
    pos = None
    
    for idx, row in df.itertuples():
        if pos is None and row.long_signal:
            entry = row.close * (1 + SLIPPAGE_PCT)
            sl = row.low * (1 - SLIPPAGE_PCT * 2)
            tp = entry + (entry - sl) * 2.0
            
            qty = max(0.001, (capital * RISK_PER_TRADE) / (entry - sl))
            comm = entry * qty * COMMISSION_BPS
            
            pos = {"entry": entry, "qty": qty, "sl": sl, "tp": tp, "comm": comm, "idx": idx}
            df.at[idx, "position_side"] = "LONG"
            df.at[idx, "entry_price"] = entry
            df.at[idx, "sl_price"] = sl
            df.at[idx, "tp_price"] = tp
            df.at[idx, "equity"] = capital + ((row.close - entry) * qty)
            
        elif pos is not None:
            floating = (row.close - pos["entry"]) * pos["qty"]
            df.at[idx, "equity"] = capital + floating
            
            pnl_comm = -pos["comm"]
            if row.low <= pos["sl"]:
                pnl = (pos["sl"] - pos["entry"]) * pos["qty"] + pnl_comm
                df.at[idx, "trade_pnl"] = pnl
                df.at[idx, "cumulative_pnl"] = df["cumulative_pnl"].shift(1).fillna(0) + pnl
                capital += pnl
                pos = None
                df.at[idx, "position_side"] = None
                
            elif row.high >= pos["tp"]:
                pnl = (pos["tp"] - pos["entry"]) * pos["qty"] + pnl_comm
                df.at[idx, "trade_pnl"] = pnl
                df.at[idx, "cumulative_pnl"] = df["cumulative_pnl"].shift(1).fillna(0) + pnl
                capital += pnl
                pos = None
                df.at[idx, "position_side"] = None
                
            elif row.close < pos["entry"] and row.close < row.open:
                pnl = (row.close - pos["entry"]) * pos["qty"] + pnl_comm
                df.at[idx, "trade_pnl"] = pnl
                df.at[idx, "cumulative_pnl"] = df["cumulative_pnl"].shift(1).fillna(0) + pnl
                capital += pnl
                pos = None
                df.at[idx, "position_side"] = None
                
        else:
            df.at[idx, "equity"] = capital
            
    df["equity"] = df["equity"].fillna(INITIAL_CAPITAL)
    return df

# ───────── METRICS REPORTER ─────────
def report_metrics(df: pd.DataFrame) -> dict:
    trades = df[df["trade_pnl"] != 0].copy()
    wins = trades[trades["trade_pnl"] > 0]
    losses = trades[trades["trade_pnl"] <= 0]
    
    total = len(trades)
    wr = len(wins)/total if total else 0
    aw = wins["trade_pnl"].mean() if len(wins) else 0
    al = abs(losses["trade_pnl"].mean()) if len(losses) else 0
    pf = wins["trade_pnl"].sum()/abs(losses["trade_pnl"].sum()) if len(losses) and losses["trade_pnl"].sum() != 0 else 0
    eq = df["equity"]
    dd = eq.pct_change().expanding().max().sub(eq.pct_change().expanding().apply(lambda x: 1 if isinstance(x, float) else 0).cumprod()).max()*100
    sharpe = (eq.pct_change().mean() / eq.pct_change().std()) * (252**0.5) if eq.pct_change().std() else 0
    final_eq = df["equity"].iloc[-1]
    ret = (final_eq/INITIAL_CAPITAL - 1)*100
    
    return {
        "Total Trades": total,
        "Win Rate (%)": round(wr*100, 2),
        "Avg Win (₹)": round(aw, 2),
        "Avg Loss (₹)": round(-al, 2),
        "Profit Factor": round(pf, 3),
        "Max Drawdown (%)": round(dd, 2),
        "Sharpe Ratio": round(sharpe, 2),
        "Final Equity (₹)": round(final_eq, 2),
        "Total Return (%)": round(ret, 2)
    }

# ───────── NTFY ALERT SENDER ─────────
def send_ntfy_alert(metrics: dict) -> None:
    if not NTFY_TOPIC_URL:
        logging.warning("⚠️ NTFY_TOPIC_URL not set. Skipping push notification.")
        return
    try:
        msg = (
            f"📊 Delta Backtest Complete\n"
            f"🔹 Symbol: `{SYMBOL}`\n"
            f"🔹 Range: `{START_DATE.date()} → {END_DATE.date()}`\n"
            f"🔹 Trades: `{metrics['Total Trades']}`\n"
            f"🔹 Win Rate: `{metrics['Win Rate (%)']}%`\n"
            f"🔹 Profit Factor: `{metrics['Profit Factor']}`\n"
            f"🔹 Max DD: `{metrics['Max Drawdown (%)']}%`\n"
            f"🔹 Return: `{metrics['Total Return (%)']}%`\n\n"
            f"📎 CSVs exported! Check repo artifacts."
        )
        headers = {
            "X-Priority": "3",
            "Title": "📊 Strategy Backtest Result",
            "Tags": "chart-increasing,pushpin"
        }
        resp = requests.post(NTFY_TOPIC_URL, data=msg.encode('utf-8'), headers=headers, timeout=10)
        resp.raise_for_status()
        logging.info("✅ NTFY alert sent successfully.")
    except Exception as e:
        logging.error(f"❌ ntfy push failed: {e}")

# ───────── MAIN RUNNER ─────────
def main():
    logging.info(f"🚀 Starting backtest for {SYMBOL} | {START_DATE.date()} → {END_DATE.date()}")
    s1 = pd.Timestamp.now()
    
    df_5m = calc_indicators(fetch_delta_candles("5m", START_DATE.timestamp(), END_DATE.timestamp()))
    df_15m = calc_indicators(fetch_delta_candles("15m", START_DATE.timestamp(), END_DATE.timestamp()))
    df_1h = calc_indicators(fetch_delta_candles("1h", START_DATE.timestamp(), END_DATE.timestamp()))
    
    master = build_master_frame(df_5m, df_15m, df_1h)
    signals = generate_signals(master)
    result = run_backtest(signals)
    
    result.to_csv("delta_backtest_equity.csv", index=True)
    trades = result[result["trade_pnl"] != 0][["time","equity","trade_pnl","position_side"]]
    trades.to_csv("delta_backtest_trades.csv", index=False)
    
    metrics = report_metrics(result)
    print("\n" + "="*42)
    print("📊 BACKTEST RESULTS")
    print("="*42)
    print(tabulate(metrics.items(), headers=["Metric", "Value"], tablefmt="grid"))
    print("="*42)
    print(f"⏱️ Completed in {pd.Timestamp.now()-s1}")
    logging.info("✅ Saved: delta_backtest_equity.csv | delta_backtest_trades.csv")
    
    # Push to nfy
    send_ntfy_alert(metrics)

if __name__ == "__main__":
    main()
  
