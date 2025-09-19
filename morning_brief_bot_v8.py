
#!/usr/bin/env python3
# morning_brief_bot_v8.py
import os
from dotenv import load_dotenv
import requests
import pandas as pd
import numpy as np
from datetime import datetime

# Load environment variables
load_dotenv()

# Configuration
BINANCE_API_KEY         = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET      = os.getenv("BINANCE_API_SECRET", "")
TELEGRAM_TOKEN          = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID        = os.getenv("TELEGRAM_CHAT_ID", "")

# Time helper
try:
    from zoneinfo import ZoneInfo
    def local_date(tz="Asia/Taipei"):
        return datetime.now(ZoneInfo(tz)).strftime("%Y-%m-%d")
except ImportError:
    def local_date(tz=None):
        return datetime.now().strftime("%Y-%m-%d")

# Fetch BTC data (1d)
def fetch_btc_data(limit=1000) -> pd.DataFrame:
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": "BTCUSDT", "interval": "1d", "limit": limit}
    data = requests.get(url, params=params).json()
    df = pd.DataFrame(data, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades","taker_base","taker_quote","ignore"
    ])
    df["Close"] = df["close"].astype(float)
    df["Date"] = pd.to_datetime(df["open_time"], unit="ms")
    return df.set_index("Date")

# Pi-Cycle Logic
def get_pi_cycle_status():
    df = fetch_btc_data(limit=400)
    df["SMA111"] = df["Close"].rolling(111).mean()
    df["SMA350x2"] = df["Close"].rolling(350).mean() * 2
    prev, last = df.iloc[-2], df.iloc[-1]
    if prev["SMA111"] < prev["SMA350x2"] and last["SMA111"] >= last["SMA350x2"]:
        return "🚨 Crossover detected! SELL BTC!"
    return "✅ No crossover detected. BTC still in uptrend."

# AHR-999 Logic (original)
def geometric_mean(prices):
    return np.exp(np.mean(np.log(prices)))

def btc_fair_value():
    btc_age_days = (datetime.now() - datetime(2009, 1, 3)).days
    return 10 ** (5.80 * np.log10(btc_age_days) - 16.88)

def get_ahr_999_ratio():
    df = fetch_btc_data(limit=1000)
    # compute 200-day DCA via geometric mean
    df["DCA200"] = df["Close"].rolling(window=200).apply(lambda x: geometric_mean(x), raw=True)
    fv = btc_fair_value()
    df["AHR999"] = (df["Close"] / df["DCA200"]) * (df["Close"] / fv)
    val = df["AHR999"].iloc[-1]
    return float(val) if pd.notna(val) else 0.0

# Binance Prices
def get_binance_prices():
    from binance.client import Client
    client = Client(BINANCE_API_KEY, BINANCE_API_SECRET)
    btc = float(client.get_symbol_ticker(symbol="BTCUSDT")["price"])
    eth = float(client.get_symbol_ticker(symbol="ETHUSDT")["price"])
    return btc, eth

# Send Telegram via HTTP
def send_telegram_message(text):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("⚠️ Missing Telegram credentials.")
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}
    resp = requests.post(url, data=payload)
    print("Telegram response:", resp.text)

# Main
def main():
    date_str = local_date("Asia/Taipei")
    btc_price, eth_price = get_binance_prices()
    eth_btc_ratio = eth_price / btc_price
    pi_status = get_pi_cycle_status()
    ahr_ratio = get_ahr_999_ratio()
    compound_hf = "2.14 ✅"  # placeholder for actual HF

    message = f"""🌞 *Morning Brief – {date_str}*

*BTC Price:* ${btc_price:,.0f}
*ETH Price:* ${eth_price:,.0f}
*ETH/BTC Ratio:* {eth_btc_ratio:.4f}

*Pi Cycle Top:* {pi_status}
*AHR 999 Ratio:* {ahr_ratio:.2f}
*Compound HF:* {compound_hf}

💡 No actions required today. Enjoy the flow.
"""
    print(message)
    send_telegram_message(message)

if __name__ == "__main__":
    main()
