import pandas as pd
import numpy as np
import requests
import time
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Binance API URL for BTCUSDT historical daily data
BINANCE_URL = "https://api.binance.com/api/v3/klines"

def fetch_btc_data():
    """ Fetch daily BTC price data from Binance API. """
    params = {
        "symbol": "BTCUSDT",
        "interval": "1d",
        "limit": 1000  # Gets last 1000 days of data
    }
    response = requests.get(BINANCE_URL, params=params)
    data = response.json()
    
    df = pd.DataFrame(data, columns=[
        "Timestamp", "Open", "High", "Low", "Close", "Volume",
        "CloseTime", "QuoteAssetVolume", "Trades", "TakerBuyBase", "TakerBuyQuote", "Ignore"
    ])
    
    # Convert timestamp to date
    df["Date"] = pd.to_datetime(df["Timestamp"], unit="ms")
    
    # Convert to numeric values
    df["Close"] = pd.to_numeric(df["Close"])
    df["Volume"] = pd.to_numeric(df["Volume"])
    
    # Keep only relevant columns
    df = df[["Date", "Close", "Volume"]]
    
    # Save CSV (optional)
    df.to_csv("BTC_Daily_Data.csv", index=False)
    
    return df

def calculate_pi_cycle_top(df):
    """ Calculate 111-day SMA and 350-day SMA x2 """
    df["111-day SMA"] = df["Close"].rolling(window=111).mean()
    df["350-day SMA x2"] = df["Close"].rolling(window=350).mean() * 2
    
    return df

def check_pi_cycle_top(df):
    """ Check if 111-day SMA crosses 350-day SMA x2 and SCREAM if true """
    last_row = df.iloc[-1]
    prev_row = df.iloc[-2]
    
    if prev_row["111-day SMA"] < prev_row["350-day SMA x2"] and last_row["111-day SMA"] >= last_row["350-day SMA x2"]:
        print("🚨🚨🚨 **ALERT!!! Pi Cycle Top Cross Detected! SELL BTC IMMEDIATELY!!!** 🚨🚨🚨")
    else:
        print("✅ No crossover detected. BTC still in uptrend.")

def plot_pi_cycle_top(df):
    """ Plot the BTC price along with Pi Cycle Top Indicator """
    plt.figure(figsize=(12, 6))

    plt.plot(df["Date"], df["Close"], label="BTC Price", color="black", alpha=0.7)
    plt.plot(df["Date"], df["111-day SMA"], label="111-day SMA", color="blue")
    plt.plot(df["Date"], df["350-day SMA x2"], label="350-day SMA x2", color="red")

    plt.title("Bitcoin Pi Cycle Top Indicator")
    plt.xlabel("Date")
    plt.ylabel("BTC Price (Log Scale)")
    plt.yscale("log")  # Log scale for better visualization
    plt.legend()
    plt.grid()

    # 📌 Save the figure BEFORE showing it, remember to change the folder file path if changed! 
    today_date = datetime.now().strftime("%Y%m%d")
    filename = f"/Users/robin/Documents/mypy/png/{today_date} BTC_Top_Fig.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved chart as {filename}")

    plt.show()  # Now show the figure (AFTER saving)

    plt.close()  # Close the figure to free memory

def main():
    print("🚀 Fetching BTC Data...")
    df = fetch_btc_data()
    
    print("📊 Calculating Pi Cycle Top Indicator...")
    df = calculate_pi_cycle_top(df)
    
    print("🔍 Checking for crossover...")
    check_pi_cycle_top(df)
    
    print("📈 Plotting chart...")
    plot_pi_cycle_top(df)

if __name__ == "__main__":
    main()
