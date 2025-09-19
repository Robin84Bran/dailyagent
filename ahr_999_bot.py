import pandas as pd
import numpy as np
import datetime
import requests
import matplotlib.pyplot as plt

# Fetch BTC historical data from Binance API
def fetch_btc_data():
    url = "https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=1d&limit=1000"
    data = requests.get(url).json()
    df = pd.DataFrame(data, columns=['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'CloseTime', 'QuoteAssetVolume', 'Trades', 'TBB', 'TBA', 'Ignore'])
    df['Date'] = pd.to_datetime(df['Timestamp'], unit='ms')
    df['Close'] = df['Close'].astype(float)
    return df[['Date', 'Close']]

# Calculate 200-day DCA (Geometric Mean)
def geometric_mean(prices):
    return np.exp(np.mean(np.log(prices)))

# Calculate BTC Fair Value using AHR 999 formula
def btc_fair_value():
    btc_age = (datetime.datetime.now() - datetime.datetime(2009, 1, 3)).days  # BTC Age in days
    fair_value = 10 ** (5.80 * np.log10(btc_age) - 16.88)
    return fair_value

# Calculate AHR 999 Index
def calculate_ahr_999(df):
    df['200D_DCA'] = df['Close'].rolling(window=200).apply(geometric_mean, raw=True)
    fair_value = btc_fair_value()
    df['AHR_999'] = (df['Close'] / df['200D_DCA']) * (df['Close'] / fair_value)
    return df

# Check AHR 999 Sell Signals
def check_ahr_999(df):
    last_value = df.iloc[-1]['AHR_999']
    if last_value > 2.0:
        print("🚨🚨 EXTREME SELL WARNING! AHR 999 =", round(last_value, 2))
    elif last_value > 1.2:
        print("⚠️ AHR 999 in SELL Zone! Current Value =", round(last_value, 2))
    elif last_value < 0.45:
        print("✅ AHR 999 in BUY Zone! Current Value =", round(last_value, 2))
    else:
        print("🔵 AHR 999 in Neutral Zone. Current Value =", round(last_value, 2))

# Plot AHR 999 Index
def plot_ahr_999(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['AHR_999'], label="AHR 999 Index", color="blue")
    plt.axhline(1.2, color="red", linestyle="dashed", label="SELL Zone (1.2)")
    plt.axhline(2.0, color="darkred", linestyle="dashed", label="EXTREME SELL (2.0)")
    plt.axhline(0.45, color="green", linestyle="dashed", label="BUY Zone (0.45)")
    plt.title("Bitcoin AHR 999 Index")
    plt.xlabel("Date")
    plt.ylabel("AHR 999 Value")
    plt.legend()
    plt.grid()

    # Save the figure
    today_date = datetime.datetime.now().strftime("%Y%m%d")
    filename = f"/Users/robin/Documents/mypy/png/{today_date} BTC AHR 999 Fig.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✅ Saved chart as {filename}")

    plt.show()

# Main execution
def main():
    print("📡 Fetching BTC Data...")
    df = fetch_btc_data()
    
    print("📊 Calculating AHR 999 Index...")
    df = calculate_ahr_999(df)
    
    print("🔍 Checking AHR 999 signals...")
    check_ahr_999(df)

    print("📈 Plotting AHR 999 chart...")
    plot_ahr_999(df)

if __name__ == "__main__":
    main()
