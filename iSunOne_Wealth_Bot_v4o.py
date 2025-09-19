#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iSunOne Wealth Bot — v4o detail
Purpose: Keep v4o behavior, add 'The Score' component breakdowns to Telegram + JSON.
- v4o-style regime: S3 iff score >= 80 (no 2/5 override logic), else S2, with S1/S5 discounts.
- Same weight mix you’ve been using.
- Mapping is the gentler v4o-style (less generous on AHR & Over200D than v4q), which is why ETH
  typically scores lower in v4o (e.g., ~73 vs ~82 in v4q).
- Compact Telegram formatting. UTC timestamps (timezone-aware).

Author: Teddy L. Turing (for Robin)
"""

import os, json, math, statistics, traceback
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List

import requests
import numpy as np
import pandas as pd
import ccxt

# ------------------------------
# TELEGRAM (hard-coded per your working config)
# ------------------------------
TELEGRAM_BOT_TOKEN = "8105826927:AAF3EIAa0I4rw9Nkyw22e9h88Tpxb1GD5Ds"
TELEGRAM_CHAT_ID   = "760905049"
TELEGRAM_ENABLED   = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

def send_telegram(text: str) -> Optional[int]:
    if not TELEGRAM_ENABLED:
        return None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        resp = requests.post(url, json={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown"
        }, timeout=10)
        if resp.status_code == 200:
            j = resp.json()
            return j.get("result", {}).get("message_id")
        return None
    except Exception:
        return None

# ------------------------------
# EXCHANGES
# ------------------------------
BINANCE = ccxt.binance()
BINANCE_FUT = ccxt.binance({"options": {"defaultType": "future"}})

# ------------------------------
# TIME
# ------------------------------
def now_utc_str() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

# ------------------------------
# DATA FETCH
# ------------------------------
def fetch_ohlcv_daily(symbol: str, limit: int = 400) -> pd.DataFrame:
    ohlcv = BINANCE.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
    df = pd.DataFrame(ohlcv, columns=["ts","open","high","low","close","vol"])
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df

def fetch_perp_funding_history(symbol: str, limit: int = 60) -> pd.DataFrame:
    try:
        rows = BINANCE_FUT.fetch_funding_rate_history(symbol, since=None, limit=limit)
        df = pd.DataFrame(rows)
        if "fundingRate" in df.columns:
            df["fundingRate"] = pd.to_numeric(df["fundingRate"], errors="coerce")
        else:
            df["fundingRate"] = pd.to_numeric(df["info"].apply(lambda r: float(r["fundingRate"])), errors="coerce")
        df["dt"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df[["dt","fundingRate"]].dropna()
    except Exception:
        return pd.DataFrame(columns=["dt","fundingRate"])

def fetch_ticker_price(symbol: str) -> float:
    t = BINANCE.fetch_ticker(symbol)
    return float(t["last"])

def fetch_fear_greed() -> Optional[int]:
    try:
        r = requests.get("https://api.alternative.me/fng/", timeout=8)
        j = r.json()
        return int(j["data"][0]["value"])
    except Exception:
        return None

# ------------------------------
# INDICATORS
# ------------------------------
@dataclass
class AssetSnapshot:
    symbol: str
    price: float
    sma50: float
    sma200: float
    sma50_slope_up: bool
    over_200d_pct: float        # % above 200D
    ahr: float                  # price / 200D proxy
    vol30_ann_pct: float        # annualized 30D close-to-close vol (%)
    funding_z: Optional[float]  # funding zscore
    score: Optional[float] = None
    state: Optional[str] = None
    components: Optional[Dict[str, Dict[str, float]]] = None

def calc_sma(arr: pd.Series, n: int) -> float:
    return float(arr.tail(n).mean())

def slope_up(series: pd.Series, n: int = 5) -> bool:
    if len(series) < n + 1:
        return True
    return float(series.iloc[-1]) >= float(series.iloc[-n-1])

def calc_vol_annualized(closes: pd.Series, lookback: int = 30) -> float:
    rets = closes.pct_change().dropna().tail(lookback)
    if len(rets) == 0:
        return float("nan")
    daily_std = float(rets.std())
    return daily_std * math.sqrt(365.0) * 100.0

def zscore_last(values: List[float], window: int = 48) -> Optional[float]:
    if len(values) < 10:
        return None
    arr = values[-window:]
    if len(arr) < 10:
        return None
    x = arr[-1]
    ref = arr[:-1]
    mu = statistics.mean(ref)
    sd = statistics.pstdev(ref)
    if sd == 0 or math.isnan(sd):
        return 0.0
    z = (x - mu) / sd
    return max(min(z, 4.0), -4.0)

def build_snapshot(symbol_spot: str, symbol_perp: str) -> AssetSnapshot:
    df = fetch_ohlcv_daily(symbol_spot, 400)
    closes = df["close"]
    price = float(closes.iloc[-1])
    sma50 = calc_sma(closes, 50)
    sma200 = calc_sma(closes, 200)
    over_pct = ((price - sma200) / sma200) * 100.0
    ahr = price / sma200 if sma200 > 0 else float("nan")
    vol30 = calc_vol_annualized(closes, 30)
    sma50_series = closes.rolling(50).mean()
    up = slope_up(sma50_series, 5)

    fdf = fetch_perp_funding_history(symbol_perp, 60)
    if len(fdf) == 0:
        fz = None
    else:
        fz = zscore_last(list(fdf["fundingRate"].astype(float)), 48)

    return AssetSnapshot(
        symbol=symbol_spot, price=price, sma50=sma50, sma200=sma200,
        sma50_slope_up=up, over_200d_pct=over_pct, ahr=ahr,
        vol30_ann_pct=vol30, funding_z=fz
    )

# ------------------------------
# v4o SCORING (gentler mapping than v4q)
# ------------------------------
WEIGHTS = {
    "fear_greed": 0.25,
    "over_200d": 0.20,
    "funding_z": 0.20,
    "ahr": 0.20,
    "vol": 0.10,
    "breadth": 0.05
}
S3_TRIGGER = 80  # v4o rule: S3 iff score >= 80

def map_fear_greed_v4o(x: Optional[int]) -> float:
    if x is None: return 50.0
    return float(max(0, min(100, x)))

def map_over200d_pct_v4o(pct: float) -> float:
    # Softer than v4q: -20%->0, 0%->55, +50%->95
    if pct <= -20: return 0.0
    if pct >= 50:  return 95.0
    # piecewise linear between
    if pct <= 0:
        return 0.0 + (pct + 20.0) / 20.0 * 55.0     # [-20..0] => [0..55]
    return 55.0 + pct / 50.0 * 40.0                 # [0..50]  => [55..95]

def map_funding_z_v4o(z: Optional[float]) -> float:
    # Narrower band around neutral than v4q (less amplitude):
    # -2 → 40, 0 → 55, +2 → 80 (clipped)
    if z is None: return 55.0
    zc = max(-2.0, min(2.0, float(z)))
    return 55.0 + zc * (25.0 / 2.0)  # slope 12.5 per Z; yields [~30..80], but we clamp next
    # (We will clamp in compute to [0..100] anyway.)

def map_ahr_v4o(ratio: float) -> float:
    # v4o gentler: 0.7->0, 1.0->50, 1.35->80, 2.0->95 (cap)
    if ratio <= 0.70: return 0.0
    if ratio >= 2.00: return 95.0
    if ratio <= 1.0:
        return (ratio - 0.70) / 0.30 * 50.0
    if ratio <= 1.35:
        return 50.0 + (ratio - 1.0) / 0.35 * 30.0
    return 80.0 + (ratio - 1.35) / (0.65) * 15.0

def map_vol30_ann_v4o(vpct: float) -> float:
    # Peak around 50% (~75), penalize too low/high more than v4q
    if math.isnan(vpct): return 55.0
    v = max(5.0, min(200.0, vpct))
    if v <= 10: return 40.0
    if v <= 40: return 60.0 + (v - 10) / 30.0 * 10.0   # 10→40 => 60→70
    if v <= 60: return 70.0 + (v - 40) / 20.0 * 5.0    # 40→60 => 70→75
    if v <= 90: return 75.0 - (v - 60) / 30.0 * 15.0   # 60→90 => 75→60
    if v <= 150: return 60.0 - (v - 90) / 60.0 * 10.0  # 90→150 => 60→50
    return 48.0

def map_breadth_v4o(frac: float) -> float:
    return float(max(0.0, min(1.0, frac)) * 100.0)

def compute_score_components_v4o(
    snap: AssetSnapshot,
    fg_now: Optional[int],
    breadth_frac: float
) -> Tuple[float, Dict[str, Dict[str, float]]]:
    comp = {}
    fg_sub = map_fear_greed_v4o(fg_now);         comp["fear_greed"] = {"raw": float(fg_now or 50.0), "sub": fg_sub, "w": WEIGHTS["fear_greed"]}
    o2_sub = map_over200d_pct_v4o(snap.over_200d_pct); comp["over_200d"] = {"raw": snap.over_200d_pct, "sub": o2_sub, "w": WEIGHTS["over_200d"]}
    fz_sub = map_funding_z_v4o(snap.funding_z); comp["funding_z"] = {"raw": float(snap.funding_z if snap.funding_z is not None else 0.0), "sub": max(0.0, min(100.0, fz_sub)), "w": WEIGHTS["funding_z"]}
    ah_sub = map_ahr_v4o(snap.ahr);             comp["ahr"] = {"raw": snap.ahr, "sub": ah_sub, "w": WEIGHTS["ahr"]}
    vo_sub = map_vol30_ann_v4o(snap.vol30_ann_pct); comp["vol"] = {"raw": snap.vol30_ann_pct, "sub": vo_sub, "w": WEIGHTS["vol"]}
    br_sub = map_breadth_v4o(breadth_frac);     comp["breadth"] = {"raw": breadth_frac, "sub": br_sub, "w": WEIGHTS["breadth"]}

    score = 0.0
    for k, d in comp.items():
        d["contrib"] = round(d["sub"] * d["w"], 2)
        score += d["sub"] * d["w"]
    score = round(score, 1)
    return score, comp

# ------------------------------
# v4o REGIME PICKER (simple & strict)
# ------------------------------
def pick_regime_v4o(over_200d_pct: float, score: float) -> str:
    if over_200d_pct <= -20.0:
        return "S5"
    if over_200d_pct <= -5.0:
        return "S1"
    if score >= S3_TRIGGER:
        return "S3"
    return "S2"

# ------------------------------
# BREAKDOWN RENDERING
# ------------------------------
def build_breakdown_lines(tag: str, snap: AssetSnapshot, score: float, comp: Dict[str, Dict[str, float]]) -> str:
    def one(k, label):
        d = comp[k]
        raw = d["raw"]; sub = d["sub"]; w = d["w"]; c = d["contrib"]
        if k == "over_200d":
            rawtxt = f"{raw:.1f}%"
        elif k == "breadth":
            rawtxt = f"{raw:.2f}"
        elif k == "vol":
            rawtxt = f"{raw:.1f}%"
        elif k == "ahr":
            rawtxt = f"{raw:.2f}"
        elif k == "funding_z":
            rawtxt = f"{raw:.2f}"
        else:
            rawtxt = f"{raw:.0f}"
        return f"{label}: {rawtxt} → {sub:.0f} × {w:.2f} = {c:.2f}"
    lines = [
        f"*{tag} Components* (raw → sub × w):",
        one("fear_greed", "F&G"),
        one("over_200d", "Over200D%"),
        one("funding_z", "Funding‑Z"),
        one("ahr", "AHR"),
        one("vol", "Vol30d"),
        one("breadth", "Breadth"),
        f"*TOTAL* = {score:.1f}"
    ]
    return "\n".join(lines)

# ------------------------------
# MAIN
# ------------------------------
def run() -> None:
    os.makedirs("out", exist_ok=True)
    ts_utc = now_utc_str()

    # Fetch market series
    btc_df = fetch_ohlcv_daily("BTC/USDT", 400)
    eth_df = fetch_ohlcv_daily("ETH/USDT", 400)
    btc = build_snapshot("BTC/USDT", "BTC/USDT:USDT")
    eth = build_snapshot("ETH/USDT", "ETH/USDT:USDT")

    # Rotation / sentiment
    try:
        ethbtc = fetch_ticker_price("ETH/BTC")
    except Exception:
        ethbtc = float(eth.price / btc.price) if btc.price else float("nan")
    fg = fetch_fear_greed()
    if fg is None: fg = 50

    # Breadth across BTC & ETH
    breadth = ((1.0 if btc.price > btc.sma200 else 0.0) + (1.0 if eth.price > eth.sma200 else 0.0)) / 2.0

    # Scores (v4o-style) + component dicts
    btc_score, btc_comp = compute_score_components_v4o(btc, fg, breadth)
    eth_score, eth_comp = compute_score_components_v4o(eth, fg, breadth)
    btc.score, btc.components = btc_score, btc_comp
    eth.score, eth.components = eth_score, eth_comp

    # Regimes (v4o simple)
    btc_state = pick_regime_v4o(btc.over_200d_pct, btc_score)
    eth_state = pick_regime_v4o(eth.over_200d_pct, eth_score)

    # ---- JSON EXPORT (with breakdowns) ----
    out_json = {
        "timestamp_utc": ts_utc,
        "prices": {"BTC": round(btc.price,2), "ETH": round(eth.price,2), "ETH/BTC": ethbtc},
        "sma": {
            "BTC": {"SMA50": round(btc.sma50,2), "SMA200": round(btc.sma200,2), "Spot/200D": round(btc.price/btc.sma200, 3)},
            "ETH": {"SMA50": round(eth.sma50,2), "SMA200": round(eth.sma200,2), "Spot/200D": round(eth.price/eth.sma200, 3)},
        },
        "scores": {"BTC": btc_score, "ETH": eth_score},
        "states": {"BTC": btc_state, "ETH": eth_state},
        "components": {"BTC": btc_comp, "ETH": eth_comp},
        "funding_z": {"BTC": btc.funding_z, "ETH": eth.funding_z},
        "sentiment": {"FearGreed_Now": fg},
        "rotation": {"ETHBTC": ethbtc}
    }
    export_path = os.path.join("out", "iSun_Wealth_Brief_v4o_detail.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    # ---- TELEGRAM (compact v4o style + breakdowns) ----
    arrow_b = "↗️" if btc.sma50_slope_up else "↘️"
    arrow_e = "↗️" if eth.sma50_slope_up else "↘️"
    header_line = f"*iSun Wealth — Morning Brief* ({ts_utc})\nS2 — HOLD / Do Nothing."  # keep v4o vibe

    btc_block = (f"\n\n*BTC*\n"
                 f"Price: {btc.price:,.2f}\n"
                 f"SMA50: {btc.sma50:,.2f} {arrow_b} | SMA200: {btc.sma200:,.2f}\n"
                 f"Spot/200D: {btc.price/btc.sma200:.2f}x\n"
                 f"The Score: {btc_score:.1f} / 100\n"
                 f"Regime: {btc_state}")

    eth_block = (f"\n\n*ETH*\n"
                 f"Price: {eth.price:,.2f}\n"
                 f"SMA50: {eth.sma50:,.2f} {arrow_e} | SMA200: {eth.sma200:,.2f}\n"
                 f"Spot/200D: {eth.price/eth.sma200:.2f}x\n"
                 f"The Score: {eth_score:.1f} / 100\n"
                 f"Regime: {eth_state}")

    rotation = f"\n\n*Rotation*: ETH/BTC = {ethbtc:.5f} | *F&G*: {fg}"

    # Component breakdowns (new)
    btc_bd = build_breakdown_lines("BTC", btc, btc_score, btc_comp)
    eth_bd = build_breakdown_lines("ETH", eth, eth_score, eth_comp)

    text = f"{header_line}\n{btc_block}\n{eth_block}\n{rotation}\n\n{btc_bd}\n\n{eth_bd}"

    msg_id = send_telegram(text)
    print(f"[iSunOne_Wealth_Bot] Export: {export_path} | Telegram: {msg_id if msg_id else 'sent' if TELEGRAM_ENABLED else 'disabled'}")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("Run error:", e)
        traceback.print_exc()


