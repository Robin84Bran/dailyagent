#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
iSunOne Wealth Bot — v4s (Telegram fallback enabled)
- Output JSON structure IDENTICAL to v4q
- Same scoring pipeline as v4q; stricter S3/S5 triggers only
- Telegram now: env vars OR fall back to ruleset.yml: telegram.bot_token/chat_id
"""

import os, json, math, statistics, traceback, requests
import datetime as dt
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import pandas as pd
import ccxt

try:
    import yaml
    HAVE_YAML = True
except Exception:
    HAVE_YAML = False

# ------------------------------
# Defaults = v4q + advanced knobs (overridable via ruleset.yml: advanced:)
# ------------------------------
DEFAULT_RULESET: Dict[str, Any] = {
    "header": {"mode": "global"},  # global | any_S3 | both_S3 | weighted
    "score": {
        "weights": {
            "fear_greed": 0.25,
            "over_200d": 0.20,
            "funding_z": 0.20,
            "ahr": 0.20,
            "vol": 0.10,
            "breadth": 0.05
        },
        "thresholds": {"s3_up": 80, "s3_down": 70},
        "funding_map": {"neg": -2.0, "neu": 0.0, "pos": 2.0},
        "ahr_hot": 1.35,
        "fg_euphoria": 85,
        "over200d_hot_pct": 50.0
    },
    "advanced": {
        "ath_band": {"low": -0.03, "high": 0.02},                 # -3%…+2%
        "cycle":    {"L": 15500.0, "H": 124450.0, "fib": 0.786, "s3_pullback_exit_pct": -0.08},
        "s3_need_overrides": 2,
        "s5_need_coolers": 2,
        "s5": {"funding_leq_zero_days": 7, "ethbtc_slope_days": 7, "need_sma200_reclaim": True},
        "s3_oi_rise_7d_pct": 10.0
    },
    # NEW: optional Telegram fallback (only used if env vars are absent)
    "telegram": {
        "bot_token": "",
        "chat_id": ""
    }
}

def load_ruleset(path: str = "ruleset.yml") -> Dict[str, Any]:
    cfg = DEFAULT_RULESET.copy()
    if HAVE_YAML and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            # v4q-style shallow merges
            cfg["header"] = {**cfg["header"], **(data.get("header", {}) or {})}
            sc = {**cfg["score"], **(data.get("score", {}) or {})}
            sc["weights"]     = {**cfg["score"]["weights"], **(sc.get("weights", {}) or {})}
            sc["thresholds"]  = {**cfg["score"]["thresholds"], **(sc.get("thresholds", {}) or {})}
            sc["funding_map"] = {**cfg["score"]["funding_map"], **(sc.get("funding_map", {}) or {})}
            cfg["score"] = sc
            adv = {**cfg["advanced"], **(data.get("advanced", {}) or {})}
            adv["ath_band"] = {**cfg["advanced"]["ath_band"], **(adv.get("ath_band", {}) or {})}
            adv["cycle"]    = {**cfg["advanced"]["cycle"], **(adv.get("cycle", {}) or {})}
            adv["s5"]       = {**cfg["advanced"]["s5"], **(adv.get("s5", {}) or {})}
            cfg["advanced"] = adv
            # telegram fallback
            if "telegram" in data:
                cfg["telegram"] = {**cfg["telegram"], **(data.get("telegram") or {})}
        except Exception:
            pass
    return cfg

RULESET = load_ruleset()

# ------------------------------
# CCXT connections
# ------------------------------
BINANCE     = ccxt.binance()
BINANCE_FUT = ccxt.binance({"options": {"defaultType": "future"}})

# ------------------------------
# Telegram: env first, else ruleset fallback (so "off" disappears)
# ------------------------------
TELEGRAM_BOT_TOKEN = "8105826927:AAF3EIAa0I4rw9Nkyw22e9h88Tpxb1GD5Ds"
TELEGRAM_CHAT_ID   = "760905049"
TELEGRAM_ENABLED   = bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)

def send_telegram(text: str):
    if not TELEGRAM_ENABLED: return None
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        r = requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": text, "parse_mode": "Markdown"}, timeout=10)
        if r.status_code == 200:
            return r.json().get("result", {}).get("message_id")
    except Exception:
        pass
    return None

# ------------------------------
# Data & indicators (as in v4q)
# ------------------------------
@dataclass
class AssetSnapshot:
    symbol: str
    price: float
    sma50: float
    sma200: float
    sma50_slope_up: bool
    over_200d_pct: float
    ahr: float
    vol30_ann_pct: float
    funding_z: Optional[float]
    score: Optional[float] = None
    state: Optional[str] = None
    components: Optional[Dict[str, Dict[str, float]]] = None

def now_utc_str() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

def fetch_ohlcv_daily(symbol: str, limit: int = 400) -> pd.DataFrame:
    o = BINANCE.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
    df = pd.DataFrame(o, columns=["ts","open","high","low","close","vol"])
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
        df["dt"] = pd.to_datetime(df["timestamp"] if "timestamp" in df.columns else df["info"].apply(lambda r: int(r["fundingTime"])), unit="ms", utc=True)
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

def fetch_oi_hist_daily_binance(symbol: str, limit: int = 8) -> pd.DataFrame:
    try:
        url = "https://fapi.binance.com/futures/data/openInterestHist"
        params = {"symbol": symbol, "period": "1d", "limit": limit}
        j = requests.get(url, params=params, timeout=8).json()
        df = pd.DataFrame(j)
        if len(df) == 0: return df
        df["ts"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df["oi_val"] = pd.to_numeric(df["sumOpenInterestValue"], errors="coerce")
        return df[["ts","oi_val"]]
    except Exception:
        return pd.DataFrame()

def calc_sma(arr: pd.Series, n: int) -> float:
    return float(arr.tail(n).mean())

def calc_vol_annualized(closes: pd.Series, lookback: int = 30) -> float:
    rets = closes.pct_change().dropna().tail(lookback)
    if len(rets) == 0: return float("nan")
    daily = float(rets.std())
    return daily * math.sqrt(365) * 100.0

def slope_up(series: pd.Series, n: int = 5) -> bool:
    if len(series) < n+1: return False
    s = series.tail(n+1).reset_index(drop=True)
    return bool(s.iloc[-1] > s.iloc[0])

def zscore_last(values: List[float], window: int = 48) -> Optional[float]:
    arr = values[-window:] if len(values) >= window else values
    if len(arr) < 5: return None
    mu = statistics.mean(arr[:-1]); sd = statistics.pstdev(arr[:-1]) or 1e-9
    return max(min((arr[-1] - mu) / sd, 4.0), -4.0)

def build_snapshot(symbol_spot: str, symbol_perp: str) -> AssetSnapshot:
    df = fetch_ohlcv_daily(symbol_spot, 400)
    closes = df["close"]
    price = float(closes.iloc[-1])
    sma50 = calc_sma(closes, 50)
    sma200 = calc_sma(closes, 200)
    over200 = (price - (sma200 or 1e-9)) / (sma200 or 1e-9) * 100.0
    ahr = price / (sma200 or 1e-9)
    vol30a = calc_vol_annualized(closes, 30)
    fdf = fetch_perp_funding_history(symbol_perp, 60)
    fz = None
    if len(fdf) > 0:
        fz = zscore_last(list(pd.to_numeric(fdf["fundingRate"], errors="coerce").dropna().tail(48)))
    return AssetSnapshot(symbol_spot, price, sma50, sma200, slope_up(closes.rolling(50).mean(), 5),
                         over200, ahr, vol30a, fz)

# ------------------------------
# Score (identical to v4q)
# ------------------------------
def map_fear_greed(x: Optional[int]) -> float:
    if x is None: return 50.0
    return float(x)

def map_over200d_pct(pct: float) -> float:
    if pct <= -20: return 0.0
    if pct >= 50:  return 100.0
    return (pct + 20) / 70.0 * 100.0

def map_funding_z(z: Optional[float]) -> float:
    if z is None: return 50.0
    zc = max(-2.0, min(2.0, float(z)))
    return (zc + 2.0) / 4.0 * 100.0

def map_ahr(ratio: float, hot: float) -> float:
    if ratio <= 0.70: return 0.0
    if ratio >= 2.00: return 100.0
    if ratio <= 1.0:  return (ratio - 0.70) / 0.30 * 60.0
    if ratio <= hot:  return 60.0 + (ratio - 1.0) / (hot - 1.0) * 25.0
    return 85.0 + (ratio - hot) / (2.0 - hot) * 15.0

def map_vol30_ann(vpct: float) -> float:
    if math.isnan(vpct): return 50.0
    v = max(5.0, min(200.0, vpct))
    if v <= 10: return 40.0
    if v <= 40: return 60.0 + (v - 10) / 30.0 * 20.0
    if v <= 90: return 80.0 - (v - 40) / 50.0 * 20.0
    return 55.0 - min(45.0, (v - 90) / 110.0 * 25.0)

def map_breadth(frac: float) -> float:
    return float(max(0.0, min(1.0, frac)) * 100.0)

def compute_score_components(snap: AssetSnapshot, fg_now: Optional[int],
                             breadth_frac: float, cfg: Dict[str, Any]) -> (float, Dict[str, Dict[str, float]]):
    w  = cfg["score"]["weights"]; thr = cfg["score"]
    comp = {}
    comp["fear_greed"] = {"raw": float(fg_now or 50), "sub": map_fear_greed(fg_now), "w": w["fear_greed"]}
    comp["over_200d"]  = {"raw": snap.over_200d_pct, "sub": map_over200d_pct(snap.over_200d_pct), "w": w["over_200d"]}
    comp["funding_z"]  = {"raw": float(snap.funding_z or 0.0), "sub": map_funding_z(snap.funding_z), "w": w["funding_z"]}
    comp["ahr"]        = {"raw": snap.ahr, "sub": map_ahr(snap.ahr, float(thr["ahr_hot"])), "w": w["ahr"]}
    comp["vol"]        = {"raw": snap.vol30_ann_pct, "sub": map_vol30_ann(snap.vol30_ann_pct), "w": w["vol"]}
    comp["breadth"]    = {"raw": breadth_frac, "sub": map_breadth(breadth_frac), "w": w["breadth"]}
    score = 0.0
    for k, d in comp.items():
        d["contrib"] = d["sub"] * d["w"]
        score += d["contrib"]
    return score, comp

# ------------------------------
# Confirmers / coolers
# ------------------------------
def s3_overrides(snap: AssetSnapshot, fg_now: Optional[int], cfg: Dict[str, Any],
                 pi_top_cross: bool, oi_hot: bool) -> int:
    thr = cfg["score"]; count = 0
    if snap.ahr >= float(thr["ahr_hot"]): count += 1
    if (snap.funding_z or 0.0) >= float(thr["funding_map"]["pos"]): count += 1
    if (fg_now or 50) >= int(thr["fg_euphoria"]): count += 1
    if snap.over_200d_pct >= float(thr["over200d_hot_pct"]): count += 1
    if pi_top_cross: count += 1
    if oi_hot: count += 1
    return count

def detect_sideways(snap: AssetSnapshot) -> bool:
    near_200d = abs(snap.over_200d_pct) < 5.0
    flat_50d  = (not snap.sma50_slope_up) and abs((snap.sma50 - (snap.sma200 or 1e-9)) / (snap.sma200 or 1e-9)) < 0.02
    return bool(near_200d and flat_50d)

def ethbtc_slope_up(ethbtc_series: pd.Series, days: int) -> bool:
    if len(ethbtc_series) < days + 1: return False
    return float(ethbtc_series.iloc[-1]) >= float(ethbtc_series.iloc[-days-1])

def near_ath_flag(price: float, series_ath: float, cfg_adv: Dict[str, Any]) -> (bool, float, float):
    H_cfg = float(cfg_adv["cycle"]["H"])
    ATH = max(float(series_ath), H_cfg)
    dist = price/ATH - 1.0
    low, high = float(cfg_adv["ath_band"]["low"]), float(cfg_adv["ath_band"]["high"])
    return (low <= dist <= high), dist, ATH

def fib_0786_target(cfg_adv: Dict[str, Any]) -> float:
    L = float(cfg_adv["cycle"]["L"]); H = float(cfg_adv["cycle"]["H"]); r = float(cfg_adv["cycle"]["fib"])
    return H - r*(H - L)

# ------------------------------
# Cache (same filename as v4q)
# ------------------------------
STATE_CACHE = "out/state_cache.json"

def load_last_state() -> Dict[str, Any]:
    try:
        with open(STATE_CACHE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_last_state(d: Dict[str, Any]) -> None:
    os.makedirs("out", exist_ok=True)
    with open(STATE_CACHE, "w", encoding="utf-8") as f:
        json.dump(d, f, indent=2)

# ------------------------------
# Pi-cycle proxy
# ------------------------------
def pi_cycle_top_cross(df: pd.DataFrame) -> bool:
    ma111 = df["close"].rolling(111).mean()
    ma350 = df["close"].rolling(350).mean()
    if len(ma111.dropna()) < 2 or len(ma350.dropna()) < 2: return False
    return bool((ma111.iloc[-2] <= ma350.iloc[-2]) and (ma111.iloc[-1] > ma350.iloc[-1]))

# ------------------------------
# Classifier (stricter triggers; outputs unchanged)
# ------------------------------
def pick_state(asset: str, snap: AssetSnapshot, score: float,
               fg_now: Optional[int], cfg: Dict[str, Any],
               pi_top_cross: bool, last_state: str,
               near_ath: bool, dd_from_ath: float,
               s5_ready: bool, overrides_count: int) -> str:

    s3_up   = int(cfg["score"]["thresholds"]["s3_up"])
    s3_down = int(cfg["score"]["thresholds"]["s3_down"])
    adv     = cfg["advanced"]

    if s5_ready:
        return "S5"

    if snap.over_200d_pct <= -5.0:
        return "S1"

    if detect_sideways(snap):
        return "S4"

    if last_state == "S3":
        if (score >= s3_down or overrides_count >= int(adv["s3_need_overrides"])) and (dd_from_ath > float(adv["cycle"]["s3_pullback_exit_pct"])):
            return "S3"
    else:
        if near_ath and (score >= s3_up) and (overrides_count >= int(adv["s3_need_overrides"])):
            return "S3"

    return "S2"

# ------------------------------
# Brief formatting (same as v4q)
# ------------------------------
def build_breakdown_lines(tag: str, snap: AssetSnapshot, score: float, comp: Dict[str, Dict[str, float]]) -> str:
    def one(k, label):
        d = comp[k]; raw=d["raw"]; sub=d["sub"]; w=d["w"]; contrib=d["contrib"]
        if k == "over_200d": rawtxt=f"{raw:.1f}%"
        elif k == "vol":    rawtxt=f"{raw:.1f}%"
        elif k == "ahr":    rawtxt=f"{raw:.2f}"
        elif k == "funding_z": rawtxt=f"{raw:.2f}"
        elif k == "breadth": rawtxt=f"{raw:.2f}"
        else: rawtxt=str(raw)
        return f"{label}: raw {rawtxt} → {sub:.0f}×{w:.2f} = {contrib:.0f}"
    lines = [
        f"*{tag}* — px {snap.price:,.0f} | 50D {snap.sma50:,.0f} | 200D {snap.sma200:,.0f} | over200 {snap.over_200d_pct:.1f}%",
        one("fear_greed","F&G"), one("over_200d","Over200D"), one("funding_z","Funding-Z"),
        one("ahr","AHR"), one("vol","Vol30a"), one("breadth","Breadth"),
        f"Score: *{score:.0f}*"
    ]
    return "\n".join(lines)

# ------------------------------
# Main run (keeps v4q’s export JSON schema & filenames)
# ------------------------------
def run() -> None:
    os.makedirs("out", exist_ok=True)
    ts_utc = now_utc_str()
    cfg = RULESET
    adv = cfg["advanced"]

    btc_df = fetch_ohlcv_daily("BTC/USDT", 400)
    eth_df = fetch_ohlcv_daily("ETH/USDT", 400)

    btc = build_snapshot("BTC/USDT", "BTC/USDT:USDT")
    eth = build_snapshot("ETH/USDT", "ETH/USDT:USDT")

    try:
        ethbtc = fetch_ticker_price("ETH/BTC")
    except Exception:
        ethbtc = float(eth.price / (btc.price or 1e-9))
    fg = fetch_fear_greed() or 50

    breadth = ((1.0 if btc.price > btc.sma200 else 0.0) + (1.0 if eth.price > eth.sma200 else 0.0)) / 2.0

    btc_score, btc_comp = compute_score_components(btc, fg, breadth, cfg)
    eth_score, eth_comp = compute_score_components(eth, fg, breadth, cfg)
    btc.score, btc.components = btc_score, btc_comp
    eth.score, eth.components = eth_score, eth_comp

    pi_btc = pi_cycle_top_cross(btc_df)
    pi_eth = pi_cycle_top_cross(eth_df)

    oi_df = fetch_oi_hist_daily_binance("BTCUSDT", 8)
    btc_oi_hot = False
    if len(oi_df) >= 2:
        rise = (oi_df["oi_val"].iloc[-1] - oi_df["oi_val"].iloc[0]) / max(oi_df["oi_val"].iloc[0], 1e-9) * 100.0
        btc_oi_hot = bool(rise >= float(adv["s3_oi_rise_7d_pct"]))

    ethbtc_series = (eth_df["close"] / btc_df["close"]).dropna()
    ethbtc_up = ethbtc_slope_up(ethbtc_series, int(adv["s5"]["ethbtc_slope_days"]))

    series_ath = float(btc_df["close"].max())
    near_ath, dd_from_ath, ATH_used = near_ath_flag(btc.price, series_ath, adv)

    fib_target = fib_0786_target(adv)
    fdf = fetch_perp_funding_history("BTC/USDT:USDT", 90)
    funding_nonpos = False
    if len(fdf) > 0:
        fr = pd.to_numeric(fdf["fundingRate"], errors="coerce").dropna()
        need = int(adv["s5"]["funding_leq_zero_days"])
        funding_nonpos = bool(len(fr.tail(need)) == need and (fr.tail(need) <= 0).all())
    sma200_reclaim = bool((btc.price >= btc.sma200) if adv["s5"]["need_sma200_reclaim"] else True)
    s5_cool_count = int(funding_nonpos) + int(ethbtc_up) + int(sma200_reclaim)
    s5_ready = bool((btc.price <= fib_target) and (s5_cool_count >= int(adv["s5_need_coolers"])))

    last = load_last_state()
    last_btc = last.get("BTC_state", "S2")
    last_eth = last.get("ETH_state", "S2")

    btc_over = s3_overrides(btc, fg, cfg, pi_btc, btc_oi_hot)
    eth_over = s3_overrides(eth, fg, cfg, pi_eth, btc_oi_hot)

    btc_state = pick_state("BTC", btc, btc_score, fg, cfg, pi_btc, last_btc,
                           near_ath, dd_from_ath, s5_ready, btc_over)
    near_ath_e, dd_from_ath_e, _ = near_ath_flag(eth.price, float(eth_df["close"].max()), adv)
    eth_state = pick_state("ETH", eth, eth_score, fg, cfg, pi_eth, last_eth,
                           near_ath_e, dd_from_ath_e, s5_ready, eth_over)

    def merge_states(btc_state: str, eth_state: str, mode: str = "global") -> str:
        if mode == "any_S3":
            if btc_state == "S3" or eth_state == "S3": return "S3"
            if "S5" in (btc_state, eth_state): return "S5"
            if "S1" in (btc_state, eth_state): return "S1"
            if "S4" in (btc_state, eth_state): return "S4"
            return "S2"
        if mode == "both_S3":
            if btc_state == "S3" and eth_state == "S3": return "S3"
            if "S5" in (btc_state, eth_state): return "S5"
            if "S1" in (btc_state, eth_state): return "S1"
            if "S4" in (btc_state, eth_state): return "S4"
            return "S2"
        if mode == "weighted":
            sev = {"S5":0, "S1":1, "S4":2, "S2":2, "S3":3}
            s = (sev.get(btc_state,2) + sev.get(eth_state,2)) / 2.0
            return "S3" if s>=2.5 else ("S1" if s<=0.5 else ("S5" if s<0.5 else "S2"))
        if eth_state == "S3": return "S3"
        return btc_state

    header_mode  = RULESET.get("header", {}).get("mode", "global")
    header_state = merge_states(btc_state, eth_state, header_mode)

    save_last_state({"BTC_state": btc_state, "ETH_state": eth_state, "ts": ts_utc})

    out_json = {
        "timestamp_utc": ts_utc,
        "header_mode": header_mode,
        "header_state": header_state,
        "prices": {"BTC": round(btc.price,2), "ETH": round(eth.price,2), "ETH/BTC": ethbtc},
        "sma": {
            "BTC": {"SMA50": round(btc.sma50,2), "SMA200": round(btc.sma200,2), "Spot/200D": round(btc.price/(btc.sma200 or 1e-9), 3)},
            "ETH": {"SMA50": round(eth.sma50,2), "SMA200": round(eth.sma200,2), "Spot/200D": round(eth.price/(eth.sma200 or 1e-9), 3)},
        },
        "scores": {"BTC": btc_score, "ETH": eth_score},
        "states": {"BTC": btc_state, "ETH": eth_state, "header": header_state},
        "components": { "BTC": btc_comp, "ETH": eth_comp },
        "funding_z": {"BTC": btc.funding_z, "ETH": eth.funding_z},
        "sentiment": {"FearGreed_Now": fg},
        "rotation": {"ETHBTC": ethbtc}
    }
    os.makedirs("out", exist_ok=True)
    with open(os.path.join("out","iSun_Wealth_Brief_v4s.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    # backward-compat drop (same name as v4q)
    with open(os.path.join("out","iSun_Wealth_Brief_v4q.json"), "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)

    arrow_b = "↗️" if btc.sma50_slope_up else "↘️"
    arrow_e = "↗️" if eth.sma50_slope_up else "↘️"
    header_line = f"*iSun Wealth — Morning Brief* ({ts_utc})\n*Regime*: {header_state} | *Header.mode*: {header_mode}"
    market = (f"\n\n*BTC* {btc.price:,.0f} {arrow_b} | SMA50 {btc.sma50:,.0f} | SMA200 {btc.sma200:,.0f} | Spot/200D {btc.price/(btc.sma200 or 1e-9):.3f}"
              f"\n*ETH* {eth.price:,.0f} {arrow_e} | SMA50 {eth.sma50:,.0f} | SMA200 {eth.sma200:,.0f} | Spot/200D {eth.price/(eth.sma200 or 1e-9):.3f}")
    rotation = f"\nETH/BTC {ethbtc:.4f} | Funding-Z {btc.funding_z or 0:.2f}"

    btc_bd = build_breakdown_lines("BTC", btc, btc_score, btc_comp)
    eth_bd = build_breakdown_lines("ETH", eth, eth_score, eth_comp)

    text = f"{header_line}\n{market}\n{rotation}\n\n{btc_bd}\n\n{eth_bd}"
    msg_id = send_telegram(text)

    print(f"[iSunOne_Wealth_Bot_v4s] {header_state} | Export: out/iSun_Wealth_Brief_v4s.json"
          f" | Telegram: {'ok' if msg_id else ('off' if not TELEGRAM_ENABLED else 'fail')}")

if __name__ == "__main__":
    try:
        run()
    except Exception as e:
        print("Run error:", e)
        traceback.print_exc()
