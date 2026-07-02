
import re
import time
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://merolagani.com/handlers/TechnicalChartHandler.ashx"
MARKET_PAGE_URL = "https://merolagani.com/LatestMarket.aspx"

NEPAL_TZ = ZoneInfo("Asia/Kathmandu")
MARKET_OPEN = dtime(11, 0)
MARKET_CLOSE = dtime(15, 0)
MARKET_WEEKDAYS = {6, 0, 1, 2, 3}  # Sun=6, Mon=0 ... Thu=3 (Python's Monday=0 convention)


def is_market_open_now():
    """NEPSE trades Sun-Thu, 11:00-15:00 Nepal time. Best-effort — no official holiday calendar."""
    now = datetime.now(NEPAL_TZ)
    return now.weekday() in MARKET_WEEKDAYS and MARKET_OPEN <= now.time() <= MARKET_CLOSE


def fetch_ohlcv(symbol, resolution="1D", start=None, end=None, currency="NPR"):
    """Fetch OHLCV bars for a NEPSE symbol.

    Returns a DataFrame indexed by date with columns open/high/low/close/volume.
    Raises ValueError if the API has no data for the symbol/range.
    """
    if end is None:
        end = int(time.time())
    if start is None:
        start = end - 5 * 365 * 24 * 3600

    params = {
        "type": "get_advanced_chart",
        "symbol": symbol,
        "resolution": resolution,
        "isAdjust": 1,
        "currencyCode": currency,
        "rangeStartDate": int(start),
        "rangeEndDate": int(end),
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(BASE_URL, params=params, headers=headers, timeout=15)
    resp.raise_for_status()
    payload = resp.json()

    status = payload.get("s")
    if status != "ok" or not payload.get("t"):
        raise ValueError(f"no data for symbol '{symbol}' (status={status})")

    df = pd.DataFrame(
        {
            "date": pd.to_datetime(payload["t"], unit="s"),
            "open": payload["o"],
            "high": payload["h"],
            "low": payload["l"],
            "close": payload["c"],
            "volume": payload["v"],
        }
    ).set_index("date").sort_index()
    df = df[~df.index.duplicated(keep="last")]
    return df


def fetch_symbol_list():
    """Scrape the full NEPSE symbol list (symbol -> company name) from merolagani's market page.

    Returns a dict like {"NABIL": "Nabil Bank Limited", ...}. Raises on network failure.
    """
    resp = requests.get(MARKET_PAGE_URL, headers={"User-Agent": "Mozilla/5.0"}, timeout=20)
    resp.raise_for_status()
    pattern = re.compile(
        r"CompanyDetail\.aspx\?symbol=([A-Z0-9]+)'\s+title='[A-Z0-9]+\s+\(([^)]+)\)'"
    )
    symbols = {}
    for sym, name in pattern.findall(resp.text):
        symbols[sym] = name.strip()
    if not symbols:
        raise ValueError("no symbols parsed from market page (page layout may have changed)")
    return dict(sorted(symbols.items()))


def liquidity_stats(df):
    """Liquidity snapshot for one symbol's OHLCV frame: turnover, volume, dead days."""
    turnover = df["close"] * df["volume"]
    recent = df.tail(20)
    return {
        "avg_daily_turnover_npr": turnover.mean(),
        "recent_avg_turnover_npr": (recent["close"] * recent["volume"]).mean(),
        "median_daily_volume": df["volume"].median(),
        "zero_volume_days_pct": (df["volume"] <= 0).mean() * 100,
        "high_52w": df["high"].tail(252).max(),
        "low_52w": df["low"].tail(252).min(),
    }


def compute_metrics(close, periods_per_year=252):
    """Return total return / CAGR / annualized vol / naive Sharpe / max drawdown for a close-price series."""
    empty = {
        "total_return_pct": np.nan,
        "cagr_pct": np.nan,
        "ann_volatility_pct": np.nan,
        "sharpe_naive": np.nan,
        "max_drawdown_pct": np.nan,
    }
    if len(close) < 2:
        return empty

    ret = close.pct_change().dropna()
    cum = (1 + ret).cumprod()
    if not len(cum):
        return empty
    drawdown = cum / cum.cummax() - 1

    total_return = cum.iloc[-1] - 1
    years = (close.index[-1] - close.index[0]).days / 365.25
    cagr = (cum.iloc[-1]) ** (1 / years) - 1 if years > 0 else np.nan
    vol_annual = ret.std() * np.sqrt(periods_per_year)
    sharpe = (ret.mean() * periods_per_year) / vol_annual if vol_annual else np.nan

    return {
        "total_return_pct": total_return * 100,
        "cagr_pct": cagr * 100,
        "ann_volatility_pct": vol_annual * 100,
        "sharpe_naive": sharpe,
        "max_drawdown_pct": drawdown.min() * 100,
    }


def cumulative_growth(close):
    ret = close.pct_change().dropna()
    return (1 + ret).cumprod()


def drawdown_series(close):
    cum = cumulative_growth(close)
    return cum / cum.cummax() - 1


def sma_crossover_backtest(df, fast=20, slow=50):
    """Backtest a long-only SMA crossover rule against buy & hold. Not a forecast."""
    d = df.copy()
    d["sma_fast"] = d["close"].rolling(fast).mean()
    d["sma_slow"] = d["close"].rolling(slow).mean()
    d["signal"] = (d["sma_fast"] > d["sma_slow"]).astype(int)
    d["strategy_ret"] = d["signal"].shift(1) * d["close"].pct_change()
    d["buy_hold_cum"] = (1 + d["close"].pct_change()).cumprod()
    d["strategy_cum"] = (1 + d["strategy_ret"].fillna(0)).cumprod()
    return d


def interpret_volatility(vol_pct):
    if np.isnan(vol_pct):
        return "not enough data yet"
    if vol_pct < 20:
        return "low — price moves gently relative to typical NEPSE stocks"
    if vol_pct < 45:
        return "moderate — roughly typical swings for a NEPSE-listed stock"
    return "high — this stock swings much more than average, expect bigger ups and downs"


def interpret_drawdown(dd_pct):
    if np.isnan(dd_pct):
        return "not enough data yet"
    dd_pct = abs(dd_pct)
    if dd_pct < 15:
        return "shallow — this stock hasn't fallen far from its peak during the window shown"
    if dd_pct < 40:
        return "meaningful — a holder would have seen a real, uncomfortable dip at some point"
    return "severe — this stock lost more than 40% from a prior peak at some point in this window"


def sma(series, window):
    return series.rolling(window).mean()


def ema(series, span):
    return series.ewm(span=span, adjust=False).mean()


def bollinger_bands(series, window=20, num_std=2):
    mid = sma(series, window)
    std = series.rolling(window).std()
    return mid, mid + num_std * std, mid - num_std * std


def rsi(series, window=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window).mean()
    avg_loss = loss.rolling(window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def macd(series, fast=12, slow=26, signal=9):
    macd_line = ema(series, fast) - ema(series, slow)
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def summarize_indicators(close):
    """Plain-text snapshot of RSI/SMA-trend/MACD state, for feeding to the AI chat as context."""
    if len(close) < 20:
        return "insufficient history for indicators"

    parts = [f"RSI(14)={rsi(close).iloc[-1]:.1f}"]

    if len(close) >= 50:
        sma20, sma50 = sma(close, 20).iloc[-1], sma(close, 50).iloc[-1]
        trend = "above" if sma20 > sma50 else "below"
        parts.append(f"SMA20 {trend} SMA50 ({sma20:.2f} vs {sma50:.2f})")

    _, _, hist = macd(close)
    hist_valid = hist.dropna()
    if len(hist_valid):
        latest_hist = hist_valid.iloc[-1]
        sign = "positive" if latest_hist >= 0 else "negative"
        parts.append(f"MACD histogram {sign} ({latest_hist:.3f})")

    return ", ".join(parts)


def interpret_correlation(corr):
    if corr > 0.7:
        return "move together strongly — holding both adds little diversification"
    if corr > 0.3:
        return "somewhat linked — partial diversification benefit"
    if corr > -0.3:
        return "largely independent — decent diversification benefit"
    return "move in opposite directions — strong diversification benefit, but check this isn't a data artifact"
