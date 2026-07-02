"""Torendo — NEPSE market analyzer on top of merolagani's chart API."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from ai import build_context, get_client, stream_reply
from data import (
    NEPAL_TZ,
    bollinger_bands,
    compute_metrics,
    cumulative_growth,
    drawdown_series,
    ema,
    fetch_ohlcv,
    fetch_symbol_list,
    interpret_correlation,
    interpret_drawdown,
    interpret_volatility,
    is_market_open_now,
    liquidity_stats,
    macd,
    rsi,
    sma,
    sma_crossover_backtest,
    summarize_indicators,
)

LIVE_REFRESH_SECONDS = 30

# Palette validated for lightness band, chroma, CVD separation, and 3:1 contrast
# on a light surface — keep the order fixed (assign by slot, never re-cycle).
COLORS = ["#3366CC", "#DC3912", "#109618", "#990099", "#B45309", "#0099C6", "#DD4477"]
CANDLE_UP = "#2E7D32"
CANDLE_DOWN = "#C62828"

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToAdd": ["drawline", "drawopenpath", "drawrect", "eraseshape"],
}


def color_for(i):
    return COLORS[i % len(COLORS)]


def get_mistral_api_key():
    try:
        key = st.secrets.get("MISTRAL_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("MISTRAL_API_KEY")


st.set_page_config(page_title="Torendo", layout="wide", initial_sidebar_state="expanded")

st.title("Torendo")
st.caption(
    "NEPSE market analyzer — price history via merolagani's public chart endpoint. This is an "
    "unofficial data source: treat it as directionally useful, not a source of truth for real "
    "trading decisions."
)


def parse_weights_param(raw):
    weights = {}
    for part in raw.split(","):
        if ":" not in part:
            continue
        sym, _, val = part.partition(":")
        try:
            weights[sym] = float(val)
        except ValueError:
            continue
    return weights


RESOLUTIONS = ["1D", "1W", "1M"]
qp = st.query_params
qp_symbols = [s for s in qp.get("symbols", "").split(",") if s]
qp_weights = parse_weights_param(qp.get("weights", ""))
qp_resolution = qp.get("res") if qp.get("res") in RESOLUTIONS else "1D"
try:
    qp_start = date.fromisoformat(qp.get("from", ""))
except ValueError:
    qp_start = date.today() - timedelta(days=5 * 365)
try:
    qp_end = date.fromisoformat(qp.get("to", ""))
except ValueError:
    qp_end = date.today()

# ---------------------------------------------------------------------------
# Inputs (top of page)
# ---------------------------------------------------------------------------


@st.cache_data(ttl=86400, show_spinner=False)
def load_symbol_list():
    return fetch_symbol_list()


try:
    symbol_names = load_symbol_list()
except Exception:
    symbol_names = {}

if "extra_symbols" not in st.session_state:
    st.session_state.extra_symbols = []

with st.container(border=True):
    all_options = sorted(set(symbol_names) | set(qp_symbols) | set(st.session_state.extra_symbols))

    selected_symbols = st.multiselect(
        "Stocks — type to search all NEPSE-listed symbols",
        options=all_options,
        default=[s for s in qp_symbols if s in all_options],
        format_func=lambda s: f"{s} — {symbol_names[s]}" if s in symbol_names else s,
        placeholder="Start typing a ticker or company name (e.g. NABIL, hydro...)",
    )

    if not symbol_names:
        st.warning("Couldn't fetch the NEPSE symbol list right now — add tickers manually below.")
        with st.form("manual_add_form", clear_on_submit=True):
            manual_sym = st.text_input("Add a symbol manually")
            if st.form_submit_button("Add") and manual_sym.strip():
                sym = manual_sym.strip().upper()
                if sym not in st.session_state.extra_symbols:
                    st.session_state.extra_symbols.append(sym)
                    st.rerun()

    c1, c2, c3 = st.columns(3)
    with c1:
        start_date = st.date_input("From", value=qp_start)
    with c2:
        end_date = st.date_input("To", value=qp_end)
    with c3:
        resolution = st.selectbox("Bar size", RESOLUTIONS, index=RESOLUTIONS.index(qp_resolution))

    st.caption(
        "💾 The page URL tracks everything you set here — bookmark or share the address bar to "
        "save/restore this exact view (stocks, dates, weights included)."
    )

st.query_params.update(
    {
        "symbols": ",".join(selected_symbols),
        "from": start_date.isoformat(),
        "to": end_date.isoformat(),
        "res": resolution,
    }
)

if not selected_symbols:
    st.info("Search and select one or more NEPSE stocks above to begin.")
    st.stop()

covers_today = end_date >= date.today()
market_open = is_market_open_now()
now_npt = datetime.now(NEPAL_TZ)

if covers_today and market_open:
    badge_col, toggle_col = st.columns([3, 1])
    with toggle_col:
        live_enabled = st.toggle("Auto-refresh", value=True, help="Pause this while using the AI chat if you don't want the page refreshing mid-conversation.")
    with badge_col:
        if live_enabled:
            st.success(f"🟢 Live — market open, refreshing every {LIVE_REFRESH_SECONDS}s (last checked {now_npt:%H:%M:%S} NPT)")
        else:
            st.info("🟡 Market open — auto-refresh paused")
    if live_enabled:
        st_autorefresh(interval=LIVE_REFRESH_SECONDS * 1000, key="live_refresh")
    is_live = live_enabled
else:
    is_live = False
    if covers_today:
        st.info(f"⚫ Market closed — NEPSE trades Sun–Thu, 11:00–15:00 NPT (now {now_npt:%a %H:%M} NPT). Showing last available data.")


@st.cache_data(ttl=LIVE_REFRESH_SECONDS if is_live else 3600, show_spinner=False)
def load(symbol, resolution, start_ts, end_ts):
    return fetch_ohlcv(symbol, resolution=resolution, start=start_ts, end=end_ts)


start_ts = int(pd.Timestamp(start_date).timestamp())
end_ts = int(pd.Timestamp(end_date).timestamp()) + 86400

data = {}
errors = {}
with st.spinner("Fetching price data..."):
    with ThreadPoolExecutor(max_workers=min(8, len(selected_symbols))) as pool:
        futures = {sym: pool.submit(load, sym, resolution, start_ts, end_ts) for sym in selected_symbols}
        for sym, future in futures.items():
            try:
                data[sym] = future.result()
            except Exception as e:
                errors[sym] = str(e)

for sym, err in errors.items():
    st.warning(f"Couldn't load **{sym}**: {err}")

if not data:
    st.stop()

close_df = pd.DataFrame({sym: df["close"] for sym, df in data.items()}).dropna(how="all")

metrics_rows = {sym: compute_metrics(df["close"]) for sym, df in data.items()}
indicators_by_symbol = {sym: summarize_indicators(df["close"]) for sym, df in data.items()}

tab_overview, tab_chart, tab_risk, tab_corr, tab_liquidity, tab_portfolio, tab_backtest = st.tabs(
    ["Overview", "Chart", "Risk & Returns", "Correlation", "Liquidity", "Portfolio", "Backtest"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Price history")
    fig = go.Figure()
    for i, (sym, df) in enumerate(data.items()):
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name=sym, line=dict(color=color_for(i), width=2)))
    fig.update_layout(
        height=480,
        hovermode="x unified",
        yaxis_title="Price (NPR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=10),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            )
        ),
    )
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    st.caption("🖱️ Scroll to zoom, drag to box-zoom, double-click to reset. Drawing tools (trendline, freehand, rectangle) are in the chart's toolbar, top-right on hover.")

    with st.expander("How to read this"):
        st.markdown(
            "- This is the **raw closing price** for each stock, not a return — a stock at "
            "NPR 2000 isn't 'more expensive to own' than one at NPR 200, since a portfolio is "
            "sized in rupees, not shares.\n"
            "- Use this chart to spot the big picture: trend direction, big gaps (often "
            "dividend/bonus adjustments or halts), and how volatile the price path looks.\n"
            "- Hover over the chart to see exact prices on any date."
        )

    st.subheader("At a glance")
    cols = st.columns(len(data))
    for i, (sym, df) in enumerate(data.items()):
        m = metrics_rows[sym]
        liq = liquidity_stats(df)
        last_close = df["close"].iloc[-1]
        pct_off_high = (last_close / liq["high_52w"] - 1) * 100 if liq["high_52w"] else float("nan")
        with cols[i]:
            st.metric(sym, f"NPR {last_close:.2f}", f"{m['total_return_pct']:.1f}% over window")
            st.caption(f"52w range {liq['low_52w']:.0f}–{liq['high_52w']:.0f} · {pct_off_high:.1f}% from 52w high")

# ---------------------------------------------------------------------------
# Chart (candlesticks + indicators)
# ---------------------------------------------------------------------------


@st.fragment
def render_chart_tab():
    st.subheader("Candlestick chart with indicators")

    chart_symbol = st.selectbox("Symbol to chart", options=list(data.keys()), key="chart_symbol")
    cdf = data[chart_symbol]

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        sma_windows = st.multiselect("SMA overlays", [20, 50, 100, 200], default=[20, 50], key="chart_sma")
    with ic2:
        ema_windows = st.multiselect("EMA overlays", [12, 26, 50], default=[], key="chart_ema")
    with ic3:
        show_bbands = st.checkbox("Bollinger Bands (20, 2σ)", key="chart_bb")
    with ic4:
        show_rsi = st.checkbox("RSI (14)", value=True, key="chart_rsi")
    show_macd = st.checkbox("MACD (12, 26, 9)", value=True, key="chart_macd")

    n_rows = 2 + int(show_rsi) + int(show_macd)
    row_heights = [0.5, 0.15] + ([0.175] if show_rsi else []) + ([0.175] if show_macd else [])
    specs_titles = ["Price", "Volume"] + (["RSI"] if show_rsi else []) + (["MACD"] if show_macd else [])

    fig = make_subplots(
        rows=n_rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights,
        subplot_titles=specs_titles,
    )

    fig.add_trace(
        go.Candlestick(
            x=cdf.index,
            open=cdf["open"],
            high=cdf["high"],
            low=cdf["low"],
            close=cdf["close"],
            name=chart_symbol,
            increasing_line_color=CANDLE_UP,
            decreasing_line_color=CANDLE_DOWN,
        ),
        row=1,
        col=1,
    )

    color_i = 1  # slot 0 is visually close to the RSI/MACD line colors; start overlays at slot 1
    for w in sorted(sma_windows):
        fig.add_trace(
            go.Scatter(x=cdf.index, y=sma(cdf["close"], w), name=f"SMA{w}", line=dict(color=color_for(color_i), width=1.3)),
            row=1,
            col=1,
        )
        color_i += 1
    for w in sorted(ema_windows):
        fig.add_trace(
            go.Scatter(x=cdf.index, y=ema(cdf["close"], w), name=f"EMA{w}", line=dict(color=color_for(color_i), width=1.3, dash="dot")),
            row=1,
            col=1,
        )
        color_i += 1
    if show_bbands:
        mid, upper, lower = bollinger_bands(cdf["close"])
        fig.add_trace(go.Scatter(x=cdf.index, y=upper, name="BB upper", line=dict(color="gray", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=lower, name="BB lower", line=dict(color="gray", width=1), fill="tonexty", fillcolor="rgba(150,150,150,0.1)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=mid, name="BB mid (SMA20)", line=dict(color="gray", width=1, dash="dash")), row=1, col=1)

    vol_colors = [CANDLE_UP if c >= o else CANDLE_DOWN for o, c in zip(cdf["open"], cdf["close"])]
    fig.add_trace(go.Bar(x=cdf.index, y=cdf["volume"], name="Volume", marker_color=vol_colors, showlegend=False), row=2, col=1)

    next_row = 3
    if show_rsi:
        r = rsi(cdf["close"])
        fig.add_trace(go.Scatter(x=cdf.index, y=r, name="RSI(14)", line=dict(color="#6A1B9A")), row=next_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=next_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", opacity=0.5, row=next_row, col=1)
        fig.update_yaxes(range=[0, 100], row=next_row, col=1)
        next_row += 1
    if show_macd:
        macd_line, signal_line, hist = macd(cdf["close"])
        hist_colors = [CANDLE_UP if v >= 0 else CANDLE_DOWN for v in hist.fillna(0)]
        fig.add_trace(go.Bar(x=cdf.index, y=hist, name="MACD hist", marker_color=hist_colors, showlegend=False), row=next_row, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=macd_line, name="MACD", line=dict(color=COLORS[0])), row=next_row, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=signal_line, name="Signal", line=dict(color=COLORS[4])), row=next_row, col=1)

    fig.update_layout(
        height=340 + 180 * (n_rows - 1),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=10),
        xaxis=dict(
            rangeselector=dict(
                buttons=[
                    dict(count=1, label="1M", step="month", stepmode="backward"),
                    dict(count=3, label="3M", step="month", stepmode="backward"),
                    dict(count=6, label="6M", step="month", stepmode="backward"),
                    dict(count=1, label="1Y", step="year", stepmode="backward"),
                    dict(step="all", label="All"),
                ]
            )
        ),
    )
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)
    st.caption("🖱️ Scroll to zoom, drag to box-zoom, double-click to reset. Use the toolbar (top-right on hover) to draw trendlines, freehand marks, or rectangles on the chart — the eraser removes them.")

    with st.expander("How to read a candlestick chart"):
        st.markdown(
            "- Each **candle** covers one bar (day/week/month depending on 'Bar size' above): the "
            "thick body spans open→close (green = closed higher than it opened, red = closed "
            "lower), the thin wicks show the high/low extremes traded during that bar.\n"
            "- **SMA/EMA** lines smooth the price to show trend direction — price above a rising "
            "moving average is generally read as an uptrend, below a falling one as a downtrend. "
            "EMA reacts faster to recent price than SMA.\n"
            "- **Bollinger Bands** widen when volatility rises and squeeze when it falls; price "
            "pressing the upper/lower band is often read as short-term overbought/oversold, not "
            "as a reversal guarantee.\n"
            "- **RSI** (0–100) above 70 is conventionally 'overbought', below 30 'oversold' — but "
            "in a strong trend RSI can stay pinned near an extreme for a long time, so treat it as "
            "one input, not a standalone signal.\n"
            "- **MACD** crossing above its signal line is a classic bullish momentum cue (and vice "
            "versa); the histogram is just the gap between the two, useful for spotting momentum "
            "building or fading before an actual crossover happens."
        )


with tab_chart:
    render_chart_tab()

# ---------------------------------------------------------------------------
# Risk & Returns
# ---------------------------------------------------------------------------
with tab_risk:
    st.subheader("Return & risk metrics")

    metrics_df = pd.DataFrame(metrics_rows).T
    metrics_df.columns = ["Total return %", "CAGR %", "Ann. volatility %", "Sharpe (naive)", "Max drawdown %"]
    st.dataframe(metrics_df.style.format("{:.2f}"), width='stretch')

    with st.expander("What do these mean?"):
        st.markdown(
            "- **Total return** — how much NPR 100 invested at the start would be worth now, "
            "as a percent gain/loss.\n"
            "- **CAGR** — total return smoothed into a compounding annual rate, so stocks held "
            "over different lengths of time are comparable.\n"
            "- **Annualized volatility** — how much the price typically swings around its own "
            "trend in a year, in percentage terms. Higher = bumpier ride, not necessarily worse "
            "returns.\n"
            "- **Sharpe (naive)** — return earned per unit of volatility taken on (no risk-free "
            "rate subtracted, so treat it as a rough comparison between your own stocks, not an "
            "absolute benchmark).\n"
            "- **Max drawdown** — the worst peak-to-trough decline in the window. This is the "
            "number that answers 'how bad could this have felt to hold?'"
        )

    st.markdown("**In plain language:**")
    for sym in data:
        m = metrics_rows[sym]
        st.markdown(
            f"- **{sym}**: volatility is {interpret_volatility(m['ann_volatility_pct'])}; "
            f"worst drawdown in this window was {interpret_drawdown(m['max_drawdown_pct'])} "
            f"({m['max_drawdown_pct']:.1f}%)."
        )

    st.subheader("Cumulative growth (NPR 1 invested at the start)")
    fig = go.Figure()
    for i, (sym, df) in enumerate(data.items()):
        cum = cumulative_growth(df["close"])
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name=sym, line=dict(color=color_for(i), width=2)))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(height=420, hovermode="x unified", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    st.subheader("Drawdown")
    fig = go.Figure()
    for i, (sym, df) in enumerate(data.items()):
        dd = drawdown_series(df["close"]) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=sym, fill="tozeroy", line=dict(color=color_for(i))))
    fig.update_layout(height=380, hovermode="x unified", yaxis_title="Drawdown %", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with st.expander("How to read drawdown"):
        st.markdown(
            "Each line shows how far below its **own previous peak** that stock was on a given "
            "day. It always starts at 0% and dips negative; it only returns to 0% when the stock "
            "makes a new all-time high (within this window). Wide, deep dips = long, painful "
            "periods underwater if you'd bought at the top."
        )

# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------
with tab_corr:
    if len(data) < 2:
        st.info("Add a second symbol above to see correlation between stocks.")
    else:
        st.subheader("Correlation of daily returns")
        rets = close_df.pct_change().dropna(how="all")
        corr = rets.corr()

        fig = go.Figure(
            data=go.Heatmap(
                z=corr.values,
                x=corr.columns,
                y=corr.index,
                zmin=-1,
                zmax=1,
                colorscale="RdBu",
                reversescale=True,
                text=np.round(corr.values, 2),
                texttemplate="%{text}",
            )
        )
        fig.update_layout(height=400 + 20 * len(corr), margin=dict(t=10, b=10))
        st.plotly_chart(fig, width='stretch')

        with st.expander("How to read this / what it means for diversification"):
            st.markdown(
                "Correlation ranges from **-1** (perfectly opposite) to **+1** (move in lockstep). "
                "Holding multiple stocks only reduces your portfolio's bumpiness if they *aren't* "
                "highly correlated — two stocks that always move together behave like one bigger "
                "position, not two diversified ones."
            )
            pairs = []
            cols_list = list(corr.columns)
            for i in range(len(cols_list)):
                for j in range(i + 1, len(cols_list)):
                    a, b = cols_list[i], cols_list[j]
                    pairs.append((a, b, corr.loc[a, b]))
            for a, b, c in sorted(pairs, key=lambda x: -abs(x[2])):
                st.markdown(f"- **{a} vs {b}** ({c:.2f}): {interpret_correlation(c)}")

# ---------------------------------------------------------------------------
# Liquidity
# ---------------------------------------------------------------------------


@st.fragment
def render_liquidity_tab():
    st.subheader("Liquidity & volume profile")
    st.caption(
        "How easily (and at what prices) this stock actually trades. Note: NEPSE order-book / "
        "market-depth data isn't available from any free public source, so this is built from "
        "traded volume history — where volume actually happened, not resting orders."
    )

    liq_symbol = st.selectbox("Symbol", options=list(data.keys()), key="liq_symbol")
    ldf = data[liq_symbol]
    stats = liquidity_stats(ldf)
    last_close = ldf["close"].iloc[-1]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Avg daily turnover", f"NPR {stats['avg_daily_turnover_npr']:,.0f}")
    m2.metric(
        "Recent (20-bar) turnover",
        f"NPR {stats['recent_avg_turnover_npr']:,.0f}",
        f"{(stats['recent_avg_turnover_npr'] / stats['avg_daily_turnover_npr'] - 1) * 100:.0f}% vs window avg"
        if stats["avg_daily_turnover_npr"]
        else None,
    )
    m3.metric("Median daily volume", f"{stats['median_daily_volume']:,.0f} shares")
    m4.metric("No-trade bars", f"{stats['zero_volume_days_pct']:.1f}%")

    prof_col, tw_col = st.columns(2)

    with prof_col:
        st.markdown("**Volume profile** — traded volume by price level")
        closes = ldf["close"].to_numpy()
        vols = ldf["volume"].to_numpy()
        counts, edges = np.histogram(closes, bins=30, weights=vols)
        centers = (edges[:-1] + edges[1:]) / 2
        fig = go.Figure(go.Bar(x=counts, y=centers, orientation="h", marker_color=COLORS[0], name="Volume"))
        fig.add_hline(y=last_close, line_dash="dash", line_color=COLORS[1], annotation_text=f"Last: {last_close:.1f}")
        fig.update_layout(
            height=460,
            xaxis_title="Total volume traded",
            yaxis_title="Price (NPR)",
            margin=dict(t=10, b=10),
            bargap=0.15,
            showlegend=False,
        )
        st.plotly_chart(fig, width='stretch')

    with tw_col:
        st.markdown("**Turnover over time** — 20-bar rolling avg (NPR)")
        turnover = (ldf["close"] * ldf["volume"]).rolling(20).mean()
        fig = go.Figure(go.Scatter(x=turnover.index, y=turnover, line=dict(color=COLORS[0], width=2), name="Turnover"))
        fig.update_layout(height=460, yaxis_title="NPR / bar (20-bar avg)", margin=dict(t=10, b=10), hovermode="x unified", showlegend=False)
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    with st.expander("How to read this"):
        st.markdown(
            "- **Volume profile**: long bars mark price zones where lots of shares changed hands — "
            "these often act as 'sticky' zones (support/resistance) because many holders have a "
            "cost basis there. The dashed line is the latest close: if it sits far above the "
            "biggest bars, most past buyers are in profit; far below, many are underwater.\n"
            "- **Turnover** (price × volume) is the honest liquidity measure in rupee terms — a "
            "stock can print thousands of shares but still be hard to exit if turnover is tiny.\n"
            "- **No-trade bars %** is a thin-stock warning: high values mean there are whole days "
            "when you simply couldn't have traded at any price.\n"
            "- A falling turnover trend while price rises can mean the move is driven by few "
            "participants — historically more fragile."
        )


with tab_liquidity:
    render_liquidity_tab()

# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------


@st.fragment
def render_portfolio_tab():
    st.subheader("Combine into a portfolio")
    st.caption("Set weights (they'll be normalized to sum to 100%).")

    weight_cols = st.columns(len(data))
    raw_weights = {}
    for i, sym in enumerate(data):
        with weight_cols[i]:
            raw_weights[sym] = st.number_input(
                f"{sym} weight", min_value=0.0, value=qp_weights.get(sym, 1.0), step=0.5, key=f"w_{sym}"
            )

    st.query_params["weights"] = ",".join(f"{sym}:{w}" for sym, w in raw_weights.items())

    total_w = sum(raw_weights.values())
    # Forward-fill before computing returns: NEPSE stocks don't all trade the same days, and
    # dropping any row with a gap could collapse illiquid combinations to 0 rows.
    aligned_rets = close_df.ffill().dropna().pct_change().dropna()
    if total_w <= 0:
        st.warning("At least one weight must be greater than 0.")
    elif aligned_rets.empty:
        st.warning("These stocks don't have enough overlapping trading history to combine into a portfolio yet.")
    else:
        weights = {sym: w / total_w for sym, w in raw_weights.items()}
        port_ret = (aligned_rets * pd.Series(weights)).sum(axis=1)
        port_cum = (1 + port_ret).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Portfolio", line=dict(color="black", width=3)))
        for i, sym in enumerate(data):
            cum = (1 + aligned_rets[sym]).cumprod()
            fig.add_trace(go.Scatter(x=cum.index, y=cum, name=sym, opacity=0.5, line=dict(color=color_for(i))))
        fig.update_layout(height=460, hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

        port_metrics = compute_metrics(port_cum * 100)
        st.session_state["portfolio_weights"] = weights
        st.session_state["port_metrics"] = port_metrics

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Portfolio total return", f"{port_metrics['total_return_pct']:.1f}%")
        m2.metric("Portfolio volatility (ann.)", f"{port_metrics['ann_volatility_pct']:.1f}%")
        m3.metric("Portfolio max drawdown", f"{port_metrics['max_drawdown_pct']:.1f}%")
        m4.metric("Naive Sharpe", f"{port_metrics['sharpe_naive']:.2f}")

        with st.expander("How to read this"):
            st.markdown(
                "The **black line** is your weighted blend; the faded lines are the individual "
                "stocks for comparison. A well-diversified portfolio's black line should look "
                "*smoother* than its bumpiest component — if it doesn't, your holdings are too "
                "correlated (check the Correlation tab) or too concentrated in one weight."
            )


with tab_portfolio:
    render_portfolio_tab()

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------


@st.fragment
def render_backtest_tab():
    st.subheader("Simple moving-average crossover backtest")
    st.warning(
        "This is a **backtest of a rule on past data**, not a prediction of future prices. "
        "NEPSE small-caps are thinly traded and gappy — a strategy that worked historically can "
        "easily fail going forward. Use this to build intuition about trend-following, not as "
        "trading advice."
    )

    bt_symbol = st.selectbox("Symbol to backtest", options=list(data.keys()), key="bt_symbol")
    c1, c2 = st.columns(2)
    with c1:
        fast = st.slider("Fast SMA window", 5, 50, 20, key="bt_fast")
    with c2:
        slow = st.slider("Slow SMA window", 10, 200, 50, key="bt_slow")

    if fast >= slow:
        st.error("Fast window must be smaller than slow window.")
        return

    bt = sma_crossover_backtest(data[bt_symbol], fast=fast, slow=slow)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=bt.index, y=bt["close"], name="Price", line=dict(color="lightgray")))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["sma_fast"], name=f"SMA{fast}", line=dict(color=COLORS[0], width=2)))
    fig.add_trace(go.Scatter(x=bt.index, y=bt["sma_slow"], name=f"SMA{slow}", line=dict(color=COLORS[1], width=2)))
    fig.update_layout(height=380, hovermode="x unified", title="Price & moving averages", margin=dict(t=40, b=10))
    st.plotly_chart(fig, width='stretch', config=PLOTLY_CONFIG)

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_cum"], name="Buy & hold", line=dict(color=COLORS[0], width=2)))
    fig2.add_trace(go.Scatter(x=bt.index, y=bt["strategy_cum"], name=f"SMA({fast}/{slow}) crossover", line=dict(color=COLORS[1], width=2)))
    fig2.update_layout(height=380, hovermode="x unified", title="Strategy vs buy & hold", margin=dict(t=40, b=10))
    st.plotly_chart(fig2, width='stretch', config=PLOTLY_CONFIG)

    bh_total = (bt["buy_hold_cum"].iloc[-1] - 1) * 100
    strat_total = (bt["strategy_cum"].iloc[-1] - 1) * 100
    c1, c2 = st.columns(2)
    c1.metric("Buy & hold total return", f"{bh_total:.1f}%")
    c2.metric("Strategy total return", f"{strat_total:.1f}%", f"{strat_total - bh_total:.1f}% vs buy & hold")

    with st.expander("How this strategy works, in plain language"):
        st.markdown(
            f"- The **fast average** (SMA{fast}) tracks price closely; the **slow average** "
            f"(SMA{slow}) tracks it loosely.\n"
            "- The rule: be **long (holding)** whenever the fast average is above the slow "
            "average — this usually means the recent trend is up — and be **flat (in cash)** "
            "otherwise.\n"
            "- This is a classic *trend-following* rule. It tends to underperform buy & hold "
            "in choppy, sideways markets (whipsaws in and out) and can outperform in markets "
            "with long, sustained trends. Try changing the fast/slow windows above and watch "
            "how sensitive the result is — that sensitivity itself is a warning sign about "
            "overfitting a rule to history."
        )


with tab_backtest:
    render_backtest_tab()

# ---------------------------------------------------------------------------
# AI Chat (sidebar — available from every tab)
# ---------------------------------------------------------------------------


@st.fragment
def render_sidebar_chat():
    st.subheader("💬 AI Chat")

    api_key = get_mistral_api_key()
    if not api_key:
        st.info(
            "No Mistral API key configured. Add `MISTRAL_API_KEY` to this app's Streamlit "
            "secrets (or a local `.streamlit/secrets.toml`) to enable chat."
        )
        return

    if "ai_chat_history" not in st.session_state:
        st.session_state.ai_chat_history = []

    # Fixed-height scrollable message area ABOVE the input, so new messages
    # always land in the right place instead of below the input box.
    messages = st.container(height=430)
    with messages:
        if not st.session_state.ai_chat_history:
            with st.chat_message("assistant"):
                st.markdown(
                    "Hi! I can see the data for your selected stocks — performance, "
                    "indicators, portfolio. Ask me anything about them."
                )
        for msg in st.session_state.ai_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    pending_question = None
    if not st.session_state.ai_chat_history:
        suggestions = [
            "Summarize my stocks' performance",
            "What do the indicators suggest?",
            "Is my portfolio diversified?",
        ]
        for suggestion in suggestions:
            if st.button(suggestion, key=f"suggestion_{suggestion}", width='stretch'):
                pending_question = suggestion

    typed_question = st.chat_input("Ask about your selected stocks...", key="sidebar_chat_input")
    question = typed_question or pending_question

    if question:
        st.session_state.ai_chat_history.append({"role": "user", "content": question})
        context = build_context(
            data,
            metrics_rows,
            indicators_by_symbol,
            st.session_state.get("portfolio_weights"),
            st.session_state.get("port_metrics"),
        )
        with messages:
            with st.chat_message("user"):
                st.markdown(question)
            with st.chat_message("assistant"):
                try:
                    client = get_client(api_key)
                    reply = st.write_stream(
                        stream_reply(client, context, st.session_state.ai_chat_history[:-1], question)
                    )
                except Exception as e:
                    reply = f"Couldn't reach Mistral: {e}"
                    st.error(reply)
        st.session_state.ai_chat_history.append({"role": "assistant", "content": reply})
        if pending_question:
            # Suggestion buttons should disappear after first use.
            st.rerun()

    st.caption("Grounded in the data computed for your selected stocks — not live web access, not financial advice.")
    if st.session_state.ai_chat_history and st.button("Clear chat", key="clear_chat"):
        st.session_state.ai_chat_history = []
        st.rerun()


with st.sidebar:
    render_sidebar_chat()
