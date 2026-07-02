"""NEPSE portfolio analyzer — interactive Streamlit app on top of merolagani's chart API."""

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
    interpret_correlation,
    interpret_drawdown,
    interpret_volatility,
    is_market_open_now,
    macd,
    rsi,
    sma,
    sma_crossover_backtest,
    summarize_indicators,
)

LIVE_REFRESH_SECONDS = 30


def get_mistral_api_key():
    try:
        key = st.secrets.get("MISTRAL_API_KEY")
    except Exception:
        key = None
    return key or os.environ.get("MISTRAL_API_KEY")

st.set_page_config(page_title="NEPSE Portfolio Analyzer", layout="wide")

st.title("NEPSE Portfolio Analyzer")
st.caption(
    "Pulls price history from merolagani's public chart endpoint. This is an unofficial data "
    "source — treat it as directionally useful, not a source of truth for real trading decisions."
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

if "symbols" not in st.session_state:
    st.session_state.symbols = qp_symbols or ["SSHL"]

with st.container(border=True):
    st.subheader("1. Choose stocks")

    add_col, list_col = st.columns([1, 2])
    with add_col:
        with st.form("add_symbol_form", clear_on_submit=True):
            new_symbol = st.text_input("Add a symbol (ticker as listed on NEPSE)", "")
            submitted = st.form_submit_button("Add")
            if submitted and new_symbol.strip():
                sym = new_symbol.strip().upper()
                if sym not in st.session_state.symbols:
                    st.session_state.symbols.append(sym)

    with list_col:
        selected_symbols = st.multiselect(
            "Selected stocks (pick which of your added symbols to analyze together)",
            options=st.session_state.symbols,
            default=st.session_state.symbols,
        )

    st.caption(
        "No built-in stock list is provided here — type the exact ticker as it appears on NEPSE "
        "(e.g. SSHL, NABIL). Wrong or delisted tickers will just fail to fetch below."
    )

    st.markdown("**2. Time window & granularity**")
    c1, c2, c3 = st.columns(3)
    with c1:
        start_date = st.date_input("From", value=qp_start)
    with c2:
        end_date = st.date_input("To", value=qp_end)
    with c3:
        resolution = st.selectbox("Bar size", RESOLUTIONS, index=RESOLUTIONS.index(qp_resolution))

    st.caption(
        "💾 This page's URL updates as you change things here — bookmark or copy the address "
        "bar anytime to save or share this exact setup (stocks, dates, weights included)."
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
    st.info("Add and select at least one symbol above to see analysis.")
    st.stop()

covers_today = end_date >= date.today()
market_open = is_market_open_now()
is_live = covers_today and market_open

if is_live:
    st_autorefresh(interval=LIVE_REFRESH_SECONDS * 1000, key="live_refresh")

now_npt = datetime.now(NEPAL_TZ)
if is_live:
    st.success(f"🟢 Live — market open, refreshing every {LIVE_REFRESH_SECONDS}s (last checked {now_npt:%H:%M:%S} NPT)")
elif covers_today:
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
portfolio_weights = None
port_metrics = None

COLORS = ["#3366CC", "#DC3912", "#109618", "#FF9900", "#990099", "#0099C6", "#DD4477"]


def color_for(i):
    return COLORS[i % len(COLORS)]


tab_overview, tab_chart, tab_risk, tab_corr, tab_portfolio, tab_backtest, tab_ai = st.tabs(
    ["Overview", "Chart", "Risk & Returns", "Correlation", "Portfolio", "Backtest", "💬 AI Chat"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader("Price history")
    fig = go.Figure()
    for i, (sym, df) in enumerate(data.items()):
        fig.add_trace(go.Scatter(x=df.index, y=df["close"], name=sym, line=dict(color=color_for(i))))
    fig.update_layout(
        height=480,
        hovermode="x unified",
        yaxis_title="Price (NPR)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=10, b=10),
    )
    st.plotly_chart(fig, width='stretch')

    with st.expander("How to read this"):
        st.markdown(
            "- This is the **raw closing price** for each stock, not a return — a stock at "
            "NPR 2000 isn't 'more expensive to own' than one at NPR 200, since a portfolio is "
            "sized in rupees, not shares.\n"
            "- Use this chart to spot the big picture: trend direction, big gaps (often "
            "dividend/bonus adjustments or halts), and how volatile the price path looks.\n"
            "- Hover over the chart to see exact prices on any date; drag to zoom into a range."
        )

    st.subheader("At a glance")
    cols = st.columns(len(data))
    for i, (sym, df) in enumerate(data.items()):
        m = compute_metrics(df["close"])
        with cols[i]:
            st.metric(
                sym,
                f"NPR {df['close'].iloc[-1]:.2f}",
                f"{m['total_return_pct']:.1f}% over window",
            )

# ---------------------------------------------------------------------------
# Chart (candlesticks + indicators)
# ---------------------------------------------------------------------------
with tab_chart:
    st.subheader("Candlestick chart with indicators")

    chart_symbol = st.selectbox("Symbol to chart", options=list(data.keys()), key="chart_symbol")
    cdf = data[chart_symbol]

    ic1, ic2, ic3, ic4 = st.columns(4)
    with ic1:
        sma_windows = st.multiselect("SMA overlays", [20, 50, 100, 200], default=[20, 50])
    with ic2:
        ema_windows = st.multiselect("EMA overlays", [12, 26, 50], default=[])
    with ic3:
        show_bbands = st.checkbox("Bollinger Bands (20, 2σ)")
    with ic4:
        show_rsi = st.checkbox("RSI (14)", value=True)
    show_macd = st.checkbox("MACD (12, 26, 9)", value=True)

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
            increasing_line_color="#2E7D32",
            decreasing_line_color="#C62828",
        ),
        row=1,
        col=1,
    )

    overlay_colors = ["#3366CC", "#FF9900", "#990099", "#0099C6", "#DD4477", "#109618"]
    color_i = 0
    for w in sorted(sma_windows):
        fig.add_trace(
            go.Scatter(x=cdf.index, y=sma(cdf["close"], w), name=f"SMA{w}", line=dict(color=overlay_colors[color_i % len(overlay_colors)], width=1.3)),
            row=1,
            col=1,
        )
        color_i += 1
    for w in sorted(ema_windows):
        fig.add_trace(
            go.Scatter(x=cdf.index, y=ema(cdf["close"], w), name=f"EMA{w}", line=dict(color=overlay_colors[color_i % len(overlay_colors)], width=1.3, dash="dot")),
            row=1,
            col=1,
        )
        color_i += 1
    if show_bbands:
        mid, upper, lower = bollinger_bands(cdf["close"])
        fig.add_trace(go.Scatter(x=cdf.index, y=upper, name="BB upper", line=dict(color="gray", width=1)), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=lower, name="BB lower", line=dict(color="gray", width=1), fill="tonexty", fillcolor="rgba(150,150,150,0.1)"), row=1, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=mid, name="BB mid (SMA20)", line=dict(color="gray", width=1, dash="dash")), row=1, col=1)

    vol_colors = ["#2E7D32" if c >= o else "#C62828" for o, c in zip(cdf["open"], cdf["close"])]
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
        hist_colors = ["#2E7D32" if v >= 0 else "#C62828" for v in hist.fillna(0)]
        fig.add_trace(go.Bar(x=cdf.index, y=hist, name="MACD hist", marker_color=hist_colors, showlegend=False), row=next_row, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=macd_line, name="MACD", line=dict(color="#3366CC")), row=next_row, col=1)
        fig.add_trace(go.Scatter(x=cdf.index, y=signal_line, name="Signal", line=dict(color="#FF9900")), row=next_row, col=1)

    fig.update_layout(
        height=320 + 180 * (n_rows - 1),
        hovermode="x unified",
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=10),
    )
    st.plotly_chart(fig, width='stretch')

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
        fig.add_trace(go.Scatter(x=cum.index, y=cum, name=sym, line=dict(color=color_for(i))))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray")
    fig.update_layout(height=420, hovermode="x unified", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')

    st.subheader("Drawdown")
    fig = go.Figure()
    for i, (sym, df) in enumerate(data.items()):
        dd = drawdown_series(df["close"]) * 100
        fig.add_trace(go.Scatter(x=dd.index, y=dd, name=sym, fill="tozeroy", line=dict(color=color_for(i))))
    fig.update_layout(height=380, hovermode="x unified", yaxis_title="Drawdown %", margin=dict(t=10, b=10))
    st.plotly_chart(fig, width='stretch')

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
# Portfolio
# ---------------------------------------------------------------------------
with tab_portfolio:
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
    # dropping any row with a gap (the old behavior) could collapse illiquid combinations to 0 rows.
    aligned_rets = close_df.ffill().dropna().pct_change().dropna()
    if total_w <= 0:
        st.warning("At least one weight must be greater than 0.")
    elif aligned_rets.empty:
        st.warning("These stocks don't have enough overlapping trading history to combine into a portfolio yet.")
    else:
        weights = {sym: w / total_w for sym, w in raw_weights.items()}
        portfolio_weights = weights
        port_ret = (aligned_rets * pd.Series(weights)).sum(axis=1)
        port_cum = (1 + port_ret).cumprod()

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=port_cum.index, y=port_cum, name="Portfolio", line=dict(color="black", width=3)))
        for i, sym in enumerate(data):
            cum = (1 + aligned_rets[sym]).cumprod()
            fig.add_trace(go.Scatter(x=cum.index, y=cum, name=sym, opacity=0.5, line=dict(color=color_for(i))))
        fig.update_layout(height=460, hovermode="x unified", margin=dict(t=10, b=10))
        st.plotly_chart(fig, width='stretch')

        port_cum_price = port_cum * 100
        port_metrics = compute_metrics(port_cum_price)
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

# ---------------------------------------------------------------------------
# Backtest
# ---------------------------------------------------------------------------
with tab_backtest:
    st.subheader("Simple moving-average crossover backtest")
    st.warning(
        "This is a **backtest of a rule on past data**, not a prediction of future prices. "
        "NEPSE small-caps are thinly traded and gappy — a strategy that worked historically can "
        "easily fail going forward. Use this to build intuition about trend-following, not as "
        "trading advice."
    )

    bt_symbol = st.selectbox("Symbol to backtest", options=list(data.keys()))
    c1, c2 = st.columns(2)
    with c1:
        fast = st.slider("Fast SMA window", 5, 50, 20)
    with c2:
        slow = st.slider("Slow SMA window", 10, 200, 50)

    if fast >= slow:
        st.error("Fast window must be smaller than slow window.")
    else:
        bt = sma_crossover_backtest(data[bt_symbol], fast=fast, slow=slow)

        fig = make_subplots(specs=[[{"secondary_y": False}]])
        fig.add_trace(go.Scatter(x=bt.index, y=bt["close"], name="Price", line=dict(color="lightgray")))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["sma_fast"], name=f"SMA{fast}", line=dict(color="#3366CC")))
        fig.add_trace(go.Scatter(x=bt.index, y=bt["sma_slow"], name=f"SMA{slow}", line=dict(color="#DC3912")))
        fig.update_layout(height=380, hovermode="x unified", title="Price & moving averages", margin=dict(t=40, b=10))
        st.plotly_chart(fig, width='stretch')

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=bt.index, y=bt["buy_hold_cum"], name="Buy & hold"))
        fig2.add_trace(go.Scatter(x=bt.index, y=bt["strategy_cum"], name=f"SMA({fast}/{slow}) crossover"))
        fig2.update_layout(height=380, hovermode="x unified", title="Strategy vs buy & hold", margin=dict(t=40, b=10))
        st.plotly_chart(fig2, width='stretch')

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

# ---------------------------------------------------------------------------
# AI Chat
# ---------------------------------------------------------------------------
with tab_ai:
    st.subheader("Ask AI about your selected stocks")
    st.caption(
        "Answers are generated by Mistral AI from the data already computed above (prices, "
        "indicators, metrics) — not live web access, and not financial advice."
    )

    api_key = get_mistral_api_key()
    if not api_key:
        st.info(
            "No Mistral API key configured. Add `MISTRAL_API_KEY` to this app's Streamlit "
            "secrets (or a local `.streamlit/secrets.toml`) to enable this tab."
        )
    else:
        if "ai_chat_history" not in st.session_state:
            st.session_state.ai_chat_history = []

        for msg in st.session_state.ai_chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        pending_question = None
        if not st.session_state.ai_chat_history:
            st.caption("Try asking:")
            suggestions = [
                "Summarize how my selected stocks have performed",
                "What do the current RSI/MACD readings suggest?",
                "How correlated are these stocks, and does my portfolio look diversified?",
            ]
            cols = st.columns(len(suggestions))
            for col, suggestion in zip(cols, suggestions):
                if col.button(suggestion, key=f"suggestion_{suggestion}"):
                    pending_question = suggestion

        typed_question = st.chat_input("Ask about the stocks/portfolio you've selected...")
        question = typed_question or pending_question

        if question:
            st.session_state.ai_chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            context = build_context(data, metrics_rows, indicators_by_symbol, portfolio_weights, port_metrics)
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

        if st.session_state.ai_chat_history and st.button("Clear chat"):
            st.session_state.ai_chat_history = []
            st.rerun()
