"""Mistral-backed chat that answers questions using the app's current computed data."""

from mistralai.client import Mistral

MODEL = "mistral-medium-latest"

SYSTEM_PROMPT = (
    "You are a financial data assistant embedded in a NEPSE (Nepal Stock Exchange) analysis app. "
    "You are given a snapshot of computed data (prices, indicators, return/risk metrics) for the "
    "stocks the user currently has selected, plus their portfolio weights if set. Answer using "
    "ONLY this data plus general financial literacy — don't invent numbers you weren't given. Be "
    "direct and concise. Never claim certainty about future prices; frame any pattern as "
    "descriptive of the past, not predictive. If asked for investment advice, give balanced "
    "analysis and explicitly note you are not a licensed financial advisor."
)


def get_client(api_key):
    return Mistral(api_key=api_key)


def build_context(data, metrics_by_symbol, indicators_by_symbol, portfolio_weights=None, portfolio_metrics=None):
    lines = []
    for sym, df in data.items():
        m = metrics_by_symbol.get(sym, {})
        lines.append(
            f"{sym}: last close NPR {df['close'].iloc[-1]:.2f} on {df.index[-1].date()} "
            f"(history {df.index[0].date()} to {df.index[-1].date()}, {len(df)} bars). "
            f"Total return {m.get('total_return_pct', float('nan')):.1f}%, "
            f"CAGR {m.get('cagr_pct', float('nan')):.1f}%, "
            f"annualized volatility {m.get('ann_volatility_pct', float('nan')):.1f}%, "
            f"max drawdown {m.get('max_drawdown_pct', float('nan')):.1f}%. "
            f"Indicators: {indicators_by_symbol.get(sym, 'n/a')}"
        )
    if portfolio_weights:
        w_str = ", ".join(f"{s}={w * 100:.0f}%" for s, w in portfolio_weights.items())
        lines.append(f"Portfolio weights: {w_str}")
    if portfolio_metrics:
        lines.append(
            f"Portfolio total return {portfolio_metrics.get('total_return_pct', float('nan')):.1f}%, "
            f"volatility {portfolio_metrics.get('ann_volatility_pct', float('nan')):.1f}%, "
            f"max drawdown {portfolio_metrics.get('max_drawdown_pct', float('nan')):.1f}%."
        )
    return "\n".join(lines)


def stream_reply(client, context, history, question):
    """Yield response text chunks as they arrive from Mistral."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT + "\n\nCurrent data snapshot:\n" + context}]
    messages.extend(history)
    messages.append({"role": "user", "content": question})
    stream = client.chat.stream(model=MODEL, messages=messages, max_tokens=800)
    for event in stream:
        delta = event.data.choices[0].delta
        if delta.content:
            yield delta.content
