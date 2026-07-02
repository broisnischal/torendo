# Torendo

NEPSE market analyzer built with Streamlit.

## Features

- Searchable picker over the full NEPSE symbol list
- Candlestick charts with SMA/EMA/Bollinger/RSI/MACD, scroll-zoom, range presets, and on-chart drawing tools
- Return/risk metrics, correlation heatmap, and weighted portfolio analysis
- Liquidity view: volume profile by price level, rolling turnover, thin-stock stats
- Moving-average crossover backtesting
- AI chat (sidebar) grounded in the data computed for your selected stocks
- Live auto-refresh during market hours (Sun–Thu, 11:00–15:00 NPT), toggleable
- Shareable views — stocks, dates, and weights are encoded in the page URL

## Run locally

```
uv sync
uv run streamlit run app.py
```

The AI chat needs a Mistral API key in `.streamlit/secrets.toml` (gitignored):

```toml
MISTRAL_API_KEY = "your-key-here"
```

## Deploy

Deploys as a standard Streamlit app (dependencies in `requirements.txt`).
On Streamlit Community Cloud, set `MISTRAL_API_KEY` under the app's
Settings → Secrets.

## Disclaimer

Market data comes from unofficial public sources and may be delayed,
incomplete, or break without notice. Nothing here is financial advice.
