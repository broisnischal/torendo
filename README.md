# NEPSE Portfolio Analyzer

Interactive Streamlit app for pulling NEPSE stock price history (via merolagani's
public chart endpoint), charting candlesticks with SMA/EMA/Bollinger/RSI/MACD,
analyzing return/risk/correlation, combining stocks into a weighted portfolio,
and backtesting a simple moving-average crossover.

While NEPSE is open (Sun–Thu, 11:00–15:00 Nepal time) and the selected date
range includes today, the app auto-refreshes every 30s. Outside those hours
it just shows the last available cached data — no point polling a closed market.

An older `analysis.ipynb` notebook covering the same core analysis also lives
in this repo for quick one-off scripting.

## Data source caveats

`merolagani.com`'s chart endpoint is unofficial and undocumented — there's no
ToS guarantee it keeps working. Every free NEPSE data source we found (official
site, Sharesansar, NepseAlpha) is similarly unofficial; none offer genuine
push/WebSocket live data, only pollable endpoints. If this endpoint ever breaks,
`data.py`'s `fetch_ohlcv` is the single place to swap in a replacement.

## Run locally

```
uv sync
uv run streamlit run app.py
```

## Saving / sharing a view

There's no login — your chosen stocks, date range, resolution, and portfolio
weights are all encoded in the page URL as you change them. Bookmark or copy
the browser address bar to save or share an exact setup; opening that URL
restores it.

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub (public, or private if linked to your Streamlit account):
   ```
   git remote add origin <your-github-repo-url>
   git branch -M main
   git push -u origin main
   ```
2. Go to https://share.streamlit.io, sign in with GitHub, and click "New app".
3. Pick this repo/branch, set the main file path to `app.py`, and deploy.
4. Streamlit Cloud installs from `requirements.txt` automatically — no extra config needed.

Each subsequent `git push` to the connected branch auto-redeploys the app.
