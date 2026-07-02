# NEPSE Portfolio Analyzer

Interactive Streamlit app for pulling NEPSE stock price history (via merolagani's
public chart endpoint), charting candlesticks with SMA/EMA/Bollinger/RSI/MACD,
analyzing return/risk/correlation, combining stocks into a weighted portfolio,
backtesting a simple moving-average crossover, and an AI chat tab (Mistral) that
answers questions grounded in the data already computed for your selected stocks.

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

## AI chat setup

The AI Chat tab needs a Mistral API key (get one at https://console.mistral.ai).
It reads from Streamlit's secrets mechanism, which is **not** the same as GitHub
repo secrets — a GitHub Actions secret on this repo is not visible to the running
app at all.

**Local dev**: create `.streamlit/secrets.toml` (already gitignored — never commit
this file) with:
```toml
MISTRAL_API_KEY = "your-key-here"
```

**Streamlit Community Cloud**: open the deployed app → bottom-right "Manage app"
→ Settings → Secrets, and paste the same `MISTRAL_API_KEY = "..."` line there.
This is the only way to get the key into the running Cloud app — there's no CLI
for it, and GitHub secrets don't reach it.

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
