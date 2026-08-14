# Personal Monte Carlo Simulation

A portfolio simulation that projects net worth forward under uncertainty, given a
spending plan and an asset allocation. Runs as a local web app.

## Running it

```bash
.venv/bin/streamlit run app.py
```

This opens a browser window at http://localhost:8501.

## Setup from scratch

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Layout

| File | Role |
|---|---|
| `app.py` | Streamlit UI — inputs, layout, chart |
| `montecarlo/assets.py` | Asset classes, return/volatility assumptions, correlation matrix |
| `montecarlo/simulation.py` | The simulation engine (no UI dependencies) |
| `.streamlit/config.toml` | Theme |

## The model

- Annual timesteps; the portfolio rebalances to the target weights each year.
- Spending is withdrawn at the **start** of each year, then returns are applied.
- Asset returns are drawn from **correlated lognormals** calibrated to the
  arithmetic mean and volatility in `assets.py`, so a single year can lose value
  but never more than 100%.
- All figures are **real** (inflation-adjusted), so today's dollars throughout.
- A path that hits zero stays at zero — no borrowing.

"Money lasts" is the share of paths that still had a balance at the end of the
horizon.

## Tuning the assumptions

Everything lives in `montecarlo/assets.py` — the five asset classes, their
expected real returns and volatilities, and the correlation matrix between them.
Edit that file and the app picks it up on the next run.

The correlation matrix must stay positive semi-definite. To check after editing:

```bash
.venv/bin/python -c "import numpy as np; from montecarlo.assets import CORRELATION; np.linalg.cholesky(CORRELATION); print('OK')"
```
