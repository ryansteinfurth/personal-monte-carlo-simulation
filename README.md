# Personal Monte Carlo Simulation

A portfolio simulation that projects net worth forward under uncertainty, given a
spending plan and an asset allocation. Runs as a local desktop app.

## Running it

```bash
.venv/bin/python run.py
```

This is the only way to start it. The app opens in its own desktop window;
closing the window stops the server.

`app.py` is a Streamlit script, but it will not render on its own — it checks for
an environment variable that only `run.py` sets, so `streamlit run app.py`
refuses and tells you to use `run.py` instead.

## Setup from scratch

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

## Layout

| File | Role |
|---|---|
| `run.py` | Desktop-window launcher (starts Streamlit, wraps it in a native window) |
| `app.py` | Streamlit UI — inputs, layout, chart (launched by `run.py`) |
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

"Percent chance of success" is the share of paths that still had a balance at the
end of the horizon. The chart draws 100 individual paths behind the percentile
bands; the y-axis is framed on the bands, so a few very lucky paths run off the
top rather than flattening everything else.

## Tuning the assumptions

Everything lives in `montecarlo/assets.py` — the five asset classes, their
expected real returns and volatilities, and the correlation matrix between them.
Edit that file and the app picks it up on the next run.

The correlation matrix must stay positive semi-definite. To check after editing:

```bash
.venv/bin/python -c "import numpy as np; from montecarlo.assets import CORRELATION; np.linalg.cholesky(CORRELATION); print('OK')"
```
