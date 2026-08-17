# Personal Monte Carlo Simulation

A portfolio simulation that projects net worth forward under uncertainty, given a
spending plan and an asset allocation. Runs as a local desktop app.

## Running it

```bash
.venv/bin/python run.py
```

This is the only way to start it. The app opens in its own desktop window;
closing the window stops the server.

`app.py` is a Streamlit script, but it will not render on its own. It checks for
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
| `app.py` | Streamlit UI: inputs, layout, chart (launched by `run.py`) |
| `montecarlo/assets.py` | Asset classes, return/volatility assumptions, correlation matrix |
| `montecarlo/simulation.py` | The simulation engine (no UI dependencies) |
| `.streamlit/config.toml` | Theme |

## The model

- Annual timesteps; the portfolio rebalances to the target weights each year.
- Spending is withdrawn at the start of each year, then returns are applied.
- Asset returns are drawn from correlated lognormals calibrated to the
  arithmetic mean and volatility in `assets.py`, so a single year can lose value
  but never more than 100%.
- A crisis regime (always on) scales every asset's shock by 2x in 12% of years,
  drawn once per year and shared across assets. This fattens the tails (US
  stocks: excess kurtosis 0.4 -> 3.6, worst year -56% -> -71%) and makes assets
  crash together. The chance bonds are in their worst 1% given stocks are rises
  from 5.5% to 15.2%. Mean and volatility stay exactly on target;
  `_lognormal_parameters` bisects for the sigma that reproduces them.
- The engine always runs in real terms. A sidebar toggle switches the display to
  future dollars, inflating every figure by 2.5% a year. That is a change of
  units only. The chance of success is identical either way, since a path runs
  dry in the same year whatever you print it in.
- A path that hits zero stays at zero. There is no borrowing.

"Percent chance of success" is the share of paths that still had a balance at the
end of the horizon. The chart draws 100 individual paths behind the percentile
bands; the y-axis is framed on the bands, so a few very lucky paths run off the
top rather than flattening everything else.

## Where the assumptions come from

`montecarlo/assets.py` holds the five asset classes and is sourced from the
J.P. Morgan Long-Term Capital Market Assumptions, 2026 edition (USD matrix,
data as of 30 Sep 2025). Each asset records the LTCMA line item it maps to:

| Asset class | LTCMA row | Nominal compound | Volatility | Real arithmetic |
|---|---|---|---|---|
| US Stocks | U.S. Large Cap | 6.70% | 16.47% | 5.45% |
| International Stocks | EAFE Equity | 7.50% | 17.63% | 6.43% |
| Bonds | U.S. Aggregate Bonds | 4.80% | 4.76% | 2.36% |
| Real Estate | U.S. Core Real Estate | 8.20% | 11.39% | 6.21% |
| Cash | U.S. Cash | 3.10% | 0.67% | 0.59% |

The published figures are stored verbatim so they can be checked against the
sheet. Two conversions happen in code, because LTCMA quotes something different
from what the engine needs:

- Nominal to real, deflating by the LTCMA U.S. inflation assumption (2.50%).
- Compound to arithmetic, adding sigma^2 / 2. This is worth 1.4pp for US Large
  Cap, so it is not a rounding detail.

Correlations come from the same matrix. Volatilities are used as published.

To update to a later edition, replace the compound return / volatility pairs and
the correlations. The conversions run off those numbers, so nothing else needs
to change.

The correlation matrix must stay positive semi-definite. To check after editing:

```bash
.venv/bin/python -c "import numpy as np; from montecarlo.assets import CORRELATION; np.linalg.cholesky(CORRELATION); print('OK')"
```
