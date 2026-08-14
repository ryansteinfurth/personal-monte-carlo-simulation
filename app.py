"""Streamlit front end for the Monte Carlo portfolio simulation.

Not meant to be run directly — ``run.py`` starts this and wraps it in a window.
"""

import os

import plotly.graph_objects as go
import streamlit as st

from montecarlo import ASSET_CLASSES, DEFAULT_ALLOCATION, run_simulation

# Single-hue violet ramp on a warm cream surface. The fills are translucent so
# the individual paths show through, so the ramp was validated on the COMPOSITED
# colours (#b6a3f4 / #8067de / #3d2b8f over #fdfaf3) rather than these raw hexes:
# monotone lightness, visible step gaps, light end at 2.11:1 against the surface.
BAND_OUTER = "rgba(124, 92, 245, 0.55)"  # 10th-90th percentile
BAND_INNER = "rgba(92, 63, 208, 0.60)"  # 25th-75th percentile
MEDIAN_LINE = "#3d2b8f"
PATH_LINE = "rgba(61, 43, 143, 0.14)"  # individual paths, stacked to show density
PATH_LINE_LEGEND = "rgba(61, 43, 143, 0.55)"  # same hue, readable in the legend
SURFACE = "#fdfaf3"
GRIDLINE = "#ece3d2"
AXIS = "#d2c7ae"
MUTED_INK = "#8a8277"
PRIMARY_INK = "#0b0b0b"

N_VISIBLE_PATHS = 100  # individual simulations drawn through the summary bands

# Fixed so the same inputs always give the same answer -- nudging an unrelated
# slider shouldn't make the headline number twitch. The sampling error this
# leaves behind is reported under the success rate instead.
SEED = 42

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")

if not os.environ.get("MONTE_CARLO_DESKTOP"):
    st.error("This app runs in its own window. Start it with:  python run.py")
    st.stop()


def money(value: float) -> str:
    return f"${value:,.0f}"


def axis_tick_format(value: float) -> str:
    if value >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value / 1_000:.0f}K"


with st.sidebar:
    st.header("Simulation settings")
    years = st.slider("Years to project", 5, 60, 30, step=5)
    n_sims = st.select_slider(
        "Number of simulations", options=[1_000, 5_000, 10_000, 25_000], value=10_000
    )
    st.caption(
        "Returns are real (inflation-adjusted). Portfolio rebalances annually. "
        "12% of years are crisis years: every asset takes double-volatility "
        "shocks together."
    )

st.title("Monte Carlo Simulation")

# --- Row 1: starting position -------------------------------------------------
top_left, top_right = st.columns(2)
with top_left:
    net_worth = st.number_input(
        "Net worth", min_value=0, value=2_000_000, step=50_000, format="%d"
    )
    st.caption(money(net_worth))
with top_right:
    annual_spending = st.number_input(
        "Annual spending", min_value=0, value=80_000, step=5_000, format="%d"
    )
    withdrawal_rate = annual_spending / net_worth if net_worth else 0.0
    st.caption(f"{money(annual_spending)} — {withdrawal_rate:.1%} withdrawal rate")

st.divider()

# --- Row 2: allocation on the left, chart on the right ------------------------
alloc_col, chart_col = st.columns([1, 2], gap="large")

with alloc_col:
    st.subheader("Allocation")
    allocation: dict[str, float] = {}
    for asset in ASSET_CLASSES:
        allocation[asset.key] = st.number_input(
            f"{asset.name} (%)",
            min_value=0.0,
            max_value=100.0,
            value=DEFAULT_ALLOCATION.get(asset.key, 0.0),
            step=5.0,
            format="%.0f",
            key=f"alloc_{asset.key}",
            help=(
                f"Expected real return {asset.mean_return:.1%}, "
                f"volatility {asset.volatility:.0%}"
            ),
        )

    total = sum(allocation.values())
    if abs(total - 100.0) < 0.01:
        st.success(f"Total: {total:.0f}%")
    else:
        st.warning(f"Total: {total:.0f}% — weights are normalised to 100%.")

with chart_col:
    if total <= 0:
        st.info("Allocate to at least one asset class to run the simulation.")
        st.stop()

    result = run_simulation(
        net_worth=net_worth,
        annual_spending=annual_spending,
        allocation=allocation,
        years=years,
        n_sims=int(n_sims),
        seed=SEED,
    )
    bands = result.percentile_bands()
    x = list(range(years + 1))

    stat_a, stat_b, stat_c = st.columns(3)
    stat_a.metric(
        "Percent chance of success",
        f"{result.success_rate:.0%}",
        help="Share of paths that never ran out",
    )
    # Each path either survives or doesn't, so the sampling error on that share
    # is the binomial one. Two standard errors ~ the 95% interval.
    p_hat = result.success_rate
    margin = 2.0 * (p_hat * (1.0 - p_hat) / result.n_sims) ** 0.5
    stat_a.caption(
        f"±{margin * 100:.1f} pts from random sampling — "
        "raise the simulation count to narrow it"
    )
    stat_b.metric("Median ending", money(result.median_ending))
    stat_c.metric("Downside (10th pct)", money(bands[10][-1]))

    fig = go.Figure()

    # 10th-90th percentile band.
    fig.add_trace(
        go.Scatter(
            x=x, y=bands[10], line=dict(width=0), hoverinfo="skip", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bands[90],
            line=dict(width=0),
            fill="tonexty",
            fillcolor=BAND_OUTER,
            name="10th–90th percentile",
            hoverinfo="skip",
        )
    )

    # 25th-75th percentile band.
    fig.add_trace(
        go.Scatter(
            x=x, y=bands[25], line=dict(width=0), hoverinfo="skip", showlegend=False
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bands[75],
            line=dict(width=0),
            fill="tonexty",
            fillcolor=BAND_INNER,
            name="25th–75th percentile",
            hoverinfo="skip",
        )
    )

    # A sample of individual paths. One trace with None between paths keeps the
    # figure light; spline smoothing takes the edge off the annual steps.
    path_x: list[float | None] = []
    path_y: list[float | None] = []
    for path in result.sample_paths(N_VISIBLE_PATHS):
        path_x.extend([*x, None])
        path_y.extend([*path.tolist(), None])

    fig.add_trace(
        go.Scatter(
            x=path_x,
            y=path_y,
            mode="lines",
            line=dict(color=PATH_LINE, width=1, shape="spline", smoothing=0.8),
            connectgaps=False,
            hoverinfo="skip",
            showlegend=False,
        )
    )
    # Legend proxy: the real paths are too faint to read as a swatch.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=PATH_LINE_LEGEND, width=1),
            name=f"{N_VISIBLE_PATHS} individual paths",
            hoverinfo="skip",
        )
    )

    # Median path.
    fig.add_trace(
        go.Scatter(
            x=x,
            y=bands[50],
            line=dict(color=MEDIAN_LINE, width=2),
            name="Median",
            hovertemplate="Year %{x}<br>Median %{y:$,.0f}<extra></extra>",
        )
    )

    max_value = float(bands[90].max())
    tick_step = max(round(max_value / 5 / 500_000) * 500_000, 500_000)
    tick_values = [i * tick_step for i in range(int(max_value // tick_step) + 2)]

    fig.update_layout(
        height=460,
        margin=dict(l=8, r=8, t=8, b=8),
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        font=dict(family="system-ui, -apple-system, sans-serif", color=MUTED_INK, size=12),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            x=0,
            font=dict(color=PRIMARY_INK),
        ),
    )
    fig.update_xaxes(
        title_text="Years from today",
        showgrid=False,
        linecolor=AXIS,
        ticks="outside",
        tickcolor=AXIS,
    )
    fig.update_yaxes(
        title_text="Portfolio value (today's dollars)",
        gridcolor=GRIDLINE,
        zeroline=True,
        zerolinecolor=AXIS,
        showline=False,
        tickvals=tick_values,
        ticktext=[axis_tick_format(v) for v in tick_values],
        # Anchored to the percentile bands: a handful of lucky paths run off the
        # top rather than compressing everything else into the floor.
        range=[0, tick_values[-1]],
    )

    st.plotly_chart(fig, width="stretch")

    with st.expander("View data table"):
        import pandas as pd

        table = pd.DataFrame(
            {f"{p}th pct": [money(v) for v in bands[p]] for p in sorted(bands)},
            index=pd.Index(x, name="Year"),
        )
        st.dataframe(table, width="stretch")
