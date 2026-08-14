"""Streamlit front end for the Monte Carlo portfolio simulation.

Run with:  streamlit run app.py
"""

import plotly.graph_objects as go
import streamlit as st

from montecarlo import ASSET_CLASSES, DEFAULT_ALLOCATION, run_simulation

# Validated single-hue blue ramp (see dataviz palette, light surface #fcfcfb).
BAND_OUTER = "#86b6ef"  # 10th-90th percentile
BAND_INNER = "#3987e5"  # 25th-75th percentile
MEDIAN_LINE = "#184f95"
SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"

st.set_page_config(page_title="Monte Carlo Simulation", layout="wide")


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
    seed = st.number_input("Random seed", value=42, step=1)
    st.caption("Returns are real (inflation-adjusted). Portfolio rebalances annually.")

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
        seed=int(seed),
    )
    bands = result.percentile_bands()
    x = list(range(years + 1))

    stat_a, stat_b, stat_c = st.columns(3)
    stat_a.metric("Money lasts", f"{result.success_rate:.0%}", help="Share of paths that never ran out")
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
    )

    st.plotly_chart(fig, width="stretch")

    with st.expander("View data table"):
        import pandas as pd

        table = pd.DataFrame(
            {f"{p}th pct": [money(v) for v in bands[p]] for p in sorted(bands)},
            index=pd.Index(x, name="Year"),
        )
        st.dataframe(table, width="stretch")
