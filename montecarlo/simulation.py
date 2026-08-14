"""Monte Carlo engine for retirement / portfolio depletion paths.

Model: annual timesteps, portfolio rebalanced to target weights each year,
spending withdrawn at the start of the year. Asset returns are drawn from
correlated lognormals calibrated to the arithmetic mean/volatility in
``assets.py``, so a path can lose value but never go below -100% in a year.
"""

from dataclasses import dataclass

import numpy as np

from .assets import ASSET_CLASSES, CORRELATION, mean_returns, volatilities

PERCENTILES = (10, 25, 50, 75, 90)


@dataclass(frozen=True)
class SimulationResult:
    balances: np.ndarray  # (n_sims, years + 1) real dollars, index 0 = today
    years: int
    n_sims: int

    @property
    def success_rate(self) -> float:
        """Share of paths that never ran out of money."""
        return float((self.balances[:, -1] > 0).mean())

    @property
    def median_ending(self) -> float:
        return float(np.median(self.balances[:, -1]))

    def percentile_bands(self) -> dict[int, np.ndarray]:
        """Percentile of balance across paths, at each year."""
        return {p: np.percentile(self.balances, p, axis=0) for p in PERCENTILES}

    def depletion_year(self, percentile: float = 50.0) -> int | None:
        """Year the given percentile of paths has run dry, or None."""
        band = np.percentile(self.balances, percentile, axis=0)
        exhausted = np.flatnonzero(band <= 0)
        return int(exhausted[0]) if exhausted.size else None


def _lognormal_parameters(
    mean: np.ndarray, vol: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arithmetic mean/stdev of returns to log-space mu/sigma."""
    growth = 1.0 + mean
    sigma_log = np.sqrt(np.log1p((vol / growth) ** 2))
    mu_log = np.log(growth) - 0.5 * sigma_log**2
    return mu_log, sigma_log


def _draw_asset_returns(
    years: int, n_sims: int, rng: np.random.Generator
) -> np.ndarray:
    """Correlated annual real returns, shape (n_sims, years, n_assets)."""
    mu_log, sigma_log = _lognormal_parameters(mean_returns(), volatilities())
    chol = np.linalg.cholesky(CORRELATION)

    shocks = rng.standard_normal((n_sims, years, len(ASSET_CLASSES)))
    correlated = shocks @ chol.T
    return np.exp(mu_log + sigma_log * correlated) - 1.0


def run_simulation(
    net_worth: float,
    annual_spending: float,
    allocation: dict[str, float],
    years: int = 30,
    n_sims: int = 10_000,
    seed: int | None = 42,
) -> SimulationResult:
    """Project ``net_worth`` forward, spending ``annual_spending`` each year.

    ``allocation`` maps asset keys to percentages; it is normalised to sum to 1.
    """
    weights = np.array([allocation.get(a.key, 0.0) for a in ASSET_CLASSES], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Allocation must include at least one non-zero weight.")
    weights = weights / total

    rng = np.random.default_rng(seed)
    asset_returns = _draw_asset_returns(years, n_sims, rng)
    portfolio_returns = asset_returns @ weights  # (n_sims, years)

    balances = np.zeros((n_sims, years + 1))
    balances[:, 0] = net_worth

    current = np.full(n_sims, float(net_worth))
    for year in range(years):
        after_spending = np.maximum(current - annual_spending, 0.0)
        current = after_spending * (1.0 + portfolio_returns[:, year])
        current = np.maximum(current, 0.0)
        balances[:, year + 1] = current

    return SimulationResult(balances=balances, years=years, n_sims=n_sims)
