"""Monte Carlo engine for retirement / portfolio depletion paths.

Model: annual timesteps, portfolio rebalanced to target weights each year,
spending withdrawn at the start of the year. Asset returns are drawn from
correlated lognormals calibrated to the arithmetic mean/volatility in
``assets.py``, so a path can lose value but never go below -100% in a year.

Optionally the shocks pass through a shared crisis regime: in a given year every
asset is hit by the same volatility multiplier, drawn as a two-state mixture.
That fattens each asset's tails AND makes them crash together, which independent
draws never do. Mean and volatility stay exactly on target either way -- see
``_lognormal_parameters``.
"""

from dataclasses import dataclass

import numpy as np

from .assets import ASSET_CLASSES, CORRELATION, mean_returns, volatilities

PERCENTILES = (10, 25, 50, 75, 90)

# Shared crisis regime. In CRISIS_PROBABILITY of years every asset's shock is
# scaled by CRISIS_SCALE, producing fat tails and joint crashes. A Student-t
# would be the textbook choice, but exp() of a t-distributed shock has no finite
# mean; a two-state scale mixture stays integrable and calibrates exactly.
CRISIS_PROBABILITY = 0.12
CRISIS_SCALE = 2.0


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

    def sample_paths(self, n: int = 100) -> np.ndarray:
        """``n`` individual paths, shape (n, years + 1), for plotting.

        Paths are i.i.d., so evenly spaced indices are an unbiased sample and
        keep the picture stable between reruns with the same seed.
        """
        if n >= self.n_sims:
            return self.balances
        index = np.linspace(0, self.n_sims - 1, n).round().astype(int)
        return self.balances[index]

    def depletion_year(self, percentile: float = 50.0) -> int | None:
        """Year the given percentile of paths has run dry, or None."""
        band = np.percentile(self.balances, percentile, axis=0)
        exhausted = np.flatnonzero(band <= 0)
        return int(exhausted[0]) if exhausted.size else None


def _mixture_moments(
    sigma: np.ndarray, crisis_prob: float, crisis_scale: float
) -> tuple[np.ndarray, np.ndarray]:
    """E[exp(sigma*S*Z)] and E[exp(sigma*S*Z)^2] for the scale mixture S."""
    states = np.array([1.0, crisis_scale])
    weights = np.array([1.0 - crisis_prob, crisis_prob])
    s2 = (states**2)[:, None] * sigma[None, :] ** 2
    first = (weights[:, None] * np.exp(0.5 * s2)).sum(axis=0)
    second = (weights[:, None] * np.exp(2.0 * s2)).sum(axis=0)
    return first, second


def _lognormal_parameters(
    mean: np.ndarray,
    vol: np.ndarray,
    crisis_prob: float = 0.0,
    crisis_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert arithmetic mean/stdev of returns to log-space mu/sigma.

    With no crisis regime this is the closed-form lognormal inversion. With one,
    the coefficient of variation is monotone in sigma, so bisection recovers the
    sigma that reproduces the target volatility; mu then follows exactly.
    """
    growth = 1.0 + mean
    if crisis_prob <= 0.0 or crisis_scale == 1.0:
        sigma_log = np.sqrt(np.log1p((vol / growth) ** 2))
        return np.log(growth) - 0.5 * sigma_log**2, sigma_log

    target_cv = vol / growth  # stdev / mean of the growth factor
    low = np.zeros_like(vol)
    high = np.full_like(vol, 5.0)
    for _ in range(200):
        mid = 0.5 * (low + high)
        first, second = _mixture_moments(mid, crisis_prob, crisis_scale)
        cv = np.sqrt(np.maximum(second - first**2, 0.0)) / first
        too_wide = cv > target_cv
        high = np.where(too_wide, mid, high)
        low = np.where(too_wide, low, mid)

    sigma_log = 0.5 * (low + high)
    first, _ = _mixture_moments(sigma_log, crisis_prob, crisis_scale)
    mu_log = np.log(growth) - np.log(first)
    return mu_log, sigma_log


def _draw_asset_returns(
    years: int,
    n_sims: int,
    rng: np.random.Generator,
    fat_tails: bool = True,
) -> np.ndarray:
    """Correlated annual real returns, shape (n_sims, years, n_assets)."""
    crisis_prob = CRISIS_PROBABILITY if fat_tails else 0.0
    crisis_scale = CRISIS_SCALE if fat_tails else 1.0
    mu_log, sigma_log = _lognormal_parameters(
        mean_returns(), volatilities(), crisis_prob, crisis_scale
    )
    chol = np.linalg.cholesky(CORRELATION)

    shocks = rng.standard_normal((n_sims, years, len(ASSET_CLASSES)))
    correlated = shocks @ chol.T

    if fat_tails:
        # One draw per year, shared across assets -- this is what makes them
        # crash together rather than each having its own bad year.
        crisis = rng.random((n_sims, years, 1)) < crisis_prob
        correlated = correlated * np.where(crisis, crisis_scale, 1.0)

    return np.exp(mu_log + sigma_log * correlated) - 1.0


def run_simulation(
    net_worth: float,
    annual_spending: float,
    allocation: dict[str, float],
    years: int = 30,
    n_sims: int = 10_000,
    seed: int | None = 42,
    fat_tails: bool = True,
) -> SimulationResult:
    """Project ``net_worth`` forward, spending ``annual_spending`` each year.

    ``allocation`` maps asset keys to percentages; it is normalised to sum to 1.
    ``fat_tails`` turns the shared crisis regime on or off.
    """
    weights = np.array([allocation.get(a.key, 0.0) for a in ASSET_CLASSES], dtype=float)
    total = weights.sum()
    if total <= 0:
        raise ValueError("Allocation must include at least one non-zero weight.")
    weights = weights / total

    rng = np.random.default_rng(seed)
    asset_returns = _draw_asset_returns(years, n_sims, rng, fat_tails)
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
