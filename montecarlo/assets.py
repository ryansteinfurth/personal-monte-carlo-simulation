"""Asset class definitions and capital market assumptions.

All returns are REAL (inflation-adjusted), expressed as arithmetic annual means.
Tune these in one place -- the simulation reads whatever is defined here.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AssetClass:
    key: str
    name: str
    mean_return: float  # arithmetic annual real return
    volatility: float  # annual standard deviation


ASSET_CLASSES: tuple[AssetClass, ...] = (
    AssetClass("us_stocks", "US Stocks", 0.065, 0.17),
    AssetClass("intl_stocks", "International Stocks", 0.060, 0.19),
    AssetClass("bonds", "Bonds", 0.015, 0.06),
    AssetClass("real_estate", "Real Estate", 0.050, 0.18),
    AssetClass("cash", "Cash", 0.005, 0.02),
)

# Correlation between asset classes, in the order listed above.
CORRELATION = np.array(
    [
        # US    INTL  BOND  RE    CASH
        [1.00, 0.85, 0.10, 0.60, 0.00],  # US Stocks
        [0.85, 1.00, 0.10, 0.55, 0.00],  # International Stocks
        [0.10, 0.10, 1.00, 0.15, 0.20],  # Bonds
        [0.60, 0.55, 0.15, 1.00, 0.00],  # Real Estate
        [0.00, 0.00, 0.20, 0.00, 1.00],  # Cash
    ]
)

DEFAULT_ALLOCATION = {
    "us_stocks": 50.0,
    "intl_stocks": 20.0,
    "bonds": 20.0,
    "real_estate": 5.0,
    "cash": 5.0,
}


def mean_returns() -> np.ndarray:
    return np.array([a.mean_return for a in ASSET_CLASSES])


def volatilities() -> np.ndarray:
    return np.array([a.volatility for a in ASSET_CLASSES])


def covariance() -> np.ndarray:
    """Covariance matrix implied by the volatilities and correlation matrix."""
    sigma = volatilities()
    return CORRELATION * np.outer(sigma, sigma)
