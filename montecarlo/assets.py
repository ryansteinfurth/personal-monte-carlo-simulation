"""Asset class definitions and capital market assumptions.

Source: J.P. Morgan Long-Term Capital Market Assumptions, 2026 edition
(``ltcma-2026-us-matrix_usd.pdf``, USD assumptions, data as of 30 Sep 2025).

Two conversions stand between the published sheet and this model, so the
published figures are stored verbatim and converted in one place:

1. LTCMA returns are NOMINAL; this model is REAL, so returns are deflated by
   the LTCMA U.S. inflation assumption.
2. LTCMA quotes COMPOUND (geometric) returns; the engine draws from arithmetic
   means, which are higher by roughly half the variance.

       real_compound = (1 + nominal_compound) / (1 + inflation) - 1
       arithmetic    = real_compound + volatility**2 / 2

Volatilities are used as published -- the real-vs-nominal difference is minor
next to the uncertainty in the estimates themselves.

To update next year: replace the compound_return/volatility pairs and the
correlations with the new edition's numbers. Nothing else needs to change.
"""

from dataclasses import dataclass

import numpy as np

# LTCMA 2026 "U.S. Inflation" compound assumption.
INFLATION = 0.0250


@dataclass(frozen=True)
class AssetClass:
    key: str
    name: str
    ltcma_row: str  # the line item this maps to on the LTCMA matrix
    compound_return: float  # nominal compound return, as published
    volatility: float  # annualized volatility, as published

    @property
    def mean_return(self) -> float:
        """Arithmetic annual REAL return -- what the simulation draws from."""
        real_compound = (1.0 + self.compound_return) / (1.0 + INFLATION) - 1.0
        return real_compound + self.volatility**2 / 2.0


ASSET_CLASSES: tuple[AssetClass, ...] = (
    AssetClass("us_stocks", "US Stocks", "U.S. Large Cap", 0.0670, 0.1647),
    AssetClass("intl_stocks", "International Stocks", "EAFE Equity", 0.0750, 0.1763),
    AssetClass("bonds", "Bonds", "U.S. Aggregate Bonds", 0.0480, 0.0476),
    AssetClass("real_estate", "Real Estate", "U.S. Core Real Estate", 0.0820, 0.1139),
    AssetClass("cash", "Cash", "U.S. Cash", 0.0310, 0.0067),
)

# Correlations between the rows above, read off the same LTCMA matrix.
CORRELATION = np.array(
    [
        # US    INTL   BOND   RE     CASH
        [1.00, 0.87, 0.30, 0.35, 0.01],  # US Stocks
        [0.87, 1.00, 0.34, 0.26, 0.05],  # International Stocks
        [0.30, 0.34, 1.00, -0.15, 0.10],  # Bonds
        [0.35, 0.26, -0.15, 1.00, -0.12],  # Real Estate
        [0.01, 0.05, 0.10, -0.12, 1.00],  # Cash
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
