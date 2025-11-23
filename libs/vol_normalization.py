"""
Volatility Normalization Module

This module provides the VolNormalization class for calculating volume multipliers
based on dollar volatility. It is used to normalize position sizes across different
volatility regimes.
"""

import pandas as pd
import numpy as np
from functools import partial
from .volatility import simple_vol


class VolNormalization:
    """
    Calculate volatility multipliers for position sizing.

    The VolNormalization class takes a price series and returns a vol_multiplier
    series that adjusts positions based on dollar volatility.

    Formula: vol_multiplier = total_dollar_size / dollar_volatility
    Where: dollar_volatility = (volatility / 100) × price

    Args:
        total_dollar_size (float): Target dollar size for position sizing
        volatility_function (callable, optional): Function to calculate volatility.
                                                   Default: simple_vol(N=20)

    Example:
        from functools import partial
        from libs.volatility import simple_vol, ewma_vol
        from libs.vol_normalization import VolNormalization

        # Default: 20-day simple volatility
        vol_norm = VolNormalization(total_dollar_size=100000)
        vol_multiplier = vol_norm(prices)

        # Custom: 30-day simple volatility
        vol_norm = VolNormalization(
            total_dollar_size=250000,
            volatility_function=partial(simple_vol, N=30)
        )
        vol_multiplier = vol_norm(prices)

        # EWMA volatility
        vol_norm = VolNormalization(
            total_dollar_size=100000,
            volatility_function=partial(ewma_vol, span=20)
        )
        vol_multiplier = vol_norm(prices)
    """

    def __init__(self, total_dollar_size, volatility_function=None):
        """
        Initialize VolNormalization.

        Args:
            total_dollar_size (float): Target dollar size for position sizing
            volatility_function (callable, optional): Function to calculate volatility.
                                                       Takes prices, returns volatility series.
                                                       Default: simple_vol(N=20)
        """
        self.total_dollar_size = total_dollar_size

        # Default to 20-day simple volatility if not provided
        if volatility_function is None:
            self.volatility_function = partial(simple_vol, N=20)
        else:
            self.volatility_function = volatility_function

    def __call__(self, prices):
        """
        Calculate vol_multiplier series from price series.

        Args:
            prices (pd.Series or pd.DataFrame): Price series or DataFrame with 'Close' column

        Returns:
            pd.Series: vol_multiplier = total_dollar_size / dollar_volatility

        Example:
            prices = fetch_stock_prices('AAPL', '2024-01-01', '2024-12-31')
            vol_norm = VolNormalization(total_dollar_size=100000)
            vol_multiplier = vol_norm(prices)

            # Use multiplier with positions
            sized_positions = raw_positions * vol_multiplier
        """
        # Extract Close prices if DataFrame
        if isinstance(prices, pd.DataFrame):
            if 'Close' not in prices.columns:
                raise ValueError("DataFrame must have 'Close' column")
            price_series = prices['Close']
        else:
            price_series = prices

        # Calculate volatility using the provided function
        volatility = self.volatility_function(prices)

        # Calculate dollar volatility
        # Dollar volatility = (return volatility / 100) × price
        # Volatility is in percentage form from volatility functions
        dollar_volatility = (volatility / 100) * price_series

        # Handle division by zero or NaN
        dollar_volatility = dollar_volatility.replace(0, np.nan)

        # Calculate vol multiplier
        vol_multiplier = self.total_dollar_size / dollar_volatility

        # Fill NaN with 0
        vol_multiplier = vol_multiplier.fillna(0)

        return vol_multiplier

    def __repr__(self):
        """String representation of VolNormalization."""
        return (f"VolNormalization(total_dollar_size=${self.total_dollar_size:,.2f}, "
                f"volatility_function={self.volatility_function.func.__name__})")


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from libs.volatility import simple_vol, ewma_vol, parkinson_vol

    print("=" * 80)
    print("VolNormalization Examples")
    print("=" * 80)

    # Create sample price data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    # Simulate random walk prices
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices_close = 100 * np.exp(np.cumsum(returns))

    # Create price DataFrame
    prices = pd.DataFrame({
        'Close': prices_close,
        'High': prices_close * (1 + np.abs(np.random.normal(0, 0.01, len(dates)))),
        'Low': prices_close * (1 - np.abs(np.random.normal(0, 0.01, len(dates)))),
    }, index=dates)

    print("\nSample price data:")
    print(prices.head())

    # Example 1: Default vol normalization (20-day simple vol)
    print("\n1. Default VolNormalization (20-day simple vol, $100k)")
    print("-" * 80)
    vol_norm = VolNormalization(total_dollar_size=100000)
    print(vol_norm)
    vol_multiplier = vol_norm(prices)
    print(f"\nLatest vol multiplier: {vol_multiplier.iloc[-1]:.2f}")
    print(f"Mean vol multiplier: {vol_multiplier.mean():.2f}")
    print(f"Std vol multiplier: {vol_multiplier.std():.2f}")

    # Example 2: Custom volatility function (30-day)
    print("\n2. Custom VolNormalization (30-day simple vol, $250k)")
    print("-" * 80)
    vol_norm_30 = VolNormalization(
        total_dollar_size=250000,
        volatility_function=partial(simple_vol, N=30)
    )
    print(vol_norm_30)
    vol_multiplier_30 = vol_norm_30(prices)
    print(f"\nLatest vol multiplier: {vol_multiplier_30.iloc[-1]:.2f}")
    print(f"Mean vol multiplier: {vol_multiplier_30.mean():.2f}")

    # Example 3: EWMA volatility
    print("\n3. EWMA VolNormalization (span=20, $100k)")
    print("-" * 80)
    vol_norm_ewma = VolNormalization(
        total_dollar_size=100000,
        volatility_function=partial(ewma_vol, span=20)
    )
    print(vol_norm_ewma)
    vol_multiplier_ewma = vol_norm_ewma(prices)
    print(f"\nLatest vol multiplier: {vol_multiplier_ewma.iloc[-1]:.2f}")
    print(f"Mean vol multiplier: {vol_multiplier_ewma.mean():.2f}")

    # Example 4: Parkinson volatility
    print("\n4. Parkinson VolNormalization (20-day, $100k)")
    print("-" * 80)
    vol_norm_park = VolNormalization(
        total_dollar_size=100000,
        volatility_function=partial(parkinson_vol, N=20)
    )
    print(vol_norm_park)
    vol_multiplier_park = vol_norm_park(prices)
    print(f"\nLatest vol multiplier: {vol_multiplier_park.iloc[-1]:.2f}")
    print(f"Mean vol multiplier: {vol_multiplier_park.mean():.2f}")

    # Example 5: Compare different vol normalizations
    print("\n5. Comparison of Different VolNormalization Methods")
    print("-" * 80)

    comparison = pd.DataFrame({
        'Simple_20': vol_multiplier,
        'Simple_30': vol_multiplier_30 * (100000 / 250000),  # Normalize to same dollar size
        'EWMA': vol_multiplier_ewma,
        'Parkinson': vol_multiplier_park,
    })

    print("\nLast 10 days:")
    print(comparison.tail(10).to_string())

    print("\nSummary statistics:")
    print(comparison.describe().to_string())

    # Example 6: Using with positions
    print("\n6. Using VolNormalization with Positions")
    print("-" * 80)

    # Simulate raw positions (e.g., from a strategy)
    raw_positions = pd.Series(1, index=dates)  # Simple buy-and-hold

    vol_norm = VolNormalization(total_dollar_size=100000)
    vol_multiplier = vol_norm(prices)

    # Calculate sized positions
    sized_positions = raw_positions * vol_multiplier

    print(f"\nRaw position: {raw_positions.iloc[-1]}")
    print(f"Vol multiplier: {vol_multiplier.iloc[-1]:.2f}")
    print(f"Sized position: {sized_positions.iloc[-1]:.2f}")

    # Calculate dollar exposure
    dollar_exposure = sized_positions * prices['Close']
    print(f"\nTarget dollar size: $100,000")
    print(f"Actual dollar exposure: ${dollar_exposure.iloc[-1]:,.2f}")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
