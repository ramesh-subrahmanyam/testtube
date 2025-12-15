"""
Exposure Management Module

This module provides classes for managing position exposure sizing.
Exposure managers compute a multiplier that is applied to raw strategy positions
to determine the actual number of shares to hold.

The module includes:
- ExposureManager: Abstract base class
- VolatilityTargetExposure: Size positions to target constant dollar volatility
- ConstantDollarExposure: Size positions to target constant dollar exposure
- Periodic wrapper support via .periodic() method

All exposure managers return a multiplier series that, when multiplied by
raw positions, gives the number of shares to hold.
"""

import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from functools import partial
from .volatility import simple_vol
from .periodic import periodic


class ExposureManager(ABC):
    """
    Abstract base class for exposure management strategies.

    An exposure manager computes a multiplier series from price data.
    This multiplier is applied to raw strategy positions (e.g., 1, 0, -1)
    to determine the actual number of shares to hold.

    Formula:
        actual_shares = raw_position × multiplier

    Where:
        actual_shares: Number of shares to hold
        raw_position: Strategy signal (typically 1=long, 0=flat, -1=short)
        multiplier: Computed by the exposure manager

    Subclasses must implement:
        __call__(prices): Returns multiplier series
    """

    @abstractmethod
    def __call__(self, prices):
        """
        Calculate exposure multiplier from price data.

        Args:
            prices (pd.Series or pd.DataFrame): Price data

        Returns:
            pd.Series: Multiplier series with same index as prices
        """
        pass

    def periodic(self, frequency="daily", when="end", day=None):
        """
        Wrap this exposure manager with periodic updates.

        This is a convenience method that wraps the exposure manager
        with the Periodic wrapper from libs.periodic.

        Args:
            frequency (str): Update frequency - "daily", "weekly", "monthly"
            when (str): Update at period "start" or "end"
            day (str, optional): For weekly - day of week

        Returns:
            Periodic: Wrapped exposure manager with periodic updates

        Example:
            # Monthly rebalancing at start of month
            exp = VolatilityTargetExposure(100000).periodic(
                frequency="monthly",
                when="start"
            )

            # Weekly rebalancing on Mondays
            exp = ConstantDollarExposure(100000).periodic(
                frequency="weekly",
                day="monday"
            )
        """
        return periodic(self, frequency=frequency, when=when, day=day)


class VolatilityTargetExposure(ExposureManager):
    """
    Size positions to target constant dollar volatility.

    This exposure manager adjusts position sizes so that the dollar volatility
    of the position remains approximately constant over time.

    Formula:
        multiplier = total_dollar_size / dollar_volatility
        dollar_volatility = (volatility / 100) × price

    Where:
        total_dollar_size: Target dollar volatility (e.g., $100,000)
        volatility: Return volatility in % (e.g., 20 for 20% annualized)
        price: Current stock price

    This results in larger positions (more shares) when volatility is low,
    and smaller positions (fewer shares) when volatility is high.

    Example:
        from functools import partial
        from libs.volatility import simple_vol, ewma_vol
        from libs.exposure_management import VolatilityTargetExposure

        # Default: 20-day simple volatility
        exp = VolatilityTargetExposure(total_dollar_size=100000)
        multiplier = exp(prices)

        # Custom: 30-day simple volatility
        exp = VolatilityTargetExposure(
            total_dollar_size=250000,
            volatility_function=partial(simple_vol, N=30)
        )
        multiplier = exp(prices)

        # EWMA volatility
        exp = VolatilityTargetExposure(
            total_dollar_size=100000,
            volatility_function=partial(ewma_vol, span=20)
        )
        multiplier = exp(prices)

        # Monthly rebalancing
        exp = VolatilityTargetExposure(100000).periodic(
            frequency="monthly",
            when="start"
        )
    """

    name = "VolatilityTargetExposure"

    def __init__(self, total_dollar_size, volatility_function=None):
        """
        Initialize VolatilityTargetExposure.

        Args:
            total_dollar_size (float): Target dollar volatility for position sizing
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
        Calculate volatility-targeted multiplier from price series.

        Args:
            prices (pd.Series or pd.DataFrame): Price series or DataFrame with 'Close' column

        Returns:
            pd.Series: multiplier = total_dollar_size / dollar_volatility

        Example:
            prices = fetch_stock_prices('AAPL', '2024-01-01', '2024-12-31')
            exp = VolatilityTargetExposure(total_dollar_size=100000)
            multiplier = exp(prices)

            # Use multiplier with positions
            actual_shares = raw_positions * multiplier
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

        # Calculate multiplier
        multiplier = self.total_dollar_size / dollar_volatility

        # Fill NaN with 0
        multiplier = multiplier.fillna(0)

        return multiplier

    def __repr__(self):
        """String representation of VolatilityTargetExposure."""
        vol_func_name = getattr(self.volatility_function.func, '__name__', 'custom')
        return (f"VolatilityTargetExposure(total_dollar_size=${self.total_dollar_size:,.2f}, "
                f"volatility_function={vol_func_name})")


class ConstantDollarExposure(ExposureManager):
    """
    Size positions to target constant dollar exposure.

    This exposure manager adjusts position sizes so that the dollar value
    of the position remains approximately constant over time, regardless
    of volatility.

    Formula:
        multiplier = target_dollar_exposure / price

    Where:
        target_dollar_exposure: Target dollar value of position (e.g., $100,000)
        price: Current stock price

    This results in fewer shares when price is high, and more shares when
    price is low, maintaining approximately constant dollar exposure.

    Example:
        from libs.exposure_management import ConstantDollarExposure

        # Target $100k exposure
        exp = ConstantDollarExposure(target_dollar_exposure=100000)
        multiplier = exp(prices)

        # With raw position of 1 (long), actual_shares adjusts to maintain
        # ~$100k exposure as price changes

        # Monthly rebalancing
        exp = ConstantDollarExposure(100000).periodic(
            frequency="monthly",
            when="start"
        )

        # Weekly rebalancing on Fridays
        exp = ConstantDollarExposure(50000).periodic(
            frequency="weekly",
            day="friday"
        )
    """

    name = "ConstantDollarExposure"

    def __init__(self, target_dollar_exposure):
        """
        Initialize ConstantDollarExposure.

        Args:
            target_dollar_exposure (float): Target dollar exposure for positions
        """
        self.target_dollar_exposure = target_dollar_exposure

    def __call__(self, prices):
        """
        Calculate constant-dollar multiplier from price series.

        Args:
            prices (pd.Series or pd.DataFrame): Price series or DataFrame with 'Close' column

        Returns:
            pd.Series: multiplier = target_dollar_exposure / price

        Example:
            prices = fetch_stock_prices('AAPL', '2024-01-01', '2024-12-31')
            exp = ConstantDollarExposure(target_dollar_exposure=100000)
            multiplier = exp(prices)

            # Use multiplier with positions
            actual_shares = raw_positions * multiplier

            # Dollar exposure verification
            dollar_exposure = actual_shares * prices['Close']
            # dollar_exposure will be approximately constant at $100k
        """
        # Extract Close prices if DataFrame
        if isinstance(prices, pd.DataFrame):
            if 'Close' not in prices.columns:
                raise ValueError("DataFrame must have 'Close' column")
            price_series = prices['Close']
        else:
            price_series = prices

        # Handle division by zero or NaN
        price_series_clean = price_series.replace(0, np.nan)

        # Calculate multiplier
        multiplier = self.target_dollar_exposure / price_series_clean

        # Fill NaN with 0
        multiplier = multiplier.fillna(0)

        return multiplier

    def __repr__(self):
        """String representation of ConstantDollarExposure."""
        return f"ConstantDollarExposure(target_dollar_exposure=${self.target_dollar_exposure:,.2f})"


# Backward compatibility alias (old name -> new name)
# This allows existing code using "VolNormalization" to continue working
VolNormalization = VolatilityTargetExposure


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from libs.volatility import simple_vol, ewma_vol

    print("=" * 80)
    print("Exposure Management Examples")
    print("=" * 80)

    # Create sample price data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    # Simulate random walk prices with varying volatility
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

    # Example 1: VolatilityTargetExposure (default)
    print("\n1. Volatility Target Exposure (20-day simple vol, $100k)")
    print("-" * 80)
    vol_exp = VolatilityTargetExposure(total_dollar_size=100000)
    print(vol_exp)
    multiplier = vol_exp(prices)
    print(f"\nLatest multiplier: {multiplier.iloc[-1]:.2f} shares")
    print(f"Mean multiplier: {multiplier.mean():.2f} shares")
    print(f"Std multiplier: {multiplier.std():.2f} shares")

    # Example 2: ConstantDollarExposure
    print("\n2. Constant Dollar Exposure ($100k)")
    print("-" * 80)
    const_exp = ConstantDollarExposure(target_dollar_exposure=100000)
    print(const_exp)
    const_multiplier = const_exp(prices)
    print(f"\nLatest multiplier: {const_multiplier.iloc[-1]:.2f} shares")
    print(f"Mean multiplier: {const_multiplier.mean():.2f} shares")

    # Verify constant dollar exposure
    raw_position = 1  # Long position
    actual_shares = raw_position * const_multiplier
    dollar_exposure = actual_shares * prices['Close']
    print(f"\nDollar exposure verification:")
    print(f"  Target: $100,000")
    print(f"  Actual (latest): ${dollar_exposure.iloc[-1]:,.2f}")
    print(f"  Actual (mean): ${dollar_exposure.mean():,.2f}")

    # Example 3: Monthly rebalancing with VolatilityTargetExposure
    print("\n3. Volatility Target with Monthly Rebalancing")
    print("-" * 80)
    monthly_vol_exp = VolatilityTargetExposure(100000).periodic(
        frequency="monthly",
        when="start"
    )
    print(monthly_vol_exp)
    monthly_multiplier = monthly_vol_exp(prices)

    # Compare daily vs monthly
    comparison = pd.DataFrame({
        'Price': prices['Close'],
        'Daily_Multiplier': multiplier,
        'Monthly_Multiplier': monthly_multiplier,
    })
    print("\nFirst 15 days (monthly updates at month start):")
    print(comparison.head(15).to_string())

    # Example 4: Weekly rebalancing with ConstantDollarExposure
    print("\n4. Constant Dollar with Weekly Rebalancing (Mondays)")
    print("-" * 80)
    weekly_const_exp = ConstantDollarExposure(100000).periodic(
        frequency="weekly",
        day="monday"
    )
    print(weekly_const_exp)
    weekly_const_multiplier = weekly_const_exp(prices)

    weekly_comparison = pd.DataFrame({
        'Price': prices['Close'],
        'Daily': const_multiplier,
        'Weekly_Mon': weekly_const_multiplier,
    })
    print("\nFirst 10 trading days:")
    print(weekly_comparison.head(10).to_string())

    # Example 5: Compare different exposure strategies
    print("\n5. Comparison: Vol Target vs Constant Dollar")
    print("-" * 80)

    # Calculate volatility for context
    vol_series = simple_vol(prices, N=20)

    strategy_comparison = pd.DataFrame({
        'Price': prices['Close'],
        'Volatility': vol_series,
        'Vol_Target_Shares': multiplier,
        'Const_Dollar_Shares': const_multiplier,
        'Vol_Target_$': multiplier * prices['Close'],
        'Const_Dollar_$': const_multiplier * prices['Close'],
    })

    print("\nLast 10 days:")
    print(strategy_comparison.tail(10).to_string())

    print("\nSummary statistics:")
    print(strategy_comparison[['Vol_Target_Shares', 'Const_Dollar_Shares',
                               'Vol_Target_$', 'Const_Dollar_$']].describe().to_string())

    # Example 6: Using with raw positions
    print("\n6. Using Exposure Managers with Strategy Positions")
    print("-" * 80)

    # Simulate raw positions from a strategy (e.g., simple trend following)
    raw_positions = pd.Series(index=dates, dtype=int)
    raw_positions[:] = 1  # Simple buy-and-hold for demonstration

    # Apply different exposure managers
    vol_target_shares = raw_positions * multiplier
    const_dollar_shares = raw_positions * const_multiplier

    position_comparison = pd.DataFrame({
        'Raw_Position': raw_positions,
        'Vol_Target_Shares': vol_target_shares,
        'Const_Dollar_Shares': const_dollar_shares,
        'Vol_Target_Exposure': vol_target_shares * prices['Close'],
        'Const_Dollar_Exposure': const_dollar_shares * prices['Close'],
    })

    print("\nLast 5 days:")
    print(position_comparison.tail(5).to_string())

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("\nKey Concepts:")
    print("- VolatilityTargetExposure: Sizes positions for constant dollar volatility")
    print("- ConstantDollarExposure: Sizes positions for constant dollar exposure")
    print("- .periodic(): Wraps any exposure manager for calendar-based rebalancing")
    print("- Multiplier × Raw Position = Actual Shares to hold")
    print("=" * 80)


# Exposure Manager Registry
EXPOSURE_MANAGER_REGISTRY = {
    'VolatilityTargetExposure': VolatilityTargetExposure,
    'ConstantDollarExposure': ConstantDollarExposure,
}


def get_exposure_manager_class(name):
    """
    Get exposure manager class by name.

    Args:
        name (str): Exposure manager name (e.g., 'ConstantDollarExposure', 'VolatilityTargetExposure')

    Returns:
        class: Exposure manager class

    Raises:
        ValueError: If exposure manager name not found
    """
    if name not in EXPOSURE_MANAGER_REGISTRY:
        available = ', '.join(EXPOSURE_MANAGER_REGISTRY.keys())
        raise ValueError(f"Exposure manager '{name}' not found. Available: {available}")
    return EXPOSURE_MANAGER_REGISTRY[name]
