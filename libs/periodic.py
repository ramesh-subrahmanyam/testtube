"""
Periodic Update Wrapper

This module provides a general-purpose wrapper that caches and updates
the result of any callable on a periodic schedule (daily, weekly, monthly).

The Periodic wrapper can be used with any callable that takes time-indexed
data and returns time-indexed results - functions, callable classes, lambdas, etc.

This is similar to functools.partial in spirit - it wraps a callable and
modifies its behavior without changing its interface.
"""

import pandas as pd
import numpy as np
from datetime import datetime


class Periodic:
    """
    Wrap any callable to cache and update its result periodically.

    On update dates, the wrapped function is called and its result is cached.
    Between updates, the cached value is forward-filled from the last update.

    This is useful for:
    - Rebalancing exposure/position sizes on a calendar schedule
    - Recalculating expensive computations less frequently
    - Updating trading signals periodically instead of daily

    Example:
        # With a function
        def my_exposure(prices):
            return 100000 / prices['Close']

        monthly_exposure = Periodic(my_exposure, frequency="monthly", when="start")
        result = monthly_exposure(prices)

        # With a callable instance
        exp_mgr = VolatilityTargetExposure(100000)
        weekly_exp = Periodic(exp_mgr, frequency="weekly", day="monday")
        result = weekly_exp(prices)
    """

    def __init__(self, func, frequency="daily", when="end", day=None):
        """
        Initialize the Periodic wrapper.

        Args:
            func (callable): Callable to wrap. Must accept time-indexed data
                           and return time-indexed results.
            frequency (str): Update frequency - "daily", "weekly", "monthly"
                           Default: "daily"
            when (str): Update at period "start" or "end"
                       Default: "end"
            day (str, optional): For weekly frequency - day of week to update
                               ("monday", "tuesday", "wednesday", "thursday", "friday")
                               Ignored for daily/monthly frequencies

        Example:
            # Update monthly at start of month
            p = Periodic(my_func, frequency="monthly", when="start")

            # Update weekly on Mondays
            p = Periodic(my_func, frequency="weekly", day="monday")

            # Update daily at end of day (same as no wrapping)
            p = Periodic(my_func, frequency="daily", when="end")
        """
        self.func = func
        self.frequency = frequency.lower()
        self.when = when.lower()
        self.day = day.lower() if day else None

        # Validate inputs
        if self.frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError(f"frequency must be 'daily', 'weekly', or 'monthly', got '{frequency}'")

        if self.when not in ["start", "end"]:
            raise ValueError(f"when must be 'start' or 'end', got '{when}'")

        if self.frequency == "weekly" and self.day:
            valid_days = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if self.day not in valid_days:
                raise ValueError(f"day must be one of {valid_days}, got '{self.day}'")

    def __call__(self, prices):
        """
        Compute the wrapped function with periodic updates.

        Args:
            prices (pd.Series or pd.DataFrame): Time-indexed price data

        Returns:
            pd.Series: Result with same index as input, updated periodically

        Implementation:
            1. Call wrapped function to get full result series
            2. Determine update dates based on frequency/when/day
            3. Create output series that only updates on update dates
            4. Forward-fill between updates
        """
        # Compute full result from underlying function
        full_result = self.func(prices)

        # If daily frequency, just return the full result (no caching needed)
        if self.frequency == "daily":
            return full_result

        # Determine update dates
        update_dates = self._get_update_dates(full_result.index)

        # Create result series with periodic updates
        result = pd.Series(index=full_result.index, dtype=float)
        last_value = None

        for date in full_result.index:
            if date in update_dates:
                # Update on scheduled date
                last_value = full_result.loc[date]
            result.loc[date] = last_value

        return result

    def _get_update_dates(self, date_index):
        """
        Determine which dates trigger computation updates.

        Args:
            date_index (pd.DatetimeIndex): Time index of the data

        Returns:
            set: Set of dates on which to update the computation
        """
        if self.frequency == "daily":
            return set(date_index)
        elif self.frequency == "weekly":
            return self._weekly_update_dates(date_index)
        elif self.frequency == "monthly":
            return self._monthly_update_dates(date_index)

    def _weekly_update_dates(self, date_index):
        """
        Determine weekly update dates.

        Args:
            date_index (pd.DatetimeIndex): Time index of the data

        Returns:
            set: Dates falling on the specified day of week

        Implementation:
            - If day is specified: filter to that day of week
            - If day is None and when="start": filter to Mondays
            - If day is None and when="end": filter to Fridays
        """
        # Map day names to weekday numbers (0=Monday, 6=Sunday)
        day_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }

        if self.day:
            target_weekday = day_map[self.day]
        else:
            # Default: Monday for start, Friday for end
            target_weekday = 0 if self.when == "start" else 4

        # Filter dates to the target weekday
        update_dates = set()
        for date in date_index:
            if date.weekday() == target_weekday:
                update_dates.add(date)

        return update_dates

    def _monthly_update_dates(self, date_index):
        """
        Determine monthly update dates.

        Args:
            date_index (pd.DatetimeIndex): Time index of the data

        Returns:
            set: Dates at start or end of each month

        Implementation:
            - when="start": First trading day of each month
            - when="end": Last trading day of each month
        """
        update_dates = set()

        if self.when == "start":
            # Group by month and take first date in each month
            grouped = date_index.to_series().groupby([date_index.year, date_index.month])
            for (year, month), group in grouped:
                first_date = group.index[0]
                update_dates.add(first_date)
        else:  # when == "end"
            # Group by month and take last date in each month
            grouped = date_index.to_series().groupby([date_index.year, date_index.month])
            for (year, month), group in grouped:
                last_date = group.index[-1]
                update_dates.add(last_date)

        return update_dates

    def __repr__(self):
        """String representation of Periodic wrapper."""
        func_name = getattr(self.func, '__name__', repr(self.func))

        if self.frequency == "daily":
            return f"Periodic({func_name}, frequency='daily')"
        elif self.frequency == "weekly":
            day_str = f", day='{self.day}'" if self.day else ""
            return f"Periodic({func_name}, frequency='weekly'{day_str})"
        elif self.frequency == "monthly":
            return f"Periodic({func_name}, frequency='monthly', when='{self.when}')"


def periodic(func, frequency="daily", when="end", day=None):
    """
    Wrap any callable to update its result periodically.

    This is a convenience function that creates a Periodic wrapper.
    Works like functools.partial - returns a new callable with modified behavior.

    Args:
        func (callable): Callable to wrap
        frequency (str): Update frequency - "daily", "weekly", "monthly"
        when (str): Update at period "start" or "end"
        day (str, optional): For weekly - day of week

    Returns:
        Periodic: Wrapped callable with periodic update behavior

    Examples:
        # Monthly rebalancing
        monthly_exp = periodic(exposure_func, frequency="monthly", when="start")

        # Weekly recalculation on Mondays
        weekly_vol = periodic(volatility_func, frequency="weekly", day="monday")

        # Works with callable instances
        exp_mgr = VolatilityTargetExposure(100000)
        weekly_exp = periodic(exp_mgr, frequency="weekly", day="monday")

        # Works with lambdas
        const_exp = periodic(lambda p: 100000 / p['Close'], frequency="monthly")
    """
    return Periodic(func, frequency, when, day)


if __name__ == "__main__":
    # Example usage
    import sys
    from pathlib import Path

    # Add parent directory to path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    print("=" * 80)
    print("Periodic Wrapper Examples")
    print("=" * 80)

    # Create sample price data
    dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
    np.random.seed(42)

    # Simulate random walk prices
    returns = np.random.normal(0.0005, 0.02, len(dates))
    prices_close = 100 * np.exp(np.cumsum(returns))

    prices = pd.DataFrame({
        'Close': prices_close,
    }, index=dates)

    print("\nSample price data (first 10 days):")
    print(prices.head(10))

    # Example 1: Simple function - constant dollar exposure
    print("\n1. Constant Dollar Exposure (Daily vs Monthly)")
    print("-" * 80)

    def const_exposure(prices):
        """Simple constant dollar exposure."""
        return 100000 / prices['Close']

    # Daily updates (no wrapping needed)
    daily_exp = const_exposure(prices)

    # Monthly updates
    monthly_exp = periodic(const_exposure, frequency="monthly", when="start")
    monthly_result = monthly_exp(prices)

    comparison = pd.DataFrame({
        'Price': prices['Close'],
        'Daily_Shares': daily_exp,
        'Monthly_Shares': monthly_result,
    })

    print("\nFirst 15 days (note monthly updates on 1st of month):")
    print(comparison.head(15).to_string())

    # Example 2: Weekly updates on Mondays
    print("\n2. Weekly Updates on Mondays")
    print("-" * 80)

    weekly_exp = periodic(const_exposure, frequency="weekly", day="monday")
    weekly_result = weekly_exp(prices)

    # Show a week's worth
    jan_data = comparison.copy()
    jan_data['Weekly_Shares'] = weekly_result

    print("\nFirst 10 trading days in January:")
    print(jan_data.head(10).to_string())

    # Example 3: Using with a class instance
    print("\n3. Using Periodic with Callable Classes")
    print("-" * 80)

    class SimpleExposure:
        """Example callable class."""
        def __init__(self, target_dollars):
            self.target_dollars = target_dollars

        def __call__(self, prices):
            return self.target_dollars / prices['Close']

        def __repr__(self):
            return f"SimpleExposure({self.target_dollars})"

    exp_instance = SimpleExposure(250000)
    monthly_instance = periodic(exp_instance, frequency="monthly", when="end")

    print(f"Original: {exp_instance}")
    print(f"Wrapped: {monthly_instance}")

    result = monthly_instance(prices)
    print(f"\nResult on 2024-01-31 (month end): {result.loc['2024-01-31']:.2f} shares")
    print(f"Result on 2024-02-01 (next day, same): {result.loc['2024-02-01']:.2f} shares")

    # Example 4: Comparison of update frequencies
    print("\n4. Comparison of Different Update Frequencies")
    print("-" * 80)

    freq_comparison = pd.DataFrame({
        'Price': prices['Close'],
        'Daily': const_exposure(prices),
        'Weekly_Mon': periodic(const_exposure, frequency="weekly", day="monday")(prices),
        'Monthly_Start': periodic(const_exposure, frequency="monthly", when="start")(prices),
        'Monthly_End': periodic(const_exposure, frequency="monthly", when="end")(prices),
    })

    # Show first 20 days
    print("\nFirst 20 days (Jan 2024):")
    print(freq_comparison.head(20).to_string())

    # Count unique values (updates)
    print("\n\nNumber of unique values (updates) over full year:")
    for col in freq_comparison.columns:
        if col != 'Price':
            n_updates = freq_comparison[col].nunique()
            print(f"  {col}: {n_updates} updates")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("\nKey Features:")
    print("- Works with any callable (functions, classes, lambdas)")
    print("- Supports daily, weekly, monthly update frequencies")
    print("- Forward-fills between updates")
    print("- Similar to functools.partial in design philosophy")
    print("=" * 80)


class PeriodicStrategy:
    """
    Wrapper that applies periodic signal updates to a strategy.

    This wrapper causes a strategy to only update its positions on specific dates
    (e.g., monthly rebalancing) rather than daily. Between update dates, the
    position is held constant.
    """

    def __init__(self, strategy, frequency='daily', when='end', day=None):
        """
        Initialize PeriodicStrategy wrapper.

        Args:
            strategy: Base strategy to wrap
            frequency (str): Update frequency - "daily", "weekly", "monthly"
            when (str): Update at period "start" or "end"
            day (str, optional): For weekly - day of week
        """
        self.strategy = strategy
        self.frequency = frequency.lower()
        self.when = when.lower()
        self.day = day.lower() if day else None

        # Validate inputs
        if self.frequency not in ["daily", "weekly", "monthly"]:
            raise ValueError(f"frequency must be 'daily', 'weekly', or 'monthly', got '{frequency}'")

        if self.when not in ["start", "end"]:
            raise ValueError(f"when must be 'start' or 'end', got '{when}'")

    def __call__(self, symbol, start_date, end_date):
        """
        Run the wrapped strategy with periodic signal updates.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date

        Returns:
            pd.DataFrame: Strategy results with periodic updates
        """
        # Run the underlying strategy
        df = self.strategy(symbol, start_date, end_date)

        if df is None:
            return None

        # If daily frequency, just return as-is
        if self.frequency == "daily":
            return df

        # Create periodic wrapper for the position series
        periodic_wrapper = Periodic(
            func=lambda prices: df['Position_At_Close'],
            frequency=self.frequency,
            when=self.when,
            day=self.day
        )

        # Apply periodic updates to positions
        df['Position_At_Close'] = periodic_wrapper(df)

        return df
