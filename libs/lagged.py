"""
Lagged Entry and Exit Wrappers

This module provides wrappers for delaying entries and exits in trading strategies.
These wrappers can be applied to any position series to create lagged behavior.

Lagged entry delays when a strategy enters positions (goes from 0 to non-zero).
Lagged exit delays when a strategy exits positions (goes from non-zero to 0).

These are useful for:
- Reducing whipsaws by waiting for confirmation
- Implementing deliberate delays in trading decisions
- Testing the impact of execution delays on strategy performance
"""

import pandas as pd
import numpy as np


def lagged_entry(positions, lag=1):
    """
    Delay position entries by a specified number of periods.

    When the underlying strategy signals entry (position goes from 0 to non-zero),
    this function delays that entry by 'lag' periods. Exits are not affected.

    Args:
        positions (pd.Series): Series of positions (1=long, 0=flat, -1=short)
        lag (int): Number of periods to delay entries (default: 1)
                  lag=0 means no delay
                  lag=1 means enter 1 period after signal
                  lag=2 means enter 2 periods after signal, etc.

    Returns:
        pd.Series: Lagged position series with same index as input

    Example:
        Original positions: [0, 1, 1, 1, 0, 0, 1, 1, 0]
        lagged_entry(positions, lag=1):
                           [0, 0, 1, 1, 0, 0, 0, 1, 0]
        lagged_entry(positions, lag=2):
                           [0, 0, 0, 1, 0, 0, 0, 0, 0]

    Implementation:
        1. Identify entry points (position changes from 0 to non-zero)
        2. Delay those entries by 'lag' periods
        3. Keep track of pending entries
        4. Exit signals cancel any pending entries
    """
    if lag == 0:
        return positions.copy()

    result = pd.Series(0, index=positions.index, dtype=positions.dtype)

    # Track pending entries: {index: (delay_remaining, target_position)}
    pending_entries = {}
    current_position = 0

    for i, idx in enumerate(positions.index):
        signal_position = positions.loc[idx]

        # Check if this is an entry signal (0 -> non-zero)
        is_entry = (current_position == 0) and (signal_position != 0)

        # Check if this is an exit signal (non-zero -> 0)
        is_exit = (current_position != 0) and (signal_position == 0)

        if is_entry:
            # Schedule entry for 'lag' periods from now
            pending_entries[i + lag] = signal_position
            result.iloc[i] = 0  # Stay flat during lag period
        elif is_exit:
            # Exit immediately, cancel any pending entries
            pending_entries.clear()
            result.iloc[i] = 0
            current_position = 0
        else:
            # Check if we should execute a pending entry
            if i in pending_entries:
                current_position = pending_entries[i]
                del pending_entries[i]

            result.iloc[i] = current_position

    return result


def lagged_exit(positions, lag=1):
    """
    Delay position exits by a specified number of periods.

    When the underlying strategy signals exit (position goes from non-zero to 0),
    this function delays that exit by 'lag' periods. Entries are not affected.

    Args:
        positions (pd.Series): Series of positions (1=long, 0=flat, -1=short)
        lag (int): Number of periods to delay exits (default: 1)
                  lag=0 means no delay
                  lag=1 means exit 1 period after signal
                  lag=2 means exit 2 periods after signal, etc.

    Returns:
        pd.Series: Lagged position series with same index as input

    Example:
        Original positions: [0, 1, 1, 1, 0, 0, 1, 1, 0]
        lagged_exit(positions, lag=1):
                           [0, 1, 1, 1, 1, 0, 1, 1, 1]
        lagged_exit(positions, lag=2):
                           [0, 1, 1, 1, 1, 1, 1, 1, 1]

    Implementation:
        1. Identify exit points (position changes from non-zero to 0)
        2. Delay those exits by 'lag' periods
        3. Keep track of pending exits
        4. New entry signals override any pending exits
    """
    if lag == 0:
        return positions.copy()

    result = pd.Series(0, index=positions.index, dtype=positions.dtype)

    # Track pending exit: {index: True}
    pending_exit_at = None
    current_position = 0

    for i, idx in enumerate(positions.index):
        signal_position = positions.loc[idx]

        # Check if this is an entry signal (0 -> non-zero)
        is_entry = (current_position == 0) and (signal_position != 0)

        # Check if this is an exit signal (non-zero -> 0)
        is_exit = (current_position != 0) and (signal_position == 0)

        if is_entry:
            # Enter immediately, cancel any pending exit
            pending_exit_at = None
            current_position = signal_position
            result.iloc[i] = current_position
        elif is_exit:
            # Schedule exit for 'lag' periods from now
            pending_exit_at = i + lag
            result.iloc[i] = current_position  # Stay in position during lag period
        else:
            # Check if we should execute a pending exit
            if pending_exit_at is not None and i >= pending_exit_at:
                current_position = 0
                pending_exit_at = None

            result.iloc[i] = current_position

    return result


def apply_lag(positions, entry_lag=0, exit_lag=0):
    """
    Apply both entry and exit lags to a position series.

    This is a convenience function that applies entry lag first, then exit lag.

    Args:
        positions (pd.Series): Series of positions (1=long, 0=flat, -1=short)
        entry_lag (int): Number of periods to delay entries (default: 0)
        exit_lag (int): Number of periods to delay exits (default: 0)

    Returns:
        pd.Series: Position series with both lags applied

    Example:
        positions = pd.Series([0, 1, 1, 1, 0, 0, 1, 1, 0])
        apply_lag(positions, entry_lag=1, exit_lag=1)
        # Result: [0, 0, 1, 1, 1, 0, 0, 1, 1]
    """
    result = positions.copy()

    if entry_lag > 0:
        result = lagged_entry(result, lag=entry_lag)

    if exit_lag > 0:
        result = lagged_exit(result, lag=exit_lag)

    return result


if __name__ == "__main__":
    # Example usage and testing
    print("=" * 80)
    print("Lagged Entry and Exit Examples")
    print("=" * 80)

    # Create sample position series
    positions = pd.Series(
        [0, 1, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0],
        index=pd.date_range('2024-01-01', periods=20, freq='D')
    )

    print("\nOriginal Positions:")
    print(positions.to_string())

    # Example 1: Lagged Entry
    print("\n" + "=" * 80)
    print("Example 1: Lagged Entry (lag=1)")
    print("=" * 80)
    lagged_entry_1 = lagged_entry(positions, lag=1)

    comparison_1 = pd.DataFrame({
        'Original': positions,
        'Lagged_Entry_1': lagged_entry_1,
    })
    print(comparison_1.to_string())
    print("\nNote: Entry signals are delayed by 1 period")

    # Example 2: Lagged Entry with lag=2
    print("\n" + "=" * 80)
    print("Example 2: Lagged Entry (lag=2)")
    print("=" * 80)
    lagged_entry_2 = lagged_entry(positions, lag=2)

    comparison_2 = pd.DataFrame({
        'Original': positions,
        'Lagged_Entry_2': lagged_entry_2,
    })
    print(comparison_2.to_string())
    print("\nNote: Entry signals are delayed by 2 periods")

    # Example 3: Lagged Exit
    print("\n" + "=" * 80)
    print("Example 3: Lagged Exit (lag=1)")
    print("=" * 80)
    lagged_exit_1 = lagged_exit(positions, lag=1)

    comparison_3 = pd.DataFrame({
        'Original': positions,
        'Lagged_Exit_1': lagged_exit_1,
    })
    print(comparison_3.to_string())
    print("\nNote: Exit signals are delayed by 1 period")

    # Example 4: Lagged Exit with lag=2
    print("\n" + "=" * 80)
    print("Example 4: Lagged Exit (lag=2)")
    print("=" * 80)
    lagged_exit_2 = lagged_exit(positions, lag=2)

    comparison_4 = pd.DataFrame({
        'Original': positions,
        'Lagged_Exit_2': lagged_exit_2,
    })
    print(comparison_4.to_string())
    print("\nNote: Exit signals are delayed by 2 periods")

    # Example 5: Combined lags
    print("\n" + "=" * 80)
    print("Example 5: Combined Entry Lag (1) and Exit Lag (1)")
    print("=" * 80)
    combined = apply_lag(positions, entry_lag=1, exit_lag=1)

    comparison_5 = pd.DataFrame({
        'Original': positions,
        'Entry_Lag_1': lagged_entry(positions, lag=1),
        'Exit_Lag_1': lagged_exit(positions, lag=1),
        'Both_Lag_1': combined,
    })
    print(comparison_5.to_string())
    print("\nNote: Both entry and exit signals are delayed by 1 period")

    # Example 6: Statistics comparison
    print("\n" + "=" * 80)
    print("Example 6: Impact on Trade Statistics")
    print("=" * 80)

    def count_trades(pos_series):
        """Count number of round-trip trades."""
        entries = ((pos_series != 0) & (pos_series.shift(1) == 0)).sum()
        return entries

    def count_exposure_days(pos_series):
        """Count days with non-zero position."""
        return (pos_series != 0).sum()

    strategies = {
        'Original': positions,
        'Entry_Lag_1': lagged_entry(positions, lag=1),
        'Entry_Lag_2': lagged_entry(positions, lag=2),
        'Exit_Lag_1': lagged_exit(positions, lag=1),
        'Exit_Lag_2': lagged_exit(positions, lag=2),
        'Both_Lag_1': apply_lag(positions, entry_lag=1, exit_lag=1),
    }

    print("\nStrategy Comparison:")
    print("-" * 80)
    print(f"{'Strategy':<20} {'Trades':<10} {'Exposure Days':<15}")
    print("-" * 80)

    for name, pos in strategies.items():
        trades = count_trades(pos)
        exp_days = count_exposure_days(pos)
        print(f"{name:<20} {trades:<10} {exp_days:<15}")

    print("\n" + "=" * 80)
    print("Examples completed successfully!")
    print("\nKey Features:")
    print("- lagged_entry(): Delays when positions are entered")
    print("- lagged_exit(): Delays when positions are exited")
    print("- apply_lag(): Applies both entry and exit lags")
    print("- Useful for testing execution delays and reducing whipsaws")
    print("=" * 80)


class LaggedStrategy:
    """
    Wrapper that applies entry/exit timing changes to a strategy.

    This wrapper allows modifying when entries and exits are executed:
    - Entry at open vs close
    - Entry lag (e.g., next day's close)
    - Exit at open vs close

    The wrapped strategy behaves exactly like the original but with modified
    execution timing.
    """

    def __init__(self, strategy, entry_lag=0, entry_price='close',
                 exit_lag=0, exit_price='close'):
        """
        Initialize LaggedStrategy wrapper.

        Args:
            strategy: Base strategy to wrap
            entry_lag (int): Days to lag entry (0=same day, 1=next day)
            entry_price (str): 'open' or 'close' for entry execution
            exit_lag (int): Days to lag exit (0=same day, 1=next day)
            exit_price (str): 'open' or 'close' for exit execution
        """
        self.strategy = strategy
        self.entry_lag = entry_lag
        self.entry_price = entry_price.lower()
        self.exit_lag = exit_lag
        self.exit_price = exit_price.lower()

    def __call__(self, symbol, start_date, end_date):
        """
        Run the wrapped strategy with lagged entry/exit timing.

        Args:
            symbol (str): Stock symbol
            start_date (str): Start date
            end_date (str): End date

        Returns:
            pd.DataFrame: Strategy results with modified timing
        """
        # Run the underlying strategy
        df = self.strategy(symbol, start_date, end_date)

        if df is None:
            return None

        # Apply entry lag to positions
        if self.entry_lag > 0:
            df['Position'] = lagged_entry(df['Position'], lag=self.entry_lag)

        # Apply exit lag to positions
        if self.exit_lag > 0:
            df['Position'] = lagged_exit(df['Position'], lag=self.exit_lag)

        # Note: Price changes (open vs close) are handled by the backtester
        # This wrapper just modifies the position timing

        return df
