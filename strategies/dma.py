"""
DMA Strategy - Daily Moving Average Strategy

This strategy implements a simple long-only trend following approach:
- Position = 1 (long) if yesterday's price is above the N-day moving average
- Position = 0 (flat) if yesterday's price is below the N-day moving average
"""

import sys
from pathlib import Path

# Add parent directory to path to import libs
sys.path.insert(0, str(Path(__file__).parent.parent))

from strategies.base import BaseStrategy
from signals.technical import SMA
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


class DMA(BaseStrategy):
    """
    Daily Moving Average Strategy

    A simple trend-following strategy that goes long when the price is above
    its N-day moving average and goes flat when below.

    Parameters:
        lookback (int): Moving average lookback period (default 200)
    """

    def __init__(self, lookback=200, **kwargs):
        """
        Initialize the DMA strategy.

        Args:
            lookback (int): Lookback period for moving average calculation (default 200)
            **kwargs: Additional parameters passed to BaseStrategy
        """
        super().__init__(lookback=lookback, **kwargs)
        self.lookback = lookback

    def generate_signals(self, prices):
        """
        Generate trading signals based on N-day moving average.

        Logic:
        - If yesterday's close is above the N-day MA: Position = 1 (long)
        - If yesterday's close is below the N-day MA: Position = 0 (flat)

        Args:
            prices (pd.DataFrame): OHLCV data with DatetimeIndex

        Returns:
            pd.DataFrame: DataFrame with Close, DMA, Signal, Target_Position, Position_At_Close columns
        """
        df = prices.copy()

        # Calculate N-day moving average using SMA factory function
        sma_func = SMA(self.lookback)
        df[f'DMA'] = sma_func(self.symbol, df['Close'])

        # Generate signals based on yesterday's price vs N-day MA
        # We use yesterday's close to determine today's position
        df['Prev_Close'] = df['Close'].shift(1)

        # Real-valued signal: distance from MA as percentage
        df['Signal'] = (df['Prev_Close'] - df['DMA']) / df['DMA']

        # Convert to discrete positions first
        df['_temp_position'] = 0
        df.loc[df['Prev_Close'] > df['DMA'], '_temp_position'] = 1
        df.loc[df['Prev_Close'] <= df['DMA'], '_temp_position'] = 0

        # Only set Target_Position when it CHANGES from prior day
        df['Target_Position'] = pd.NA
        position_changes = df['_temp_position'] != df['_temp_position'].shift(1)
        df.loc[position_changes, 'Target_Position'] = df.loc[position_changes, '_temp_position']

        # Forward-fill to get position at close every day
        df['Position_At_Close'] = df['Target_Position'].ffill().fillna(0)

        # Drop temporary column
        df.drop(columns=['_temp_position'], inplace=True)

        # Default entry timing
        df['Entry_Time'] = 'close'
        df['Entry_Price'] = df['Close']

        # Return only the strategy period (after indicators are calculated)
        result = df.loc[self.start_date:self.end_date]

        logger.info(f"Generated {len(result)} signals for DMA strategy (lookback={self.lookback})")
        logger.info(f"Long positions: {(result['Position_At_Close'] == 1).sum()}")
        logger.info(f"Flat positions: {(result['Position_At_Close'] == 0).sum()}")

        return result


if __name__ == "__main__":
    # Example usage and testing
    import logging
    logging.basicConfig(level=logging.INFO)

    print("=" * 60)
    print("Testing DMA Strategy")
    print("=" * 60)

    # Create strategy instance
    strategy = DMA(lookback=200)

    # Run strategy on a symbol
    print("\nRunning DMA strategy (lookback=200) on AAPL from 2023-01-01 to 2024-12-31")
    print("-" * 60)

    signals = strategy('AAPL', '2023-01-01', '2024-12-31')

    if signals is not None:
        print(f"\nGenerated {len(signals)} trading days")
        print(f"\nStrategy Parameters: {strategy.get_parameter_summary()}")

        print("\nFirst 10 signals:")
        print(signals[['Close', 'DMA', 'Prev_Close', 'Signal', 'Position']].head(10))

        print("\nLast 10 signals:")
        print(signals[['Close', 'DMA', 'Prev_Close', 'Signal', 'Position']].tail(10))

        print(f"\nPosition distribution:")
        print(signals['Position'].value_counts())

        # Count transitions
        transitions = (signals['Position'] != signals['Position'].shift()).sum()
        print(f"\nNumber of position transitions: {transitions}")

        # Show some key statistics
        long_days = (signals['Position'] == 1).sum()
        flat_days = (signals['Position'] == 0).sum()
        total_days = len(signals)

        print(f"\nStrategy Statistics:")
        print(f"  Long days: {long_days} ({100*long_days/total_days:.1f}%)")
        print(f"  Flat days: {flat_days} ({100*flat_days/total_days:.1f}%)")
    else:
        print("Failed to generate signals")
